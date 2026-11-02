import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Helper to create the target tensor for n x m x p
    def create_target_tensor(n, m, p):
        # Tensor shape: (n*m, m*p, p*n)
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[u_idx, v_idx, w_idx] = 1.0
        return T

    # Helper to check if a decomposition is valid
    def check_decomposition(U, V, W, T, tol=1e-9):
        # Compute the decomposed tensor
        # T_decomp[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a tensor contraction.
        # We can use einsum
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_decomp, T, atol=tol)

    # Helper to round to half-integers
    def round_to_half_int(M):
        return np.round(M * 2) / 2

    # Try to find a decomposition for 3x3x2
    def find_decomposition_332(rank, max_iter=5000):
        n, m, p = 3, 3, 2
        T = create_target_tensor(n, m, p)
        
        # Flatten T for optimization? No, we optimize U, V, W directly.
        # Variables: U (9xR), V (6xR), W (6xR)
        
        best_error = float('inf')
        best_params = None
        
        # Multi-start
        for restart in range(10):
            if time.time() > deadline - 20: break
            
            np.random.seed(restart)
            U0 = np.random.randn(9, rank)
            V0 = np.random.randn(6, rank)
            W0 = np.random.randn(6, rank)
            
            # Scale initialization
            U0 *= 0.5
            V0 *= 0.5
            W0 *= 0.5
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(params):
                U = params[:9*rank].reshape(9, rank)
                V = params[9*rank:9*rank+6*rank].reshape(6, rank)
                W = params[9*rank+6*rank:].reshape(6, rank)
                T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
                return np.sum((T_decomp - T)**2)
            
            from scipy.optimize import minimize
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
            if best_error < 1e-9:
                break
        
        if best_error < 1e-9:
            U = best_params[:9*rank].reshape(9, rank)
            V = best_params[9*rank:9*rank+6*rank].reshape(6, rank)
            W = best_params[9*rank+6*rank:].reshape(6, rank)
            
            # Round to half-integers
            U_r = round_to_half_int(U)
            V_r = round_to_half_int(V)
            W_r = round_to_half_int(W)
            
            if check_decomposition(U_r, V_r, W_r, T):
                return U_r, V_r, W_r
        
        return None

    # Try ranks 5, 6, 7 for 3x3x2
    decomp_332 = None
    for r in [5, 6, 7, 8]:
        if time.time() > deadline - 30: break
        decomp_332 = find_decomposition_332(r)
        if decomp_332:
            print(f"Found 3x3x2 decomposition with rank {r}")
            break
    else:
        print("Failed to find 3x3x2 decomposition, using schoolbook")
        # Fallback to schoolbook for 3x3x8 directly
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    U332, V332, W332 = decomp_332
    r332 = U332.shape[1]
    
    # Construct 3x3x8 from 3x3x2
    # 3x3x8 is 4 blocks of 3x3x2? No, p=8 is 4 blocks of p=2.
    # So we need 4 copies of the 3x3x2 decomposition.
    # Total rank = 4 * r332.
    
    R_total = 4 * r332
    
    U = np.zeros((n * m, R_total))
    V = np.zeros((m * p, R_total))
    W = np.zeros((p * n, R_total))
    
    for block in range(4):
        # Block b corresponds to columns 2*b to 2*b+1 of B and C
        col_start = 2 * block
        
        # U is the same for all blocks (since A is the same)
        # But we need to place it in the correct columns of U
        U[:, block*r332:(block+1)*r332] = U332
        
        # V needs to be shifted
        # V332 is 6xR (rows are j*p+k for p=2, so j*2+k)
        # We need V for p=8, so rows are j*8+k.
        # For block b, k in B is 2*b + k_local.
        # So row index in V_full is j*8 + (2*b + k_local).
        for j in range(m):
            for k_local in range(2):
                v_row_full = j * p + (col_start + k_local)
                v_row_332 = j * 2 + k_local
                V[v_row_full, block*r332:(block+1)*r332] = V332[v_row_332, :]
                
        # W needs to be shifted
        # W332 is 6xR (rows are k*n+i for p=2, so k*3+i)
        # We need W for p=8, so rows are k*3+i.
        # Wait, W index is k*n+i.
        # In 3x3x2, k is 0 or 1.
        # In 3x3x8, k is 0..7.
        # For block b, k in C is 2*b + k_local?
        # No, C has columns k=0..7.
        # The block b corresponds to C columns 2*b and 2*b+1.
        # So k_full = 2*b + k_local.
        for k_local in range(2):
            w_row_full = (col_start + k_local) * n
            w_row_332 = k_local * n
            for i in range(n):
                W[w_row_full + i, block*r332:(block+1)*r332] = W332[w_row_332 + i, :]

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
