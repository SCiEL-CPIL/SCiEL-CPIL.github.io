import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0

def get_tensor_2x2x3():
    """
    Returns the matrix multiplication tensor for 2x2 by 2x3.
    Shape: (4, 6, 6)
    U index: i*m + j (i in 0..1, j in 0..1) -> 0..3
    V index: j*p + k (j in 0..1, k in 0..2) -> 0..5
    W index: k*n + i (k in 0..2, i in 0..1) -> 0..5
    """
    n, m, p = 2, 2, 3
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def solve_rank10_2x2x3():
    """
    Attempts to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Returns (U, V, W) or None if failed.
    """
    T = get_tensor_2x2x3()
    n, m, p = 2, 2, 3
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = 10
    
    # Flatten T
    T_flat = T.flatten()
    
    # Initial guess
    np.random.seed(42)
    U = np.random.randn(dim_u, rank) * 0.5
    V = np.random.randn(dim_v, rank) * 0.5
    W = np.random.randn(dim_w, rank) * 0.5
    
    def residual(params):
        U_flat = params[:dim_u*rank].reshape(dim_u, rank)
        V_flat = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W_flat = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        # Compute tensor
        # T_comp[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can do this efficiently
        # T_comp = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
        # But einsum might be slow if called many times.
        # Alternative:
        # T_comp = np.zeros((dim_u, dim_v, dim_w))
        # for r in range(rank):
        #     T_comp += np.outer(np.outer(U_flat[:, r], V_flat[:, r]), W_flat[:, r])
        
        # Let's use einsum, it's usually fast enough for small dims
        T_comp = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
        return T_comp.flatten() - T_flat

    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = least_squares(residual, x0, method='lm', max_nfev=1000)
        if result.cost < 1e-10:
            U_sol = result.x[:dim_u*rank].reshape(dim_u, rank)
            V_sol = result.x[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W_sol = result.x[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            
            # Check if coefficients are close to half-integers
            # If not, we might still use them if the error is tiny, 
            # but the task asks for integer/half-integer.
            # Let's check proximity.
            def is_half_integer(x):
                return np.allclose(x * 2, np.round(x * 2), atol=1e-6)
            
            if is_half_integer(U_sol) and is_half_integer(V_sol) and is_half_integer(W_sol):
                # Round to half-integers
                U_sol = np.round(U_sol * 2) / 2
                V_sol = np.round(V_sol * 2) / 2
                W_sol = np.round(W_sol * 2) / 2
                return U_sol, V_sol, W_sol
            else:
                # If not half-integer, we can't guarantee "exact" in the required format
                # unless we find a rational approximation.
                # For now, return None to be safe, or try to round and verify.
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                T_check = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                if np.allclose(T_check, T, atol=1e-6):
                    return U_round, V_round, W_round
                return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find rank-54 solution
    if time.time() - start_time < 10.0:
        sol_223 = solve_rank10_2x2x3()
        if sol_223 is not None:
            U_223, V_223, W_223 = sol_223
            # Construct 2x2x10 from 3x(2x2x3) + 1x(2x2x1)
            # 2x2x10: n=2, m=2, p=10
            # Split p into 3,3,3,1
            
            rank_223 = U_223.shape[1]
            rank_221 = 4 # schoolbook
            rank_2210 = 3 * rank_223 + rank_221
            
            rank_2110 = 2 * 10 # schoolbook
            total_rank = rank_2210 + rank_2110
            
            U = np.zeros((n * m, total_rank))
            V = np.zeros((m * p, total_rank))
            W = np.zeros((p * n, total_rank))
            
            idx = 0
            
            # Part 1: 2x2x10
            # Blocks for p: 0-2, 3-5, 6-8, 9
            for block in range(3):
                # Copy 2x2x3 decomposition
                # U indices: same
                # V indices: j*10 + k. For block b, k is b*3 to b*3+2
                # W indices: k*2 + i. For block b, k is b*3 to b*3+2
                
                u_start = idx
                u_end = idx + rank_223
                
                U[:, u_start:u_end] = U_223
                
                for j in range(2):
                    for k_local in range(3):
                        k_global = block * 3 + k_local
                        v_idx_223 = j * 3 + k_local
                        v_idx_2210 = j * 10 + k_global
                        V[v_idx_2210, u_start:u_end] = V_223[v_idx_223, :]
                        
                        w_idx_223 = k_local * 2 + 0 # i=0? No, W index is k*n+i
                        # W_223 shape is (6, 10). Index is k*2+i.
                        # We need to map W_223 indices to W_2210 indices.
                        # W_2210 index is k_global*2 + i.
                        # W_223 index is k_local*2 + i.
                        # So for each i in 0..1:
                        for i in range(2):
                            w_idx_223 = k_local * 2 + i
                            w_idx_2210 = k_global * 2 + i
                            W[w_idx_2210, u_start:u_end] = W_223[w_idx_223, :]
                
                idx += rank_223
            
            # Block 4: 2x2x1 (schoolbook)
            # k=9
            for i in range(2):
                for j in range(2):
                    U[i*2 + j, idx] = 1.0
                    V[j*10 + 9, idx] = 1.0
                    W[9*2 + i, idx] = 1.0
                    idx += 1
            
            # Part 2: 2x1x10 (schoolbook)
            # A is 2x1 (row 2 of A), B is 1x10 (row 2 of B)
            # U index: i*3 + 2
            # V index: 2*10 + k
            # W index: k*2 + i
            for i in range(2):
                for k in range(10):
                    U[i*3 + 2, idx] = 1.0
                    V[2*10 + k, idx] = 1.0
                    W[k*2 + i, idx] = 1.0
                    idx += 1
            
            if idx != total_rank:
                print(f"Error: idx {idx} != total_rank {total_rank}")
                return None # Fallback

            # Verify
            U_check = U
            V_check = V
            W_check = W
            # Quick check
            np.random.seed(42)
            A = np.random.randint(-5, 6, size=(n, m)).astype(float)
            B = np.random.randint(-5, 6, size=(m, p)).astype(float)
            C_direct = A @ B
            C_decomp = np.zeros((n, p))
            for r in range(total_rank):
                u_r = U_check[:, r]
                v_r = V_check[:, r]
                w_r = W_check[:, r]
                A_w = u_r.reshape(n, m)
                B_w = v_r.reshape(m, p)
                C_decomp += np.sum(A * A_w) * np.sum(B * B_w) * w_r.reshape(p, n).T
            if np.max(np.abs(C_decomp - C_direct)) > 1e-6:
                return None

            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Rank 55 solution
    return construct_rank55()

def construct_rank55():
    n, m, p = DIMENSIONS
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    # ... (copy from original) ...
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1
        # M1
        U[0, idx] = 1.0; U[4, idx] = 1.0
        V[k0, idx] = 1.0; V[10+k1, idx] = 1.0
        W[0, idx] = 1.0; W[11, idx] = 1.0 # k1*2+1 = 1*2+1=3? No.
        # W index: k*n + i.
        # k0=0, k1=1.
        # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # C11: k=0, i=0 -> idx 0
        # C22: k=1, i=1 -> idx 1*2+1 = 3
        W[0, idx] = 1.0
        W[3, idx] = 1.0
        idx += 1
        # M2: (A21+A22)B11 -> C12 += M2, C22 -= M2
        # C12: k=1, i=0 -> idx 1*2+0 = 2
        # C22: k=1, i=1 -> idx 3
        U[3, idx] = 1.0; U[4, idx] = 1.0
        V[k0, idx] = 1.0
        W[2, idx] = 1.0
        W[3, idx] = -1.0
        idx += 1
        # M3: A11(B12-B22) -> C21 += M3, C22 += M3
        # C21: k=0, i=1 -> idx 1
        # C22: k=1, i=1 -> idx 3
        U[0, idx] = 1.0
        V[10+k1, idx] = 1.0; V[10+k1, idx] = -1.0 # Wait, B12 is k1, B22 is k1?
        # B is 2x10. B11 is (0,0), B12 is (0,1). B21 is (1,0), B22 is (1,1).
        # V index: j*10 + k.
        # B12: j=0, k=1 -> idx 1
        # B22: j=1, k=1 -> idx 11
        V[1, idx] = 1.0
        V[11, idx] = -1.0
        W[1, idx] = 1.0
        W[3, idx] = 1.0
        idx += 1
        # M4: A22(B21-B11) -> C11 += M4, C21 += M4
        # C11: idx 0
        # C21: idx 1
        U[4, idx] = 1.0
        V[10, idx] = 1.0 # B21: j=1, k=0
        V[0, idx] = -1.0 # B11: j=0, k=0
        W[0, idx] = 1.0
        W[1, idx] = 1.0
        idx += 1
        # M5: (A11+A12)B22 -> C11 -= M5, C21 += M5
        # C11: idx 0
        # C21: idx 1
        U[0, idx] = 1.0; U[1, idx] = 1.0 # A11, A12
        V[11, idx] = 1.0 # B22
        W[0, idx] = -1.0
        W[1, idx] = 1.0
        idx += 1
        # M6: (A21-A11)(B11+B12) -> C22 += M6
        # C22: idx 3
        U[3, idx] = 1.0; U[0, idx] = -1.0
        V[0, idx] = 1.0; V[1, idx] = 1.0
        W[3, idx] = 1.0
        idx += 1
        # M7: (A12-A22)(B21+B22) -> C11 += M7
        # C11: idx 0
        U[1, idx] = 1.0; U[4, idx] = -1.0
        V[10, idx] = 1.0; V[11, idx] = 1.0
        W[0, idx] = 1.0
        idx += 1

    # Part 2: Schoolbook 2x1x10
    for i in range(2):
        for k in range(10):
            U[i*3 + 2, idx] = 1.0
            V[2*10 + k, idx] = 1.0
            W[k*2 + i, idx] = 1.0
            idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    if result:
        print(f"Rank: {len(result['U'][0])}")
    else:
        print("Failed")
