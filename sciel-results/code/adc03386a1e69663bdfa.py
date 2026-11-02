import time
import numpy as np
import scipy.optimize

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Test with random matrices
    for trial in range(5):
        A = np.random.randint(-3, 4, size=(n, m)).astype(float)
        B = np.random.randint(-3, 4, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _find_2x2x3_rank10():
    """
    Find a rank-10 decomposition for 2x2x3 matrix multiplication.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Target tensor M[i, j, k]
    # M[i, j, k] = 1 for all i, j, k in 2x2x3?
    # No, the tensor is defined by the multiplication structure.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The decomposition is sum_r U[i, m+j, r] V[j, p+k, r] W[k, n+i, r]
    # Wait, indices in the problem description:
    # U: (n*m, rank) -> U[i*m + j, r]
    # V: (m*p, rank) -> V[j*p + k, r]
    # W: (p*n, rank) -> W[k*n + i, r]
    
    # The tensor entry T[i, j, k] corresponding to U[i*m+j], V[j*p+k], W[k*n+i]
    # must be 1 if the product A[i,j]B[j,k] contributes to C[i,k].
    # In matrix multiplication, every product A[i,j]B[j,k] contributes to C[i,k].
    # So T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    # Let's verify this interpretation.
    # If T[i,j,k] = 1, then sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
    
    target = np.ones((n, m, p))
    
    deadline = time.time() + 60
    
    best_error = float('inf')
    best_params = None
    
    # We will search for U, V, W.
    # To reduce search space, we can fix the first row of U to be 1s?
    # No, let's just search all.
    
    # Use a local search strategy:
    # 1. Random init with small integers/half-integers.
    # 2. Refine using Nelder-Mead.
    # 3. Round and verify.
    
    for trial in range(50):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Values likely to be small: -1.5 to 1.5
        init_scale = 1.0
        U_init = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], size=(n*m, rank))
        V_init = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], size=(m*p, rank))
        W_init = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], size=(p*n, rank))
        
        # Flatten parameters
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def error_func(x):
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Compute tensor
            tensor = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        tensor[i, j, k] = s
            return np.sum((tensor - target) ** 2)
        
        # Optimize
        try:
            result = scipy.optimize.minimize(error_func, x0, method='Nelder-Mead', 
                                             options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_error:
                best_error = result.fun
                best_params = result.x.copy()
                
                if best_error < 1e-6:
                    # Extract and round
                    U_opt = best_params[:n*m*rank].reshape(n*m, rank)
                    V_opt = best_params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = best_params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    tensor_check = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for r in range(rank):
                                    s += U_round[i*m+j, r] * V_round[j*p+k, r] * W_round[k*n+i, r]
                                tensor_check[i, j, k] = s
                                
                    if np.allclose(tensor_check, target, atol=1e-9):
                        return U_round.tolist(), V_round.tolist(), W_round.tolist()
                        
        except Exception:
            continue
            
    return None

def _construct_strassen_53(n, m, p):
    """Fallback: Strassen blocks + schoolbook, rank 53."""
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        for idx, val in u_coeffs:
            U[idx][col] += val
        for idx, val in v_coeffs:
            V[idx][col] += val
        for idx, val in w_coeffs:
            W[idx][col] += val

    col = 0
    # 7 blocks of 2x2x2 Strassen
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for block b
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> offset by j*p
        # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> offset by k*n
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        add_term([(0, 1), (3, 1)], [(k0, 1), (p + k1, 1)], [(k0 * n + 0, 1), (k1 * n + 1, 1)], col)
        col += 1
        
        # Term 2: (A10+A11)B00 -> C10, C11 (with -)
        add_term([(2, 1), (3, 1)], [(k0, 1)], [(k0 * n + 1, 1), (k1 * n + 1, -1)], col)
        col += 1
        
        # Term 3: A00(B01-B11) -> C01, C11 (with -)
        add_term([(0, 1)], [(k1, 1), (p + k1, -1)], [(k1 * n + 0, 1), (k1 * n + 1, 1)], col)
        col += 1
        
        # Term 4: A11(B10-B00) -> C10, C00 (with -)
        add_term([(3, 1)], [(p + k0, 1), (k0, -1)], [(k0 * n + 0, 1), (k0 * n + 1, 1)], col)
        col += 1
        
        # Term 5: (A00+A01)B11 -> C00, C01
        add_term([(0, 1), (1, 1)], [(p + k1, 1)], [(k0 * n + 0, -1), (k1 * n + 0, 1)], col)
        col += 1
        
        # Term 6: (A10-A00)(B00+B01) -> C01, C11 (with -)
        add_term([(2, 1), (0, -1)], [(k0, 1), (k1, 1)], [(k1 * n + 1, 1)], col)
        col += 1
        
        # Term 7: (A01-A11)(B10+B11) -> C00
        add_term([(1, 1), (3, -1)], [(p + k0, 1), (p + k1, 1)], [(k0 * n + 0, 1)], col)
        col += 1
        
    # Last column (k=14)
    k_last = 14
    # C01 = A00 B01 + A01 B11 (using B col 14, which is B[0,14] and B[1,14])
    # Wait, the last block is 2x1? No, we did 7 blocks of 2x2, covering cols 0..13.
    # Col 14 is left.
    # C[0, 14] = A[0,0]B[0,14] + A[0,1]B[1,14]
    # C[1, 14] = A[1,0]B[0,14] + A[1,1]B[1,14]
    # 4 terms.
    
    add_term([(0, 1)], [(k_last, 1)], [(k_last * n + 0, 1)], col)
    col += 1
    add_term([(1, 1)], [(p + k_last, 1)], [(k_last * n + 0, 1)], col)
    col += 1
    add_term([(2, 1)], [(k_last, 1)], [(k_last * n + 1, 1)], col)
    col += 1
    add_term([(3, 1)], [(p + k_last, 1)], [(k_last * n + 1, 1)], col)
    col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Try to find 2x2x3 rank-10 decomposition
    block_decomp = _find_2x2x3_rank10()
    
    if block_decomp is not None:
        U_block, V_block, W_block = block_decomp
        block_rank = 10
        num_blocks = p // 3  # 5 blocks
        rank = num_blocks * block_rank
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for b in range(num_blocks):
            k0 = 3 * b
            col_offset = b * block_rank
            
            for r in range(block_rank):
                # U: same for each block
                for i in range(n * m):
                    U[i][col_offset + r] += U_block[i][r]
                
                # V: shift k indices
                for j in range(m):
                    for k in range(3):
                        v_idx = j * p + (k0 + k)
                        v_block_idx = j * 3 + k
                        V[v_idx][col_offset + r] += V_block[v_block_idx][r]
                
                # W: shift k indices
                for k in range(3):
                    for i in range(n):
                        w_idx = (k0 + k) * n + i
                        w_block_idx = k * n + i
                        W[w_idx][col_offset + r] += W_block[w_block_idx][r]
        
        # Verify
        if time.time() < deadline - 5:
            if not _verify_decomposition(U, V, W, n, m, p):
                # Fallback to Strassen
                return _construct_strassen_53(n, m, p)
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to Strassen rank 53
        return _construct_strassen_53(n, m, p)

# EVOLVE-BLOCK-END
