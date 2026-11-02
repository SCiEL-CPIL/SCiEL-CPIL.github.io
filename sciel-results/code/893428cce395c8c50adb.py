"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # Strategy:
    # 1. Start with baseline rank-53 decomposition (7 Strassen blocks + 1 schoolbook)
    # 2. Use non-archimedean valuation heuristic to search for sparser patterns
    #    by assigning distinct powers of epsilon to each column and looking for
    #    coefficient matching that reveals global linear dependencies
    
    # Build baseline rank-53 decomposition
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53

    # Strassen coefficients for M_{2,2,2}
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    V_school = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    
    W_school = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    col_offset = 0
    for block in range(7):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        k0 = 2 * block
        k1 = 2 * block + 1
        
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        
        col_offset += 7
    
    k_last = 14
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    else:
        for r in range(4):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_school[i * 2 + j][r]
            
            for j in range(2):
                V[j * p + k_last][c] = V_school[j][r]
            
            for i in range(2):
                W[k_last * n + i][c] = W_school[i][r]
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Verify baseline is correct
    def verify_decomposition(U_m, V_m, W_m):
        """Verify that the decomposition exactly represents matrix multiplication."""
        # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # should equal delta[i,k] * delta[j,j] for matmul
        # Actually for matmul: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] should be 1 if we're computing A[i,j]*B[j,k]
        # The decomposition means: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # for all i,j,k
        
        n_val, m_val, p_val = n, m, p
        # Check all (i,j,k) triples
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    u_idx = i * m_val + j
                    v_idx = j * p_val + k
                    w_idx = k * n_val + i
                    s = np.dot(U_m[u_idx], V_m[v_idx] * W_m[w_idx])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify_decomposition(U_np, V_np, W_np):
        # Baseline is wrong, return it anyway (shouldn't happen)
        return {"U": U, "V": V, "W": W}
    
    # Non-archimedean valuation heuristic:
    # Assign each column k of B a distinct power of epsilon: eps^k
    # This transforms the problem into matching coefficients in a formal power series
    # We look for sparse patterns where rank-1 terms have support across multiple columns
    
    # The key insight: if we can find a decomposition where some rank-1 terms
    # have V entries that are non-zero in multiple columns with specific power patterns,
    # we may be able to merge terms and reduce rank.
    
    # Try to find a lower-rank decomposition by searching over sparse patterns
    # We'll use a greedy approach: try to remove columns from the decomposition
    # and see if we can compensate with modified U, V, W entries
    
    # First, let's try a different block decomposition:
    # Instead of 7 blocks of 2x2 + 1 block of 2x1, try other groupings
    
    # Try: 5 blocks of 2x3 (if we can find rank-10 for 2x3) + 1 block of 2x1
    # Or: 3 blocks of 2x5 (if we can find rank-15 for 2x5)
    
    # Let's try to find a rank-10 decomposition for M_{2,2,3}
    # This would give: 5*10 + 4 = 54 (worse than 53)
    # Or rank-9: 5*9 + 4 = 49 (better!)
    
    # Search for M_{2,2,3} rank-9 decomposition
    def find_m223_rank9():
        """Try to find a rank-9 decomposition for 2x2 by 2x3 matrix multiplication."""
        n_val, m_val, p_val = 2, 2, 3
        rank_val = 9
        
        # Initialize with random values
        np.random.seed(42)
        U_try = np.random.randn(n_val * m_val, rank_val)
        V_try = np.random.randn(m_val * p_val, rank_val)
        W_try = np.random.randn(p_val * n_val, rank_val)
        
        # Gradient descent to minimize the error
        lr = 0.01
        for iteration in range(5000):
            if time.time() > deadline - 20:
                return None
            
            # Compute gradients
            U_grad = np.zeros_like(U_try)
            V_grad = np.zeros_like(V_try)
            W_grad = np.zeros_like(W_try)
            
            for i in range(n_val):
                for j in range(m_val):
                    for k in range(p_val):
                        u_idx = i * m_val + j
                        v_idx = j * p_val + k
                        w_idx = k * n_val + i
                        
                        # Error: dot(U[u_idx], V[v_idx] * W[w_idx]) - 1
                        vw = V_try[v_idx] * W_try[w_idx]
                        err = np.dot(U_try[u_idx], vw) - 1.0
                        
                        U_grad[u_idx] += err * vw
                        V_grad[v_idx] += err * U_try[u_idx] * W_try[w_idx]
                        W_grad[w_idx] += err * U_try[u_idx] * V_try[v_idx]
            
            # Update
            U_try -= lr * U_grad
            V_try -= lr * V_grad
            W_try -= lr * W_grad
            
            # Check convergence
            if iteration % 500 == 0:
                max_err = 0
                for i in range(n_val):
                    for j in range(m_val):
                        for k in range(p_val):
                            u_idx = i * m_val + j
                            v_idx = j * p_val + k
                            w_idx = k * n_val + i
                            s = np.dot(U_try[u_idx], V_try[v_idx] * W_try[w_idx])
                            max_err = max(max_err, abs(s - 1.0))
                if max_err < 1e-8:
                    # Round to nearest integers/half-integers
                    U_rounded = np.round(U_try * 2) / 2
                    V_rounded = np.round(V_try * 2) / 2
                    W_rounded = np.round(W_try * 2) / 2
                    
                    # Verify rounded version
                    max_err_rounded = 0
                    for i in range(n_val):
                        for j in range(m_val):
                            for k in range(p_val):
                                u_idx = i * m_val + j
                                v_idx = j * p_val + k
                                w_idx = k * n_val + i
                                s = np.dot(U_rounded[u_idx], V_rounded[v_idx] * W_rounded[w_idx])
                                max_err_rounded = max(max_err_rounded, abs(s - 1.0))
                    
                    if max_err_rounded < 1e-10:
                        return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try to find M_{2,2,3} rank-9
    m223_result = find_m223_rank9()
    
    if m223_result is not None and time.time() < deadline - 5:
        U_m223, V_m223, W_m223 = m223_result
        new_rank = 5 * 9 + 4  # 5 blocks of 2x3 + 1 block of 2x1
        if new_rank < rank:
            # Build the new decomposition
            U_new = [[0] * new_rank for _ in range(n * m)]
            V_new = [[0] * new_rank for _ in range(m * p)]
            W_new = [[0] * new_rank for _ in range(p * n)]
            
            col_offset = 0
            for block in range(5):
                k0 = 3 * block
                k1 = k0 + 1
                k2 = k0 + 2
                
                for r in range(9):
                    c = col_offset + r
                    for i in range(2):
                        for j in range(2):
                            U_new[i * m + j][c] = float(U_m223[i * 2 + j, r])
                    
                    for j in range(2):
                        V_new[j * p + k0][c] = float(V_m223[j * 3 + 0, r])
                        V_new[j * p + k1][c] = float(V_m223[j * 3 + 1, r])
                        V_new[j * p + k2][c] = float(V_m223[j * 3 + 2, r])
                    
                    for k in range(3):
                        kk = k0 + k
                        for i in range(2):
                            W_new[kk * n + i][c] = float(W_m223[k * 2 + i, r])
                
                col_offset += 9
            
            # Last block: 2x1
            k_last = 15
            for r in range(4):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U_new[i * m + j][c] = float(U_school[i * 2 + j][r])
                
                for j in range(2):
                    V_new[j * p + k_last][c] = float(V_school[j][r])
                
                for i in range(2):
                    W_new[k_last * n + i][c] = float(W_school[i][r])
            
            # Verify
            U_new_np = np.array(U_new, dtype=np.float64)
            V_new_np = np.array(V_new, dtype=np.float64)
            W_new_np = np.array(W_new, dtype=np.float64)
            
            if verify_decomposition(U_new_np, V_new_np, W_new_np):
                return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try another approach: look for linear dependencies in the baseline decomposition
    # to merge terms
    # Check if any columns of U, V, W are linearly dependent
    
    # Try to find a rank-8 decomposition for M_{2,2,2} (unlikely, but worth trying)
    # Or look for cross-block dependencies
    
    # For now, return the baseline if no improvement found
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
