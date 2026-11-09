"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time
_start_time = time.time()
TIME_LIMIT = 140.0  # 150s limit, 10s safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (n*m, R) = (6, R)
    # V shape: (m*p, R) = (18, R)  
    # W shape: (p*n, R) = (12, R)
    
    def check_time():
        return (time.time() - _start_time) > TIME_LIMIT
    
    best_result = None
    best_rank = float('inf')
    
    def make_vec(size, idx_vals):
        vec = [0] * size
        for idx, val in idx_vals.items():
            vec[idx] = val
        return vec
    
    def build_from_terms(terms):
        rank = len(terms)
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for c, (u,v,w) in enumerate(terms):
            for i in range(n*m):
                U[i][c] = u[i]
            for i in range(m*p):
                V[i][c] = v[i]
            for i in range(p*n):
                W[i][c] = w[i]
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Construction 1: Trivial rank 36
    def trivial_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Construction 2: Strassen on 2x2 block (rows 0-1 of B) + scalar terms for row 2
    def strassen_based_decomposition():
        terms = []
        # 6 scalar terms for j=2 (third row of B, which is 1x6)
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                terms.append((make_vec(n*m, {u_idx: 1}),
                             make_vec(m*p, {v_idx: 1}),
                             make_vec(p*n, {w_idx: 1})))
        
        # 7 Strassen terms for the 2x2 block: A[0:2,0:2] * B[0:2,0:6] -> C[0:2,0:6]
        # B[0:2,0:6] is a 2x6 matrix, split into three 2x2 blocks
        # For each 2x2 output block, apply Strassen to A[0:2,0:2] * B_sub[0:2,0:2]
        # But we need to handle the full 2x6, so we process each pair of k-columns
        
        # Actually, let's think differently. The tensor is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # For i in {0,1}, k in {0,...,5}, j in {0,1,2}
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        
        # The j=2 part is already handled (6 terms).
        # For j in {0,1}: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # This is a 2x2 * 2x6 multiplication, which can be done with rank 7 per 2x2 output block.
        # But 2x6 output has 3 pairs of k-values, so we need 3 * 7 = 21? No...
        
        # Actually Strassen works on square matrices. For 2x2 * 2x2, rank is 7.
        # For 2x2 * 2x6, we can split the 2x6 into three 2x2 blocks and apply Strassen to each.
        # That gives 3 * 7 = 21 terms for the j={0,1} part.
        # Total: 6 + 21 = 27.
        
        # Let me implement this properly:
        for k_block in range(3):
            # B_sub = B[0:2, 2*k_block:2*k_block+2], shape 2x2
            # A_sub = A[0:2, 0:2], shape 2x2
            # C_sub = C[0:2, 2*k_block:2*k_block+2], shape 2x2
            
            b00 = 0 * p + 2*k_block      # B[0, 2*k_block]
            b01 = 0 * p + 2*k_block + 1  # B[0, 2*k_block+1]
            b10 = 1 * p + 2*k_block      # B[1, 2*k_block]
            b11 = 1 * p + 2*k_block + 1  # B[1, 2*k_block+1]
            
            c00 = (2*k_block) * n + 0    # C[0, 2*k_block]
            c01 = (2*k_block + 1) * n + 0 # C[0, 2*k_block+1]
            c10 = (2*k_block) * n + 1    # C[1, 2*k_block]
            c11 = (2*k_block + 1) * n + 1 # C[1, 2*k_block+1]
            
            a00 = 0 * m + 0  # A[0,0]
            a01 = 0 * m + 1  # A[0,1]
            a10 = 1 * m + 0  # A[1,0]
            a11 = 1 * m + 1  # A[1,1]
            
            # Strassen's 7 products for 2x2 * 2x2:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            terms.append((make_vec(6, {a00:1, a11:1}),
                         make_vec(18, {b00:1, b11:1}),
                         make_vec(12, {c00:1, c11:1})))
            terms.append((make_vec(6, {a10:1, a11:1}),
                         make_vec(18, {b00:1}),
                         make_vec(12, {c10:1, c11:-1})))
            terms.append((make_vec(6, {a00:1}),
                         make_vec(18, {b01:1, b11:-1}),
                         make_vec(12, {c01:1, c11:1})))
            terms.append((make_vec(6, {a11:1}),
                         make_vec(18, {b10:1, b00:-1}),
                         make_vec(12, {c00:1, c10:1})))
            terms.append((make_vec(6, {a00:1, a01:1}),
                         make_vec(18, {b11:1}),
                         make_vec(12, {c00:-1, c01:1})))
            terms.append((make_vec(6, {a10:1, a00:-1}),
                         make_vec(18, {b00:1, b01:1}),
                         make_vec(12, {c11:1})))
            terms.append((make_vec(6, {a01:1, a11:-1}),
                         make_vec(18, {b10:1, b11:1}),
                         make_vec(12, {c00:1})))
        
        return build_from_terms(terms)
    
    # Construction 3: Fast structured search for lower rank using linear algebra
    def try_optimization_search():
        """Use linear algebra to find lower-rank decompositions"""
        import numpy as np
        
        # The tensor T[i,j,k] = 1 for all i,j,k
        # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # For fixed j, this becomes: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is a bilinear form in (U[:,j], V[:,j], W[:,j])
        
        # Key insight: if we fix U and V, the equations in W are linear.
        # Similarly, we can use alternating minimization.
        
        def verify_np(U, V, W):
            """Fast verification using numpy"""
            R = U.shape[1]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        if abs(s - 1) > 1e-9:
                            return False
            return True
        
        # Try to find a rank-20 solution using alternating minimization
        # Start from the Strassen construction and try to reduce rank
        
        for target_rank in range(min(best_rank - 1, 26), 19, -1):
            if check_time():
                break
            
            for restart in range(5):
                if check_time():
                    break
                
                np.random.seed(target_rank * 1000 + restart)
                
                # Initialize with Strassen structure + random
                U = np.zeros((6, target_rank))
                V = np.zeros((18, target_rank))
                W = np.zeros((12, target_rank))
                
                # First 6 columns: j=2 scalar terms
                for i in range(n):
                    for k in range(p):
                        col = i * p + k
                        if col < target_rank:
                            U[i*m+2, col] = 1
                            V[2*p+k, col] = 1
                            W[k*n+i, col] = 1
                
                # Remaining columns: Strassen for j={0,1}
                strassen_patterns = [
                    ({0:1, 3:1}, {0:1, 11:1}, {0:1, 11:1}),
                    ({3:1, 4:1}, {0:1}, {4:1, 11:-1}),
                    ({0:1}, {1:1, 11:-1}, {1:1, 11:1}),
                    ({3:1}, {10:1, 0:-1}, {0:1, 4:1}),
                    ({0:1, 1:1}, {11:1}, {0:-1, 1:1}),
                    ({4:1, 0:-1}, {0:1, 1:1}, {11:1}),
                    ({1:1, 3:-1}, {10:1, 11:1}, {0:1}),
                ]
                
                col = 6
                for k_block in range(3):
                    for (u_pat, v_pat, w_pat) in strassen_patterns:
                        if col >= target_rank:
                            break
                        for idx, val in u_pat.items():
                            U[idx, col] = val
                        for idx, val in v_pat.items():
                            V[idx, col] = val
                        for idx, val in w_pat.items():
                            W[idx, col] = val
                        col += 1
                
                # Add small perturbation to remaining columns
                for c in range(col, target_rank):
                    U[:, c] = 0.1 * np.random.randn(6)
                    V[:, c] = 0.1 * np.random.randn(18)
                    W[:, c] = 0.1 * np.random.randn(12)
                
                # Alternating minimization: fix two, solve for third
                # This is complex, so just do a few gradient steps
                for iteration in range(50):
                    if check_time():
                        break
                    
                    # Compute residual
                    residual = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                                residual[i,j,k] = s - 1
                    
                    if np.linalg.norm(residual) < 1e-10:
                        break
                    
                    # Gradient descent step (simplified)
                    lr = 0.01
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                grad_U = residual[i,j,k] * np.outer(V[j*p+k] * W[k*n+i], np.ones(target_rank))
                                grad_U = (V[j*p+k] * W[k*n+i])
                                U[i*m+j] -= lr * grad_U * residual[i,j,k]
                    
                    # Check if we converged
                    residual2 = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                                residual2[i,j,k] = s - 1
                    
                    if np.linalg.norm(residual2) < 1e-10:
                        break
                
                # Try rounding
                for scale in [1, 2, 4]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    
                    if verify_np(U_r, V_r, W_r):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Try integer rounding
                U_r = np.round(U)
                V_r = np.round(V)
                W_r = np.round(W)
                if verify_np(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        return None
    
    # Try constructions
    for name, decomp_func in [("trivial", trivial_decomposition), 
                               ("strassen", strassen_based_decomposition)]:
        if check_time():
            break
        try:
            result = decomp_func()
            rank = len(result["U"][0])
            if verify_decomposition(result["U"], result["V"], result["W"]):
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        except Exception:
            pass
    
    # Try optimization-based search only if we haven't found a good solution
    if not check_time() and best_rank > 20:
        try:
            result = try_optimization_search()
            if result is not None:
                rank = len(result["U"][0])
                if verify_decomposition(result["U"], result["V"], result["W"]):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = result
        except Exception:
            pass
    
    if best_result is None:
        best_result = trivial_decomposition()
        best_rank = len(best_result["U"][0])
    
    return best_result
# EVOLVE-BLOCK-END
