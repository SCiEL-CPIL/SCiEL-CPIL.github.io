"""Matrix multiplication 2x4 x 4x4 tensor decomposition - attempting rank 27."""

import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # 1. Build the rank-28 Strassen decomposition as a baseline
    def build_strassen_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        # Strassen 2x2 terms
        # Local indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
        # M1 = (x00 + x11)(y00 + y11)
        # M2 = (x10 + x11)y00
        # M3 = x00(y01 - y11)
        # M4 = x11(y10 - y00)
        # M5 = (x00 + x01)y11
        # M6 = (x10 - x00)(y00 + y01)
        # M7 = (x01 - x11)(y10 + y11)
        
        # P00 = M1 + M4 - M5 + M7
        # P01 = M3 + M5
        # P10 = M2 + M4
        # P11 = M1 - M2 + M3 + M6
        
        # We have 4 blocks:
        # 1. A_L (cols 0,1) * B_TL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
        # 2. A_R (cols 2,3) * B_BL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
        # 3. A_L (cols 0,1) * B_TR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
        # 4. A_R (cols 2,3) * B_BR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
        
        def add_strassen_block(a_side, b_block, c_side, start_col):
            # a_side: 'L' (0,1) or 'R' (2,3)
            # b_block: 'TL' (0,1), 'TR' (0,1), 'BL' (2,3), 'BR' (2,3)
            # c_side: 'L' (0,1) or 'R' (2,3)
            
            # Map local 2x2 indices to global indices
            # A (2x4): i in {0,1}, j in {0..3}
            # Local j: 0,1 for L; 2,3 for R
            # Global j = local_j if L else local_j + 2
            # U index = i * 4 + j
            
            # B (4x4): j in {0..3}, k in {0..3}
            # Local j: 0,1 for TL/TR; 2,3 for BL/BR
            # Local k: 0,1 for TL/BL; 2,3 for TR/BR
            # Global j = local_j + (2 if BL/BR else 0)
            # Global k = local_k + (2 if TR/BR else 0)
            # V index = j * 4 + k
            
            # C (2x4): i in {0,1}, k in {0..3}
            # Local k: 0,1 for L; 2,3 for R
            # Global k = local_k if L else local_k + 2
            # W index = k * 2 + i (Trace convention W[k*n+i])
            
            def get_u_idx(i, local_j):
                j = local_j if a_side == 'L' else local_j + 2
                return i * 4 + j
            
            def get_v_idx(local_j, local_k):
                j = local_j + (2 if b_block in ['BL', 'BR'] else 0)
                k = local_k + (2 if b_block in ['TR', 'BR'] else 0)
                return j * 4 + k
            
            def get_w_idx(local_k, i):
                k = local_k if c_side == 'L' else local_k + 2
                return k * 2 + i

            # Define the 7 Strassen terms
            # Each term is (u_indices, v_indices, w_indices)
            # indices are (local_idx, coeff)
            
            terms = [
                # M1
                ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
                # M2
                ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
                # M3
                ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
                # M4
                ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
                # M5
                ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
                # M6
                ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
                # M7
                ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
            ]
            
            col = start_col
            for u_loc, v_loc, w_loc in terms:
                for idx, coeff in u_loc:
                    i = idx // 2
                    lj = idx % 2
                    U[get_u_idx(i, lj), col] += coeff
                
                for idx, coeff in v_loc:
                    lj = idx // 2
                    lk = idx % 2
                    V[get_v_idx(lj, lk), col] += coeff
                
                for idx, coeff in w_loc:
                    i = idx // 2
                    lk = idx % 2
                    W[get_w_idx(lk, i), col] += coeff
                
                col += 1
            
            return col

        col = 0
        col = add_strassen_block('L', 'TL', 'L', col)
        col = add_strassen_block('R', 'BL', 'L', col)
        col = add_strassen_block('L', 'TR', 'R', col)
        col = add_strassen_block('R', 'BR', 'R', col)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2. Validation function
    def validate(U, V, W, rank):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Check diagonal entries: T[i,j,k] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Check off-diagonal entries: T[i,j,k] = 0
        # We can vectorize this check partially or just loop since 32*32*32 is small
        # But we only need to check where the tensor is 0.
        # The tensor is 1 only when i1==i2, j1==j2, k1==k2.
        # So we check all (a,b,d) where indices don't match.
        
        # To be fast, we can check a random subset or all. 8*16*8 = 1024 entries.
        # 1024 * 27 flops is trivial.
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b in range(m*p):
                j2 = b // p
                k1 = b % p
                for d in range(p*n):
                    i2 = d // n
                    k2 = d % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    s = np.dot(U_np[a], V_np[b] * W_np[d])
                    if abs(s) > 1e-9:
                        return False
        return True

    # 3. Baseline Rank 28
    best_result = build_strassen_28()
    best_rank = 28
    
    # 4. Try to find Rank 27
    target_rank = 27
    found = False
    
    # Precompute target tensor T as a 3D array for faster RHS construction in ALS
    # T[i,j,k] = 1 if consistent, else 0
    # We need T[a,b,d] where a=i*m+j, b=j*p+k, d=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0

    def run_als_search(rank, num_restarts, max_iters, seed_base):
        for r in range(num_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(seed_base + r)
            
            # Initialize
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Normalize to prevent divergence
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                for a in range(n*m):
                    # Build M: (m*p*p*n) x rank
                    # M[b*n + d, c] = V[b, c] * W[d, c]
                    # RHS: T[a, b, d]
                    
                    # Vectorized construction of M
                    # V is (m*p, rank), W is (p*n, rank)
                    # We need outer product of rows of V and W, then reshape
                    # M_flat = (V[:, None, :] * W[None, :, :]).reshape(-1, rank)
                    # But we only need specific entries? No, all entries of T[a,:,:]
                    # T[a, b, d] is 0 for most, 1 for one specific (b,d) pair?
                    # No, T[a, b, d] = 1 only if indices match.
                    # For fixed a (i1, j1), b (j2, k1), d (k2, i2).
                    # Match if i1==i2, j1==j2, k1==k2.
                    # So for fixed a, there is exactly ONE (b, d) pair where T=1?
                    # b = j1*p + k, d = k*n + i1.
                    # Yes, for a fixed a (i1, j1), the only non-zero T entries are for
                    # k in 0..p-1.
                    # b_k = j1*p + k
                    # d_k = k*n + i1
                    # T[a, b_k, d_k] = 1
                    # All other T[a, b, d] = 0.
                    
                    # So the system is:
                    # Sum_c U[a,c] V[b,c] W[d,c] = 1 for (b,d) in consistent set
                    # Sum_c U[a,c] V[b,c] W[d,c] = 0 for all other (b,d)
                    
                    # This is an overdetermined system.
                    # M is (16*8) x 27 = 128 x 27.
                    M = np.zeros((m*p*p*n, rank))
                    rhs = np.zeros(m*p*p*n)
                    
                    # Fill M
                    # M[b*p*n + d, c] = V[b, c] * W[d, c]
                    # We can use broadcasting
                    # V is (16, 27), W is (8, 27)
                    # V[:, None, :] is (16, 1, 27)
                    # W[None, :, :] is (1, 8, 27)
                    # Prod is (16, 8, 27)
                    # Reshape to (128, 27)
                    M = (V[:, None, :] * W[None, :, :]).reshape(-1, rank)
                    
                    # Fill RHS
                    # For k in 0..p-1:
                    # b = j1*p + k
                    # d = k*n + i1
                    # idx = b*p*n + d
                    i1 = a // m
                    j1 = a % m
                    for k in range(p):
                        b = j1 * p + k
                        d = k * n + i1
                        idx = b * p * n + d
                        rhs[idx] = 1.0
                    
                    # Solve Least Squares
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass

                # Update V
                for b in range(m*p):
                    j2 = b // p
                    k1 = b % p
                    M = np.zeros((n*m*p*n, rank))
                    rhs = np.zeros(n*m*p*n)
                    
                    # M[a*p*n + d, c] = U[a, c] * W[d, c]
                    M = (U[:, None, :] * W[None, :, :]).reshape(-1, rank)
                    
                    # RHS: T[a, b, d] = 1 if i1==i2, j1==j2, k1==k2
                    # Fixed b (j2, k1).
                    # Match if j1==j2, k1==k2.
                    # So j1 must be j2.
                    # i1 can be 0..n-1.
                    # k2 must be k1.
                    # d = k2*n + i1 = k1*n + i1.
                    # a = i1*m + j1 = i1*m + j2.
                    # idx = a*p*n + d.
                    
                    for i1 in range(n):
                        a = i1 * m + j2
                        d = k1 * n + i1
                        idx = a * p * n + d
                        rhs[idx] = 1.0
                        
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass

                # Update W
                for d in range(p*n):
                    i2 = d // n
                    k2 = d % n
                    M = np.zeros((n*m*m*p, rank))
                    rhs = np.zeros(n*m*m*p)
                    
                    # M[a*m*p + b, c] = U[a, c] * V[b, c]
                    M = (U[:, None, :] * V[None, :, :]).reshape(-1, rank)
                    
                    # RHS: T[a, b, d] = 1 if i1==i2, j1==j2, k1==k2
                    # Fixed d (i2, k2).
                    # Match if i1==i2, k1==k2.
                    # j1 can be 0..m-1.
                    # a = i2*m + j1.
                    # b = j1*p + k2.
                    # idx = a*m*p + b.
                    
                    for j1 in range(m):
                        a = i2 * m + j1
                        b = j1 * p + k2
                        idx = a * m * p + b
                        rhs[idx] = 1.0
                        
                    try:
                        W[d, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
            
            # Check result
            if validate(U.tolist(), V.tolist(), W.tolist(), rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Run search for rank 27
    # Use more restarts and iterations
    # Each ALS iteration is ~ 8+16+8 = 32 LSTSQ calls of size ~128x27.
    # This is very fast. ~1ms per iteration?
    # 100 iterations ~ 100ms.
    # 100 restarts ~ 10s.
    # We have 110s.
    
    result_27 = run_als_search(27, 500, 20, 12345)
    
    if result_27:
        best_rank = 27
        best_result = result_27
        found = True

    # If not found, try rank 26? (Unlikely)
    if not found and time.time() < deadline - 5:
        result_26 = run_als_search(26, 100, 10, 99999)
        if result_26:
            best_rank = 26
            best_result = result_26

    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
