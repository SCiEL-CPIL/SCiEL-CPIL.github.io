"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools

DIMENSIONS = [2, 2, 5]
_START_TIME = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Build the target tensor T of shape (n*m, m*p, p*n) is not quite right.
    # The tensor is T_{a, b, c} where:
    # a indexes A (n*m = 4)
    # b indexes B (m*p = 10)
    # c indexes C (p*n = 10)
    # T[a, b, c] = 1 if A[a] * B[b] contributes to C[c]
    # A is 2x2, indices (i, j) -> a = i*m + j = i*2 + j
    # B is 2x5, indices (j, k) -> b = j*p + k = j*5 + k
    # C is 2x5, indices (i, k) -> c = k*n + i = k*2 + i
    
    # T[a, b, c] is non-zero only if:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # In that case, the contribution is A[i,j] * B[j,k] to C[i,k].
    # So T[a, b, c] = 1 if the indices match, else 0.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            a = i*m + j
            for k in range(p):
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
                
    # Helper to check if a decomposition is valid
    def check_decomposition(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reconstructed tensor: sum_r U[:, r] * V[:, r] * W[:, r]
        # This is an outer product sum.
        # Reshape to (n*m, m*p, p*n)
        # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # We can compute this efficiently.
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # But for speed, we can check specific entries or use matrix multiplication tricks.
        # Since dimensions are small, einsum is fine.
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)

    def is_half_integer(x):
        # Check if x is close to a multiple of 0.5
        y = x * 2
        return np.allclose(y, np.round(y), atol=1e-6)

    def snap_to_half_int(M):
        return np.round(M * 2) / 2.0

    # Initial solution: Rank 18
    # We can reuse the logic from the previous block or generate it.
    # For speed, let's just use the known rank 18 as a fallback.
    
    def build_rank18():
        total_rank = 18
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # Strassen block 1 (cols 0,1)
        # M1 = (a+d)(e+h)
        U[0, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[1*p+1, 0] += 1
        W[0*n+0, 0] += 1; W[1*n+1, 0] += 1
        
        # M2 = (c+d)e
        U[2, 1] += 1; U[3, 1] += 1
        V[0, 1] += 1
        W[0*n+1, 1] += 1; W[1*n+1, 1] += -1
        
        # M3 = a(f-h)
        U[0, 2] += 1
        V[0*p+1, 2] += 1; V[1*p+1, 2] += -1
        W[1*n+0, 2] += 1; W[1*n+1, 2] += 1
        
        # M4 = d(g-e)
        U[3, 3] += 1
        V[1*p+0, 3] += 1; V[0*p+0, 3] += -1
        W[0*n+0, 3] += 1; W[0*n+1, 3] += 1
        
        # M5 = (a+b)h
        U[0, 4] += 1; U[1, 4] += 1
        V[1*p+1, 4] += 1
        W[0*n+0, 4] += -1; W[1*n+0, 4] += 1
        
        # M6 = (c-a)(e+f)
        U[2, 5] += 1; U[0, 5] += -1
        V[0*p+0, 5] += 1; V[0*p+1, 5] += 1
        W[1*n+1, 5] += 1
        
        # M7 = (b-d)(g+h)
        U[1, 6] += 1; U[3, 6] += -1
        V[1*p+0, 6] += 1; V[1*p+1, 6] += 1
        W[0*n+0, 6] += 1
        
        # Strassen block 2 (cols 2,3) - Shift indices
        # Col 2,3
        # M1
        U[0, 7] += 1; U[3, 7] += 1
        V[0*p+2, 7] += 1; V[1*p+3, 7] += 1
        W[2*n+0, 7] += 1; W[3*n+1, 7] += 1
        
        # M2
        U[2, 8] += 1; U[3, 8] += 1
        V[0*p+2, 8] += 1
        W[2*n+1, 8] += 1; W[3*n+1, 8] += -1
        
        # M3
        U[0, 9] += 1
        V[0*p+3, 9] += 1; V[1*p+3, 9] += -1
        W[3*n+0, 9] += 1; W[3*n+1, 9] += 1
        
        # M4
        U[3, 10] += 1
        V[1*p+2, 10] += 1; V[0*p+2, 10] += -1
        W[2*n+0, 10] += 1; W[2*n+1, 10] += 1
        
        # M5
        U[0, 11] += 1; U[1, 11] += 1
        V[1*p+3, 11] += 1
        W[2*n+0, 11] += -1; W[3*n+0, 11] += 1
        
        # M6
        U[2, 12] += 1; U[0, 12] += -1
        V[0*p+2, 12] += 1; V[0*p+3, 12] += 1
        W[3*n+1, 12] += 1
        
        # M7
        U[1, 13] += 1; U[3, 13] += -1
        V[1*p+2, 13] += 1; V[1*p+3, 13] += 1
        W[2*n+0, 13] += 1
        
        # Standard block (col 4)
        # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
        # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
        
        U[0, 14] = 1
        V[0*p+4, 14] = 1
        W[4*n+0, 14] = 1
        
        U[1, 15] = 1
        V[1*p+4, 15] = 1
        W[4*n+0, 15] = 1
        
        U[2, 16] = 1
        V[0*p+4, 16] = 1
        W[4*n+1, 16] = 1
        
        U[3, 17] = 1
        V[1*p+4, 17] = 1
        W[4*n+1, 17] = 1
        
        return U, V, W

    # Try to find a lower rank solution using ALS
    def try_rank(rank):
        if time.time() - _START_TIME > _DEADLINE:
            return None
            
        # Initialize with random or from rank 18
        # Let's try dropping terms from rank 18
        U18, V18, W18 = build_rank18()
        
        # Try different subsets of 18 terms to initialize rank 17
        # Or just random initialization
        for attempt in range(20):
            if time.time() - _START_TIME > _DEADLINE:
                break
                
            if attempt == 0:
                # Use first 17 terms of rank 18
                U = U18[:, :rank].copy()
                V = V18[:, :rank].copy()
                W = W18[:, :rank].copy()
            else:
                # Random initialization
                U = np.random.randn(n*m, rank)
                V = np.random.randn(m*p, rank)
                W = np.random.randn(p*n, rank)
                # Scale to help convergence
                U /= np.linalg.norm(U)
                V /= np.linalg.norm(V)
                W /= np.linalg.norm(W)

            # ALS iterations
            for it in range(100):
                if time.time() - _START_TIME > _DEADLINE:
                    break
                
                # Update U: U[:, r] = (T * V * W) / (V^2 * W^2)
                # For each r, U[:, r] is the solution to:
                # sum_b sum_c T[a, b, c] V[b, r] W[c, r] = sum_r' U[a, r'] V[b, r'] W[c, r']
                # This is not a simple linear system for U directly because V and W are fixed.
                # Actually, we can reshape T to (n*m, m*p*p*n) and use matrix multiplication.
                
                # Let's use the property:
                # T_recon = sum_r U[:, r] V[:, r] W[:, r]
                # For fixed V, W, we want to find U that minimizes || T - sum_r U_r (V_r \otimes W_r) ||^2
                # Let M_r = V_r \otimes W_r (size m*p * p*n = 10*10=100)
                # Let T_flat = T.reshape(n*m, -1) (size 4 x 100)
                # We want T_flat approx sum_r U[:, r] M_r^T ? No.
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # T_flat[a, (b,c)] = sum_r U[a, r] (V[b, r] W[c, r])
                # Let K_r be the vector V_r \otimes W_r.
                # Then T_flat[a, :] = sum_r U[a, r] K_r^T
                # This is a linear least squares problem: T_flat approx U M
                # Where M is a matrix of size (rank, 100) where M[r, :] = K_r^T
                
                M = np.einsum('br,cr->rc', V, W).T # Shape (100, rank)
                # T_flat is (4, 100)
                T_flat = T.reshape(n*m, -1)
                
                # Solve T_flat = U M
                # U = T_flat M^+
                U_new, _, _, _ = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
                U = U_new.T
                
                # Update V
                # T[a, b, c] = sum_r V[b, r] (U[a, r] W[c, r])
                # T_flat2[b, (a,c)] = sum_r V[b, r] (U[a, r] W[c, r])
                # Let L_r = U_r \otimes W_r
                # M2 = L^T
                M2 = np.einsum('ar,cr->rc', U, W).T
                T_flat2 = T.transpose(1, 0, 2).reshape(m*p, -1)
                V_new, _, _, _ = np.linalg.lstsq(M2.T, T_flat2.T, rcond=None)
                V = V_new.T
                
                # Update W
                # T[a, b, c] = sum_r W[c, r] (U[a, r] V[b, r])
                # T_flat3[c, (a,b)] = sum_r W[c, r] (U[a, r] V[b, r])
                M3 = np.einsum('ar,br->rc', U, V).T
                T_flat3 = T.transpose(2, 0, 1).reshape(p*n, -1)
                W_new, _, _, _ = np.linalg.lstsq(M3.T, T_flat3.T, rcond=None)
                W = W_new.T
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_recon - T)
                if err < 1e-9:
                    # Check half-integer
                    if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                        U = snap_to_half_int(U)
                        V = snap_to_half_int(V)
                        W = snap_to_half_int(W)
                        if check_decomposition(U, V, W):
                            return {"U": U, "V": V, "W": W}
                    break

        return None

    best_result = None
    for rank in [17, 16, 15]:
        if time.time() - _START_TIME > _DEADLINE:
            break
        res = try_rank(rank)
        if res:
            best_result = res
            break # Found a lower rank, stop

    if best_result is None:
        # Fallback to rank 18
        U, V, W = build_rank18()
        best_result = {"U": U, "V": V, "W": W}
        
    return best_result
# EVOLVE-BLOCK-END
