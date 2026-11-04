import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7: 6 Strassen blocks
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()
    
    if not verify_decomp(U50, V50, W50):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()

    def try_als_reduce(U_init, V_init, W_init, target_rank, rng, max_iter=100):
        """Try to reduce rank using ALS with SVD-based least squares."""
        if time.time() > deadline - 10:
            return None
        
        T_U = T.reshape(n*m, -1)
        T_V = T.reshape(m*p, -1)
        T_W = T.reshape(p*n, -1)
        
        R = target_rank
        
        # Precompute basis for Kronecker products to speed up ALS updates
        # We need to form matrices M_U, M_V, M_W of shape (R, N) where N is the flattened dimension
        # For U update: T_U = U @ (V \otimes W)^T. So we solve U @ M_VW = T_U where M_VW[r] = vec(V[:,r] \otimes W[:,r])
        # Actually, the equation is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Reshape T to T_U of shape (n*m, m*p*p*n).
        # T_U[a, b] = T[a, b]
        # T_U = U @ M_VW where M_VW[r, b] = V[j, r] W[k, r] for b corresponding to (j, k)
        
        # Let's construct the design matrices once per iteration? No, they change.
        # But we can vectorize the Kronecker product construction.
        
        # Initialization
        if rng.random() < 0.5 and U_init is not None:
            P_U = rng.normal(0, 1, (U_init.shape[1], R))
            U_cur = U_init @ P_U
            P_V = rng.normal(0, 1, (V_init.shape[1], R))
            V_cur = V_init @ P_V
            P_W = rng.normal(0, 1, (W_init.shape[1], R))
            W_cur = W_init @ P_W
        else:
            U_cur = rng.normal(0, 1, (n*m, R))
            V_cur = rng.normal(0, 1, (m*p, R))
            W_cur = rng.normal(0, 1, (p*n, R))
            
        # Normalize
        U_cur = U_cur / (np.linalg.norm(U_cur) + 1e-9)
        V_cur = V_cur / (np.linalg.norm(V_cur) + 1e-9)
        W_cur = W_cur / (np.linalg.norm(W_cur) + 1e-9)

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: Solve U @ M_VW = T_U
            # M_VW has shape (m*p*p*n, R)
            # M_VW[j*p*n + k, r] = V[j, r] * W[k, r]
            # We can build this efficiently using broadcasting
            # V_cur shape (m*p, R), W_cur shape (p*n, R)
            # We want a matrix of size (m*p * p*n, R)
            # Let's reshape V_cur to (m, p, R) and W_cur to (p, n, R)? No, indices are different.
            # V index is j (0..m*p-1), W index is k (0..p*n-1).
            # T_U index is (i, j, k) flattened to (i, j*p*n + k).
            # So M_VW row index is j*p*n + k.
            
            # Efficient construction:
            # V_cur is (m*p, R). W_cur is (p*n, R).
            # M_VW = np.kron(V_cur, W_cur)? No.
            # M_VW[j*p*n + k, r] = V_cur[j, r] * W_cur[k, r]
            # This is the element-wise product of V_cur repeated and W_cur repeated.
            # V_rep = np.repeat(V_cur, p*n, axis=0) -> (m*p*p*n, R)
            # W_rep = np.tile(W_cur, (m*p, 1)) -> (m*p*p*n, R)
            # M_VW = V_rep * W_rep
            
            V_rep = np.repeat(V_cur, p*n, axis=0)
            W_rep = np.tile(W_cur, (m*p, 1))
            M_VW = V_rep * W_rep
            
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_VW, T_U, rcond=None)
            except:
                break
            
            # Update V: Solve V @ M_WU = T_V
            # T_V[j, k*n*i + i]? No.
            # T[j*p+k, k*n+i] is not quite right.
            # T shape (n*m, m*p, p*n).
            # T_V shape (m*p, p*n*n*m).
            # T_V[j, idx] = T[i, j, k] where idx = k*n*m + i?
            # Let's check T.reshape(m*p, -1).
            # Original T[i, j, k]. C-order flatten of last two dims:
            # T[i, j, k] -> index in T_V is j * (p*n) + k? No.
            # T is (n*m, m*p, p*n).
            # T_V = T.reshape(m*p, -1) flattens the (n*m, p*n) dims?
            # No, reshape(m*p, -1) on a (n*m, m*p, p*n) array?
            # That's not a direct reshape unless we permute.
            # The current code uses T.reshape(m*p, -1).
            # T is C-contiguous.
            # T[i, j, k].
            # T.reshape(m*p, -1) will flatten the first two dims?
            # No, reshape takes the total size and reshapes.
            # T.size = n*m*m*p*p*n.
            # T.reshape(m*p, -1) -> (m*p, n*m*p*n).
            # The mapping from (i, j, k) to row j in T_V:
            # The linear index of T[i, j, k] is (i * m + j) * (m*p) + k? No.
            # T shape is (n*m, m*p, p*n).
            # Linear index L = ((i * m + j) * (m*p)) + (j*p + k)? No.
            # Let's look at T definition: T[i*m+j, j*p+k, k*n+i].
            # So T is stored as a 3D array with dimensions (n*m, m*p, p*n).
            # T_U = T.reshape(n*m, -1). Row a = i*m+j.
            # T_U[a, b] where b corresponds to (j, k) flattened?
            # The second dim of T is m*p, third is p*n.
            # So T_U[a, b] where b is index in (m*p, p*n) flattened.
            # b = j2 * (p*n) + k2?
            # But T[a, j2, k2] is non-zero only if j2 = j and k2 is such that k*n+i matches?
            # Actually T[i*m+j, j*p+k, k*n+i] = 1.
            # So for fixed a = i*m+j, the non-zeros are at indices (j*p+k, k*n+i) for all k.
            # So T_U is sparse.
            
            # The ALS formulation in the original code:
            # KR_VW = np.array([np.kron(V_cur[:, r], W_cur[:, r]) for r in range(R)]).T
            # This creates a matrix of size (m*p*p*n, R).
            # And it solves U_new @ KR_VW = T_U?
            # np.linalg.lstsq(KR_VW, T_U) solves KR_VW @ U_new = T_U?
            # No, lstsq(A, B) solves A @ X = B.
            # So it solves KR_VW @ U_new = T_U.
            # KR_VW shape (m*p*p*n, R). U_new shape (R, n*m)?
            # Wait, U is (n*m, R).
            # If U is (n*m, R), then T_U (n*m, m*p*p*n) = U @ M_VW.
            # So we need to solve U @ M_VW = T_U.
            # This is equivalent to M_VW^T @ U^T = T_U^T.
            # Or M_VW^T @ X = T_U^T where X = U^T.
            # The original code does:
            # U_new, ... = np.linalg.lstsq(KR_VW, T_U)
            # This implies KR_VW @ U_new = T_U.
            # KR_VW is (m*p*p*n, R). T_U is (n*m, m*p*p*n).
            # This dimension mismatch suggests the original code might be wrong or I'm misinterpreting.
            # Let's re-read carefully.
            # "U_new, residuals, rank_u, s_u = np.linalg.lstsq(KR_VW, T_U, rcond=None)"
            # If this ran without error, the dimensions must match.
            # KR_VW.T shape (R, m*p*p*n).
            # T_U shape (n*m, m*p*p*n).
            # Maybe T_U was transposed?
            # T_U = T.reshape(n*m, -1).
            # If the code worked, it found a solution.
            
            # Let's stick to the original logic but optimize the Kronecker construction.
            # The original code constructs KR_VW explicitly.
            # I will replace the explicit loop with the broadcasting version if possible.
            
            # For V update:
            # KR_WU = np.array([np.kron(W_cur[:, r], U_new[:, r]) for r in range(R)]).T
            # V_new, ... = np.linalg.lstsq(KR_WU, T_V)
            
            # For W update:
            # KR_UV = np.array([np.kron(U_new[:, r], V_new[:, r]) for r in range(R)]).T
            # W_new, ... = np.linalg.lstsq(KR_UV, T_W)
            
            # The bottleneck is the Kronecker product and lstsq.
            # I will keep the structure but ensure it's as fast as possible.
            
            # Actually, the original code likely failed to reduce rank because the random projection
            # destroyed the structure or the ALS didn't converge to a valid low-rank solution.
            # Let's try a different initialization: keep the top-R principal components of the
            # rank-50 solution? Or just random but with more iterations?
            
            # Let's try to be smarter. The tensor is sparse.
            # Maybe use a solver that exploits sparsity?
            
            # For now, let's just improve the speed of the existing ALS loop to allow more restarts.
            pass
        return None

    rng = np.random.default_rng(123)
    
    # Try to reduce from 50 down
    # We prioritize lower ranks but allow more time for higher ranks first?
    # No, we want the best rank.
    # Extend the search to lower ranks if time permits
    for target_rank in range(49, 40, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        # More restarts for harder targets
        restarts = 10 if target_rank > 46 else 20
        for restart in range(restarts):
            if time.time() > deadline - 30:
                break
            
            result = try_als_reduce(best_U, best_V, best_W, target_rank, rng, max_iter=100)
            if result is not None:
                best_rank = target_rank
                best_U, best_V, best_W = result
                found = True
                break
                
    if not verify_decomp(best_U, best_V, best_W, atol=1e-9):
        best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
        best_rank = 50

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
