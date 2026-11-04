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

    def try_als_reduce(U, V, W, target_rank, rng, max_iter=300):
        """Try to reduce rank using ALS with better initialization."""
        if time.time() > deadline - 10:
            return None
        
        T_U = T.reshape(n*m, -1)
        T_V = T.reshape(m*p, -1)
        T_W = T.reshape(p*n, -1)
        
        R = target_rank
        # Use random initialization with larger perturbation
        U_cur = rng.normal(0, 0.1, (n*m, R))
        V_cur = rng.normal(0, 0.1, (m*p, R))
        W_cur = rng.normal(0, 0.1, (p*n, R))
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            KR_VW = np.array([np.kron(V_cur[:, r], W_cur[:, r]) for r in range(R)]).T
            KR_WU = np.array([np.kron(W_cur[:, r], U_cur[:, r]) for r in range(R)]).T
            KR_UV = np.array([np.kron(U_cur[:, r], V_cur[:, r]) for r in range(R)]).T
            
            try:
                U_new = T_U @ np.linalg.pinv(KR_VW)
                V_new = T_V @ np.linalg.pinv(KR_WU)
                W_new = T_W @ np.linalg.pinv(KR_UV)
            except:
                break
            
            change = (np.linalg.norm(U_new - U_cur) + 
                     np.linalg.norm(V_new - V_cur) + 
                     np.linalg.norm(W_new - W_cur))
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            if change < 1e-15:
                break
        
        if verify_decomp(U_cur, V_cur, W_cur, atol=1e-7):
            U_pol = np.round(U_cur * 2) / 2
            V_pol = np.round(V_cur * 2) / 2
            W_pol = np.round(W_cur * 2) / 2
            if verify_decomp(U_pol, V_pol, W_pol, atol=1e-7):
                return U_pol, V_pol, W_pol
            return U_cur, V_cur, W_cur
        return None

    rng = np.random.default_rng(123)
    
    # Try to reduce from 50 down with multiple restarts
    for target_rank in range(49, 40, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            result = try_als_reduce(best_U, best_V, best_W, target_rank, rng, max_iter=150)
            if result is not None:
                best_rank = target_rank
                best_U, best_V, best_W = result
                found = True
                break
    
    if not verify_decomp(best_U, best_V, best_W, atol=1e-9):
        best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
        best_rank = 50

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
