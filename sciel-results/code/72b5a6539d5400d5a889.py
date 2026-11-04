import numpy as np
import time
from scipy.linalg import qr

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-8):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        # Use einsum for speed
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

        # Tile 2x4 x 4x7:
        # Blocks for k=0,1
        add_strassen(0, 0, 0)
        add_strassen(0, 2, 0)
        # Blocks for k=2,3
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 2)
        # Blocks for k=4,5
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1  # A[0,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 0, col] = 1  # C[0,6]
            col += 1
            U[1 * m + j, col] = 1  # A[1,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 1, col] = 1  # C[1,6]
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()
    
    # Verify rank 50
    if not verify_decomp(U50, V50, W50):
        # Fallback to schoolbook if construction failed (should not happen)
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
    best_U, best_V, best_W = U50, V50, W50

    # Precompute unfoldings of T
    T1 = T.reshape(n * m, -1)  # 8 x 392
    T2 = T.reshape(m * p, -1)  # 28 x 112
    T3 = T.reshape(p * n, -1)  # 14 x 224

    def khatri_rao(A, B):
        """
        Compute Khatri-Rao product of A (M x R) and B (N x R).
        Result is (M*N) x R.
        """
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def solve_least_squares(A, B):
        """
        Solve A X = B in least squares sense using QR for stability and speed.
        A is (M, R), B is (M, N). Returns X (R, N).
        """
        Q, R = qr(A, mode='economic')
        # R is upper triangular (R, R)
        # Q.T @ A = R
        # Q.T @ B = R @ X  => X = R^{-1} @ (Q.T @ B)
        try:
            X = np.linalg.solve(R, Q.T @ B)
            return X
        except np.linalg.LinAlgError:
            # Fallback to pinv if singular
            return np.linalg.pinv(A) @ B

    def als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=100, tol=1e-10):
        """
        Alternating Least Squares for CP decomposition using QR solver.
        """
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            # Solve for U: T1 = U @ (V ⊙ W)^T
            # T1 (8, 392) = U (8, R) @ KR_VW (R, 392)^T
            # So U = T1 @ (KR_VW)^+
            KR_VW = khatri_rao(V, W)
            U_new = solve_least_squares(KR_VW, T1.T).T
            
            # Solve for V
            KR_WU = khatri_rao(W, U)
            V_new = solve_least_squares(KR_WU, T2.T).T
            
            # Solve for W
            KR_UV = khatri_rao(U, V)
            W_new = solve_least_squares(KR_UV, T3.T).T
            
            # Check convergence
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            
            if change < tol:
                break
        
        return U, V, W

    def polish_decomposition(U, V, W, max_iters=50, tol=1e-12):
        """
        High-precision polish.
        """
        R = U.shape[1]
        for _ in range(max_iters):
            KR_VW = khatri_rao(V, W)
            U = solve_least_squares(KR_VW, T1.T).T
            
            KR_WU = khatri_rao(W, U)
            V = solve_least_squares(KR_WU, T2.T).T
            
            KR_UV = khatri_rao(U, V)
            W = solve_least_squares(KR_UV, T3.T).T
            
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(recon - T)
            if residual < tol:
                break
        
        return U, V, W

    def round_to_half_integers(X):
        """
        Round entries to nearest half-integer.
        """
        return np.round(X * 2) / 2

    def try_lattice_snap(U, V, W, target_rank):
        """
        Try to snap the solution to the nearest half-integer lattice point.
        """
        U_r = round_to_half_integers(U)
        V_r = round_to_half_integers(V)
        W_r = round_to_half_integers(W)
        
        if verify_decomp(U_r, V_r, W_r, atol=1e-6):
            return U_r, V_r, W_r, True
        
        # Try a local search around the rounded points?
        # Given time constraints, just return the rounded points if they are close,
        # but verification must be exact.
        return U, V, W, False

    # Try to reduce rank
    # Start from 49 down to 28
    rng = np.random.default_rng(42)
    
    # Time budget allocation
    # We have ~130s. 
    # Each ALS run takes ~0.5-2s depending on convergence.
    # We can afford ~50-100 runs.
    
    for target_rank in range(49, 27, -1):
        if time.time() > deadline - 10:
            break
            
        # Number of restarts based on remaining time
        # Assume ~1s per restart
        remaining_time = deadline - time.time() - 5
        max_restarts = int(remaining_time / 1.0)
        max_restarts = max(2, min(20, max_restarts)) # Cap at 20 for sanity
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            if target_rank == 49 and restart == 0:
                # Drop 1 column from rank 50
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                # Add small noise
                U_init += rng.normal(0, 0.01, U_init.shape)
                V_init += rng.normal(0, 0.01, V_init.shape)
                W_init += rng.normal(0, 0.01, W_init.shape)
            elif restart < 5:
                # Random init
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            else:
                # Perturb rank 50
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                U_init += rng.normal(0, 0.1, U_init.shape)
                V_init += rng.normal(0, 0.1, V_init.shape)
                W_init += rng.normal(0, 0.1, W_init.shape)
            
            U_sol, V_sol, W_sol = als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=100)
            
            # Check residual
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            
            if residual < 1e-8:
                # Polish
                U_sol, V_sol, W_sol = polish_decomposition(U_sol, V_sol, W_sol)
                
                # Try to snap to lattice
                U_snap, V_snap, W_snap, snapped = try_lattice_snap(U_sol, V_sol, W_sol, target_rank)
                
                if snapped:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    found = True
                    break
                elif verify_decomp(U_sol, V_sol, W_sol, atol=1e-8):
                    # If not snapped but valid, keep it (though task prefers integer/half-integer)
                    # The task says "integer/half-integer entries". 
                    # If we can't snap, we might need to return the float version? 
                    # "score is the rank". "entries that exactly decompose".
                    # Usually, if it's not integer/half-integer, it might be invalid for the strict checker?
                    # The prompt says: "factor matrices ... with integer/half-integer entries that exactly decompose"
                    # So we MUST have integer/half-integer entries.
                    # If we can't snap, we can't use this solution.
                    pass
            else:
                # If residual is small but not tiny, try polishing more aggressively?
                if residual < 1e-4:
                     U_sol, V_sol, W_sol = polish_decomposition(U_sol, V_sol, W_sol, max_iters=100, tol=1e-14)
                     recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                     residual = np.linalg.norm(recon - T)
                     if residual < 1e-8:
                         U_snap, V_snap, W_snap, snapped = try_lattice_snap(U_sol, V_sol, W_sol, target_rank)
                         if snapped:
                             best_rank = target_rank
                             best_U, best_V, best_W = U_snap, V_snap, W_snap
                             found = True
                             break

        if found:
            # Continue to lower ranks
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
