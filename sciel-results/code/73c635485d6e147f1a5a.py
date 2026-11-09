"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the full multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly reconstruct T."""
        r = U.shape[1]
        # Efficient contraction
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Use einsum
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = np.abs(recon - T)
        return np.max(diff) < tol

    def to_rational_matrix(A, tol=1e-9):
        """Convert to half-integers if possible, else return as-is."""
        B = np.round(A * 2) / 2
        if np.max(np.abs(B - A)) < tol:
            return B
        B = np.round(A)
        if np.max(np.abs(B - A)) < tol:
            return B
        return A

    def find_decomposition(rank, max_attempts=2000, seed=0):
        """
        Search for a decomposition of given rank.
        Strategy: Random U, V with small integers. Solve for W. Round W. Check.
        """
        rng = np.random.RandomState(seed)
        
        for attempt in range(max_attempts):
            # Generate random U and V with entries in {-2, -1, 0, 1, 2}
            # This helps in finding integer solutions
            U = rng.randint(-2, 3, size=(nm, rank))
            V = rng.randint(-2, 3, size=(mp, rank))
            
            # Construct M such that T_flat = W @ M^T ?
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # Fix U, V.
            # For each c, T[:, :, c] = sum_r W[c, r] (U[:, r] * V[:, r])
            # Let Z_r = U[:, r] * V[:, r] (shape nm*mp)
            # T_reshaped (pn, nm*mp) = W @ Z^T
            # Z is (rank, nm*mp) where Z[r, a*mp+b] = U[a, r] * V[b, r]
            
            Z = np.zeros((rank, nm * mp))
            for r in range(rank):
                Z[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
            T_flat = np.transpose(T, (2, 0, 1)).reshape(pn, nm * mp)
            
            # Solve W @ Z^T = T_flat
            # W = T_flat @ pinv(Z^T)
            try:
                # Use least squares
                W_sol, residuals, rank_z, s = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
                W_sol = W_sol.T # shape (pn, rank)
            except Exception:
                continue
                
            # Round W to half-integers
            W_r = to_rational_matrix(W_sol)
            
            # Check if rounded W works with original U, V
            if check_decomposition(U, V, W_r):
                # Also try rounding U and V just in case they drifted (they are ints so no drift)
                return U, V, W_r
                
        return None

    # Try ranks from 40 to 95
    # Schoolbook is 96.
    # We expect a significant improvement.
    best_rank = nm * mp
    best_U = None
    best_V = None
    best_W = None

    # First, verify schoolbook works as fallback
    U_school = np.zeros((nm, nm * mp))
    V_school = np.zeros((mp, nm * mp))
    W_school = np.zeros((pn, nm * mp))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Search for lower ranks
    # Start from a reasonable lower bound, e.g., 40
    for rank in range(40, 96):
        found = False
        # Try multiple seeds for this rank
        for seed in range(5):
            res = find_decomposition(rank, max_attempts=500, seed=seed)
            if res:
                U, V, W = res
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        if found:
            break

    if best_U is not None:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        U_list = [[float(x) for x in row] for row in U_school]
        V_list = [[float(x) for x in row] for row in V_school]
        W_list = [[float(x) for x in row] for row in W_school]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
