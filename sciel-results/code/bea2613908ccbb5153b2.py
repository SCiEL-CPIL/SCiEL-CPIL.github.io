"""Trimming-based rank reduction for 3x4x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    # Build tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True

def _trim_one(U, V, W, n, m, p):
    """
    Try to remove one term. For each candidate term t, check if we can
    adjust the remaining terms to still represent the multiplication tensor.
    Returns (U', V', W', removed_index) or None.
    """
    rank = U.shape[1]
    # For each candidate term t to remove, we need to find new U', V', W'
    # for the remaining rank-1 terms such that the tensor is preserved.
    # This is a bilinear/trilinear feasibility problem.
    # Simplification: fix two of the three factor matrices (e.g., keep V and W fixed),
    # and solve for U. Then check if the residual can be absorbed.
    # Actually, a cleaner approach: the tensor T must equal sum_{c in S} u_c v_c^T w_c^T.
    # If we remove term t, we need sum_{c in S\{t}} u_c v_c^T w_c^T = T.
    # This is underdetermined. Let's use a specific strategy:
    # Fix V and W, solve for U. The equation is:
    #   For each (i,j,k): sum_{c in S\{t}} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is linear in U. Let's set up the linear system.
    
    for t in range(rank):
        S = [c for c in range(rank) if c != t]
        if len(S) == 0:
            continue
        # Build linear system: unknowns are U[i*m+j, c] for i in range(n), j in range(m), c in S
        # Number of unknowns: n * m * len(S)
        # Number of equations: n * m * p
        num_unknowns = n * m * len(S)
        num_eqs = n * m * p
        # If num_unknowns < num_eqs, the system is overdetermined; may not have a solution.
        # If num_unknowns >= num_eqs, we can try to solve.
        
        # Actually, let's think differently. The tensor T is fixed. We want:
        #   T = sum_{c in S} u_c ⊗ v_c ⊗ w_c
        # where v_c and w_c are fixed (from current V, W), and u_c are to be determined.
        # This is: for each (j,k), the matrix T[:,:,j,k]... no.
        # Let me reorganize. T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        # Group by (j,k): for fixed (j,k), we have a vector over i:
        #   T[:, j, k] = sum_c (V[j*p+k, c] * W[k*n + :, c]) * U[:, m*j ... ] 
        # This is getting complex. Let me use a different approach.
        
        # Alternative: use the fact that we can also modify V and W.
        # A simpler heuristic: try removing term t and see if the "correction"
        # tensor T - sum_{c!=t} u_c v_c w_c is zero (it should be, since we started
        # with a valid decomposition). The issue is whether we can redistribute.
        
        # Let me try a concrete approach: for a given t, fix V and W, and solve for U.
        # The system is: for all (i,j,k):
        #   sum_{c in S} U[i*m+j, c] * A_{(i,j,k), c} = 1
        # where A_{(i,j,k), c} = V[j*p+k, c] * W[k*n+i, c].
        # This is a linear system in the variables U[i*m+j, c].
        
        # Build the matrix.
        # Index equations by (i,j,k), variables by (i, j, c) for c in S.
        # But note: U[i*m+j, c] appears in equations for all k.
        
        # Let's flatten: equation index e = (i, j, k), variable index v = (i, j, c) where c in S.
        # Coefficient of variable (i', j', c) in equation (i, j, k) is:
        #   A_{(i,j,k), c} if i'=i and j'=j, else 0.
        # So the matrix is block diagonal over (i,j)! For each (i,j), we have p equations
        # and len(S) variables, with matrix M_{(i,j)}[k, c] = V[j*p+k, c] * W[k*n+i, c].
        
        # For each (i,j), we need to solve M_{(i,j)} * x_{(i,j)} = 1 (vector of ones of length p).
        # This is solvable if the rows of M_{(i,j)} span the all-ones vector.
        
        feasible = True
        new_U = np.zeros((n * m, rank), dtype=object)
        for i in range(n):
            for j in range(m):
                # Build M: p x len(S)
                M = np.zeros((p, len(S)), dtype=object)
                for k in range(p):
                    for idx_c, c in enumerate(S):
                        M[k, idx_c] = V[j * p + k, c] * W[k * n + i, c]
                # Solve M * x = ones(p)
                # Use numpy with float conversion for solving, then check exactness
                M_float = np.array([[float(x) for x in row] for row in M], dtype=float)
                b = np.ones(p, dtype=float)
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M_float, b, rcond=None)
                except:
                    feasible = False
                    break
                # Check if the solution is exact (residual ~ 0)
                if len(residuals) > 0 and residuals[0] > 1e-10:
                    feasible = False
                    break
                # Check if x entries are "nice" (close to integers or half-integers)
                # For now, accept any real solution, but we need exact rational.
                # Let's store as Fraction for exactness check.
                for idx_c, c in enumerate(S):
                    val = x[idx_c]
                    # Round to nearest rational with small denominator
                    frac = Fraction(val).limit_denominator(1000)
                    if abs(float(frac) - val) > 1e-9:
                        # Not a nice rational; this removal may not work exactly
                        feasible = False
                        break
                    new_U[i * m + j, c] = frac
                if not feasible:
                    break
            if not feasible:
                break
        
        if feasible:
            # Build new V and W (just remove column t)
            new_V = V[:, S].copy()
            new_W = W[:, S].copy()
            # Verify
            if _verify(new_U, new_V, new_W, n, m, p):
                return new_U, new_V, new_W, t
    return None

def construct():
    deadline = time.time() + 140.0  # safety margin
    n, m, p = DIMENSIONS
    
    U, V, W = _schoolbook()
    rank = U.shape[1]
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Iteratively try to remove terms
    progress = True
    while progress and time.time() < deadline:
        progress = False
        result = _trim_one(U, V, W, n, m, p)
        if result is not None:
            U, V, W, removed = result
            rank = U.shape[1]
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            progress = True
    
    # Convert to lists of lists
    def to_list(mat):
        return [[mat[i, j] for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
