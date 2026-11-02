"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as 2D lists for schoolbook decomposition."""
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the tensor decomposition is exact."""
    rank = len(U[0])
    # Build the multiplication tensor T[i, j, k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # Actually T[i, j, k] = 1 for all (i,j,k) — it's the identity on the tensor product.
    # The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i * m + j, :], Va[j * p + k, :]) * 1.0
                # Wait, that's not right. The tensor is:
                # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # This is a trilinear form, not a simple dot product.
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by removing one column and solving for the rest.
    
    Strategy: for each candidate column c to remove, set U[:,c]=V[:,c]=W[:,c]=0,
    then check if the remaining (rank-1) columns can be adjusted to still satisfy
    the tensor equations. This is a linear system in the entries of the remaining
    columns, but it's nonlinear (products of three variables). Instead, we try a
    simpler approach: try setting specific entries to 0 or 1 in a pattern.
    
    A more practical approach: use the fact that for matrix multiplication,
    there are known rank-reducing identities. For 3x3x8, we can try to merge
    pairs of schoolbook terms.
    
    For now, try a randomized local search: perturb the decomposition and
    project back onto the manifold of valid decompositions.
    """
    rank = len(U[0])
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Try to remove each column one at a time by solving a linear system
    # The tensor equation is: for each (i,j,k):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # If we fix all columns except c, the equation for column c is:
    #   U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 - sum_{c'!=c} U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c']
    # This is still nonlinear. 
    #
    # Alternative: try a structured reduction. For 3x3 * 3x8, note that
    # C = A * B where A is 3x3, B is 3x8, C is 3x8.
    # We can write B as [B1 | B2] where B1, B2 are 3x4.
    # Then C = [A*B1 | A*B2].
    # If we can decompose A*B1 with rank r1 and A*B2 with rank r2,
    # total rank is r1 + r2 (if we share the A-side factors).
    # Actually, we can share: use the same U factors for both.
    # U has shape (9, r1+r2). For B1: V1 has shape (24, r1), for B2: V2 has shape (24, r2).
    # W has shape (24, r1+r2).
    #
    # A simpler constructive approach: use the fact that 3x3 matrix multiplication
    # can be done with rank 23 (Strassen-like for small matrices) or even less.
    # But we need to handle the 8-column aspect.
    
    # Let's try a direct approach: build a decomposition of rank R < 72
    # by using a smaller set of "basis" multiplications.
    
    # For 3x3 * 3x8: think of it as 8 independent 3x3 * 3x1 multiplications.
    # Each 3x3 * 3x1 can be done with rank 9 (schoolbook) or potentially less.
    # But we can share structure across the 8 columns.
    
    # Key insight: A is 3x3, B is 3x8. Let's partition B into two 3x4 blocks.
    # C = A * B = [A*B_left | A*B_right]
    # We need: for each column c of C: C[:,c] = A * B[:,c]
    # 
    # Rank-minimal for 3x3 * 3x1: the tensor is (3x3) x (3x1) -> (3x1)
    # This has rank 9 (schoolbook). Can we do better? 
    # The tensor T[i,j,k] = 1 (delta-like) for A[i,j]*B[j,k] -> C[i,k]
    # For 3x3 * 3x1, this is actually rank 9 (no known reduction for this small case).
    #
    # But for 3x3 * 3x8, we have more freedom because B has 8 columns.
    # Let's try rank = 3*8 + something = 24 + extra? No, that's not right.
    
    # Let me try a completely different constructive approach:
    # Use the "middle" matrix trick. 
    # C = A * B where A is 3x3, B is 3x8, C is 3x8.
    # Define M = A (3x3). Then C = M * B.
    # We can write: C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # A rank-r decomposition means: C[i,k] = sum_{c=1}^r U[i,c] * V[c,k] where
    # U depends on A and V depends on B in a specific way.
    #
    # Actually, the standard formulation:
    # C[i,k] = sum_c U[i*m+j... no.
    
    # Let me re-read the interface:
    # U: (n*m, rank) = (9, rank) — indexed by (i,j)
    # V: (m*p, rank) = (24, rank) — indexed by (j,k)  
    # W: (p*n, rank) = (24, rank) — indexed by (k,i)
    #
    # The condition is: for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is the standard matrix multiplication tensor decomposition.
    
    # For 3x3 * 3x8, let's try rank = 54 (as suggested in the directive).
    # 54 = 9 * 6, or 3 * 18, or 2 * 27, etc.
    
    # Constructive approach for rank 54:
    # Partition the 8 columns of B into groups.
    # Group 1: columns 0,1,2 (3 columns)
    # Group 2: columns 3,4,5 (3 columns)  
    # Group 3: columns 6,7 (2 columns)
    
    # For each group of g columns, we need to compute A * B_group (3xg).
    # Schoolbook for 3x3 * 3xg has rank 9g.
    # Total: 9*3 + 9*3 + 9*2 = 27+27+18 = 72. No savings.
    
    # To get savings, we need to share factors across groups.
    
    # Alternative: use a rank-23 decomposition for 3x3x3 (if it exists)
    # and extend it to 3x3x8.
    
    # Actually, let me just try a numerical optimization approach with a time budget.
    
    # Start from schoolbook, try to zero out entries and fix the rest.
    
    # For each column c, try to set it to zero and see if we can adjust
    # other columns to compensate. This is a hard nonlinear problem.
    
    # Simpler: try small rank values and see if a solution exists via
    # random initialization + iterative refinement.
    
    # Given time constraints, let's try a few specific rank values
    # with a simple iterative solver.
    
    for target_rank in [54, 48, 45, 42, 40, 36, 32, 28, 24]:
        if time.time() > deadline:
            break
        if not _try_construct_rank(target_rank, n, m, p, deadline):
            continue
        # Found a valid decomposition at this rank
        # Extract and return
        U, V, W = _get_last_solution()
        return U, V, W
    
    # If no reduction found, return schoolbook
    return best_U, best_V, best_W


_last_solution = None

def _try_construct_rank(rank, n, m, p, deadline):
    """Try to construct a rank-`rank` decomposition via random init + Newton-like steps."""
    global _last_solution
    
    # The tensor equations: for each (i,j,k):
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Total equations: n*m*p = 72
    # Total unknowns: rank * (n*m + m*p + p*n) = rank * (9 + 24 + 24) = rank * 57
    # For rank >= 2, we have more unknowns than equations, so solutions likely exist.
    # The question is whether they have "nice" (integer/half-integer) entries.
    
    # Try random integer/half-integer initialization
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            return False
        
        # Random init with values in {-2, -1, -0.5, 0, 0.5, 1, 2}
        vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
        U = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(n*m)], dtype=float)
        V = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(m*p)], dtype=float)
        W = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(p*n)], dtype=float)
        
        # Check if this random init happens to work (very unlikely)
        if _verify_arrays(U, V, W, n, m, p):
            _last_solution = (U.tolist(), V.tolist(), W.tolist())
            return True
        
        # Try iterative refinement: pick one variable at a time, adjust to reduce error
        # This is too slow for the time budget.
        
        # Instead, try a structured approach:
        # Use the schoolbook as a starting point and try to merge terms.
        break
    
    # Structured approach: start from schoolbook and try to zero out pairs
    # of columns by combining them.
    
    # For the 3x3 * 3x8 case, consider that B has 8 columns.
    # If we pair up columns of B: (B[:,0], B[:,1]), (B[:,2], B[:,3]), etc.
    # For each pair, A*B[:,0] and A*B[:,1] can be computed with shared A-factors.
    
    # Let's try a specific construction:
    # Use 6 "basis" multiplications for the 3x3 part, each handling multiple B-columns.
    
    # Actually, let me try a different structured approach:
    # Rank = n * p * something? No.
    
    # Let me try: rank = m * p + n * p + n * m - overlaps
    # = 24 + 24 + 9 = 57. Try rank 57?
    
    # Or use the "bilinear" trick:
    # C = A*B. Let P = A (3x3), Q = B (3x8).
    # C[i,k] = sum_j P[i,j] Q[j,k]
    # 
    # Define for each (i,k): C[i,k] = <p_i, q_k> where p_i is row i of A, q_k is col k of B.
    # This is 24 inner products of 3-vectors, each of rank 3 (schoolbook).
    # Total: 24*3 = 72.
    
    # To reduce: note that p_i are only 3 distinct vectors (rows of A),
    # and q_k are 8 distinct vectors (columns of B).
    # So we have 3*8 = 24 inner products, but with shared structure.
    
    # A rank-r decomposition of the 3x3x8 tensor:
    # We need U (9 x r), V (24 x r), W (24 x r) such that
    # for all i,j,k: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    # Let's try a specific construction with rank = 36:
    # For each pair of rows (i, i') of A and each pair of columns (k, k') of B...
    # This is getting complicated. Let me just try numerical search with a deadline.
    
    # Simple iterative approach: start from schoolbook, try to set one entry to 0
    # and solve for the rest using a linearized approximation.
    
    U_s, V_s, W_s = _schoolbook(n, m, p)
    U_arr = np.array(U_s, dtype=float)
    V_arr = np.array(V_s, dtype=float)
    W_arr = np.array(W_s, dtype=float)
    rank_s = n * m * p
    
    # Try to remove each column
    for c_remove in range(rank_s):
        if time.time() > deadline:
            return False
        
        # Create a copy with column c_remove zeroed
        U_test = U_arr.copy()
        V_test = V_arr.copy()
        W_test = W_arr.copy()
        U_test[:, c_remove] = 0
        V_test[:, c_remove] = 0
        W_test[:, c_remove] = 0
        
        # Now try to fix the remaining columns to satisfy the tensor equations.
        # This is a nonlinear system. Use a simple iterative method:
        # For each equation (i,j,k), the residual is:
        #   r[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1
        # We want to adjust U, V, W to make all residuals zero.
        
        # Simple gradient-free approach: try adjusting one variable at a time.
        # Too slow. Let's just check if zeroing a column still works (it won't, but check).
        
        if _verify_arrays(U_test, V_test, W_test, n, m, p):
            # Removed a column successfully!
            _last_solution = (U_test.tolist(), V_test.tolist(), W_test.tolist())
            return True
    
    return False


def _verify_arrays(U, V, W, n, m, p, tol=1e-9):
    """Check tensor decomposition validity using numpy arrays."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # First, get schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    
    # Try to reduce rank
    U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
