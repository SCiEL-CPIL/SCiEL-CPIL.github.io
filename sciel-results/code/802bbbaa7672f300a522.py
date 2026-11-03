"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 5, 5]
_START_TIME = time.time()
_TIME_LIMIT = 120

# EVOLVE-BLOCK-START
def _verify_decomp(U, V, W, n, m, p, tol=1e-6):
    """Verify exact decomposition: for all i,k,j1,j2: sum_r W[k*n+i,r] U[i*m+j1,r] V[j2*p+k,r] == delta_{j1,j2}"""
    r = len(U[0]) if isinstance(U, list) else U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for r_ in range(r):
                        s += W[k*n+i, r_] * U[i*m+j1, r_] * V[j2*p+k, r_]
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > tol:
                        return False
    return True

def _round_half(X):
    """Round matrix to nearest half-integer."""
    return np.round(X * 2.0) / 2.0

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _als_optimize(U, V, W, n, m, p, r, max_iters=300, deadline=None):
    """ALS optimization for the 375-equation system. Returns optimized (U, V, W)."""
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Update W: for each (k,i), solve linear system
        for k in range(p):
            for i in range(n):
                A_W = np.zeros((m*m, r))
                for j1 in range(m):
                    for j2 in range(m):
                        A_W[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                b = np.zeros(m*m)
                for j in range(m):
                    b[j*m+j] = 1.0
                try:
                    w, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                    W[k*n+i, :] = w
                except np.linalg.LinAlgError:
                    pass
        # Update U: for each (i,j1)
        for i in range(n):
            for j1 in range(m):
                A_U = np.zeros((p*m, r))
                b = np.zeros(p*m)
                for k in range(p):
                    for j2 in range(m):
                        A_U[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                        b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                try:
                    u, *_ = np.linalg.lstsq(A_U, b, rcond=None)
                    U[i*m+j1, :] = u
                except np.linalg.LinAlgError:
                    pass
        # Update V: for each (j2,k)
        for j2 in range(m):
            for k in range(p):
                A_V = np.zeros((n*m, r))
                b = np.zeros(n*m)
                for i in range(n):
                    for j1 in range(m):
                        A_V[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                        b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                try:
                    v, *_ = np.linalg.lstsq(A_V, b, rcond=None)
                    V[j2*p+k, :] = v
                except np.linalg.LinAlgError:
                    pass
    return U, V, W

def _try_strassen_2x2(n, m, p):
    """
    Try to exploit Strassen-like structure: decompose the 3x5x5 multiplication
    by treating it as blocks and using Strassen's 7-multiplication formula for 2x2 blocks.
    
    For 3x5x5: think of A as 3x5, B as 5x5.
    We can partition A into rows: row 0 (1x5), rows 1-2 (2x5).
    B into columns: col 0 (5x1), cols 1-2 (5x2), cols 3-4 (5x2).
    
    Actually, let's think more carefully. The standard approach:
    A (3x5) * B (5x5) = C (3x5)
    
    Partition A = [A1; A2] where A1 is 1x5, A2 is 2x5
    Partition B = [B1, B2, B3] where B1 is 5x1, B2 is 5x2, B3 is 5x2
    
    C1 = A1*B1 (1x1) - 5 mults
    C2 = A1*B2 (1x2) - 10 mults
    C3 = A1*B3 (1x2) - 10 mults
    C4 = A2*B1 (2x1) - 10 mults
    C5 = A2*B2 (2x2) - use Strassen: 7 mults instead of 4... wait, 2x2 is 4 mults, Strassen gives 7? No, Strassen is for 2x2*2x2 = 4 mults -> 7 mults? That's worse.
    
    Wait, Strassen reduces 2x2*2x2 from 4 scalar mults to 7? No! Strassen reduces n x n * n x n. For 2x2, standard is 4 mults. Strassen gives 7 mults which is WORSE for 2x2.
    
    Strassen is for when you have larger blocks. The idea is: if you have a 4x4*4x4, standard is 16 mults, Strassen gives 7*(2*2)=28... no.
    
    Actually Strassen: to multiply two n x n matrices, use 7 multiplications of (n/2)x(n/2) matrices. So for 2x2: 7 multiplications of 1x1 = 7 scalar mults vs 4 standard. So Strassen is worse for 2x2.
    
    For 4x4: 7 multiplications of 2x2 = 7*4 = 28 scalar mults vs 16 standard. Still worse.
    
    Strassen only helps asymptotically. For small matrices, it's not helpful.
    
    Let me reconsider the inspiration: "Using Strassen's algorithm for 2x2x2 sub-blocks within a 3x5x5 decomposition reduces the rank from 75 to 71 by exploiting algebraic structure in the 2x2x2 blocks, saving 4 multiplications."
    
    So the idea is different. It's about finding a rank-7 decomposition for a 2x2x2 tensor that shares structure with the 3x5x5 problem.
    
    Actually, I think the key insight is: the 3x5x5 matrix multiplication can be decomposed into independent sub-problems. If we can find that some 2x2x2 sub-blocks can be handled with rank 7 instead of 8, we save multiplications.
    
    Let me think about this differently. The 3x5x5 multiplication has rank at most 75. We want to find a lower rank.
    
    One approach: use the fact that the problem has structure. For example, if we can group some of the 75 terms and show that a subset can be done with fewer terms.
    
    Let me try a different approach: use the ALS optimization from the inspiration, but start from the schoolbook decomposition and try to reduce rank by removing columns and re-optimizing.
    """
    pass

def _is_half_integer_matrix(M, tol=1e-9):
    """Check if all entries are integers or half-integers."""
    M2 = M * 2.0
    return np.all(np.abs(M2 - np.round(M2)) < tol)

def _snap_to_half(M):
    """Snap matrix to nearest half-integer and verify."""
    M_snapped = np.round(M * 2.0) / 2.0
    if _is_half_integer_matrix(M_snapped):
        return M_snapped
    return None

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT - 5  # safety margin

    # Start with schoolbook decomposition
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook is valid and half-integer (it is, but just in case)
    if not _verify_decomp(best_U, best_V, best_W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    if not _is_half_integer_matrix(best_U) or not _is_half_integer_matrix(best_V) or not _is_half_integer_matrix(best_W):
        raise ValueError("Schoolbook decomposition is not half-integer")

    np.random.seed(42)
    
    # Try to find a better decomposition using a randomized local search
    # We will try ranks from 48 (known optimal from knowledge) up to 74.
    # For each rank, we try a few random restarts with a fast local search.
    
    for r in range(48, best_rank):
        if time.time() > deadline - 2:
            break
            
        # Try a few random initializations
        for restart in range(3):
            if time.time() > deadline - 2:
                break
            
            # Initialize with small random half-integers
            U_init = (np.random.randint(-2, 3, size=(n*m, r)) * 0.5)
            V_init = (np.random.randint(-2, 3, size=(m*p, r)) * 0.5)
            W_init = (np.random.randint(-2, 3, size=(p*n, r)) * 0.5)
            
            # Local Search: Randomly perturb one entry at a time
            # This is slow, so we use a heuristic:
            # Solve for W given U,V (least squares), snap, verify.
            # If it fails, perturb U or V slightly and repeat.
            
            current_U, current_V, current_W = U_init.copy(), V_init.copy(), W_init.copy()
            
            # Run a few steps of "Snap and Solve"
            for step in range(50):
                if time.time() > deadline - 2:
                    break
                    
                # Update W: For each (k,i), solve linear system
                valid_step = True
                for k in range(p):
                    for i in range(n):
                        A_W = np.zeros((m*m, r))
                        for j1 in range(m):
                            for j2 in range(m):
                                A_W[j1*m+j2, :] = current_U[i*m+j1, :] * current_V[j2*p+k, :]
                        b = np.zeros(m*m)
                        for j in range(m):
                            b[j*m+j] = 1.0
                        try:
                            w, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                            # Snap w to half-integers
                            w_snapped = np.round(w * 2.0) / 2.0
                            current_W[k*n+i, :] = w_snapped
                        except np.linalg.LinAlgError:
                            valid_step = False
                            break
                    if not valid_step:
                        break
                
                if not valid_step:
                    # Perturb U or V randomly
                    if np.random.rand() < 0.5:
                        idx = np.random.choice(n*m)
                        col = np.random.choice(r)
                        current_U[idx, col] += np.random.choice([-0.5, 0.5])
                    else:
                        idx = np.random.choice(m*p)
                        col = np.random.choice(r)
                        current_V[idx, col] += np.random.choice([-0.5, 0.5])
                    continue

                # Check if current solution is valid
                if _verify_decomp(current_U, current_V, current_W, n, m, p):
                    # Ensure strict half-integer
                    if _is_half_integer_matrix(current_U) and _is_half_integer_matrix(current_V) and _is_half_integer_matrix(current_W):
                        if r < best_rank:
                            best_rank = r
                            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                        break # Success for this r, move to next lower r? No, r is increasing in loop?
                        # We want lowest rank. Loop is 48 to 74. If we find 48, we should stop.
                        # So if we find a solution, we can return immediately if r is the lowest we tried?
                        # Actually, we should try to find the *lowest* possible.
                        # So if we find r=48, we can stop.
                        # If we find r=50, we should try r=48 again?
                        # The loop goes 48, 49, 50...
                        # If we fail at 48, we try 49. If we succeed at 49, best_rank=49.
                        # Then we try 50? No, 50 is worse.
                        # So we should break if we find a solution?
                        # Yes, because we are iterating from low to high.
                        # If we find a solution at r, it's the best we can hope for in this loop (since lower r's failed).
                        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
    # If no improvement found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
