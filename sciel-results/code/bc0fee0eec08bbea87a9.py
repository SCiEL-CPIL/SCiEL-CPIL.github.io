"""Matrix multiplication 3x4x5 - explicit rank-minimal decomposition attempt."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_expected_tensor(n, m, p):
    """Build the expected matmul tensor T[x,y,z] where:
    x = i*m+j (A index), y = j*p+k (B index), z = k*n+i (C index, column-major)
    T[x,y,z] = 1 if x=i*m+j, y=j*p+k, z=k*n+i for some consistent (i,j,k), else 0.
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _compute_decomposed_tensor(U, V, W):
    """Compute D[x,y,z] = sum_r U[x,r] * V[y,r] * W[z,r]"""
    return np.einsum('xr,yr,zr->xyz', U, V, W)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    expected = _build_expected_tensor(n, m, p)
    D = _compute_decomposed_tensor(U, V, W)
    return np.allclose(D, expected, atol=tol)

def _try_rank(n, m, p, rank, num_trials=50, time_limit=2.0, seed=None):
    """Try to find a valid rank-r decomposition via random search + local refinement.
    
    Returns (U, V, W) if found, else None.
    """
    start = time.time()
    rng = np.random.default_rng(seed)
    T = _build_expected_tensor(n, m, p)
    
    for trial in range(num_trials):
        if time.time() - start > time_limit:
            break
        
        # Random initialization
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        W = rng.standard_normal((p*n, rank))
        
        # Normalize to help with convergence
        for r in range(rank):
            norm = np.sqrt(np.sum(U[:,r]**2) * np.sum(V[:,r]**2) * np.sum(W[:,r]**2))
            if norm > 1e-12:
                scale = norm ** (1.0/3.0)
                U[:,r] /= scale
                V[:,r] /= scale
                W[:,r] /= scale
        
        # Check if already close
        D = _compute_decomposed_tensor(U, V, W)
        err = np.max(np.abs(D - T))
        if err < 1e-9:
            return U, V, W
        
        # Try simple gradient-free refinement: alternating minimization
        # This is a heuristic; for exact integer/half-integer solutions,
        # random search may find them if rank is sufficient.
        # For now, just check if random init works (unlikely for rank < schoolbook)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    
    # Verify schoolbook (sanity check)
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p):
        # Should never happen, but return schoolbook anyway
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to find lower-rank decompositions
    # For 3x4x5, known bounds suggest rank might be less than 60, but it's not proven.
    # We'll try a few ranks below schoolbook with limited time.
    best_result = (schoolbook_rank, U_school, V_school, W_school)
    
    # Try ranks from schoolbook-1 down, but limit total time
    remaining_time = deadline - time.time()
    ranks_to_try = list(range(schoolbook_rank - 1, max(0, schoolbook_rank - 10), -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:  # 5s safety margin
            break
        
        time_per_rank = max(0.5, (deadline - time.time() - 5) / max(1, len(ranks_to_try)))
        result = _try_rank(n, m, p, rank, num_trials=30, time_limit=time_per_rank, seed=rank)
        
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = (rank, U, V, W)
                break  # Found a valid lower-rank decomposition
    
    rank, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
