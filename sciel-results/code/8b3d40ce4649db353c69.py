"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if the given factor matrices decompose the matrix multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Convention: W[k*n + i, r] corresponds to output C[i, k]
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # D[x, y, z] = sum_r U[x, r] * V[y, r] * W[z, r]
    # x = i*m + j
    # y = j*p + k  
    # z = k*n + i
    D = np.einsum('xr,yr,zr->xyz', U_np, V_np, W_np)
    
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                expected[x, y, z] = 1.0
                
    return np.allclose(D, expected, atol=tol)

def _generate_schoolbook(n, m, p):
    """Generate the standard schoolbook decomposition with rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _try_rank_reduction(n, m, p, max_rank, time_limit):
    """
    Attempt to find a decomposition with rank < max_rank using structured search.
    Strategy: Start from schoolbook and try to merge terms or find linear dependencies.
    """
    start_time = time.time()
    
    # For this specific case (3x4x5), we know the schoolbook rank is 60.
    # Let's try to see if we can find a decomposition with rank 59, 58, etc.
    # 
    # One approach: consider that some terms might be expressible as linear combinations
    # of others. We'll try random perturbations and check if the tensor is preserved.
    
    # Actually, a more promising approach for small matrices:
    # Try all subsets of size (max_rank - 1) from the schoolbook terms and check
    # if they span the same tensor space. But this is computationally expensive.
    
    # Instead, let's try a random search with a bias towards the schoolbook solution.
    # We'll initialize near the schoolbook and try small modifications.
    
    current_rank = max_rank - 1
    attempts_per_rank = 50
    
    while current_rank >= max(1, max_rank - 5):  # Only try a few ranks below schoolbook
        if time.time() - start_time > time_limit:
            break
            
        for attempt in range(attempts_per_rank):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with schoolbook-like structure but with some randomization
            U = np.zeros((n * m, current_rank))
            V = np.zeros((m * p, current_rank))
            W = np.zeros((p * n, current_rank))
            
            # Try to use a subset of the schoolbook terms plus some random adjustments
            # First, pick 'current_rank' terms from the schoolbook
            all_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            
            if current_rank <= len(all_terms):
                # Randomly select terms
                selected_terms = np.random.choice(len(all_terms), size=current_rank, replace=False)
                for col, idx in enumerate(selected_terms):
                    i, j, k = all_terms[idx]
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
            else:
                # Use all terms plus some extra random columns
                for col, (i, j, k) in enumerate(all_terms):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                # Add random columns for the remaining
                for col in range(len(all_terms), current_rank):
                    U[:, col] = np.random.choice([0, 0.5, 1, -0.5, -1], size=n*m)
                    V[:, col] = np.random.choice([0, 0.5, 1, -0.5, -1], size=m*p)
                    W[:, col] = np.random.choice([0, 0.5, 1, -0.5, -1], size=p*n)
            
            # Check if this works
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        current_rank -= 1
        
    return None

def construct():
    """
    Constructs the minimal rank tensor decomposition for 3x4 matrix multiplication.
    Uses a search strategy to attempt lower ranks, falling back to schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Try to find a better rank than schoolbook
    # We limit the search time to ensure we don't timeout
    search_time_allowed = 120.0
    
    result = _try_rank_reduction(n, m, p, schoolbook_rank, search_time_allowed)
    
    if result is not None:
        return result
        
    # Fallback to schoolbook if search fails or times out
    return _generate_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
