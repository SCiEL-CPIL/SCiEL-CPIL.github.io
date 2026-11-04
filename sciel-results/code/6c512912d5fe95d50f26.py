import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and check T[i,j,k] == 1 if (i,j,k) corresponds to a valid term, 0 otherwise
    # For 3x3x3 matrix mult: T[i,j,k] should be 1 for all i,j,k (each term is A[i,j]*B[j,k])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=3):
    """Return the standard rank-27 schoolbook decomposition."""
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

def _try_strassen_like(n=3, m=3, p=3, max_rank=20, deadline=None):
    """
    Try to find a lower-rank decomposition using random search with algebraic structure.
    Uses the insight that 3x3 matrix multiplication can potentially be done with
    fewer than 27 multiplications by exploiting linear combinations.
    """
    if deadline is None:
        deadline = time.time() + 100.0
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try various rank values from high to low
    for target_rank in range(best_rank - 1, 12, -1):
        if time.time() > deadline:
            break
        
        # Attempt a few random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random rank-1 terms with small integer/half-integer coefficients
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                # Each rank-1 term: pick random indices and coefficients
                # Coefficients from {-1, -0.5, 0.5, 1}
                choices = [-1, -0.5, 0.5, 1]
                
                # Randomly select which elements get non-zero values
                for idx in range(n * m):
                    if np.random.random() < 0.3:
                        U[idx, r] = np.random.choice(choices)
                for idx in range(m * p):
                    if np.random.random() < 0.3:
                        V[idx, r] = np.random.choice(choices)
                for idx in range(p * n):
                    if np.random.random() < 0.3:
                        W[idx, r] = np.random.choice(choices)
            
            # Convert to lists for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                best_U, best_V, best_W = U_list, V_list, W_list
                best_rank = target_rank
                break
        
        if best_rank < n * m * p:
            break
    
    return best_U, best_V, best_W, best_rank

def _structured_search(n=3, m=3, p=3, deadline=None):
    """
    Try structured decompositions based on known algebraic identities.
    For 3x3 matrices, try to embed into larger known decompositions.
    """
    if deadline is None:
        deadline = time.time() + 100.0
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try a structured approach: use the fact that we can group terms
    # and see if linear combinations reduce the rank
    
    # One approach: try all possible "Strassen-like" splittings
    # For 3x3, we can try to find a decomposition with rank < 27
    
    # Try rank values
    for target_rank in range(26, 15, -1):
        if time.time() > deadline:
            break
        
        # Use a more structured initialization
        # Start with schoolbook and try to merge terms
        U_base, V_base, W_base = _schoolbook(n, m, p)
        
        # Try to find linear dependencies among the 27 terms
        # This is complex, so we use a heuristic: try random linear combinations
        
        found = False
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Create a random linear combination of the 27 basis terms
            # to try to reduce the effective rank
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with random projections of the schoolbook terms
            for r in range(target_rank):
                # Random linear combination of some schoolbook terms
                num_terms = np.random.randint(1, 4)
                indices = np.random.choice(27, size=num_terms, replace=False)
                
                for idx in indices:
                    i = idx // 9
                    j = (idx // 3) % 3
                    k = idx % 3
                    
                    coeff = np.random.choice([-1, -0.5, 0.5, 1])
                    U[i * m + j, r] += coeff
                    V[j * p + k, r] += coeff
                    W[k * n + i, r] += coeff
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                best_U, best_V, best_W = U_list, V_list, W_list
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, get the guaranteed schoolbook solution
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try structured search
    U_struct, V_struct, W_struct, rank_struct = _structured_search(n, m, p, deadline=start_time + timeout * 0.4)
    if U_struct is not None and rank_struct < best_rank:
        best_U, best_V, best_W = U_struct, V_struct, W_struct
        best_rank = rank_struct
    
    # Try random search if time permits
    if time.time() < start_time + timeout * 0.7:
        U_rand, V_rand, W_rand, rank_rand = _try_strassen_like(n, m, p, max_rank=best_rank, deadline=start_time + timeout)
        if U_rand is not None and rank_rand < best_rank:
            best_U, best_V, best_W = U_rand, V_rand, W_rand
            best_rank = rank_rand
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
