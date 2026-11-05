import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def check_time():
    """Check if we should stop searching."""
    return (time.time() - START_TIME) >= TIME_LIMIT

def verify_decomposition(U, V, W, n, m, p, rank, atol=1e-8):
    """Verify that the decomposition is exact."""
    U_r = np.array(U).reshape(n, m, rank)
    V_r = np.array(V).reshape(m, p, rank)
    W_r = np.array(W).reshape(p, n, rank)
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p))
    return np.allclose(T_recon, T_target, atol=atol)

def build_schoolbook(n, m, p):
    """Build the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def build_strassen_block(n, m, p):
    """
    Build a Strassen-based decomposition.
    For 4x5 * 5x6, we can split into blocks.
    
    Strategy: Split the middle dimension (5) into 2+3 or use 2x2 blocks where possible.
    Actually, let's try a different approach: use the known rank-7 Strassen for 2x2x2.
    
    For 4x5 * 5x6:
    We can think of this as multiplying a (2*2) x (2*2+1) matrix by (2*2+1) x (3*2) matrix.
    This is complex. Let's try a simpler structured approach.
    
    Actually, let's try to use the fact that we can combine some terms.
    The key insight from the cell findings is that a block Strassen decomposition 
    achieved rank 108 (vs 120 schoolbook). Let's implement a similar idea.
    
    We'll split the matrices into blocks and use Strassen's algorithm for the 2x2 blocks.
    """
    # For 4x5 * 5x6, let's try splitting into 2x2 blocks where possible
    # n=4=2*2, m=5=2*2+1, p=6=2*3
    
    # This is getting complex. Let's try a different approach:
    # Use the fact that we can reduce the rank by combining terms that share structure.
    
    # A simpler approach: try to find a decomposition with rank < 120 using
    # a structured random search with integer/half-integer entries.
    
    # For now, let's just return the schoolbook as a baseline and try to improve
    # in the search phase.
    return build_schoolbook(n, m, p)

def try_structured_search(n, m, p, max_rank=None):
    """
    Search for a lower-rank decomposition using structured random initialization.
    We use integer and half-integer entries as suggested by the inspirations.
    """
    if max_rank is None:
        max_rank = n * m * p - 1
    
    best_result = None
    vals = np.array([-1, -0.5, 0, 0.5, 1])
    
    # Try ranks from low to high, but start with a reasonable range
    # Based on the cell findings, rank 108 was achieved, so try around there
    target_ranks = list(range(max(10, n*m*p - 20), max_rank + 1))
    
    for rank in target_ranks:
        if check_time():
            break
        
        # Multiple random restarts for each rank
        for restart in range(50):
            if check_time():
                break
            
            # Random initialization with integer/half-integer values
            U = np.random.choice(vals, size=(n * m, rank))
            V = np.random.choice(vals, size=(m * p, rank))
            W = np.random.choice(vals, size=(p * n, rank))
            
            # Check if this random decomposition works
            if verify_decomposition(U, V, W, n, m, p, rank):
                best_result = (U.copy(), V.copy(), W.copy(), rank)
                return best_result
            
            # If not, try a local refinement
            # Use gradient-free optimization to adjust the entries
            for iteration in range(20):
                if check_time():
                    break
                
                # Compute the error
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                error = np.max(np.abs(T_recon - 1.0))
                
                if error < 1e-8:
                    best_result = (U.copy(), V.copy(), W.copy(), rank)
                    return best_result
                
                # Try to improve by adjusting one factor at a time
                # This is a very basic coordinate descent
                if iteration % 5 == 0:
                    # Perturb U
                    U_new = U.copy()
                    idx = np.random.randint(0, U.shape[0])
                    col = np.random.randint(0, rank)
                    U_new[idx, col] += np.random.choice([-0.5, 0.5])
                    if verify_decomposition(U_new, V, W, n, m, p, rank):
                        U = U_new
                        break
                    # Check if error decreased
                    U_r = U_new.reshape(n, m, rank)
                    V_r = V.reshape(m, p, rank)
                    W_r = W.reshape(p, n, rank)
                    T_recon_new = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    error_new = np.max(np.abs(T_recon_new - 1.0))
                    if error_new < error:
                        U = U_new
                        error = error_new
                elif iteration % 5 == 1:
                    # Perturb V
                    V_new = V.copy()
                    idx = np.random.randint(0, V.shape[0])
                    col = np.random.randint(0, rank)
                    V_new[idx, col] += np.random.choice([-0.5, 0.5])
                    if verify_decomposition(U, V_new, W, n, m, p, rank):
                        V = V_new
                        break
                    V_r = V_new.reshape(m, p, rank)
                    U_r = U.reshape(n, m, rank)
                    W_r = W.reshape(p, n, rank)
                    T_recon_new = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    error_new = np.max(np.abs(T_recon_new - 1.0))
                    if error_new < error:
                        V = V_new
                        error = error_new
                else:
                    # Perturb W
                    W_new = W.copy()
                    idx = np.random.randint(0, W.shape[0])
                    col = np.random.randint(0, rank)
                    W_new[idx, col] += np.random.choice([-0.5, 0.5])
                    if verify_decomposition(U, V, W_new, n, m, p, rank):
                        W = W_new
                        break
                    W_r = W_new.reshape(p, n, rank)
                    U_r = U.reshape(n, m, rank)
                    V_r = V.reshape(m, p, rank)
                    T_recon_new = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    error_new = np.max(np.abs(T_recon_new - 1.0))
                    if error_new < error:
                        W = W_new
                        error = error_new
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the guaranteed schoolbook decomposition
    U_school, V_school, W_school, rank_school = build_schoolbook(n, m, p)
    
    # Verify it works
    if not verify_decomposition(U_school, V_school, W_school, n, m, p, rank_school):
        # This should never happen, but just in case
        pass
    
    # Try to find a lower-rank decomposition
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Attempt search for lower rank
    result = try_structured_search(n, m, p, max_rank=rank_school - 1)
    if result is not None:
        U_found, V_found, W_found, rank_found = result
        # Verify the found decomposition
        if verify_decomposition(U_found, V_found, W_found, n, m, p, rank_found):
            best_U, best_V, best_W, best_rank = U_found, V_found, W_found, rank_found
            print(f"Found lower-rank decomposition with rank {rank_found}")
    
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
