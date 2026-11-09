import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W, rank


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify the decomposition produces the correct tensor."""
    rank = U.shape[1]
    # Target: T[i,j,k] = 1 for all i,j,k
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    T_decomp = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        T_decomp += u * v * w
    T_target = np.ones((n, m, p))
    return np.allclose(T_decomp, T_target, atol=tol)


def _try_structured_decomposition(n, m, p, target_rank, deadline):
    """Try to find a decomposition of given rank using structured approach."""
    # Strategy: Start with schoolbook and try to merge terms
    # For 3x3x5, try to find patterns where terms can be combined
    
    np.random.seed(123)
    
    # Try various structured patterns
    for pattern in range(50):
        if time.time() > deadline - 10:
            break
        
        U = np.zeros((n * m, target_rank), dtype=np.int64)
        V = np.zeros((m * p, target_rank), dtype=np.int64)
        W = np.zeros((p * n, target_rank), dtype=np.int64)
        
        # Initialize with random sparse patterns
        for c in range(target_rank):
            # Each term covers a subset of indices
            # Try different sparsity levels
            sparsity = np.random.choice([0.1, 0.2, 0.3, 0.5])
            
            # Random entries from {-1, 0, 1}
            for idx in range(n * m):
                if np.random.random() < sparsity:
                    U[idx, c] = np.random.choice([-1, 1])
            for idx in range(m * p):
                if np.random.random() < sparsity:
                    V[idx, c] = np.random.choice([-1, 1])
            for idx in range(p * n):
                if np.random.random() < sparsity:
                    W[idx, c] = np.random.choice([-1, 1])
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None


def _try_local_refinement(U, V, W, n, m, p, max_iter=50, deadline=None):
    """Try to refine a valid decomposition to find lower rank."""
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        
        # Try removing one column and see if we can adjust others
        for c_remove in range(rank):
            if deadline and time.time() > deadline - 5:
                break
            
            U_trial = np.delete(U, c_remove, axis=1)
            V_trial = np.delete(V, c_remove, axis=1)
            W_trial = np.delete(W, c_remove, axis=1)
            
            # Check if still valid
            if _verify(U_trial, V_trial, W_trial, n, m, p):
                return U_trial, V_trial, W_trial
    
    return None


def construct():
    deadline = time.time() + 110  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Best known for 3x3x3 is 23, so for 3x3x5 we might get around 40-43
    # Try ranks from 40 up to 44
    
    for target_rank in range(40, 45):
        if time.time() > deadline - 20:
            break
        
        # Try structured random search
        result = _try_structured_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # Try local refinement on the best found
    if time.time() < deadline - 10:
        refined = _try_local_refinement(best_U, best_V, best_W, n, m, p, 
                                        max_iter=20, deadline=deadline)
        if refined is not None:
            best_U, best_V, best_W = refined
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
