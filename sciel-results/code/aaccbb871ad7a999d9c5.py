"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import numpy as np
import random

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy einsum."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    return np.allclose(T, 1.0, atol=tol)

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def _try_kronecker_block(n, m, p, deadline):
    """
    Try to exploit block structure.
    For 4x5x5: decompose 4=2*2, try 2x2 Strassen-like on the 4-dim axis.
    Pad 5 to 8 (2^3) or use 5 directly with careful indexing.
    
    Strategy: Use Kronecker product of a 2x2x2 Strassen (rank 7) with a 1x5x5 schoolbook (rank 25),
    but that gives rank 175 which is worse.
    
    Alternative: Treat the 4x5x5 as two 2x5x5 blocks and try to share computations.
    """
    # For 4x5x5, we can view it as computing two 2x5 matrices multiplied by a 5x5.
    # C_top = A_top * B (2x5 * 5x5 = 2x5)
    # C_bot = A_bot * B (2x5 * 5x5 = 2x5)
    # Total schoolbook: 2*2*5*5 = 100
    
    # Try: use a shared "middle" representation.
    # Let M = [A_top; A_bot] * B = C (4x5)
    # This is just standard multiplication, no obvious sharing.
    
    # Try padding approach: pad 5 to 6, use 4x6x6 then extract.
    # But that increases rank.
    
    # Try: 4 = 2+2, use two independent 2x5x5 computations.
    # Each 2x5x5 has schoolbook rank 50. Total 100. No gain.
    
    return None

def _local_search_reduce(U, V, W, n, m, p, target_rank, deadline, max_iters=500):
    """
    Start from a valid decomposition of rank > target_rank and try to reduce to target_rank
    by merging/eliminating columns.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Try to eliminate one column at a time
    for _ in range(current_rank - target_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        # Try removing each column
        for c in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            
            if _verify_decomposition(U_test, V_test, W_test, n, m, p):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                found = True
                break
        
        if not found:
            # Try modifying entries to allow removal
            # This is complex; for now, just break
            break
    
    if U.shape[1] == target_rank:
        return U, V, W
    return None

def _random_sparse_search(n, m, p, target_rank, num_trials, deadline):
    """Random sparse initialization with verification."""
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for c in range(target_rank):
            # Each column: sparse with 1-3 nonzeros
            nnz_u = random.randint(1, 3)
            nnz_v = random.randint(1, 3)
            nnz_w = random.randint(1, 3)
            
            u_idx = random.sample(range(n * m), nnz_u)
            v_idx = random.sample(range(m * p), nnz_v)
            w_idx = random.sample(range(p * n), nnz_w)
            
            for idx in u_idx:
                U[idx, c] = random.choice([-1, 1])
            for idx in v_idx:
                V[idx, c] = random.choice([-1, 1])
            for idx in w_idx:
                W[idx, c] = random.choice([-1, 1])
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _structured_search(n, m, p, target_rank, deadline):
    """
    Try structured decompositions.
    For 4x5x5, try to use the fact that 4 divides evenly.
    
    Idea: Use a "layered" approach where we split the 4 output rows into groups
    and share intermediate computations.
    """
    # Try: represent as sum of rank-1 tensors with structured patterns
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    # For small target_rank, try all-ones patterns with signs
    if target_rank <= 20:
        # Very low rank is unlikely, but try
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def construct():
    deadline = time.time() + 135.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: for matrix multiplication, rank >= max(n*m, m*p, p*n) = 25
    # But actual minimum is likely close to 100 for these dimensions.
    
    # Try structured approaches first
    for target_rank in range(99, 70, -1):
        if time.time() > deadline - 15:
            break
        
        # Try structured search
        result = _structured_search(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        # Try random sparse search with fewer trials for higher ranks
        num_trials = 20 if target_rank > 90 else 50
        result = _random_sparse_search(n, m, p, target_rank, num_trials, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # Try local reduction from schoolbook
    if time.time() < deadline - 10:
        result = _local_search_reduce(U_sb, V_sb, W_sb, n, m, p, best_rank - 1, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
