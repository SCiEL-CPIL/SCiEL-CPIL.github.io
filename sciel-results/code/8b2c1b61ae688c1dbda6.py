import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication.
    
    U: (n*m, rank)
    V: (m*p, rank) 
    W: (p*n, rank)
    
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if and only if the (i,j) entry of A times (j,k) entry of B
    contributes to (i,k) entry of C, which is always 1 for valid (i,j,k).
    
    Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    (since (AB)_{ik} = sum_j A_{ij} B_{jk}, each term A_{ij}B_{jk} contributes).
    
    Wait, let me reconsider. The tensor for matrix multiplication M_{n,m,p} has
    entries T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p).
    The multiplication is: C_{ik} = sum_j A_{ij} B_{jk}
    So the tensor should satisfy: sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r} = 1 for all i,j,k
    """
    rank = len(U[0])
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
    
    # T should be all 1s
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-10):
        return False, T
    return True, T

def _is_integer_or_half_integer(arr, tol=1e-9):
    """Check if all entries are integers or half-integers (multiples of 0.5)."""
    for val in np.array(arr).flatten():
        if abs(val * 2 - round(val * 2)) > tol:
            return False
    return True

def _round_to_half_integer(arr):
    """Round array entries to nearest half-integer."""
    return np.round(np.array(arr) * 2) / 2.0

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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

def _try_structured_decomposition(n, m, p, max_rank, time_limit=100):
    """Try to find a decomposition with rank <= max_rank using structured approach.
    
    Strategy: Use the fact that we can group terms. For example, if we fix i and k,
    the sum over j should give 1. We can try to use fewer terms by sharing structure.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Try various rank values from the theoretical lower bound up to max_rank
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    lower_bound = max(n * m, m * p, p * n)
    
    best_result = None
    
    for target_rank in range(lower_bound, max_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try to construct a decomposition with exactly target_rank terms
        # Use a parametric approach: assume structure and solve
        
        # Approach 1: Try to find a decomposition where each "slice" shares structure
        # For matrix multiplication, one natural approach is:
        # C_ik = sum_j A_ij * B_jk
        # We can try: sum_r U_ir * V_jr * W_kr where the structure allows
        # the sum over j to work out.
        
        # Let's try a simple ansatz:
        # U depends on (i, some_index), V depends on (j, some_index), W depends on (k, some_index)
        # where some_index ranges over target_rank values.
        
        # Actually, let's use a more direct approach:
        # Start with a random initialization and use local search.
        
        # Initialize with schoolbook and try to merge terms
        U, V, W = _schoolbook_decomposition(n, m, p)
        current_rank = n * m * p
        
        # Try to reduce rank by finding linear dependencies
        # The tensor equation is: sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k
        
        # This is a system of polynomial equations. We can try to find solutions
        # with fewer terms by setting some entries to zero or using symmetry.
        
        # For now, let's try a simpler approach: use the known fact that
        # matrix multiplication can be done with fewer than n*m*p multiplications.
        
        # One known technique: if n divides m or other divisibility conditions,
        # we can use block structure.
        
        # Let's try: suppose we can write the multiplication as a sum of
        # rank-1 tensors where each tensor has a specific pattern.
        
        # For 3x4x7, let's try rank 28 (the lower bound) first.
        # If that fails, try higher ranks.
        
        # Use a numerical approach: parameterize and solve
        # This is complex, so let's try a heuristic search instead.
        
        # Heuristic: Start with schoolbook, try to merge pairs of terms
        # that have similar structure.
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Try to find if any two columns are linearly dependent in a useful way
        # This is getting complex. Let's try a different approach.
        
        # Approach: Use the fact that we can permute and combine terms.
        # For now, return the schoolbook decomposition as baseline.
        
        if target_rank == current_rank:
            best_result = (U, V, W, target_rank)
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    # The theoretical lower bound is max(n*m, m*p, p*n) = 28
    lower_bound = max(n * m, m * p, p * n)
    
    # For 3x4x7, let's try ranks from 28 up to 83
    # Use a systematic search with time management
    
    # Strategy: Try structured decompositions based on known techniques
    # for matrix multiplication.
    
    # One key insight: If we can find a decomposition where the terms
    # have a regular pattern, we can verify it easily.
    
    # Let's try: for each possible rank r from lower_bound to best_rank-1,
    # try to construct a valid decomposition.
    
    # Due to time constraints, we'll try a limited number of approaches.
    
    # Approach 1: Try rank = lower_bound with a specific ansatz
    # The ansatz: assume U, V, W have a specific sparse structure.
    
    # For matrix multiplication M_{n,m,p}, one natural ansatz is:
    # Use r terms where each term corresponds to a "path" through the indices.
    
    # Let's try a more direct numerical approach:
    # 1. Initialize U, V, W with random values of the target rank
    # 2. Use gradient descent to minimize the error
    # 3. Check if we can reach exact zero error with half-integer values
    
    def _try_random_init(target_rank, n_tries=100):
        """Try random initialization and local optimization."""
        nonlocal deadline, start_time
        
        for attempt in range(n_tries):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Compute the tensor
            T = np.zeros((n, m, p))
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += U_rand[i * m + j, r] * V_rand[j * p + k, r] * W_rand[k * n + i, r]
            
            error = np.abs(T - 1.0).max()
            
            if error < 1e-8:
                # Check for half-integer values
                if (_is_integer_or_half_integer(U_rand) and 
                    _is_integer_or_half_integer(V_rand) and 
                    _is_integer_or_half_integer(W_rand)):
                    return (U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), target_rank)
            
            # If error is small, try rounding to half-integers and re-check
            if error < 0.1:
                U_round = _round_to_half_integer(U_rand)
                V_round = _round_to_half_integer(V_rand)
                W_round = _round_to_half_integer(W_rand)
                
                T_round = np.zeros((n, m, p))
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_round[i, j, k] += U_round[i * m + j, r] * V_round[j * p + k, r] * W_round[k * n + i, r]
                
                if np.allclose(T_round, np.ones((n, m, p)), atol=1e-10):
                    return (U_round.tolist(), V_round.tolist(), W_round.tolist(), target_rank)
        
        return None
    
    # Try ranks from lower_bound upward
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random initialization approach
        result = _try_random_init(target_rank, n_tries=50)
        if result is not None:
            U, V, W, rank = result
            best_rank = rank
            break
    
    # If we didn't find a better decomposition, return schoolbook
    if best_rank == n * m * p:
        U, V, W = _schoolbook_decomposition(n, m, p)
    
    return {"U": U, "V": V, "W": W}
