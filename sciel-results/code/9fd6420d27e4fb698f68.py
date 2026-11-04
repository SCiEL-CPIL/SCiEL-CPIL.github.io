"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank_fast(U, V, W, n=2, m=5, p=6, tol=1e-9):
    """
    Vectorized verification of tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Returns True if the decomposition is valid.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    # Check diagonal: T[i*m+j, j*p+k, k*n+i] == 1
    # Build index arrays
    i_vals = np.repeat(np.arange(n), m*p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n*m)
    
    u_idx = i_vals * m + j_vals
    v_idx = j_vals * p + k_vals
    w_idx = k_vals * n + i_vals
    
    # Compute all diagonal entries at once
    # For each triple, we need sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
    diag_vals = np.einsum('ac,bc,ac->a', U[u_idx], V[v_idx], W[w_idx])
    if not np.allclose(diag_vals, 1.0, atol=tol):
        return False
    
    # Check off-diagonal: T[a, b, c] == 0 when indices don't align
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # Non-zero only if j==j', k==k', i==i'
    # We need to check all (a,b,c) where the indices don't fully align
    
    # Strategy: For each j, check all (i,k,i',k') where not (i==i' and k==k')
    # Actually, let's be more systematic.
    # For fixed j, the relevant entries are:
    # a in {j*m, j*m+1} (for n=2)
    # b in {j*p, j*p+1, ..., j*p+p-1}
    # c in {0,1,...,p*n-1}
    
    # Let's iterate over j and check
    for j in range(m):
        # Rows of U: i*m + j for i in range(n)
        u_rows = [i * m + j for i in range(n)]
        # Rows of V: j*p + k for k in range(p)
        v_rows = [j * p + k for k in range(p)]
        # All rows of W
        w_rows = list(range(p * n))
        
        # For each combination, check if the product sums to 0
        # except when i==i' and k==k'
        for i in range(n):
            for k in range(p):
                for i_prime in range(n):
                    for k_prime in range(p):
                        if i == i_prime and k == k_prime:
                            continue  # This is the diagonal, already checked
                        a = i * m + j
                        b = j * p + k
                        c = k_prime * n + i_prime
                        val = np.dot(U[a] * V[b], W[c])
                        if abs(val) > tol:
                            return False
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    best = _schoolbook_rank()
    best_rank = n * m * p  # 60
    
    # Verify schoolbook works
    if _verify_rank_fast(best["U"], best["V"], best["W"]):
        pass
    else:
        return best
    
    import random
    random.seed(42)
    
    # Define possible values for integer/half-integer entries
    possible_values = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 2.5, -2.5, 3, -3]
    
    def _check_ones_fast(U, V, W, tol=1e-6):
        """Fast check of just the 60 diagonal entries."""
        i_vals = np.repeat(np.arange(n), m*p)
        j_vals = np.tile(np.repeat(np.arange(m), p), n)
        k_vals = np.tile(np.arange(p), n*m)
        
        u_idx = i_vals * m + j_vals
        v_idx = j_vals * p + k_vals
        w_idx = k_vals * n + i_vals
        
        diag_vals = np.einsum('ac,bc,ac->a', U[u_idx], V[v_idx], W[w_idx])
        return np.allclose(diag_vals, 1.0, atol=tol)
    
    def _structured_candidate(rank, seed=None):
        """
        Generate a structured candidate that exploits the 2-row structure.
        For 2xm * mxp, we can try to share terms between the two rows.
        """
        if seed is not None:
            random.seed(seed)
        
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # Strategy: Use m*p = 30 terms to cover all (j,k) pairs for both i values
        # Plus extra terms for flexibility
        if rank >= m * p:
            for c in range(m * p):
                j = c // p
                k = c % p
                # Assign random values that satisfy the constraints
                # U[0*m+j, c] * V[j*p+k, c] * W[k*n+0, c] should contribute to 1
                # U[1*m+j, c] * V[j*p+k, c] * W[k*n+1, c] should contribute to 1
                # Let's set V[j*p+k, c] = 1, and choose U and W appropriately
                v_val = 1.0
                V[j*p+k, c] = v_val
                # Choose random non-zero values
                u0 = random.choice([v for v in possible_values if v != 0])
                u1 = random.choice([v for v in possible_values if v != 0])
                w0 = 1.0 / (u0 * v_val) if u0 * v_val != 0 else 1.0
                w1 = 1.0 / (u1 * v_val) if u1 * v_val != 0 else 1.0
                # Round to nearest half-integer
                w0 = round(w0 * 2) / 2
                w1 = round(w1 * 2) / 2
                U[0*m+j, c] = u0
                U[1*m+j, c] = u1
                W[k*n+0, c] = w0
                W[k*n+1, c] = w1
        
        # Add random noise to extra terms
        for c in range(m * p, rank):
            # Random sparse assignment
            n_nonzero = random.randint(1, 4)
            for _ in range(n_nonzero):
                mat = random.randint(0, 2)
                if mat == 0:
                    row = random.randint(0, n*m-1)
                    U[row, c] = random.choice(possible_values)
                elif mat == 1:
                    row = random.randint(0, m*p-1)
                    V[row, c] = random.choice(possible_values)
                else:
                    row = random.randint(0, p*n-1)
                    W[row, c] = random.choice(possible_values)
        
        return U, V, W
    
    def _local_search(U, V, W, rank, max_steps=1000):
        """
        Local search: try to fix violations by adjusting random entries.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Check if valid
            if _verify_rank_fast(U, V, W):
                return U, V, W, True
            
            # Find a violation and try to fix it
            # Randomly pick a matrix and entry to modify
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                row = random.randint(0, n*m-1)
                col = random.randint(0, rank-1)
                old_val = U[row, col]
                U[row, col] = random.choice(possible_values)
            elif mat_choice == 1:
                row = random.randint(0, m*p-1)
                col = random.randint(0, rank-1)
                old_val = V[row, col]
                V[row, col] = random.choice(possible_values)
            else:
                row = random.randint(0, p*n-1)
                col = random.randint(0, rank-1)
                old_val = W[row, col]
                W[row, col] = random.choice(possible_values)
        
        return U, V, W, False
    
    # Try ranks from 59 down to 30
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline:
            break
        
        found = False
        
        # Strategy 1: Structured candidates
        for attempt in range(50):
            if time.time() > deadline or found:
                break
            
            seed = 1000 + target_rank * 100 + attempt
            U, V, W = _structured_candidate(target_rank, seed=seed)
            
            if _check_ones_fast(U, V, W):
                if _verify_rank_fast(U, V, W):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # Try local search to fix off-diagonal violations
                    U2, V2, W2, success = _local_search(U, V, W, target_rank, max_steps=200)
                    if success:
                        best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            continue
        
        # Strategy 2: Random search with more diverse initialization
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            U = np.array([[random.choice(possible_values) for _ in range(target_rank)] for _ in range(n*m)], dtype=float)
            V = np.array([[random.choice(possible_values) for _ in range(target_rank)] for _ in range(m*p)], dtype=float)
            W = np.array([[random.choice(possible_values) for _ in range(target_rank)] for _ in range(p*n)], dtype=float)
            
            if _check_ones_fast(U, V, W):
                if _verify_rank_fast(U, V, W):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    found = True
                    break
                else:
                    U2, V2, W2, success = _local_search(U, V, W, target_rank, max_steps=300)
                    if success:
                        best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                        best_rank = target_rank
                        found = True
                        break
    
    return best

# EVOLVE-BLOCK-END
