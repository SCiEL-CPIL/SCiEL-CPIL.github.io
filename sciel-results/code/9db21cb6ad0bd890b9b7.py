"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6):
    """Check if given factor matrices produce correct matmul tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i*m+j, j*p+k, k*n+i] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal: T[a, b, c] = 0 unless indices align
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # Non-zero only if j==j', k==k', i==i'
    for a in range(n*m):
        for b in range(m*p):
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(p*n):
                k_c = c // n
                i_c = c % n
                if k_b != k_c or i_a != i_c:
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val) > 1e-9:
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
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    best = _schoolbook_rank()
    best_rank = n * m * p  # 60
    
    # Try to find a lower-rank decomposition
    # Strategy: Use structured patterns and verify
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Also, for bilinear maps, there are additional lower bounds
    
    # Let's try ranks from 60 down to 30
    # For each rank, try to construct a valid decomposition
    
    # First, let's try a simple structured approach for rank 60 (schoolbook)
    # Then try to improve
    
    # Key insight: For 2xm * mxp, we can think of it as:
    # Output[0,k] = sum_j A[0,j] * B[j,k]
    # Output[1,k] = sum_j A[1,j] * B[j,k]
    # 
    # If we could share some computation between the two rows, we could reduce rank.
    # But in exact decomposition, each (i,j,k) triple needs its own term unless
    # there's algebraic structure we can exploit.
    
    # Let's try a systematic search for rank < 60
    # We'll use a greedy approach: start with schoolbook and try to merge terms
    
    # For now, let's verify that schoolbook works
    U_test = best["U"]
    V_test = best["V"]
    W_test = best["W"]
    
    if _verify_rank(U_test, V_test, W_test):
        pass  # Schoolbook is valid
    else:
        # Should never happen, but just in case
        return best
    
    # Try to find a better decomposition
    # Approach: Use random initialization + refinement with vectorized checks
    
    import random
    random.seed(12345)
    
    # Define a larger set of possible values for integer/half-integer entries
    possible_values = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2]
    
    def _random_candidate(rank):
        """Generate a random candidate with given rank."""
        U = np.array([[random.choice(possible_values) for _ in range(rank)] for _ in range(n*m)], dtype=float)
        V = np.array([[random.choice(possible_values) for _ in range(rank)] for _ in range(m*p)], dtype=float)
        W = np.array([[random.choice(possible_values) for _ in range(rank)] for _ in range(p*n)], dtype=float)
        return U, V, W
    
    # Precompute indices for the "1" constraints
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We need to check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a sum of products of 3 vectors.
    # We can vectorize the check for all (i,j,k) at once?
    # U is (10, R), V is (30, R), W is (12, R)
    # We want to compute M[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    # To speed up, we can just check the 60 "1" constraints first.
    # If they pass, we do the full check.
    
    def _check_ones(U, V, W, rank):
        """Check if the 60 diagonal entries are 1."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx], V[v_idx]) # This is not right, it's sum of U*V*W
                    # Actually: sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    # We can compute this as:
                    # prod = U[u_idx] * V[v_idx]
                    # val = np.dot(prod, W[w_idx])
                    prod = U[u_idx] * V[v_idx]
                    val = np.dot(prod, W[w_idx])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    # Try ranks from 59 down to 30
    # We will try more attempts per rank
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline:
            break
        
        # Try multiple strategies per rank
        found = False
        for strategy in range(3):
            if time.time() > deadline or found:
                break
            
            if strategy == 0:
                # Random search
                for attempt in range(200):
                    if time.time() > deadline or found:
                        break
                    
                    U, V, W = _random_candidate(target_rank)
                    
                    # Quick check: verify the 60 "1" entries
                    if _check_ones(U, V, W, target_rank):
                        # Do full verification
                        if _verify_rank(U.tolist(), V.tolist(), W.tolist()):
                            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = target_rank
                            found = True
                            break
            elif strategy == 1:
                # Structured initialization: start from schoolbook and try to merge
                # by setting some entries to 0 and adjusting others
                U, V, W = _random_candidate(target_rank)
                # Try local search: flip random entries
                for flip in range(500):
                    if time.time() > deadline:
                        break
                    
                    # Pick a random matrix and entry to flip
                    mat_choice = random.randint(0, 2)
                    if mat_choice == 0:
                        row = random.randint(0, n*m-1)
                        col = random.randint(0, target_rank-1)
                        U[row, col] = random.choice(possible_values)
                        mat = U
                    elif mat_choice == 1:
                        row = random.randint(0, m*p-1)
                        col = random.randint(0, target_rank-1)
                        V[row, col] = random.choice(possible_values)
                        mat = V
                    else:
                        row = random.randint(0, p*n-1)
                        col = random.randint(0, target_rank-1)
                        W[row, col] = random.choice(possible_values)
                        mat = W
                    
                    if _check_ones(U, V, W, target_rank):
                        if _verify_rank(U.tolist(), V.tolist(), W.tolist()):
                            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = target_rank
                            found = True
                            break
            else:
                # Try a more structured approach: use the algebraic structure
                # For 2xm * mxp, we can potentially share terms between the two rows
                # Initialize with a pattern that exploits this
                U = np.zeros((n*m, target_rank), dtype=float)
                V = np.zeros((m*p, target_rank), dtype=float)
                W = np.zeros((p*n, target_rank), dtype=float)
                
                # Try to assign terms to cover all (i,j,k) triples
                # Each term c can contribute to multiple triples if we use
                # non-trivial values
                for c in range(target_rank):
                    # Pick a random (j,k) pair to focus on
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    
                    # For this (j,k), we need to satisfy:
                    # sum_c U[0*m+j, c] * V[j*p+k, c] * W[k*n+0, c] = 1
                    # sum_c U[1*m+j, c] * V[j*p+k, c] * W[k*n+1, c] = 1
                    # 
                    # Let's set V[j*p+k, c] = 1, U[0*m+j, c] = 1, W[k*n+0, c] = 1
                    # and U[1*m+j, c] = 1, W[k*n+1, c] = 1
                    # But this uses 2 terms per (j,k), giving 2*m*p = 60 terms
                    # 
                    # Instead, let's try: U[0*m+j, c] = a, U[1*m+j, c] = b
                    # V[j*p+k, c] = v, W[k*n+0, c] = w0, W[k*n+1, c] = w1
                    # Then a*v*w0 = 1 and b*v*w1 = 1
                    # We can choose a, b, v, w0, w1 freely as long as these hold
                    # 
                    # Let's try to have each term cover one (j,k) for both i values
                    # This gives m*p = 30 terms minimum
                    if c < m*p:
                        j = c // p
                        k = c % p
                        # Set values to satisfy both i=0 and i=1
                        U[0*m+j, c] = 1
                        U[1*m+j, c] = 1
                        V[j*p+k, c] = 1
                        W[k*n+0, c] = 1
                        W[k*n+1, c] = 1
                    else:
                        # Extra terms: try to help with off-diagonal constraints
                        # or provide flexibility
                        i = random.randint(0, n-1)
                        j = random.randint(0, m-1)
                        k = random.randint(0, p-1)
                        U[i*m+j, c] = random.choice(possible_values)
                        V[j*p+k, c] = random.choice(possible_values)
                        W[k*n+i, c] = random.choice(possible_values)
                
                if _check_ones(U, V, W, target_rank):
                    if _verify_rank(U.tolist(), V.tolist(), W.tolist()):
                        best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = target_rank
                        found = True
                        break
    
    # Return the best found
    return best

# EVOLVE-BLOCK-END
