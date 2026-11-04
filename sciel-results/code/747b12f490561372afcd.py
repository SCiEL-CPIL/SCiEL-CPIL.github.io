import time
import sys
import numpy as np

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor.
    Returns True if valid, False otherwise."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rm, rank = U.shape
    sm, rank2 = V.shape
    tm, rank3 = W.shape
    
    if rank != rank2 or rank != rank3:
        return False
    if rm != n * m:
        return False
    if sm != m * p:
        return False
    if tm != p * n:
        return False
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Check all on-diagonal entries: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries are zero
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _fast_error(U, V, W, n=3, m=3, p=6):
    """Fast error computation using numpy vectorization."""
    nm, mp, pn = n * m, m * p, p * n
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # On-diagonal: need U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :] summed over d
    # Build index arrays
    on_diag_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                on_diag_err += (val - 1.0) ** 2
    
    # Off-diagonal: sample for speed
    rng = np.random.RandomState(12345)
    off_diag_err = 0.0
    for _ in range(500):
        a = rng.randint(0, nm)
        b = rng.randint(0, mp)
        c = rng.randint(0, pn)
        i_a, j_a = a // m, a % m
        j_b, k_b = b // p, b % p
        k_c, i_c = c // n, c % n
        if not (j_a == j_b and k_b == k_c and i_c == i_a):
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            off_diag_err += val * val
    
    return on_diag_err + off_diag_err


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Returns: {'U', 'V', 'W'}: factor matrices with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 3, 3, 6
    nm, mp, pn = n * m, m * p, p * n  # 9, 18, 18
    
    # Schoolbook baseline: rank 54
    best_rank = n * m * p
    rank = best_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    
    # Strategy: Use structured low-rank decompositions
    # Key idea: C[i,k] = sum_j A[i,j]*B[j,k]
    # We can exploit that A is 3x3 (small) and B is 3x6
    
    # Approach: For each "combination pattern" of A and B indices,
    # try to find rank-1 terms that cover multiple (i,j,k) triples
    
    # Let's try a systematic search with structured initialization
    # Use the fact that entries should be integer/half-integer
    
    import random
    
    def try_structured_rank(r, seed=42, max_attempts=100):
        """Try to find rank-r decomposition with structured search."""
        rng = random.Random(seed)
        
        # Initialize with a pattern-based approach
        # Start from schoolbook and try to merge terms
        
        # Method: Start with rank r, initialize with small integers/half-integers
        # Then do coordinate descent with fast numpy error
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with random small values
            vals = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
            U_c = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(nm)], dtype=float)
            V_c = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(mp)], dtype=float)
            W_c = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(pn)], dtype=float)
            
            # Coordinate descent: iterate over entries, try small perturbations
            err = _fast_error(U_c.tolist(), V_c.tolist(), W_c.tolist(), n, m, p)
            
            if err < 1e-9:
                return (U_c.tolist(), V_c.tolist(), W_c.tolist(), r)
            
            # Try to improve via coordinate descent
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Pick random coordinate
                mat_choice = rng.choice([0, 1, 2])
                if mat_choice == 0:
                    mat = U_c
                    nrows = nm
                elif mat_choice == 1:
                    mat = V_c
                    nrows = mp
                else:
                    mat = W_c
                    nrows = pn
                
                row = rng.randint(0, nrows - 1)
                col = rng.randint(0, r - 1)
                old_val = mat[row, col]
                
                # Try nearby values
                best_err = err
                best_val = old_val
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    new_val = old_val + delta
                    mat[row, col] = new_val
                    new_err = _fast_error(
                        U_c.tolist() if mat_choice == 0 else None,
                        V_c.tolist() if mat_choice == 1 else None,
                        W_c.tolist() if mat_choice == 2 else None,
                        n, m, p
                    )
                    # Fix: need to pass correct matrices
                    if mat_choice == 0:
                        new_err = _fast_error(U_c.tolist(), V_c.tolist(), W_c.tolist(), n, m, p)
                    elif mat_choice == 1:
                        new_err = _fast_error(U_c.tolist(), V_c.tolist(), W_c.tolist(), n, m, p)
                    else:
                        new_err = _fast_error(U_c.tolist(), V_c.tolist(), W_c.tolist(), n, m, p)
                    
                    if new_err < best_err:
                        best_err = new_err
                        best_val = new_val
                mat[row, col] = best_val
                err = best_err
                
                if err < 1e-9:
                    return (U_c.tolist(), V_c.tolist(), W_c.tolist(), r)
        
        return None
    
    # Try structured ranks
    # Theoretical: R(3,3,6) is between ~19 and 54
    # Try progressively higher ranks
    ranks_to_try = [27, 30, 32, 36, 38, 40, 42, 45]
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            continue
        
        result = try_structured_rank(r, seed=r, max_attempts=30)
        if result is not None:
            U_c, V_c, W_c, r_found = result
            if _verify(U_c, V_c, W_c, n, m, p):
                best_rank = r_found
                best_result = {"U": U_c, "V": V_c, "W": W_c}
                break
    
    return best_result
