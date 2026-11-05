import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Returns {'U', 'V', 'W'} with shapes (9, R), (18, R), (18, R).
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = 3, 3, 6
    nm, mp, pn = n * m, m * p, p * n # 9, 18, 18
    
    # Precompute the support of the tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
                
    # We want to find U, V, W such that sum_t U[a,t] V[b,t] W[c,t] = 1 for (a,b,c) in support
    # and 0 otherwise.
    
    # Strategy: Try ranks from 18 (lower bound) up to 54 (schoolbook).
    # We prioritize finding a valid solution for low ranks.
    # Rank 38 is a strong candidate (2 * 19).
    
    # We will use a local search / ALS approach.
    # To speed it up, we only check a subset of off-support constraints during iteration,
    # and verify fully at the end.
    
    def verify(U, V, W):
        """Exact verification."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Check support
        for a, b, c in support:
            val = np.dot(U[a], V[b] * W[c])
            if abs(val - 1.0) > 1e-8:
                return False
        
        # Check off-support (sampled for speed in full verify? No, full verify must be exact)
        # For efficiency in the final check, we can vectorize if possible, but loops are okay for 9*18*18=2916 checks.
        for a in range(nm):
            i_a, j_a = a // m, a % m
            for b in range(mp):
                j_b, k_b = b // p, b % p
                for c in range(pn):
                    k_c, i_c = c // n, c % n
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        val = np.dot(U[a], V[b] * W[c])
                        if abs(val) > 1e-8:
                            return False
        return True

    def run_als(U, V, W, max_iter=200, tol=1e-10):
        """
        Run ALS to refine U, V, W.
        Returns (U, V, W, converged)
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        prev_err = float('inf')
        
        for it in range(max_iter):
            # Update U
            for a in range(nm):
                # Find all (b, c) such that (a, b, c) is in support
                # We can precompute these lists for speed
                rows = []
                for b in range(mp):
                    j_b = b // p
                    k_b = b % p
                    # a = i*m + j. So j_a = a % m.
                    # Condition: j_a == j_b
                    if a % m != j_b:
                        continue
                    for c in range(pn):
                        k_c = c // n
                        i_c = c % n
                        # Condition: k_b == k_c and i_c == i_a (a // m)
                        if k_b == k_c and i_c == a // m:
                            rows.append(V[b] * W[c])
                
                if rows:
                    M = np.array(rows)
                    ones = np.ones(M.shape[0])
                    # Solve M^T U_a = ones  =>  U_a = (M M^T)^-1 M^T ones
                    # Use lstsq for stability
                    try:
                        U[a] = np.linalg.lstsq(M, ones, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                rows = []
                for a in range(nm):
                    i_a = a // m
                    j_a = a % m
                    if j_a != j_b:
                        continue
                    for c in range(pn):
                        k_c = c // n
                        i_c = c % n
                        if k_b == k_c and i_c == i_a:
                            rows.append(U[a] * W[c])
                
                if rows:
                    M = np.array(rows)
                    ones = np.ones(M.shape[0])
                    try:
                        V[b] = np.linalg.lstsq(M, ones, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                rows = []
                for a in range(nm):
                    i_a = a // m
                    j_a = a % m
                    if i_a != i_c:
                        continue
                    for b in range(mp):
                        j_b = b // p
                        k_b = b % p
                        if k_b == k_c and j_a == j_b:
                            rows.append(U[a] * V[b])
                
                if rows:
                    M = np.array(rows)
                    ones = np.ones(M.shape[0])
                    try:
                        W[c] = np.linalg.lstsq(M, ones, rcond=None)[0]
                    except:
                        pass
            
            # Calculate error (approximate)
            err = 0.0
            for a, b, c in support:
                val = np.dot(U[a], V[b] * W[c])
                err += (val - 1.0) ** 2
            
            if err < tol:
                return U, V, W, True
            
            if abs(prev_err - err) < 1e-14:
                return U, V, W, False
            prev_err = err
            
            if time.time() > deadline - 10:
                break
                
        return U, V, W, False

    # Precompute support neighbors for faster ALS
    # For each a, list of (b, c)
    support_by_a = {a: [] for a in range(nm)}
    support_by_b = {b: [] for b in range(mp)}
    support_by_c = {c: [] for c in range(pn)}
    
    for a, b, c in support:
        support_by_a[a].append((b, c))
        support_by_b[b].append((a, c))
        support_by_c[c].append((a, b))

    def run_als_fast(U, V, W, max_iter=500, tol=1e-10):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        prev_err = float('inf')
        
        for it in range(max_iter):
            # Update U
            for a in range(nm):
                pairs = support_by_a[a]
                if not pairs: continue
                M = np.array([V[b] * W[c] for b, c in pairs])
                ones = np.ones(M.shape[0])
                try:
                    U[a] = np.linalg.lstsq(M, ones, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b in range(mp):
                pairs = support_by_b[b]
                if not pairs: continue
                M = np.array([U[a] * W[c] for a, c in pairs])
                ones = np.ones(M.shape[0])
                try:
                    V[b] = np.linalg.lstsq(M, ones, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c in range(pn):
                pairs = support_by_c[c]
                if not pairs: continue
                M = np.array([U[a] * V[b] for a, b in pairs])
                ones = np.ones(M.shape[0])
                try:
                    W[c] = np.linalg.lstsq(M, ones, rcond=None)[0]
                except:
                    pass
            
            # Error
            err = 0.0
            for a, b, c in support:
                val = np.dot(U[a], V[b] * W[c])
                err += (val - 1.0) ** 2
                
            if err < tol:
                return U, V, W, True
                
            if abs(prev_err - err) < 1e-14:
                return U, V, W, False
            prev_err = err
            
            if time.time() > deadline - 5:
                break
                
        return U, V, W, False

    best_result = None
    best_rank = 54 # Schoolbook
    
    # Try to find a solution with rank < 54
    # We know rank 38 is possible (2 * 19).
    # We will try ranks from 18 to 53.
    # To save time, we focus on a few promising ranks: 18, 19, 20, ..., 38.
    
    # Initialize with a structured guess for rank R
    def init_structured(R):
        # Split R into two halves if possible, corresponding to k-blocks
        # For simplicity, just use random with some sparsity
        U = np.random.randn(nm, R) * 0.1
        V = np.random.randn(mp, R) * 0.1
        W = np.random.randn(pn, R) * 0.1
        return U, V, W

    # Try ranks
    # We will iterate through ranks. If we find a valid one, we stop.
    # To ensure we find *some* improvement, we start with 38 and go down? 
    # No, we want the *minimum*. So we should start from 18 and go up.
    # However, 18 is the lower bound and might be hard to hit.
    # 38 is very likely.
    # Let's try 18 to 40.
    
    for r in range(18, 45):
        if time.time() > deadline - 20:
            break
            
        # Number of restarts
        n_restarts = 5
        if r < 25:
            n_restarts = 10
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
                
            U, V, W = init_structured(r)
            
            # Run ALS
            U, V, W, converged = run_als_fast(U, V, W, max_iter=300)
            
            if converged:
                # Verify exactly
                if verify(U, V, W):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break # Found minimal for this r, break restarts
        
        if best_result is not None:
            break # Found a valid decomposition, can stop searching higher ranks if we want minimal? 
            # Wait, we are iterating from low to high. So the first one found is the minimal.
            break

    if best_result is None:
        # Fallback to schoolbook if no better found (should not happen, but safety)
        # Or use a known rank 54 solution
        U = np.zeros((nm, 54))
        V = np.zeros((mp, 54))
        W = np.zeros((pn, 54))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        best_rank = 54

    return best_result

# EVOLVE-BLOCK-END
