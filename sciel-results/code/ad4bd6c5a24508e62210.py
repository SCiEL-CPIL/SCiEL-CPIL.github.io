"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook (rank 36) as fallback.
    2. Use structured random search with better initialization.
    3. Exploit the fact that we only need to verify 36 active triples + ensure
       all other combinations sum to 0.
    4. Use local search / hill climbing to refine candidates.
    5. Add a smooth surrogate objective (sum of squared errors) to guide local search,
       plus a final polish pass with exact rational arithmetic.
    """
    start_time = time.time()
    deadline = start_time + 135  # 135s budget, leaving 15s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    # Active triples: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    active_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.add((a, b, c))
    
    # Precompute all triples that need to be checked
    all_triples = list(itertools.product(range(n * m), range(m * p), range(p * n)))
    active_set = set(active_triples)
    
    def verify(U, V, W, r):
        """Check if U, V, W with rank r satisfy all constraints."""
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            expected = 1 if (a, b, c) in active_set else 0
            if s != expected:
                return False
        return True
    
    def error_count(U, V, W, r):
        """Count how many triples are violated (for local search)."""
        errors = 0
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            expected = 1 if (a, b, c) in active_set else 0
            if s != expected:
                errors += 1
        return errors
    
    def compute_all_sums(U, V, W, r):
        """Compute the sum S[a][b][c] = sum_ci U[a][ci]*V[b][ci]*W[c][ci] for all triples."""
        sums = [[[0.0] * (p * n) for _ in range(m * p)] for _ in range(n * m)]
        for ci in range(r):
            for a in range(n * m):
                ua = U[a][ci]
                if ua == 0:
                    continue
                for b in range(m * p):
                    vb = V[b][ci]
                    if vb == 0:
                        continue
                    uv = ua * vb
                    for c in range(p * n):
                        wc = W[c][ci]
                        if wc != 0:
                            sums[a][b][c] += uv * wc
        return sums
    
    def smooth_error_from_sums(sums):
        """Compute smooth error from precomputed sums."""
        total = 0.0
        for a, b, c in all_triples:
            expected = 1 if (a, b, c) in active_set else 0
            s = sums[a][b][c]
            total += (s - expected) ** 2
        return total
    
    def smooth_error(U, V, W, r):
        """
        Compute a smooth surrogate: sum of squared deviations from target.
        This is differentiable in spirit (we can do gradient-like moves) and
        provides a continuous landscape for local search.
        """
        sums = compute_all_sums(U, V, W, r)
        return smooth_error_from_sums(sums)
    
    def local_search_smooth(U, V, W, r, max_iters=5000, time_limit=10.0):
        """
        Local search using the smooth squared-error objective with incremental updates.
        Try small perturbations to individual entries and accept if smooth error decreases.
        Use simulated annealing-style acceptance for escaping local minima.
        """
        start_ls = time.time()
        sums = compute_all_sums(U, V, W, r)
        current_err = smooth_error_from_sums(sums)
        best_err = current_err
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        
        # Temperature for simulated annealing
        T = 2.0
        T_min = 0.01
        cooling = 0.995
        
        for it in range(max_iters):
            if time.time() > time_limit or time.time() > deadline:
                break
            
            # Pick a random matrix, row, and column to perturb
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                mat = U
                nrows = n * m
            elif mat_choice == 'V':
                mat = V
                nrows = m * p
            else:
                mat = W
                nrows = p * n
            
            row = random.randint(0, nrows - 1)
            col = random.randint(0, r - 1)
            
            # Try small integer perturbations
            delta = random.choice([-2, -1, 1, 2])
            old_val = mat[row][col]
            new_val = old_val + delta
            # Keep values in a reasonable range
            if abs(new_val) > 10:
                continue
            
            # Incrementally update sums
            # delta_sum[a][b][c] = delta * V[b][col] * W[c][col] for U change
            # etc.
            mat[row][col] = new_val
            
            # Compute the change in error incrementally
            # For each triple (a,b,c) affected, the sum changes by delta * (product of other two matrices at col)
            delta_err = 0.0
            if mat_choice == 'U':
                for b in range(m * p):
                    vb = V[b][col]
                    if vb == 0:
                        continue
                    for c in range(p * n):
                        wc = W[c][col]
                        if wc == 0:
                            continue
                        old_s = sums[row][b][c]
                        new_s = old_s + delta * vb * wc
                        expected = 1 if (row, b, c) in active_set else 0
                        delta_err += (new_s - expected) ** 2 - (old_s - expected) ** 2
                        sums[row][b][c] = new_s
            elif mat_choice == 'V':
                for a in range(n * m):
                    ua = U[a][col]
                    if ua == 0:
                        continue
                    for c in range(p * n):
                        wc = W[c][col]
                        if wc == 0:
                            continue
                        old_s = sums[a][row][c]
                        new_s = old_s + delta * ua * wc
                        expected = 1 if (a, row, c) in active_set else 0
                        delta_err += (new_s - expected) ** 2 - (old_s - expected) ** 2
                        sums[a][row][c] = new_s
            else:
                for a in range(n * m):
                    ua = U[a][col]
                    if ua == 0:
                        continue
                    for b in range(m * p):
                        vb = V[b][col]
                        if vb == 0:
                            continue
                        old_s = sums[a][b][row]
                        new_s = old_s + delta * ua * vb
                        expected = 1 if (a, b, row) in active_set else 0
                        delta_err += (new_s - expected) ** 2 - (old_s - expected) ** 2
                        sums[a][b][row] = new_s
            
            new_err = current_err + delta_err
            
            # Accept if better, or with probability exp(-(new_err - current_err)/T)
            if new_err <= current_err or random.random() < pow(2.71828, -(new_err - current_err) / max(T, 1e-10)):
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                # Revert
                mat[row][col] = old_val
                # Revert sums
                if mat_choice == 'U':
                    for b in range(m * p):
                        vb = V[b][col]
                        if vb == 0:
                            continue
                        for c in range(p * n):
                            wc = W[c][col]
                            if wc == 0:
                                continue
                            sums[row][b][c] -= delta * vb * wc
                elif mat_choice == 'V':
                    for a in range(n * m):
                        ua = U[a][col]
                        if ua == 0:
                            continue
                        for c in range(p * n):
                            wc = W[c][col]
                            if wc == 0:
                                continue
                            sums[a][row][c] -= delta * ua * wc
                else:
                    for a in range(n * m):
                        ua = U[a][col]
                        if ua == 0:
                            continue
                        for b in range(m * p):
                            vb = V[b][col]
                            if vb == 0:
                                continue
                            sums[a][b][row] -= delta * ua * vb
            
            T = max(T_min, T * cooling)
        
        return best_U, best_V, best_W, best_err
    
    def polish(U, V, W, r):
        """
        Final polish: try to fix any remaining small errors by adjusting entries.
        Use exact integer arithmetic.
        """
        # First check if already valid
        if verify(U, V, W, r):
            return U, V, W, True
        
        # For each violated triple, try to find a column where we can adjust
        # to fix the error. This is a heuristic polish.
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Find the most violated triple
            worst_err = 0
            worst_triple = None
            for a, b, c in all_triples:
                s = sum(U[a][ci] * V[b][ci] * W[c][ci] for ci in range(r))
                expected = 1 if (a, b, c) in active_set else 0
                err = abs(s - expected)
                if err > worst_err:
                    worst_err = err
                    worst_triple = (a, b, c)
            
            if worst_err == 0:
                return U, V, W, True
            
            a, b, c = worst_triple
            # Try to fix by adjusting one entry
            fixed = False
            for ci in range(r):
                for delta in [-2, -1, 1, 2]:
                    # Try adjusting U[a][ci]
                    old = U[a][ci]
                    U[a][ci] = old + delta
                    if verify(U, V, W, r):
                        return U, V, W, True
                    U[a][ci] = old
                    
                    # Try adjusting V[b][ci]
                    old = V[b][ci]
                    V[b][ci] = old + delta
                    if verify(U, V, W, r):
                        return U, V, W, True
                    V[b][ci] = old
                    
                    # Try adjusting W[c][ci]
                    old = W[c][ci]
                    W[c][ci] = old + delta
                    if verify(U, V, W, r):
                        return U, V, W, True
                    W[c][ci] = old
            
            if not fixed:
                break
        
        return U, V, W, verify(U, V, W, r)
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_rank_structured(r, max_attempts=200, local_time_limit=3.0):
        """Try to find a rank-r decomposition using structured random search with smooth local search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            strategy = attempt % 4
            
            if strategy == 0:
                # Sparse random: mostly zeros with some ±1
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Dense random with small values
                scale = 2
                U = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(n * m)]
                V = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(m * p)]
                W = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(p * n)]
                
            elif strategy == 2:
                # Strassen-inspired: use block structure
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                for ci in range(r):
                    u_vals = [random.choice([-1, 0, 1]) for _ in range(n * m)]
                    v_vals = [random.choice([-1, 0, 1]) for _ in range(m * p)]
                    w_vals = [random.choice([-1, 0, 1]) for _ in range(p * n)]
                    for a in range(n * m):
                        U[a][ci] = u_vals[a]
                    for b in range(m * p):
                        V[b][ci] = v_vals[b]
                    for c in range(p * n):
                        W[c][ci] = w_vals[c]
            
            else:
                # Hybrid: start from a partial schoolbook and merge columns
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                # Initialize with random superpositions of schoolbook columns
                for ci in range(r):
                    # Pick 2-3 random schoolbook triples and superpose
                    num_terms = random.randint(1, 3)
                    for _ in range(num_terms):
                        i = random.randint(0, n - 1)
                        j = random.randint(0, m - 1)
                        k = random.randint(0, p - 1)
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        sign = random.choice([-1, 1])
                        U[a][ci] += sign
                        V[b][ci] += sign
                        W[c][ci] += sign
            
            # Strategy 3: Start from schoolbook and try to merge columns by finding
            # pairs that can be combined. This gives a near-valid starting point.
            if strategy == 3:
                # First fill with schoolbook columns up to r
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                # Use the first min(r, n*m*p) schoolbook columns
                for ci in range(min(r, n * m * p)):
                    i = ci // (m * p)
                    rem = ci % (m * p)
                    j = rem // p
                    k = rem % p
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][ci] = 1
                    V[b][ci] = 1
                    W[c][ci] = 1
                
                # If r < n*m*p, we need to combine some columns.
                # Try to add random linear combinations to fill remaining columns.
                for ci in range(r):
                    # Add a small random perturbation to encourage merging
                    if random.random() < 0.3:
                        i = random.randint(0, n - 1)
                        j = random.randint(0, m - 1)
                        k = random.randint(0, p - 1)
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a][ci] += random.choice([-1, 1])
                        V[b][ci] += random.choice([-1, 1])
                        W[c][ci] += random.choice([-1, 1])
            
            # Quick check: verify active triples first
            active_ok = True
            for (a, b, c) in active_triples:
                s = sum(U[a][ci] * V[b][ci] * W[c][ci] for ci in range(r))
                if s != 1:
                    active_ok = False
                    break
            
            if not active_ok:
                # Try smooth local search to fix
                time_remaining = deadline - time.time()
                if time_remaining > 3:
                    U, V, W, err = local_search_smooth(U, V, W, r, max_iters=5000, time_limit=min(local_time_limit, time_remaining * 0.4))
                    # Re-check
                    active_ok = True
                    for (a, b, c) in active_triples:
                        s = sum(U[a][ci] * V[b][ci] * W[c][ci] for ci in range(r))
                        if s != 1:
                            active_ok = False
                            break
                    if not active_ok:
                        continue
            
            # Full verification
            if verify(U, V, W, r):
                return {"U": U, "V": V, "W": W}
            
            # If close, try polish
            err = error_count(U, V, W, r)
            if err <= 10 and time.time() < deadline - 2:
                U, V, W, ok = polish(U, V, W, r)
                if ok:
                    return {"U": U, "V": V, "W": W}
            
            # Even if not close, if we have time, do a longer local search
            elif err <= 30 and time.time() < deadline - 3:
                time_remaining = deadline - time.time()
                U, V, W, err2 = local_search_smooth(U, V, W, r, max_iters=3000, time_limit=min(2.0, time_remaining * 0.3))
                if verify(U, V, W, r):
                    return {"U": U, "V": V, "W": W}
                err2 = error_count(U, V, W, r)
                if err2 <= 5 and time.time() < deadline - 2:
                    U, V, W, ok = polish(U, V, W, r)
                    if ok:
                        return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try to find lower rank decompositions
    # Start from a more optimistic rank based on known lower bounds
    # For 3x3*3x4, the tensor rank is at least max(n*m, m*p, p*n) = max(9,12,12) = 12
    # But realistically, we expect something between 18-30
    for r in range(18, best_rank):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Allocate more time to lower ranks since they're harder
        if r <= 20:
            max_attempts = min(800, int(time_remaining * 15))
            local_time_limit = min(5.0, time_remaining * 0.05)
        elif r <= 24:
            max_attempts = min(400, int(time_remaining * 8))
            local_time_limit = min(3.0, time_remaining * 0.04)
        elif r <= 28:
            max_attempts = min(200, int(time_remaining * 5))
            local_time_limit = min(2.0, time_remaining * 0.03)
        else:
            max_attempts = min(100, int(time_remaining * 3))
            local_time_limit = min(1.5, time_remaining * 0.02)
        
        result = try_rank_structured(r, max_attempts, local_time_limit=local_time_limit)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
