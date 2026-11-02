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
    5. Leverage the specific structure of the 3x3x4 case by searching for a 
       decomposition of rank 18 (the theoretical lower bound for some subspaces)
       or lower by representing the multiplication as a bilinear map over a finite field.
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
    
    def weighted_error(U, V, W, r):
        """Weighted error: active triples weighted more heavily."""
        total = 0.0
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            expected = 1 if (a, b, c) in active_set else 0
            diff = s - expected
            if (a, b, c) in active_set:
                total += 10.0 * diff * diff
            else:
                total += diff * diff
        return total
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try structured search for lower ranks
    # Known: for 3x3*3x4, we expect rank around 22-24
    # Try from rank 18 upward (optimistic) to best_rank-1
    
    def try_rank_structured(r, max_attempts=200):
        """Try to find a rank-r decomposition using structured random search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with structured patterns
            # Use a mix of strategies:
            # 1. Sparse random
            # 2. Based on Strassen-like patterns
            
            strategy = attempt % 5
            
            if strategy == 0:
                # Sparse random: mostly zeros with some ±1
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    num_nonzero = random.randint(1, min(3, r))
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
                
                # For each column, assign a pattern
                for ci in range(r):
                    # Random pattern with some structure
                    u_vals = [random.choice([-1, 0, 1]) for _ in range(n * m)]
                    v_vals = [random.choice([-1, 0, 1]) for _ in range(m * p)]
                    w_vals = [random.choice([-1, 0, 1]) for _ in range(p * n)]
                    for a in range(n * m):
                        U[a][ci] = u_vals[a]
                    for b in range(m * p):
                        V[b][ci] = v_vals[b]
                    for c in range(p * n):
                        W[c][ci] = w_vals[c]
            
            elif strategy == 3:
                # Semi-sparse: each row has 2-4 nonzeros, values from {-2,-1,1,2}
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                vals = [-2, -1, 1, 2]
                for a in range(n * m):
                    num_nonzero = random.randint(2, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice(vals)
                for b in range(m * p):
                    num_nonzero = random.randint(2, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice(vals)
                for c in range(p * n):
                    num_nonzero = random.randint(2, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c][ci] = random.choice(vals)
            
            else:
                # Very sparse: 1-2 nonzeros per row
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    num_nonzero = random.randint(1, 2)
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    num_nonzero = random.randint(1, 2)
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    num_nonzero = random.randint(1, 2)
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
            
            # Quick check: verify active triples first
            active_ok = True
            for (a, b, c) in active_triples:
                s = sum(U[a][ci] * V[b][ci] * W[c][ci] for ci in range(r))
                if s != 1:
                    active_ok = False
                    break
            
            if not active_ok:
                # Try local refinement if close
                err = error_count(U, V, W, r)
                if err <= 3 and time.time() < deadline:
                    refined = local_refine(U, V, W, r, max_iters=50)
                    if refined is not None:
                        return refined
                continue
            
            # Full verification
            if verify(U, V, W, r):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def local_refine(U, V, W, r, max_iters=100):
        """Local search: try flipping individual entries to reduce error."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        best_err = error_count(U, V, W, r)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try flipping random entries
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                # Pick a random matrix, row, col
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randint(0, n * m - 1)
                    ci = random.randint(0, r - 1)
                    old_val = U[idx][ci]
                    U[idx][ci] = old_val + 1 if old_val < 2 else old_val - 1
                    new_err = error_count(U, V, W, r)
                    if new_err < best_err:
                        best_err = new_err
                        improved = True
                    else:
                        U[idx][ci] = old_val
                elif mat == 'V':
                    idx = random.randint(0, m * p - 1)
                    ci = random.randint(0, r - 1)
                    old_val = V[idx][ci]
                    V[idx][ci] = old_val + 1 if old_val < 2 else old_val - 1
                    new_err = error_count(U, V, W, r)
                    if new_err < best_err:
                        best_err = new_err
                        improved = True
                    else:
                        V[idx][ci] = old_val
                else:
                    idx = random.randint(0, p * n - 1)
                    ci = random.randint(0, r - 1)
                    old_val = W[idx][ci]
                    W[idx][ci] = old_val + 1 if old_val < 2 else old_val - 1
                    new_err = error_count(U, V, W, r)
                    if new_err < best_err:
                        best_err = new_err
                        improved = True
                    else:
                        W[idx][ci] = old_val
            
            if best_err == 0:
                return {"U": U, "V": V, "W": W}
            
            if not improved and it > 20:
                break
        
        if best_err == 0:
            return {"U": U, "V": V, "W": W}
        return None
    
    # Try to find lower rank decompositions
    # Start from optimistic rank and go up
    for r in range(18, best_rank):
        if time.time() > deadline:
            break
        
        # Adaptive number of attempts based on remaining time
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Spend more time on promising ranks
        if r <= 22:
            max_attempts = min(400, int(time_remaining * 8))
        elif r <= 26:
            max_attempts = min(250, int(time_remaining * 4))
        else:
            max_attempts = min(80, int(time_remaining * 2))
        
        result = try_rank_structured(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            # We found a decomposition at rank r. Since we're going up from 18,
            # this is the lowest rank we've found so far. Stop here.
            break
    
    return best_result
# EVOLVE-BLOCK-END
