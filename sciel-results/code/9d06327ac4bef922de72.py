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
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s budget, leaving 10s margin for 150s limit
    
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
    
    # Precompute expected values for fast lookup
    expected_vals = {}
    for a, b, c in all_triples:
        expected_vals[(a, b, c)] = 1 if (a, b, c) in active_set else 0
    
    def verify_fast(U, V, W, r):
        """Fast verification using precomputed expected values."""
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            if s != expected_vals[(a, b, c)]:
                return False
        return True
    
    def compute_errors(U, V, W, r):
        """Compute error for each triple. Returns dict of (a,b,c) -> error."""
        errors = {}
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            errors[(a, b, c)] = s - expected_vals[(a, b, c)]
        return errors
    
    def error_count(U, V, W, r):
        """Count how many triples are violated."""
        errs = compute_errors(U, V, W, r)
        return sum(1 for e in errs.values() if e != 0)
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try structured search for lower ranks
    # Known: for 3x3*3x4, we expect rank around 22-24
    # Try from rank 18 upward (optimistic) to best_rank-1

    def local_refine(U, V, W, r, max_steps=5000):
        """Hill-climb on individual entries to reduce error count."""
        if time.time() > deadline:
            return None, 0
        current_errors = error_count(U, V, W, r)
        if current_errors == 0:
            return (U, V, W), 0
        
        step = 0
        improved = True
        while improved and step < max_steps and time.time() < deadline:
            improved = False
            step += 1
            
            # Randomly pick a matrix and entry to flip
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                mat = U
                rows = n * m
            elif mat_choice == 'V':
                mat = V
                rows = m * p
            else:
                mat = W
                rows = p * n
            
            row = random.randint(0, rows - 1)
            col = random.randint(0, r - 1)
            
            old_val = mat[row][col]
            new_val = -old_val  # flip sign
            
            if new_val == old_val:
                continue
            
            mat[row][col] = new_val
            new_errors = error_count(U, V, W, r)
            
            if new_errors < current_errors:
                current_errors = new_errors
                improved = True
            else:
                mat[row][col] = old_val
        
        if current_errors == 0:
            return (U, V, W), 0
        return None, current_errors

    def try_rank_structured(r, max_attempts=200):
        """Try to find a rank-r decomposition using structured random search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with structured patterns
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
                # Hybrid: start from schoolbook-like structure and perturb
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                # Assign each column to a random (i,j,k) pattern
                for ci in range(r):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    
                    # With some probability, use the exact schoolbook pattern
                    if random.random() < 0.3:
                        U[i * m + j][ci] = 1
                        V[j * p + k][ci] = 1
                        W[k * n + i][ci] = 1
                    else:
                        # Perturb: add noise to nearby indices
                        U[i * m + j][ci] = random.choice([-1, 0, 1])
                        V[j * p + k][ci] = random.choice([-1, 0, 1])
                        W[k * n + i][ci] = random.choice([-1, 0, 1])
                        
                        # Sometimes add extra non-zeros
                        if random.random() < 0.2:
                            extra_a = random.randint(0, n*m-1)
                            U[extra_a][ci] = random.choice([-1, 1])
                        if random.random() < 0.2:
                            extra_b = random.randint(0, m*p-1)
                            V[extra_b][ci] = random.choice([-1, 1])
                        if random.random() < 0.2:
                            extra_c = random.randint(0, p*n-1)
                            W[extra_c][ci] = random.choice([-1, 1])
            
            # Quick check: verify active triples first
            active_ok = True
            for (a, b, c) in active_triples:
                s = sum(U[a][ci] * V[b][ci] * W[c][ci] for ci in range(r))
                if s != 1:
                    active_ok = False
                    break
            
            if active_ok:
                # Full verification
                if verify_fast(U, V, W, r):
                    return {"U": U, "V": V, "W": W}
            else:
                # Try local refinement to fix errors
                if time.time() < deadline:
                    refined, remaining_errors = local_refine(U, V, W, r, max_steps=2000)
                    if refined is not None:
                        return {"U": refined[0], "V": refined[1], "W": refined[2]}
        
        return None
    
    # Try to find lower rank decompositions
    # Start from optimistic rank and go up
    for r in range(18, best_rank):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Spend more time on promising ranks
        if r <= 22:
            max_attempts = min(500, int(time_remaining * 10))
        elif r <= 26:
            max_attempts = min(300, int(time_remaining * 5))
        else:
            max_attempts = min(100, int(time_remaining * 2))
        
        result = try_rank_structured(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # Final polish: try to reduce rank further from current best
    if time.time() < deadline - 10 and best_rank > 18:
        for r in range(best_rank - 1, max(17, best_rank - 5), -1):
            if time.time() > deadline - 10:
                break
            result = try_rank_structured(r, max_attempts=100)
            if result is not None:
                best_result = result
                best_rank = r
                break
    
    return best_result
# EVOLVE-BLOCK-END
