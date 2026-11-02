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
    # For efficiency, we only need to check:
    # 1. All active triples must sum to 1
    # 2. All inactive triples must sum to 0
    
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
    
    def local_search(U, V, W, r, max_iters=5000):
        """Local search to fix violations by modifying single entries."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        cur_err = error_count(U, V, W, r)
        if cur_err == 0:
            return U, V, W, 0
        
        # Precompute current sums for all triples
        sums = {}
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    s = 0
                    for ci in range(r):
                        s += U[a][ci] * V[b][ci] * W[c][ci]
                    sums[(a, b, c)] = s
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find violated triples
            violated = []
            for (a, b, c), s in sums.items():
                expected = 1 if (a, b, c) in active_set else 0
                if s != expected:
                    violated.append((a, b, c))
            
            if not violated:
                return U, V, W, 0
            
            # Pick a random violated triple
            a, b, c = random.choice(violated)
            expected = 1 if (a, b, c) in active_set else 0
            cur_sum = sums[(a, b, c)]
            delta_needed = expected - cur_sum
            
            # Try to fix by changing one entry in one of the three matrices
            # at a random column
            ci = random.randint(0, r - 1)
            
            # Option 1: change U[a][ci]
            old_u = U[a][ci]
            if old_u != 0:
                # If we change U[a][ci] from old_u to new_u, the sum changes by
                # (new_u - old_u) * V[b][ci] * W[c][ci]
                factor = V[b][ci] * W[c][ci]
                if factor != 0:
                    # We need (new_u - old_u) * factor = delta_needed
                    # So new_u - old_u = delta_needed / factor
                    if delta_needed % factor == 0:
                        new_u = old_u + delta_needed // factor
                        if abs(new_u) <= 3:  # Keep values small
                            # Apply change and update sums
                            U[a][ci] = new_u
                            diff = new_u - old_u
                            for b2 in range(m * p):
                                for c2 in range(p * n):
                                    sums[(a, b2, c2)] += diff * V[b2][ci] * W[c2][ci]
                            # Check if we improved
                            new_err = 0
                            for (a2, b2, c2), s in sums.items():
                                exp2 = 1 if (a2, b2, c2) in active_set else 0
                                if s != exp2:
                                    new_err += 1
                            if new_err < cur_err:
                                cur_err = new_err
                            else:
                                # Revert
                                U[a][ci] = old_u
                                for b2 in range(m * p):
                                    for c2 in range(p * n):
                                        sums[(a, b2, c2)] -= diff * V[b2][ci] * W[c2][ci]
            
            # Option 2: change V[b][ci]
            old_v = V[b][ci]
            if old_v != 0:
                factor = U[a][ci] * W[c][ci]
                if factor != 0:
                    if delta_needed % factor == 0:
                        new_v = old_v + delta_needed // factor
                        if abs(new_v) <= 3:
                            V[b][ci] = new_v
                            diff = new_v - old_v
                            for a2 in range(n * m):
                                for c2 in range(p * n):
                                    sums[(a2, b, c2)] += U[a2][ci] * diff * W[c2][ci]
                            new_err = 0
                            for (a2, b2, c2), s in sums.items():
                                exp2 = 1 if (a2, b2, c2) in active_set else 0
                                if s != exp2:
                                    new_err += 1
                            if new_err < cur_err:
                                cur_err = new_err
                            else:
                                V[b][ci] = old_v
                                for a2 in range(n * m):
                                    for c2 in range(p * n):
                                        sums[(a2, b, c2)] -= U[a2][ci] * diff * W[c2][ci]
            
            # Option 3: change W[c][ci]
            old_w = W[c][ci]
            if old_w != 0:
                factor = U[a][ci] * V[b][ci]
                if factor != 0:
                    if delta_needed % factor == 0:
                        new_w = old_w + delta_needed // factor
                        if abs(new_w) <= 3:
                            W[c][ci] = new_w
                            diff = new_w - old_w
                            for a2 in range(n * m):
                                for b2 in range(m * p):
                                    sums[(a2, b2, c)] += U[a2][ci] * V[b2][ci] * diff
                            new_err = 0
                            for (a2, b2, c2), s in sums.items():
                                exp2 = 1 if (a2, b2, c2) in active_set else 0
                                if s != exp2:
                                    new_err += 1
                            if new_err < cur_err:
                                cur_err = new_err
                            else:
                                W[c][ci] = old_w
                                for a2 in range(n * m):
                                    for b2 in range(m * p):
                                        sums[(a2, b2, c)] -= U[a2][ci] * V[b2][ci] * diff
            
            # Option 4: try changing a zero entry to a small value
            # Pick a random matrix and position
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                a_pos = random.randint(0, n * m - 1)
                ci = random.randint(0, r - 1)
                if U[a_pos][ci] == 0:
                    # Changing U[a_pos][ci] from 0 to val changes sums for all (a_pos, b2, c2)
                    # by val * V[b2][ci] * W[c2][ci]
                    # We want to reduce error. Try small values.
                    for new_val in [1, -1, 2, -2]:
                        # Estimate error change
                        err_change = 0
                        for b2 in range(m * p):
                            for c2 in range(p * n):
                                old_s = sums[(a_pos, b2, c2)]
                                new_s = old_s + new_val * V[b2][ci] * W[c2][ci]
                                exp2 = 1 if (a_pos, b2, c2) in active_set else 0
                                old_bad = 1 if old_s != exp2 else 0
                                new_bad = 1 if new_s != exp2 else 0
                                err_change += new_bad - old_bad
                        if err_change < 0:
                            U[a_pos][ci] = new_val
                            for b2 in range(m * p):
                                for c2 in range(p * n):
                                    sums[(a_pos, b2, c2)] += new_val * V[b2][ci] * W[c2][ci]
                            cur_err += err_change
                            break
            elif mat_choice == 1:
                b_pos = random.randint(0, m * p - 1)
                ci = random.randint(0, r - 1)
                if V[b_pos][ci] == 0:
                    for new_val in [1, -1, 2, -2]:
                        err_change = 0
                        for a2 in range(n * m):
                            for c2 in range(p * n):
                                old_s = sums[(a2, b_pos, c2)]
                                new_s = old_s + U[a2][ci] * new_val * W[c2][ci]
                                exp2 = 1 if (a2, b_pos, c2) in active_set else 0
                                old_bad = 1 if old_s != exp2 else 0
                                new_bad = 1 if new_s != exp2 else 0
                                err_change += new_bad - old_bad
                        if err_change < 0:
                            V[b_pos][ci] = new_val
                            for a2 in range(n * m):
                                for c2 in range(p * n):
                                    sums[(a2, b_pos, c2)] += U[a2][ci] * new_val * W[c2][ci]
                            cur_err += err_change
                            break
            else:
                c_pos = random.randint(0, p * n - 1)
                ci = random.randint(0, r - 1)
                if W[c_pos][ci] == 0:
                    for new_val in [1, -1, 2, -2]:
                        err_change = 0
                        for a2 in range(n * m):
                            for b2 in range(m * p):
                                old_s = sums[(a2, b2, c_pos)]
                                new_s = old_s + U[a2][ci] * V[b2][ci] * new_val
                                exp2 = 1 if (a2, b2, c_pos) in active_set else 0
                                old_bad = 1 if old_s != exp2 else 0
                                new_bad = 1 if new_s != exp2 else 0
                                err_change += new_bad - old_bad
                        if err_change < 0:
                            W[c_pos][ci] = new_val
                            for a2 in range(n * m):
                                for b2 in range(m * p):
                                    sums[(a2, b2, c_pos)] += U[a2][ci] * V[b2][ci] * new_val
                            cur_err += err_change
                            break
        
        return U, V, W, cur_err
    
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
            
            strategy = attempt % 3
            
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
                
            else:
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
            
            # Quick check: verify active triples first
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
            
            # If active triples are satisfied but some inactive ones aren't,
            # try local search to fix them
            err = error_count(U, V, W, r)
            if err > 0 and err <= 20:
                U2, V2, W2, final_err = local_search(U, V, W, r, max_iters=2000)
                if final_err == 0:
                    return {"U": U2, "V": V2, "W": W2}
        
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
            max_attempts = min(300, int(time_remaining * 8))
        elif r <= 26:
            max_attempts = min(200, int(time_remaining * 4))
        else:
            max_attempts = min(80, int(time_remaining * 2))
        
        result = try_rank_structured(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            # Continue searching for even lower ranks? No, we want minimum, 
            # but we found one at rank r. Let's try lower ranks too.
            # Actually, we're going up from 18, so if we find rank r, 
            # we should check if there's something lower. But we already tried lower.
            # So we can stop here.
            break
    
    return best_result
# EVOLVE-BLOCK-END
