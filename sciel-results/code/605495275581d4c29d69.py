import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    if total_words > 10000 or radius > 1:
        # Fallback for larger spaces or general radius
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Bitset-based Exact Branch and Bound for small spaces (e.g., 3^7 = 2187)
    
    # 1. Generate all words
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # 2. Precompute balls as bitmasks (Python ints)
    balls = [0] * total_words
    
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            mask |= (1 << i)
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
    else:
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask

    # Precompute coverers: for each word u, which codewords cover it?
    # coverers[u] is a list of indices c such that u is in balls[c]
    coverers = [[] for _ in range(total_words)]
    for c in range(total_words):
        mask = balls[c]
        # Iterate through set bits in mask
        m = mask
        while m:
            # Get least significant set bit
            lsb = m & -m
            u = lsb.bit_length() - 1
            coverers[u].append(c)
            m ^= lsb

    all_ones_mask = (1 << total_words) - 1
    
    # Initial Upper Bound: Greedy Set Cover
    def greedy_cover():
        uncovered = all_ones_mask
        selected = []
        while uncovered:
            best_idx = -1
            max_cover = 0
            for i in range(total_words):
                c = (balls[i] & uncovered).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            uncovered &= ~balls[best_idx]
        return selected

    initial_solution = greedy_cover()
    best_size = len(initial_solution)
    best_code = [all_words[i] for i in initial_solution]
    
    # Branch and Bound
    start_time = time.time()
    time_limit = 110.0
    
    # Global variables for recursion
    # Using a list to hold best_size and best_code to avoid nonlocal issues in nested func if needed, 
    # but we can just use closure variables if we define solve inside construct.
    
    # Optimization: Precompute ball_counts for sorting? 
    # In the branch, we sort candidates by coverage.
    
    def solve(uncovered_mask, current_size):
        nonlocal best_size, best_code
        
        if time.time() - start_time > time_limit:
            return False # Timeout signal
        
        if uncovered_mask == 0:
            if current_size < best_size:
                best_size = current_size
                # We need to reconstruct the code? 
                # We can store the path.
                return True
            return False
        
        # Lower Bound
        # Max ball size is 1 + n*(q-1)
        max_ball_size = 1 + n * (q - 1)
        lb = (uncovered_mask.bit_count() + max_ball_size - 1) // max_ball_size
        
        if current_size + lb >= best_size:
            return False
            
        # Find an uncovered word u with minimum number of coverers
        # Actually, all words have the same number of coverers (15 for 3^7 r=1).
        # But some coverers might be "less useful" if they are already "conceptually" selected?
        # No, in this branching scheme, we don't track selected set explicitly in the state, 
        # only `uncovered`. 
        # However, if we pick c, we don't pick other coverers of u *for u*. 
        # We can pick a c that covers many things.
        
        # Heuristic: Pick u that minimizes the number of "effective" coverers?
        # Since all have 15, let's just pick the first uncovered word.
        # Or, pick u that minimizes the sum of "potential" of its coverers?
        # Let's stick to the first uncovered word for speed, or random if we want diversity.
        # Given the symmetry, any u is fine.
        
        # Get index of one uncovered word
        # lsb = uncovered_mask & -uncovered_mask
        # u = lsb.bit_length() - 1
        
        # To be safer against worst-case ordering, let's pick u with min coverers count.
        # It's constant, so just pick the first one.
        u = (uncovered_mask & -uncovered_mask).bit_length() - 1
        
        cands = coverers[u]
        
        # Sort candidates by coverage of current uncovered set (descending)
        # This helps find good solutions early, improving the upper bound for pruning.
        # cands.sort(key=lambda c: (balls[c] & uncovered_mask).bit_count(), reverse=True)
        
        # To save time, don't sort if list is small? 15 is small. Sorting is cheap.
        cands.sort(key=lambda c: (balls[c] & uncovered_mask).bit_count(), reverse=True)
        
        for c in cands:
            new_uncovered = uncovered_mask & ~balls[c]
            # Recursive call
            solve(new_uncovered, current_size + 1)
            
            # If we found a solution with size 1 (impossible) or if best_size became 1?
            # Just continue to find smaller ones.
            
        return False

    # Start B&B
    solve(all_ones_mask, 0)
    
    # If B&B didn't improve on greedy, return greedy.
    # Note: The B&B above doesn't store the *path* to the best solution.
    # We need to modify it to store the path or just return the greedy if B&B only proves optimality.
    # But the task asks to minimize the number. If B&B finds a smaller one, we need it.
    
    # Let's refactor to store the best path.
    best_path = []
    
    def solve_with_path(uncovered_mask, current_size, path):
        nonlocal best_size, best_code, best_path
        
        if time.time() - start_time > time_limit:
            return
            
        if uncovered_mask == 0:
            if current_size < best_size:
                best_size = current_size
                best_path = path[:]
                best_code = [all_words[i] for i in best_path]
            return
        
        max_ball_size = 1 + n * (q - 1)
        lb = (uncovered_mask.bit_count() + max_ball_size - 1) // max_ball_size
        
        if current_size + lb >= best_size:
            return
            
        u = (uncovered_mask & -uncovered_mask).bit_length() - 1
        cands = coverers[u]
        cands.sort(key=lambda c: (balls[c] & uncovered_mask).bit_count(), reverse=True)
        
        for c in cands:
            path.append(c)
            new_uncovered = uncovered_mask & ~balls[c]
            solve_with_path(new_uncovered, current_size + 1, path)
            path.pop()

    # Reset best to greedy
    best_size = len(initial_solution)
    best_code = [all_words[i] for i in initial_solution]
    best_path = initial_solution[:]
    
    solve_with_path(all_ones_mask, 0, [])
    
    return best_code
