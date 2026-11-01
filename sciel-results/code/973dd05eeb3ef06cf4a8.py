import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline, start_time)

    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline, start_time):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n  # 17
    
    # Precompute neighbors using vectorized operations
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Create all words as 3D array: (total, n)
    all_words = np.zeros((total, n), dtype=np.int32)
    for i in range(n):
        all_words[:, i] = np.arange(total) % q
        if i < n - 1:
            all_words[:, i] = (all_words[:, i] * q) % q  # This isn't right, let me redo
        # Actually let me just do it properly
    # Better approach:
    all_words = np.zeros((total, n), dtype=np.int32)
    for i in range(n):
        w = np.arange(total, dtype=np.int32)
        for j in range(i, -1, -1):
            w = w // q
        all_words[:, i] = w
    
    # Build balls: for each word, store its 17 neighbors
    balls = np.zeros((total, ball_size), dtype=np.int32)
    balls[:, 0] = np.arange(total)
    
    for pos in range(n):
        for delta in [-1, 1]:
            new_words = all_words.copy()
            new_words[:, pos] = (all_words[:, pos] + delta + q) % q
            # Convert back to integer index
            for i in range(n):
                new_words[:, i] *= q if i < n - 1 else 1
            # This is getting complicated, let me use a simpler approach
    
    # Simpler: just compute balls using the digit extraction approach
    balls = np.zeros((total, ball_size), dtype=np.int32)
    weights_list = [3 ** (n - 1 - i) for i in range(n)]
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights_list[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    all_indices = np.arange(total, dtype=np.int32)
    
    best_size = total
    best_set = None
    rng = np.random.default_rng(42)
    
    iteration = 0
    while time.time() < deadline - 8.0:
        iteration += 1
        if time.time() - start_time > 110:
            break
        
        # === PHASE 1: Construct initial cover using weighted greedy ===
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=np.int32)
        
        # Weight function: prioritize covering sparse areas
        # Start with uniform weights, but bias toward center of uncovered regions
        # We'll use a simple heuristic: count how many uncovered words each candidate covers
        # and weight by 1/sqrt(covered_count) to favor diverse coverage
        
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
            if np.max(uncovered) == 0:
                break
            
            # For each candidate, count how many currently uncovered words it covers
            # balls[candidate] gives the 17 words covered by candidate
            # We want sum of uncovered[balls[candidate]]
            coverage = uncovered[balls].sum(axis=1)
            
            if np.max(coverage) == 0:
                break
            
            # Among candidates with max coverage, pick one randomly
            max_cov = np.max(coverage)
            candidates = np.where(coverage == max_cov)[0]
            
            # Add a small random perturbation to break ties and encourage diversity
            scores = coverage.copy().astype(np.float64)
            scores += rng.random(total) * 0.1 * max_cov
            
            best_idx = int(np.argmax(scores))
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered = np.minimum(uncovered, cover_count)
        
        # If we didn't complete, try to fix by adding more codewords
        while np.any(uncovered > 0):
            if time.time() > deadline - 10.0:
                break
            # Find an uncovered word and add a random codeword that covers it
            uncov_idx = int(np.where(uncovered > 0)[0][0])
            # Find all codewords whose ball contains uncov_idx
            # This is slow; instead just add uncov_idx itself as a codeword
            if uncov_idx not in selected:
                selected.append(uncov_idx)
                cover_count[balls[uncov_idx]] += 1
                uncovered = np.minimum(uncovered, cover_count)
        
        if np.any(uncovered > 0):
            continue
        
        # === PHASE 2: Aggressive local search ===
        # 2a: Remove all removable codewords
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # 2b: Swap-based improvement - try to replace codewords to enable further removals
        improved = True
        max_swap_rounds = 50
        for round_num in range(max_swap_rounds):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            
            # Shuffle order for diversity
            order = rng.permutation(len(selected))
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                
                c = selected[idx]
                
                # Try to remove
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    del selected[idx]
                    improved = True
                    # Adjust order since we removed an element
                    if idx < len(order):
                        for j in range(idx, len(order) - 1):
                            order[j] = order[j + 1]
                        order = order[:-1]
                    continue
                
                # Try to swap with a randomly chosen other codeword
                # The swap is valid if: removing c leaves some words with count 1
                # (critical words), and those critical words are all covered by c'
                # AND adding c' doesn't cause issues (it can't, since we're just adding coverage)
                
                # First, identify what happens if we remove c
                cover_count[balls[c]] -= 1
                critical = balls[c][cover_count[balls[c]] == 1]
                
                if len(critical) == 0:
                    # c was removable, but we checked that above... 
                    # Actually this can happen if the min was exactly 1 but after removal
                    # some went to 0, but we checked min > 1, so this shouldn't happen
                    cover_count[balls[c]] += 1
                    continue
                
                # Now try a few random swaps
                swap_found = False
                for _ in range(20):
                    if time.time() > deadline - 5.0:
                        break
                    
                    c_prime = int(all_indices[rng.integers(total)])
                    if c_prime == c:
                        continue
                    
                    # Check if c_prime covers all critical words
                    # critical words must be in balls[c_prime]
                    # Create a set of balls[c_prime] for fast lookup
                    cp_set = set(balls[c_prime].tolist())
                    
                    valid = True
                    for w in critical:
                        if int(w) not in cp_set:
                            valid = False
                            break
                    
                    if valid:
                        # Perform the swap
                        cover_count[balls[c_prime]] += 1
                        selected[idx] = c_prime
                        swap_found = True
                        improved = True
                        break
                
                if not swap_found:
                    # Revert the removal of c
                    cover_count[balls[c]] += 1
        
        # 2c: Final cleanup - remove any newly removable codewords
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 486:
                break
    
    # Convert to output format
    if best_set is None:
        # Fallback: return all words (should never happen)
        return [list(c) for c in itertools.product(range(q), repeat=n)]
    
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
