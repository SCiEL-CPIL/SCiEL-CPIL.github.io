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

    # Precompute weights for digit extraction
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    all_indices = np.arange(total, dtype=np.int32)
    
    # Precompute balls: balls[w] contains w and all words at distance 1 from w
    balls = np.zeros((total, ball_size), dtype=np.int32)
    balls[:, 0] = all_indices
    
    for pos in range(n):
        weight = weights[pos]
        for delta in [-1, 1]:
            digits = (all_indices // weight) % q
            new_digits = (digits + delta + q) % q
            new_indices = all_indices + (new_digits - digits) * weight
            balls[:, 1 + 2 * pos + (1 if delta > 0 else 0)] = new_indices
            
    # Precompute bitmasks for each ball
    # Python integers can handle 6561 bits efficiently
    ball_masks = [0] * total
    for i in range(total):
        mask = 0
        for j in range(ball_size):
            mask |= (1 << balls[i, j])
        ball_masks[i] = mask

    best_size = total
    best_set = None
    rng = np.random.default_rng(42)
    
    # Use multiple restarts with different random seeds for greedy phase
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        # Vary the random seed for each restart to explore different solutions
        rng_restarts = np.random.default_rng(42 + restarts)
        
        # === PHASE 1: Randomized Greedy Construction ===
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=np.int32)
        
        max_greedy_iters = 5000
        for _ in range(max_greedy_iters):
            if time.time() > deadline - 12.0:
                break
            if np.max(uncovered) == 0:
                break
            
            # Count how many uncovered words each candidate covers
            coverage = uncovered[balls].sum(axis=1)
            
            if np.max(coverage) == 0:
                break
            
            # Among candidates with max coverage, pick one with random tie-breaking
            max_cov = np.max(coverage)
            candidates = np.where(coverage == max_cov)[0]
            
            # Add small random perturbation for diversity
            scores = coverage.astype(np.float64)
            scores += rng_restarts.random(total) * 0.1 * max_cov
            
            best_idx = int(np.argmax(scores))
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered = np.minimum(uncovered, cover_count)
        
        # If we didn't complete the cover, try to fix
        while np.any(uncovered > 0):
            if time.time() > deadline - 12.0:
                break
            uncov_idx = int(np.where(uncovered > 0)[0][0])
            if uncov_idx not in selected:
                selected.append(uncov_idx)
                cover_count[balls[uncov_idx]] += 1
                uncovered = np.minimum(uncovered, cover_count)
        
        if np.any(uncovered > 0):
            continue
        
        # === PHASE 2: Local Search with Bitset Acceleration ===
        
        # 2a: Remove all removable codewords
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if any word in ball(c) is covered only by c
                # We can just check min cover count
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # 2b: Swap-based improvement using Bitmasks
        improved = True
        max_swap_rounds = 100
        for round_num in range(max_swap_rounds):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            
            # Shuffle order for diversity
            order = rng_restarts.permutation(len(selected))
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                
                c = selected[idx]
                
                # Try to remove
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    del selected[idx]
                    improved = True
                    # Adjust order
                    if idx < len(order):
                        for j in range(idx, len(order) - 1):
                            order[j] = order[j + 1]
                        order = order[:-1]
                    continue
                
                # Try to swap with a randomly chosen other codeword
                # Identify critical words if we remove c
                cover_count[balls[c]] -= 1
                
                # Compute critical mask
                critical_mask = 0
                for w in balls[c]:
                    if cover_count[w] == 1:
                        critical_mask |= (1 << w)
                
                if critical_mask == 0:
                    # c was removable
                    cover_count[balls[c]] += 1
                    continue
                
                # Try a few random swaps
                swap_found = False
                # Try 50 random candidates
                for _ in range(50):
                    if time.time() > deadline - 5.0:
                        break
                    
                    c_prime = int(all_indices[rng_restarts.integers(total)])
                    if c_prime == c:
                        continue
                    
                    # Check if c_prime covers all critical words using bitset
                    if (ball_masks[c_prime] & critical_mask) == critical_mask:
                        # Perform the swap
                        cover_count[balls[c_prime]] += 1
                        selected[idx] = c_prime
                        swap_found = True
                        improved = True
                        break
                
                if not swap_found:
                    # Revert the removal of c
                    cover_count[balls[c]] += 1
        
        # 2c: Final cleanup
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
        # Fallback
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
