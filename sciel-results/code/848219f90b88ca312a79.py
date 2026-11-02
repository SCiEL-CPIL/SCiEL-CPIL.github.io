import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls: for each word, its ball (itself + all words at distance 1)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # Precompute reverse mapping: for each word, which codewords cover it
    # Actually, let's use a different approach: maintain coverage counts
    
    best_size = total
    best_set = None
    
    # Use bit-parallel representation for fast set operations
    # Each word's ball is a set of 17 words. We can represent coverage as counts.
    
    # Strategy: Multiple randomized greedy constructions with local search
    
    def greedy_construct(seed, deadline_local):
        """Randomized greedy set cover."""
        rng = random.Random(seed)
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        uncovered_count = total
        
        while uncovered_count > 0:
            if time.time() > deadline_local:
                return None
            
            # Find candidate that covers most uncovered words
            # For speed, sample a subset of candidates
            if uncovered_count > 100:
                # Sample 200 random candidates
                sample_size = min(200, total)
                candidates = rng.sample(range(total), sample_size)
                best_c = None
                best_count = -1
                for c in candidates:
                    cnt = np.sum(uncovered[balls[c]])
                    if cnt > best_count:
                        best_count = cnt
                        best_c = c
            else:
                # Check all candidates
                best_c = None
                best_count = -1
                for c in range(total):
                    cnt = np.sum(uncovered[balls[c]])
                    if cnt > best_count:
                        best_count = cnt
                        best_c = c
            
            if best_c is None or best_count <= 0:
                return None
            
            selected.append(best_c)
            cover_count[balls[best_c]] += 1
            new_uncovered = (cover_count == 0)
            uncovered_count = np.sum(new_uncovered)
            uncovered = new_uncovered
        
        return selected, cover_count
    
    def remove_redundant(selected, cover_count, deadline_local):
        """Remove codewords that are not needed."""
        changed = True
        while changed and time.time() < deadline_local:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count
    
    def local_search_swap(selected, cover_count, deadline_local, rng):
        """Try 1-swap moves: remove one, add one."""
        improved = True
        while improved and time.time() < deadline_local:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                # Find candidates that cover all critical words
                # Use intersection of balls of critical words
                candidates = set(balls[critical_words[0]].tolist())
                for cw in critical_words[1:]:
                    candidates.intersection_update(balls[cw].tolist())
                    if not candidates:
                        break
                
                candidates.discard(c)
                
                if candidates:
                    # Pick best candidate by redundancy score
                    best_c = None
                    best_score = -1
                    for c_prime in list(candidates)[:30]:
                        score = np.sum(cover_count[balls[c_prime]] > 1)
                        if score > best_score:
                            best_score = score
                            best_c = c_prime
                    if best_c is not None:
                        cover_count[balls[c]] -= 1
                        cover_count[balls[best_c]] += 1
                        new_selected.append(best_c)
                        improved = True
                    else:
                        new_selected.append(c)
                else:
                    new_selected.append(c)
            
            selected = new_selected
        
        return selected, cover_count
    
    def local_search_2exchange(selected, cover_count, deadline_local, rng):
        """Try 2-exchange moves: remove two, add one."""
        if len(selected) <= 486:
            return selected, cover_count
            
        improved = True
        while improved and time.time() < deadline_local:
            improved = False
            # Try random pairs
            num_trials = min(2000, len(selected) * (len(selected) - 1) // 2)
            
            for _ in range(num_trials):
                if time.time() > deadline_local:
                    break
                
                idx1 = rng.randint(0, len(selected) - 1)
                idx2 = rng.randint(0, len(selected) - 1)
                if idx1 == idx2:
                    continue
                    
                c1 = selected[idx1]
                c2 = selected[idx2]
                
                # Temporarily remove
                cover_count[balls[c1]] -= 1
                cover_count[balls[c2]] -= 1
                
                uncovered_after = (cover_count == 0)
                
                if np.any(uncovered_after):
                    uncovered_words = np.where(uncovered_after)[0]
                    
                    if len(uncovered_words) <= 6:
                        # Find c3 that covers all uncovered words
                        candidates = set(balls[uncovered_words[0]].tolist())
                        for uw in uncovered_words[1:]:
                            candidates.intersection_update(balls[uw].tolist())
                            if not candidates:
                                break
                        
                        candidates.discard(c1)
                        candidates.discard(c2)
                        
                        if candidates:
                            c3 = rng.choice(list(candidates))
                            cover_count[balls[c3]] += 1
                            # Remove c1, c2 from selected
                            new_selected = []
                            for c in selected:
                                if c != c1 and c != c2:
                                    new_selected.append(c)
                            new_selected.append(c3)
                            selected = new_selected
                            improved = True
                            break
                        else:
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                    else:
                        cover_count[balls[c1]] += 1
                        cover_count[balls[c2]] += 1
                else:
                    # Both can be removed
                    new_selected = []
                    for c in selected:
                        if c != c1 and c != c2:
                            new_selected.append(c)
                    selected = new_selected
                    improved = True
                    break
                        
                if not improved:
                    cover_count[balls[c1]] += 1
                    cover_count[balls[c2]] += 1
                    
        return selected, cover_count
    
    # Run multiple iterations with different random seeds
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        seed = iteration * 12345 + 67890
        
        # Greedy construction
        result = greedy_construct(seed, deadline - 10.0)
        if result is None:
            continue
            
        selected, cover_count = result
        
        # Local search phases
        selected, cover_count = remove_redundant(selected, cover_count, deadline - 5.0)
        selected, cover_count = local_search_swap(selected, cover_count, deadline - 5.0, random.Random(seed))
        selected, cover_count = local_search_2exchange(selected, cover_count, deadline - 5.0, random.Random(seed + 1))
        selected, cover_count = remove_redundant(selected, cover_count, deadline - 3.0)
        
        # Verify
        final_cov = np.zeros(total, dtype=np.int32)
        for c in selected:
            final_cov[balls[c]] += 1
            
        if np.any(final_cov == 0):
            continue
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = list(selected)
            
        if best_size <= 486:
            break
    
    if best_set is None:
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
