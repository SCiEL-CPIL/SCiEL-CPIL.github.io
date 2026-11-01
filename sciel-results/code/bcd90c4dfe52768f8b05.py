import time
import itertools
import numpy as np
import random

# EVOLVE-BLOCK-START

def _precompute_balls(q, n, radius, deadline):
    """Precompute for each word x the list of words within Hamming distance <= radius."""
    total = q ** n
    if radius == 1:
        balls = []
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
        
        for x in range(total):
            ball = [x]
            for i in range(n):
                old_digit = (x // powers[i]) % q
                for new_digit in range(q):
                    if new_digit != old_digit:
                        y = x + (new_digit - old_digit) * powers[i]
                        ball.append(y)
            balls.append(ball)
            if time.time() > deadline - 10:
                break
        return balls
    else:
        raise NotImplementedError("Only radius 1 is optimized")

def _int_to_digits(x, q, n, powers):
    digits = []
    for i in range(n):
        d = (x // powers[i]) % q
        digits.append(d)
    return digits

def _greedy_cover_fast(q, n, radius, deadline):
    """
    Fast greedy cover to get a valid initial solution.
    Returns list of integer indices.
    """
    total = q ** n
    balls = _precompute_balls(q, n, radius, deadline)
    
    covered = np.zeros(total, dtype=bool)
    # counts[i] = number of uncovered elements in ball[i]
    counts = np.array([len(b) for b in balls], dtype=np.int32)
    
    # Inverse index: inv[y] = list of ball indices i where y in balls[i]
    inv = [[] for _ in range(total)]
    for i, ball in enumerate(balls):
        for y in ball:
            inv[y].append(i)
            
    selected = []
    
    while True:
        if time.time() > deadline - 5:
            break
            
        max_idx = np.argmax(counts)
        if counts[max_idx] == 0:
            break
            
        selected.append(max_idx)
        
        ball = balls[max_idx]
        for y in ball:
            if not covered[y]:
                covered[y] = True
                for i in inv[y]:
                    counts[i] -= 1
                    
        if np.all(covered):
            break

    return selected

def _local_search_advanced(q, n, radius, deadline):
    """
    Advanced local search with targeted swaps and perturbations.
    """
    total = q ** n
    balls = _precompute_balls(q, n, radius, deadline)
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Inverse index: inv[y] = list of codeword indices c such that y in balls[c]
    inv = [[] for _ in range(total)]
    for c, ball in enumerate(balls):
        for y in ball:
            inv[y].append(c)

    # Initial solution via greedy
    current_set = set(_greedy_cover_fast(q, n, radius, deadline))
    
    # Precompute coverage counts for the current set
    # cover_count[y] = number of codewords in current_set whose ball contains y
    cover_count = np.zeros(total, dtype=np.int32)
    for c in current_set:
        for y in balls[c]:
            cover_count[y] += 1
            
    best_set = set(current_set)
    best_size = len(current_set)
    
    def is_redundant(c):
        for y in balls[c]:
            if cover_count[y] == 1:
                return False
        return True

    def remove_c(c):
        for y in balls[c]:
            cover_count[y] -= 1
        current_set.remove(c)

    def add_c(c):
        for y in balls[c]:
            cover_count[y] += 1
        current_set.add(c)

    def is_valid_swap(c_out, c_in):
        if c_in in current_set:
            return False
        
        ball_in_set = set(balls[c_in])
        for y in balls[c_out]:
            if y not in ball_in_set:
                if cover_count[y] < 2:
                    return False
        return True

    improved = True
    stuck_count = 0
    max_stuck = 500
    
    while time.time() < deadline - 5:
        if improved:
            stuck_count = 0
            improved = False
        
        # 1. Try Removals
        candidates = list(current_set)
        random.shuffle(candidates)
        
        for c in candidates:
            if c not in current_set:
                continue
            if is_redundant(c):
                remove_c(c)
                improved = True
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
        
        if improved:
            continue

        # 2. Try Targeted Swaps
        # Pick a random codeword to try to swap out
        if len(current_set) > 0:
            c_out = random.choice(list(current_set))
            critical_words = [y for y in balls[c_out] if cover_count[y] == 1]
            
            if critical_words:
                # Find candidates c_in that cover ALL critical words
                # Intersection of inv[y] for y in critical_words
                # Optimize by sorting critical words by len(inv[y]) and intersecting sets
                crit_inv_sets = [set(inv[y]) for y in critical_words]
                if crit_inv_sets:
                    # Start with the smallest set
                    crit_inv_sets.sort(key=len)
                    candidate_set = set(crit_inv_sets[0])
                    for s in crit_inv_sets[1:]:
                        candidate_set.intersection_update(s)
                    
                    # Remove current set members and c_out
                    candidate_set -= current_set
                    candidate_set.discard(c_out)
                    
                    if candidate_set:
                        # Try a few random candidates from this set
                        candidates_list = list(candidate_set)
                        random.shuffle(candidates_list)
                        found_swap = False
                        for c_in in candidates_list[:10]: # Limit attempts per c_out
                            if is_valid_swap(c_out, c_in):
                                remove_c(c_out)
                                add_c(c_in)
                                improved = True
                                found_swap = True
                                break
                        
                        if found_swap:
                            continue
            else:
                # No critical words means c_out is redundant, which should have been caught in removal phase
                # But due to random shuffle, maybe we missed it?
                # Or maybe it's redundant now but wasn't before?
                # Actually, if cover_count[y] > 1 for all y in ball[c_out], it is redundant.
                # So this branch implies is_redundant(c_out) is True.
                # We can just remove it.
                remove_c(c_out)
                improved = True
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                continue

        # 3. Perturbation if stuck
        stuck_count += 1
        if stuck_count > max_stuck:
            # Remove 5-10 random codewords and re-greedy
            num_remove = random.randint(5, 10)
            to_remove = random.sample(list(current_set), min(num_remove, len(current_set)))
            for c in to_remove:
                if c in current_set:
                    remove_c(c)
            
            # Re-greedy to restore coverage
            # We need to find words not covered
            # Instead of full re-greedy, just add back best candidates
            # Find all y with cover_count[y] == 0
            # This is expensive. 
            # Alternative: Just add random codewords until valid? No, that's dumb.
            # Better: Use the greedy logic but only on uncovered words.
            # For simplicity and speed, we'll just try to add back codewords that cover the most uncovered words.
            
            # Identify uncovered words
            uncovered = [y for y in range(total) if cover_count[y] == 0]
            if uncovered:
                # Score candidates
                # This is expensive to do for all candidates.
                # Heuristic: Just pick random candidates from the whole space and add if they help?
                # Or better: Use the precomputed 'inv' to find candidates for a specific uncovered word.
                # Pick a random uncovered word y
                y_uncovered = random.choice(uncovered)
                candidates_for_y = [c for c in inv[y_uncovered] if c not in current_set]
                if candidates_for_y:
                    # Pick the one that covers the most other uncovered words?
                    # Approximation: Just pick one.
                    c_add = random.choice(candidates_for_y)
                    add_c(c_add)
            
            stuck_count = 0
            improved = True # Force loop to continue

    # Convert best_set to digits
    codewords = []
    for idx in best_set:
        codewords.append(_int_to_digits(idx, q, n, powers))
        
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    try:
        result = _local_search_advanced(q, n, radius, deadline)
    except Exception as e:
        # Fallback to greedy if local search fails
        try:
            result_idx = _greedy_cover_fast(q, n, radius, deadline)
            powers = [1] * n
            for i in range(1, n):
                powers[i] = powers[i-1] * q
            result = []
            for idx in result_idx:
                result.append(_int_to_digits(idx, q, n, powers))
        except Exception:
            raise e

    return result

# EVOLVE-BLOCK-END
