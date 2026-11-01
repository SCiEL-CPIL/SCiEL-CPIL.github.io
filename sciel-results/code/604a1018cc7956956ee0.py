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

def _local_search_swap(q, n, radius, deadline):
    """
    Local search with Removal and Swap operations.
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
        # c is redundant if all y in balls[c] have cover_count[y] >= 2
        # If any y has count == 1, removing c makes it 0 (invalid)
        # We can vectorize this check? 
        # balls[c] is a list. 
        # min(cover_count[balls[c]]) >= 2
        # For speed, we can just iterate. Ball size is 17.
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
        # Check if S - {c_out} + {c_in} is valid.
        # 1. c_in must not be in S (if it is, it's not a swap, it's a no-op or removal)
        if c_in in current_set:
            return False
        # 2. After removing c_out, all y in balls[c_out] must still be covered by S \ {c_out} OR by c_in.
        #    Specifically, for y in balls[c_out]:
        #    If y is NOT in balls[c_in], then cover_count[y] must be >= 2 (so that removing c_out leaves >= 1).
        #    If y IS in balls[c_in], then it will be covered by c_in, so it's fine even if cover_count[y] was 1.
        #    Wait, if y is in balls[c_in], c_in covers it. So we don't care about the old count for validity of y, 
        #    EXCEPT we must ensure that words NOT in balls[c_out] are not affected (they aren't).
        #    And words in balls[c_out] \ balls[c_in] must have count >= 2.
        #    Words in balls[c_out] ∩ balls[c_in] are covered by c_in, so they are safe.
        
        # Optimization: 
        # Let U = balls[c_out] \ balls[c_in].
        # We need min(cover_count[y] for y in U) >= 2.
        # Note: cover_count is the count BEFORE the swap.
        
        # We also need to ensure that c_in doesn't "break" anything? 
        # Adding c_in only increases counts, so it never breaks validity.
        # So the only constraint is that removing c_out doesn't uncover any word that c_in doesn't cover.
        
        # Implementation:
        # Create a set of words covered by c_in? 
        # ball_in_set = set(balls[c_in])
        # for y in balls[c_out]:
        #     if y not in ball_in_set:
        #         if cover_count[y] < 2:
        #             return False
        # return True
        
        # To make it faster, we can precompute a bitmask or just use a set.
        # Since ball size is 17, set intersection is fast.
        ball_in_set = set(balls[c_in])
        for y in balls[c_out]:
            if y not in ball_in_set:
                if cover_count[y] < 2:
                    return False
        return True

    improved = True
    swap_attempts = 0
    max_swap_attempts = 10000 # Safety
    
    while improved and time.time() < deadline - 5:
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
                
        if not improved:
            # 2. Try Swaps
            # Pick a random codeword to try to swap out
            if len(current_set) > 0 and swap_attempts < max_swap_attempts:
                c_out = random.choice(list(current_set))
                # Find candidate c_in
                # Heuristic: Look for c_in that covers the "critical" words of c_out
                # Critical words: y in balls[c_out] where cover_count[y] == 1
                critical_words = [y for y in balls[c_out] if cover_count[y] == 1]
                
                if critical_words:
                    # c_in MUST cover all critical words
                    # Candidates are intersection of inv[y] for y in critical_words
                    # But inv[y] is large.
                    # Simpler: Randomly pick a c_in from the union of inv[y] for y in critical_words?
                    # Or just pick a random c_in from the whole space? Too slow.
                    
                    # Let's pick a random critical word y_c
                    y_c = random.choice(critical_words)
                    # Pick a random candidate c_in from inv[y_c]
                    # Exclude c_out itself
                    candidates_in = [c for c in inv[y_c] if c != c_out and c not in current_set]
                    if candidates_in:
                        c_in = random.choice(candidates_in)
                        if is_valid_swap(c_out, c_in):
                            # Perform swap
                            remove_c(c_out)
                            add_c(c_in)
                            improved = True # We consider a swap as progress to escape local minimum
                            swap_attempts += 1
                            # After swap, we might be able to remove something else, so we loop back to removal phase
                            continue
                # If no swap found, we are stuck in a local minimum
                # To avoid infinite loop, we break or do a perturbation
                # For now, we just stop if we can't swap
                break

    # Convert best_set to digits
    codewords = []
    for idx in best_set:
        codewords.append(_int_to_digits(idx, q, n, powers))
        
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    try:
        result = _local_search_swap(q, n, radius, deadline)
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
