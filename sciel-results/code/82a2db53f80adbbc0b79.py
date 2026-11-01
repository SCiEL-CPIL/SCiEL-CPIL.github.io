import time
import itertools
import numpy as np
import random

# EVOLVE-BLOCK-START

def _precompute_balls(q, n, radius, deadline):
    """
    Precompute for each word x the list of words within Hamming distance <= radius.
    Returns a list of lists.
    """
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

def _ils_search(q, n, radius, deadline):
    """
    Iterated Local Search for the covering code problem.
    """
    total = q ** n
    balls = _precompute_balls(q, n, radius, deadline)
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Precompute inverse balls: inv[y] = list of c such that y in balls[c]
    # This is needed for efficient "who covers y" queries during perturbation/repair
    inv = [[] for _ in range(total)]
    for i, ball in enumerate(balls):
        for y in ball:
            inv[y].append(i)
    
    # Initialize with Greedy
    current_set = set(_greedy_cover_fast(q, n, radius, deadline))
    
    # Track coverage count for each word
    # cover_count[y] = number of codewords c in current_set such that y in balls[c]
    cover_count = np.zeros(total, dtype=np.int32)
    for c in current_set:
        for y in balls[c]:
            cover_count[y] += 1
            
    best_size = len(current_set)
    best_set = set(current_set)
    
    def is_valid():
        return np.all(cover_count > 0)
        
    def try_remove(c):
        """
        Attempt to remove c. Returns True if successful (and modified state).
        """
        # Check if c is redundant: all y in balls[c] have cover_count[y] >= 2
        # If any y has cover_count[y] == 1, removing c makes it 0 (invalid)
        # Optimization: check min count in ball
        ball_c = balls[c]
        # If any count is 1, we can't remove
        # np.min(cover_count[ball_c]) >= 2
        # But ball_c is a list, so cover_count[ball_c] works
        if np.min(cover_count[ball_c]) < 2:
            return False
        
        # Safe to remove
        for y in ball_c:
            cover_count[y] -= 1
        current_set.remove(c)
        return True

    def add_c(c):
        for y in balls[c]:
            cover_count[y] += 1
        current_set.add(c)
        
    def repair():
        """
        Greedily add codewords until valid.
        """
        while not is_valid():
            if time.time() > deadline - 2:
                return False # Timeout, state might be invalid
            
            # Find uncovered words
            uncovered_indices = np.where(cover_count == 0)[0]
            if len(uncovered_indices) == 0:
                break
                
            # Pick a random uncovered word
            y = random.choice(uncovered_indices)
            
            # Find a candidate c to add.
            # Strategy: Pick c from inv[y] that covers the most *other* uncovered words?
            # For speed, just pick a random c from inv[y] or the one with highest degree?
            # Let's pick a random c from inv[y] to keep it simple and fast.
            # Actually, picking the c that covers the most uncovered is better.
            # But that's expensive.
            # Let's try: pick c randomly from inv[y].
            
            candidates = inv[y]
            if not candidates:
                # Should not happen if balls are correct
                break
            c = random.choice(candidates)
            
            # Check if adding c helps (it must cover y, so it does)
            # But maybe it's already in the set?
            if c in current_set:
                continue
                
            add_c(c)
            
        return is_valid()

    # Main ILS Loop
    no_improve_count = 0
    while time.time() < deadline - 5:
        
        # 1. Local Search: Remove redundant codewords
        # Repeat until no more removals possible
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Randomize order of candidates to avoid bias
            candidates = list(current_set)
            random.shuffle(candidates)
            
            for c in candidates:
                if c not in current_set:
                    continue
                if try_remove(c):
                    improved = True
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1 # Just a removal, not a new best? 
                        # Actually, if we remove and don't beat best, we are still improving locally.
                        # Let's reset no_improve only when we beat best.
                        
        # 2. Perturbation
        # If stuck (no improvements in local search), perturb.
        # Even if we improved, we might want to perturb to escape local optima.
        # Let's perturb with some probability, or if no_improve_count > threshold.
        
        # Perturbation: Remove a random codeword (making it invalid), then repair.
        if len(current_set) > 0:
            # How many to remove? 1 or 2?
            # Let's try removing 1.
            c_to_remove = random.choice(list(current_set))
            
            # Remove it
            for y in balls[c_to_remove]:
                cover_count[y] -= 1
            current_set.remove(c_to_remove)
            
            # Repair
            if not repair():
                # Failed to repair? Restore state?
                # If repair failed due to timeout, we should abort or restore.
                # For safety, if repair fails, we stop.
                break
            
            # After repair, we are valid again.
            # Check if we beat best
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # If we are stuck for a long time, maybe do a bigger perturbation?
            # Or just keep going.
            
    # Convert best_set to digits
    codewords = []
    for idx in best_set:
        codewords.append(_int_to_digits(idx, q, n, powers))
        
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills after 150s. Use 145s deadline for safety.
    deadline = start_time + 145
    
    try:
        # Use ILS
        result = _ils_search(q, n, radius, deadline)
    except Exception as e:
        # Fallback to greedy if ILS fails
        try:
            selected = _greedy_cover_fast(q, n, radius, deadline)
            powers = [1] * n
            for i in range(1, n):
                powers[i] = powers[i-1] * q
            codewords = []
            for idx in selected:
                codewords.append(_int_to_digits(idx, q, n, powers))
            result = codewords
        except Exception:
            # Ultimate fallback: raise
            raise e

    return result

# EVOLVE-BLOCK-END
