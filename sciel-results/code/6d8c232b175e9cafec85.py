import time
import numpy as np
import random
from scipy.optimize import linprog

# EVOLVE-BLOCK-START

def _precompute_balls(q, n, radius, deadline):
    """
    Precompute for each word x the list of words within Hamming distance <= radius.
    Returns a list of lists.
    """
    total = q ** n
    balls = []
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    if radius == 1:
        # Optimization for radius 1
        for x in range(total):
            ball = [x]
            for i in range(n):
                old_digit = (x // powers[i]) % q
                for new_digit in range(q):
                    if new_digit != old_digit:
                        y = x + (new_digit - old_digit) * powers[i]
                        ball.append(y)
            balls.append(ball)
            if time.time() > deadline - 20:
                # If we run out of time, we assume the rest follow the same pattern
                # This is risky, but for q=3, n=8, total=6561, it's fast enough.
                # Let's just break and handle it in caller or ensure it finishes.
                # 6561 * 17 operations is ~100k, which is < 0.1s. So this won't trigger.
                break
        # Ensure we have all balls if we broke early (unlikely)
        if len(balls) < total:
             # Fallback: just compute the rest normally (should be fast)
             while len(balls) < total:
                 x = len(balls)
                 ball = [x]
                 for i in range(n):
                     old_digit = (x // powers[i]) % q
                     for new_digit in range(q):
                         if new_digit != old_digit:
                             y = x + (new_digit - old_digit) * powers[i]
                             ball.append(y)
                 balls.append(ball)
    else:
        raise NotImplementedError("Only radius 1 is optimized")
        
    return balls

def _int_to_digits(x, q, n, powers):
    digits = []
    for i in range(n):
        d = (x // powers[i]) % q
        digits.append(d)
    return digits

def _construct_lp_cover(q, n, radius, deadline):
    """
    Uses LP relaxation on the dual problem to guide codeword selection.
    Dual Problem:
    Maximize sum(y_j)
    Subject to:
      sum_{j in ball(i)} y_j <= 1 for all i (codeword positions)
      y_j >= 0 for all j (word positions)
    
    The optimal y_j values indicate which words are "hard" to cover.
    We then select codewords greedily based on these weights.
    """
    total = q ** n
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    balls = _precompute_balls(q, n, radius, deadline)
    
    # Build the constraint matrix A for the dual.
    # Variables are y_j for j in 0..total-1.
    # Constraints are for each i in 0..total-1: sum_{j in ball(i)} y_j <= 1.
    # Note: ball(i) contains j if dist(i,j) <= radius.
    # Since distance is symmetric, j in ball(i) iff i in ball(j).
    # So constraint i involves variables y_j where j is in ball(i).
    
    # A is (total x total). A[i, j] = 1 if j in ball(i) else 0.
    # We can build this efficiently.
    A = np.zeros((total, total), dtype=np.float64)
    for i in range(total):
        for j in balls[i]:
            A[i, j] = 1.0
            
    b = np.ones(total, dtype=np.float64)
    c = -np.ones(total, dtype=np.float64) # Maximize sum(y) -> Minimize -sum(y)
    
    # Bounds for y_j: 0 <= y_j <= 1 (actually <=1 is implied by constraints if y>=0, but explicit is safer/faster?)
    # Actually, if sum_{j in ball(i)} y_j <= 1 and y_j >= 0, then y_j <= 1 is automatically satisfied for any j that is in some ball(i).
    # Every j is in its own ball, so y_j <= 1.
    bounds = [(0, None)] * total
    
    try:
        # Use highs solver for speed
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if not res.success:
            raise ValueError("LP failed")
        
        y_vals = res.x
    except Exception as e:
        # Fallback if LP fails or is too slow: uniform weights
        y_vals = np.ones(total, dtype=np.float64)

    # Rounding Strategy:
    # We want to select a set of codewords S.
    # Heuristic: Select codewords that cover words with high y_vals.
    # Greedy Set Cover using weights.
    
    # We maintain a "remaining weight" for each word j.
    # Initially rem[j] = y_vals[j].
    # When we select a codeword c, we "satisfy" the words in ball(c).
    # We want to pick c that maximizes the sum of y_vals of currently unsatisfied words in ball(c).
    
    # However, standard greedy set cover minimizes number of sets.
    # Here, the "cost" of a word is its y_val.
    # We want to cover all words.
    
    # Let's use a weighted greedy approach:
    # 1. Initialize covered_mask = False for all words.
    # 2. While not all covered:
    #    a. For each candidate codeword c, calculate gain = sum(y_vals[j] for j in ball(c) if not covered_mask[j]).
    #    b. Select c with max gain.
    #    c. Add c to S.
    #    d. Update covered_mask for all j in ball(c).
    
    # Optimization:
    # Instead of iterating all c (6561) and all j in ball(c) (17) every time,
    # we can update gains incrementally or just do it if it's fast enough.
    # 6561 * 17 * |S| operations. |S| ~ 500.
    # 6561 * 17 * 500 ~ 55 million operations. Might be slow in Python.
    
    # Alternative:
    # Use the y_vals to order the words.
    # Sort words by y_vals descending.
    # For each word j in sorted order:
    #   If not covered:
    #     Pick a codeword c that covers j.
    #     To minimize total count, pick c that covers the most *other* high-value words?
    #     Or just pick c that maximizes coverage of remaining uncovered words?
    
    # Let's stick to standard greedy but optimized with NumPy.
    
    covered = np.zeros(total, dtype=bool)
    selected = []
    
    # Precompute ball indices for faster access?
    # balls is already a list of lists.
    
    # We can speed up the gain calculation by maintaining an array of "active" y_vals.
    # active_y[j] = y_vals[j] if not covered[j] else 0.
    active_y = y_vals.copy()
    
    while True:
        if time.time() > deadline - 10:
            break
            
        if np.all(covered):
            break
            
        # Find best c
        # gain[c] = sum(active_y[j] for j in balls[c])
        # We can compute this by iterating c?
        # Or iterating j?
        # If we iterate j:
        #   For each uncovered j, it contributes to gain[c] for all c in balls[j].
        #   Since balls is symmetric, balls[j] contains c if dist(j,c)<=1.
        #   So for each uncovered j, add y_vals[j] to gain[c] for all c in balls[j].
        
        # This is O(total * ball_size) = 6561 * 17 ~ 100k operations per step.
        # Number of steps ~ |S| ~ 500.
        # Total ~ 50 million operations.
        # In Python, this might take 5-10 seconds. Acceptable.
        
        gains = np.zeros(total, dtype=np.float64)
        
        # Iterate over uncovered words
        # Get indices of uncovered words
        uncovered_indices = np.where(~covered)[0]
        
        for j in uncovered_indices:
            w = y_vals[j]
            if w == 0: continue
            # Add w to gains for all c that cover j
            # c covers j if c is in balls[j]
            for c in balls[j]:
                gains[c] += w
                
        # Select c with max gain
        max_gain = np.max(gains)
        if max_gain == 0:
            # No gain? Should not happen if not all covered and y_vals > 0
            # Pick arbitrary uncovered
            j = uncovered_indices[0]
            c = random.choice(balls[j])
        else:
            # Pick one of the max gain codewords (randomly if tie)
            max_indices = np.where(gains == max_gain)[0]
            c = random.choice(max_indices)
            
        selected.append(c)
        
        # Update covered and active_y
        for j in balls[c]:
            if not covered[j]:
                covered[j] = True
                active_y[j] = 0 # Not strictly needed if we use covered mask, but good for clarity
                
    # Local Search Refinement
    # Try to remove redundant codewords.
    # A codeword c is redundant if all words in ball(c) are covered by other codewords in S.
    
    # Compute coverage counts
    cover_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        for j in balls[c]:
            cover_count[j] += 1
            
    selected_set = set(selected)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        candidates = list(selected_set)
        random.shuffle(candidates)
        for c in candidates:
            if c not in selected_set:
                continue
            # Check if removing c leaves all words in ball(c) with count >= 1
            # i.e., min(cover_count[j] for j in ball(c)) >= 2?
            # If cover_count[j] >= 2 for all j in ball(c), then removing c makes it >= 1.
            # If any cover_count[j] == 1, removing c makes it 0 (invalid).
            redundant = True
            for j in balls[c]:
                if cover_count[j] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for j in balls[c]:
                    cover_count[j] -= 1
                selected_set.remove(c)
                improved = True
                
    # Convert to digits
    codewords = []
    for idx in selected_set:
        codewords.append(_int_to_digits(idx, q, n, powers))
        
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    try:
        result = _construct_lp_cover(q, n, radius, deadline)
    except Exception as e:
        # Fallback to simple greedy if LP fails
        # Implement a quick greedy fallback
        total = q ** n
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
        balls = _precompute_balls(q, n, radius, deadline)
        
        covered = np.zeros(total, dtype=bool)
        selected = []
        while True:
            if time.time() > deadline - 5:
                break
            if np.all(covered):
                break
            gains = np.zeros(total, dtype=np.int32)
            for j in range(total):
                if not covered[j]:
                    for c in balls[j]:
                        gains[c] += 1
            if np.max(gains) == 0:
                break
            c = np.argmax(gains)
            selected.append(c)
            for j in balls[c]:
                covered[j] = True
        
        codewords = []
        for idx in selected:
            codewords.append(_int_to_digits(idx, q, n, powers))
        result = codewords

    return result

# EVOLVE-BLOCK-END
