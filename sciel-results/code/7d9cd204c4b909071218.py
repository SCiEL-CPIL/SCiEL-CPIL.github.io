import time
import numpy as np
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    # Only optimized for the specific task: q=3, n=8, r=1
    # For other parameters, fall back to a simple valid construction
    if q != 3 or n != 8 or radius != 1:
        # Simple valid code: greedy for generality
        total = q ** n
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q

        # Precompute balls
        balls = []
        for x in range(total):
            ball = [x]
            digits = []
            tmp = x
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            for i in range(n):
                old_d = digits[i]
                mult = powers[i]
                for new_d in range(q):
                    if new_d != old_d:
                        y = x + (new_d - old_d) * mult
                        ball.append(y)
            balls.append(ball)

        covered = np.zeros(total, dtype=bool)
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        selected = []

        while True:
            if time.time() > deadline - 5:
                break
            max_idx = int(np.argmax(counts))
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(total):
                        if y in balls[i]:
                            counts[i] -= 1
            if np.all(covered):
                break

        result = []
        for idx in selected:
            digits = []
            tmp = idx
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            result.append(digits)
        return result

    # Specific case: q=3, n=8, r=1
    total = 3 ** 8  # 6561
    ball_size = 1 + 8 * 2  # 17

    # Precompute balls as numpy arrays for efficiency
    balls = []
    for x in range(total):
        ball = [x]
        digits = []
        tmp = x
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        for i in range(8):
            old_d = digits[i]
            mult = 3 ** i
            for new_d in range(3):
                if new_d != old_d:
                    y = x + (new_d - old_d) * mult
                    ball.append(y)
        balls.append(np.array(ball, dtype=np.int32))

    # Convert to a flat representation for faster coverage counting
    # ball_flat: shape (total, ball_size)
    ball_flat = np.array(balls, dtype=np.int32)

    # Step 1: Greedy set cover to get initial solution
    covered = np.zeros(total, dtype=bool)
    counts = np.ones(total, dtype=np.int32) * ball_size
    selected = []
    
    while True:
        if time.time() > deadline - 40:
            break
        max_idx = int(np.argmax(counts))
        if counts[max_idx] == 0:
            break
        selected.append(max_idx)
        for y in balls[max_idx]:
            if not covered[y]:
                covered[y] = True
                # Update counts for all codewords that cover y
                # We need to decrement counts for all x such that y is in balls[x]
                # This is expensive to do naively. Use the ball structure.
                # For each y in balls[max_idx], we need to find all x where y in balls[x]
                # But that's O(total * ball_size) per selected codeword, which is too slow.
                # Alternative: precompute the inverse mapping.
        if np.all(covered):
            break
    
    # The naive greedy above is too slow for updating counts. Let's use a different approach.
    # Precompute: for each point y, the list of codewords x that cover it (i.e., y in balls[x])
    # Since dist(x, y) <= 1 iff dist(y, x) <= 1, this is just balls[y]
    # So point_to_cw[y] = balls[y]
    
    # Better greedy:
    covered = np.zeros(total, dtype=bool)
    uncovered_count = np.zeros(total, dtype=np.int32)
    for i in range(total):
        uncovered_count[i] = ball_size  # initially all balls cover ball_size uncovered points
    
    selected = []
    while True:
        if time.time() > deadline - 30:
            break
        max_idx = int(np.argmax(uncovered_count))
        if uncovered_count[max_idx] == 0:
            break
        selected.append(max_idx)
        for y in balls[max_idx]:
            if not covered[y]:
                covered[y] = True
                # Decrement uncovered_count for all codewords that cover y
                for x in balls[y]:
                    uncovered_count[x] -= 1
        if np.all(covered):
            break
    
    best_size = len(selected)
    best_solution = selected[:]

    # Step 2: Local search - remove redundant codewords
    def try_remove_redundant(selected_list, max_time=None):
        if max_time is None:
            max_time = deadline - 10
        current_set = set(selected_list)
        cover_count = np.zeros(total, dtype=np.int32)
        for cw in current_set:
            cover_count[balls[cw]] += 1
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            candidates = list(current_set)
            random.shuffle(candidates)
            for cw in candidates:
                if cw not in current_set:
                    continue
                # Check if redundant: all points covered by cw are also covered by others
                redundant = True
                for y in balls[cw]:
                    if cover_count[y] == 1:
                        redundant = False
                        break
                if redundant:
                    cover_count[balls[cw]] -= 1
                    current_set.remove(cw)
                    improved = True
                    if len(current_set) < best_size:
                        nonlocal best_size, best_solution
                        best_size = len(current_set)
                        best_solution = list(current_set)
        return list(current_set)
    
    # Python 2/3 compatibility for nonlocal
    # Let's restructure to avoid nonlocal
    best_size = len(selected)
    best_solution = selected[:]
    
    current_set = set(selected)
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in current_set:
        cover_count[balls[cw]] += 1
    
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        candidates = list(current_set)
        random.shuffle(candidates)
        for cw in candidates:
            if cw not in current_set:
                continue
            redundant = True
            for y in balls[cw]:
                if cover_count[y] == 1:
                    redundant = False
                    break
            if redundant:
                cover_count[balls[cw]] -= 1
                current_set.remove(cw)
                improved = True
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = list(current_set)
    
    # Step 3: Try to add then remove to escape local optima
    # Pick a random codeword not in current_set, add it, then try to remove redundant
    for iteration in range(20):
        if time.time() > deadline - 10:
            break
        
        # Add a random codeword
        if len(current_set) < total:
            # Pick a random codeword not in current_set
            all_cw = list(range(total))
            current_list = list(current_set)
            candidates_to_add = [x for x in all_cw if x not in current_set]
            if candidates_to_add:
                add_cw = random.choice(candidates_to_add)
                current_set.add(add_cw)
                cover_count[balls[add_cw]] += 1
        
        # Now try to remove redundant
        improved2 = True
        while improved2 and time.time() < deadline - 5:
            improved2 = False
            candidates2 = list(current_set)
            random.shuffle(candidates2)
            for cw in candidates2:
                if cw not in current_set:
                    continue
                redundant2 = True
                for y in balls[cw]:
                    if cover_count[y] == 1:
                        redundant2 = False
                        break
                if redundant2:
                    cover_count[balls[cw]] -= 1
                    current_set.remove(cw)
                    improved2 = True
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_solution = list(current_set)
    
    # Convert best_solution to list of digit lists
    result = []
    for idx in best_solution:
        digits = []
        tmp = idx
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        result.append(digits)

    return result
