import time
import numpy as np
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit

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

    # Precompute balls as lists of integers
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
        balls.append(ball)

    # Build adjacency: adj[x] = list of y such that y in balls[x] (i.e., dist(x,y) <= 1)
    # This is symmetric
    adj = balls  # reuse balls as adjacency list

    # Precompute for local search: for each point y, which codewords cover it
    # point_to_cw[y] = list of x such that y in balls[x]
    # Due to symmetry, point_to_cw[y] = balls[y]
    # So we can just use balls directly

    # Step 1: Solve LP relaxation using scipy.optimize.linprog
    # Minimize sum(x_i) subject to: for each point y, sum_{x: y in balls[x]} x_i >= 1
    # x_i in [0, 1]
    
    # Build constraint matrix A where A[y, x] = 1 if y in balls[x]
    # A has shape (total, total)
    A = np.zeros((total, total), dtype=np.int8)
    for x in range(total):
        for y in balls[x]:
            A[y, x] = 1

    b = np.ones(total, dtype=np.float64)
    c = np.ones(total, dtype=np.float64)

    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1)] * total

    try:
        from scipy.optimize import linprog
        lp_result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        lp_time = time.time() - start_time
    except Exception as e:
        lp_result = None
        lp_time = time.time() - start_time

    # Step 2: Get initial solution
    if lp_result is not None and lp_result.success:
        lp_solution = lp_result.x
        # Round: select codewords with x_i > 0.5 first, then fill gaps greedily
        sorted_indices = np.argsort(-lp_solution)
        selected = []
        covered = np.zeros(total, dtype=bool)
        
        for idx in sorted_indices:
            if time.time() > deadline - 20:
                break
            idx = int(idx)
            # Check if this codeword covers any uncovered point
            for y in balls[idx]:
                if not covered[y]:
                    selected.append(idx)
                    for y2 in balls[idx]:
                        covered[y2] = True
                    break
            if np.all(covered):
                break
        
        # Greedy fill for any remaining uncovered
        if not np.all(covered):
            counts = np.array([sum(1 for y in balls[i] if not covered[y]) for i in range(total)], dtype=np.int32)
            while not np.all(covered) and time.time() < deadline - 15:
                max_idx = int(np.argmax(counts))
                if counts[max_idx] == 0:
                    break
                selected.append(max_idx)
                for y in balls[max_idx]:
                    if not covered[y]:
                        covered[y] = True
                for i in range(total):
                    if i not in selected:
                        counts[i] = sum(1 for y in balls[i] if not covered[y])
    else:
        # Fallback to greedy if LP fails
        selected = []
        covered = np.zeros(total, dtype=bool)
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        
        while True:
            if time.time() > deadline - 20:
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

    best_size = len(selected)
    best_solution = selected[:]

    # Step 3: Local search to improve - remove redundant codewords
    current_set = set(best_solution)
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in current_set:
        for y in balls[cw]:
            cover_count[y] += 1

    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        candidates = list(current_set)
        random.shuffle(candidates)
        for cw in candidates:
            if cw not in current_set:
                continue
            # Check if redundant
            redundant = True
            for y in balls[cw]:
                if cover_count[y] == 1:
                    redundant = False
                    break
            if redundant:
                for y in balls[cw]:
                    cover_count[y] -= 1
                current_set.remove(cw)
                improved = True
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = list(current_set)

    # Step 4: Try some random restarts with perturbed greedy
    for restart in range(3):
        if time.time() > deadline - 10:
            break
        
        # Perturbed greedy
        counts = np.array([len(b) for b in balls], dtype=np.float64)
        noise = np.random.uniform(0, 0.5, size=total)
        counts_pert = counts + noise
        
        covered = np.zeros(total, dtype=bool)
        selected2 = []
        
        while True:
            if time.time() > deadline - 10:
                break
            max_idx = int(np.argmax(counts_pert))
            if counts[max_idx] == 0:
                break
            selected2.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(total):
                        if y in balls[i]:
                            counts[i] -= 1
                            counts_pert[i] -= 1
            if np.all(covered):
                break
        
        # Local search on this solution
        current_set2 = set(selected2)
        cover_count2 = np.zeros(total, dtype=np.int32)
        for cw in current_set2:
            for y in balls[cw]:
                cover_count2[y] += 1
        
        improved2 = True
        while improved2 and time.time() < deadline - 5:
            improved2 = False
            candidates2 = list(current_set2)
            random.shuffle(candidates2)
            for cw in candidates2:
                if cw not in current_set2:
                    continue
                redundant2 = True
                for y in balls[cw]:
                    if cover_count2[y] == 1:
                        redundant2 = False
                        break
                if redundant2:
                    for y in balls[cw]:
                        cover_count2[y] -= 1
                    current_set2.remove(cw)
                    improved2 = True
                    if len(current_set2) < best_size:
                        best_size = len(current_set2)
                        best_solution = list(current_set2)

    # Step 5: Try swap-based local search on the best solution
    if time.time() < deadline - 5:
        current_set = set(best_solution)
        cover_count = np.zeros(total, dtype=np.int32)
        for cw in current_set:
            for y in balls[cw]:
                cover_count[y] += 1

        # Try to find a codeword that can be swapped with two others to reduce size
        # This is more complex; skip for now

    # Convert best_solution to list of digit lists
    if best_solution is None:
        # Fallback
        best_solution = []

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
