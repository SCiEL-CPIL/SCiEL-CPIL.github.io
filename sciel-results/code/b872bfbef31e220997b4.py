import numpy as np
import random
import time
import sys
from itertools import product

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return _generic_greedy(q, n, radius)

    N = 3 ** n  # 2187
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute ball membership: balls_set[i] = set of j within Hamming distance 1 of i
    balls_set = [set() for _ in range(N)]
    for i in range(N):
        for j in range(N):
            # Check Hamming distance
            # Convert to base 3 strings and compare
            # Or compute difference directly
            # For speed, let's precompute once.
            pass

    # More efficient ball precomputation
    def to_base3(x, n):
        res = [0] * n
        for i in range(n):
            res[i] = x % 3
            x //= 3
        return res

    def from_base3(digits, q=3):
        res = 0
        for d in digits:
            res = res * q + d
        return res

    balls_set = [set() for _ in range(N)]
    for i in range(N):
        digits_i = to_base3(i, n)
        for j in range(N):
            digits_j = to_base3(j, n)
            dist = sum(1 for a, b in zip(digits_i, digits_j) if a != b)
            if dist <= 1:
                balls_set[i].add(j)

    # Greedy covering algorithm with Lagrangian-inspired weighting
    # We maintain weights for each uncovered point to guide selection
    def greedy_lagrangian(seed, time_limit):
        rng = random.Random(seed)
        covered = [False] * N
        uncovered_count = N
        code = []
        
        # Weights for Lagrangian relaxation: w[j] represents the "urgency" of covering j
        # Initially all 1.0
        weights = [1.0] * N
        # Benefit of choosing codeword i: sum of weights of newly covered points
        benefits = [0.0] * N
        for i in range(N):
            for j in balls_set[i]:
                benefits[i] += weights[j]

        while uncovered_count > 0 and time.time() < time_limit:
            # Find best codeword
            max_benefit = -1
            best_idx = -1
            for i in range(N):
                if benefits[i] > max_benefit:
                    max_benefit = benefits[i]
                    best_idx = i
            
            if best_idx < 0:
                break

            # Add best_idx to code
            code.append(best_idx)
            
            # Update coverage and benefits
            newly_covered = []
            for j in balls_set[best_idx]:
                if not covered[j]:
                    covered[j] = True
                    newly_covered.append(j)
                    uncovered_count -= 1
            
            if not newly_covered:
                continue

            # Update benefits for all codewords that cover any newly covered point
            for i in range(N):
                if i == best_idx:
                    continue
                delta = 0.0
                for j in newly_covered:
                    if j in balls_set[i]:
                        delta += weights[j]
                if delta > 0:
                    benefits[i] -= delta
            
            # Lagrangian update: increase weights of uncovered points
            # to emphasize them in future selections
            for j in range(N):
                if not covered[j]:
                    weights[j] *= 1.1  # gentle increase
            
            # Renormalize weights to prevent overflow
            max_w = max(weights[j] for j in range(N) if not covered[j]) if uncovered_count > 0 else 1.0
            if max_w > 1000:
                for j in range(N):
                    if not covered[j]:
                        weights[j] /= max_w
                        benefits[j] = 0.0  # recompute below
        
        # If we exited due to time limit, we might have uncovered points
        # Use simple greedy to finish
        if uncovered_count > 0 and time.time() < time_limit:
            # Reset to simple greedy for remaining
            for i in range(N):
                # Recompute benefit as count of uncovered points covered
                count = 0
                for j in balls_set[i]:
                    if not covered[j]:
                        count += 1
                benefits[i] = count
            
            while uncovered_count > 0 and time.time() < time_limit:
                max_benefit = -1
                best_idx = -1
                for i in range(N):
                    if benefits[i] > max_benefit:
                        max_benefit = benefits[i]
                        best_idx = i
                
                if best_idx < 0:
                    break
                
                code.append(best_idx)
                newly_covered = []
                for j in balls_set[best_idx]:
                    if not covered[j]:
                        covered[j] = True
                        newly_covered.append(j)
                        uncovered_count -= 1
                
                if not newly_covered:
                    continue
                
                for i in range(N):
                    if i == best_idx:
                        continue
                    delta = 0
                    for j in newly_covered:
                        if j in balls_set[i]:
                            delta += 1
                    if delta > 0:
                        benefits[i] -= delta
        
        return code

    # Local search: try to remove or swap codewords to reduce size
    def local_search(code, time_limit):
        code = list(code)
        if not code:
            return code
        
        # Precompute cover counts
        cover_count = [0] * N
        for c in code:
            for j in balls_set[c]:
                cover_count[j] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to remove each codeword
            for i in range(len(code)):
                c = code[i]
                # Check if c is the only cover for any point
                safe = True
                for j in balls_set[c]:
                    if cover_count[j] == 1:
                        safe = False
                        break
                if safe:
                    # Remove c
                    for j in balls_set[c]:
                        cover_count[j] -= 1
                    code.pop(i)
                    improved = True
                    break
            else:
                # Try swap: remove c1, add c2
                for i in range(len(code)):
                    if time.time() > time_limit:
                        break
                    c1 = code[i]
                    # Points only covered by c1
                    only_covered_by_c1 = []
                    for j in balls_set[c1]:
                        if cover_count[j] == 1:
                            only_covered_by_c1.append(j)
                    
                    if not only_covered_by_c1:
                        continue
                    
                    # Find a codeword c2 (not in code) that covers all points in only_covered_by_c1
                    for c2 in range(N):
                        if c2 in code:
                            continue
                        # Check if c2 covers all points in only_covered_by_c1
                        covers_all = True
                        for j in only_covered_by_c1:
                            if j not in balls_set[c2]:
                                covers_all = False
                                break
                        if covers_all:
                            # Perform swap
                            for j in balls_set[c1]:
                                cover_count[j] -= 1
                            for j in balls_set[c2]:
                                cover_count[j] += 1
                            code[i] = c2
                            improved = True
                            break
                    if improved:
                        break
        return code

    # Run multiple restarts
    best_code = None
    best_len = float('inf')
    
    num_restarts = 50
    restart_time = 2.0  # seconds per restart
    
    for seed in range(num_restarts):
        if time.time() > deadline - 5:
            break
        restart_deadline = min(time.time() + restart_time, deadline - 5)
        code = greedy_lagrangian(seed, restart_deadline)
        if len(code) < best_len:
            # Try local search to improve
            ls_code = local_search(code, min(time.time() + 3, deadline - 5))
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
    
    # If no code found or all are too large, fall back to simple greedy
    if best_code is None or len(best_code) > 200:
        # Simple greedy fallback
        covered = [False] * N
        uncovered_count = N
        code = []
        while uncovered_count > 0 and time.time() < deadline - 2:
            best_idx = -1
            max_count = -1
            for i in range(N):
                count = 0
                for j in balls_set[i]:
                    if not covered[j]:
                        count += 1
                if count > max_count:
                    max_count = count
                    best_idx = i
            code.append(best_idx)
            for j in balls_set[best_idx]:
                if not covered[j]:
                    covered[j] = True
                    uncovered_count -= 1
        best_code = code
        best_len = len(best_code)

    # Convert integer indices to base-3 digit lists
    result = []
    for c in best_code:
        result.append(to_base3(c, n))
    
    return result

def _generic_greedy(q, n, radius):
    N = q ** n
    def to_base(x, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = x % q
            x //= q
        return res

    balls_set = [set() for _ in range(N)]
    for i in range(N):
        for j in range(N):
            di = to_base(i, n, q)
            dj = to_base(j, n, q)
            dist = sum(1 for a, b in zip(di, dj) if a != b)
            if dist <= radius:
                balls_set[i].add(j)

    covered = [False] * N
    uncovered_count = N
    code = []
    while uncovered_count > 0:
        best_idx = -1
        max_count = -1
        for i in range(N):
            count = 0
            for j in balls_set[i]:
                if not covered[j]:
                    count += 1
            if count > max_count:
                max_count = count
                best_idx = i
        code.append(best_idx)
        for j in balls_set[best_idx]:
            if not covered[j]:
                covered[j] = True
                uncovered_count -= 1

    return [to_base(c, n, q) for c in code]
