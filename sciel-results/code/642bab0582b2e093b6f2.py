import time
import math
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        # Use numpy for faster computation when set is large enough
        n = len(U)
        if n <= 30:
            # Small set: direct computation
            sumset = set()
            diffset = set()
            U_list = list(U)
            for a in U_list:
                for b in U_list:
                    sumset.add(a + b)
                    diffset.add(a - b)
            n_sum = len(sumset)
            n_diff = len(diffset)
        else:
            # Larger set: use numpy for speed
            U_arr = np.array(U, dtype=np.int64)
            # Sumset: all pairwise sums
            sums = U_arr[:, None] + U_arr[None, :]
            sumset = set(np.unique(sums).tolist())
            # Diffset: all pairwise differences
            diffs = U_arr[:, None] - U_arr[None, :]
            diffset = set(np.unique(diffs).tolist())
            n_sum = len(sumset)
            n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
        return s

    # Phase 1: Exhaustive search for small M (up to 12 for speed)
    for M in range(1, 13):
        if time.time() > deadline - 30:
            break
        # For M=12, that's 2^12 - 1 = 4095 candidates, manageable
        for mask in range(1, 1 << M):
            if time.time() > deadline - 30:
                break
            cand = [0]
            for i in range(M):
                if mask & (1 << i):
                    cand.append(i + 1)
            try_set(cand)
    
    # Phase 2: Heuristic Search with smarter moves
    current = list(best_set)
    
    no_improve = 0
    max_no_improve = 800
    temperature = 10  # for simulated annealing-like acceptance
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        cur_score = best_score
        
        # 1. Try adding an element - focus on sparse additions
        # Key insight: adding large gaps can increase |U-U| faster than |U+U|
        candidates_to_add = set()
        
        # Try elements that create new difference patterns
        # Add elements at strategic positions
        for i in range(1, max_cur + 30):
            if i not in current:
                candidates_to_add.add(i)
        
        # Try larger jumps - these can dramatically increase the diff/sum ratio
        # because they create new large differences without proportionally increasing sums
        for _ in range(40):
            # Exponential jumps
            jump = random.randint(max_cur + 1, max_cur + max(50, max_cur // 2))
            candidates_to_add.add(jump)
        
        # Try very large jumps
        for _ in range(20):
            jump = random.randint(max_cur + 1, max_cur * 2 + 100)
            candidates_to_add.add(jump)
            
        for new_el in sorted(candidates_to_add):
            if new_el in current:
                continue
            cand = sorted(current + [new_el])
            s = compute_score(cand)
            if s > best_score:
                best_score = s
                best_set = list(cand)
                current = list(cand)
                improved = True
                no_improve = 0
                temperature = max(1, temperature * 0.9)
                break
        
        # 2. Try replacing an element with a larger one (scaling up)
        if len(current) >= 2:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                old_el = current[i]
                # Try replacing with larger values to increase max
                for new_el in [old_el + random.randint(1, 20), old_el * 2, old_el + max_cur // 2]:
                    if new_el < 1 or new_el in current or new_el == old_el:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        temperature = max(1, temperature * 0.9)
                        break
                if improved:
                    break
        
        # 3. Try removing an element
        if len(current) > 2:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                cand = [x for j, x in enumerate(current) if j != i]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        # 4. Try swapping two elements
        if len(current) >= 3:
            for i in range(1, len(current)):
                for j in range(i + 1, len(current)):
                    cand = list(current)
                    # Try moving element j to a different position
                    old_j = cand[j]
                    new_j = old_j + random.randint(-10, 20)
                    if new_j < 1 or new_j in cand or new_j == old_j:
                        continue
                    cand[j] = new_j
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        if not improved:
            no_improve += 1
            temperature = max(1, temperature * 0.95)
            
            # Random perturbation with acceptance criteria
            if random.random() < 0.5:
                # Add a random element
                new_el = random.randint(1, max_cur + 100)
                if new_el not in current:
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score or (s > cur_score - 0.01 and random.random() < 0.3):
                        best_score = max(best_score, s)
                        if s > best_score:
                            best_score = s
                        best_set = list(cand)
                        current = list(cand)
            
            # Or try a random restructuring
            elif random.random() < 0.3:
                # Generate a random set with similar size
                n = len(current)
                if n > 1:
                    new_set = [0]
                    max_el = max(current)
                    for _ in range(n - 1):
                        new_set.append(random.randint(1, max_el + 50))
                    new_set = sorted(set(new_set))
                    if 0 in new_set and len(new_set) >= 2:
                        s = compute_score(new_set)
                        if s > best_score:
                            best_score = s
                            best_set = list(new_set)
                            current = list(new_set)
                            no_improve = 0
                            improved = True
    
    return list(best_set)
