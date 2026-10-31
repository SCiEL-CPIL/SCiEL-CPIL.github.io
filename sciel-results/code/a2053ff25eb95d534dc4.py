import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    best_score = -1
    best_set = [0, 1, 3]

    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
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
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Exhaustive search for small n and M
    # We suspect optimal n is around 4-7.
    # Exhaustive for n up to 6, M up to 25
    max_m_exhaustive = 25
    max_n_exhaustive = 6
    
    for n in range(2, max_n_exhaustive + 1):
        range_m = range(1, max_m_exhaustive + 1)
        for combo in combinations(range_m, n - 1):
            cand = [0] + list(combo)
            if time.time() > deadline - 20:
                break
            try_set(cand)
        if time.time() > deadline - 20:
            break

    # Phase 2: Add known good structures (Golomb rulers)
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 26, 29, 34, 41]
    ]
    for gr in golomb_rulers:
        if time.time() > deadline - 10:
            break
        try_set(gr)

    # Phase 3: Local search
    if time.time() < deadline - 5:
        current = list(best_set)
        no_improve_count = 0
        max_iterations = 10000
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # 1. Try adding an element
            if len(current) < 12:
                max_cur = max(current)
                # Search range: up to max_cur + 10 or 50
                search_limit = max(max_cur + 10, 50)
                for new_el in range(1, search_limit + 1):
                    if new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve_count = 0
                        break
                    elif s > best_score - 0.001:
                        # Accept if close, to explore? No, strict improvement for now.
                        pass
            
            # 2. Try replacing an element
            if len(current) >= 2:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    max_cur = max(current)
                    search_limit = max(max_cur + 10, 50)
                    for new_el in range(1, search_limit + 1):
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) < 2:
                            continue
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            no_improve_count = 0
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
                        no_improve_count = 0
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    # Random restart or perturbation
                    if random.random() < 0.2:
                        # Jump to a random Golomb-like set
                        n = random.randint(3, 8)
                        max_m = random.randint(10, 40)
                        # Generate a random set with distinct differences (approx)
                        # Just random sample
                        inner = random.sample(range(1, max_m + 1), min(n-1, max_m))
                        cand = sorted([0] + inner)
                        if len(cand) >= 2:
                            s = compute_score(cand)
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                                current = list(cand)
                                improved = True
                                no_improve_count = 0
                    else:
                        # Small perturbation
                        if len(current) > 2:
                            idx = random.randint(1, len(current)-1)
                            delta = random.randint(-3, 3)
                            new_val = current[idx] + delta
                            if new_val > 0 and new_val not in current:
                                cand = list(current)
                                cand[idx] = new_val
                                cand = sorted(set(cand))
                                if 0 in cand and len(cand) >= 2:
                                    s = compute_score(cand)
                                    if s > best_score:
                                        best_score = s
                                        best_set = list(cand)
                                        current = list(cand)
                                        improved = True
                                        no_improve_count = 0

    return list(best_set)
