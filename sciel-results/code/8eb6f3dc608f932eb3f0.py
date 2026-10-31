import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (150s limit, 5s buffer)

    best_score = -1
    best_set = [0, 1, 3]
    score_cache = {}

    def compute_score(U):
        key = tuple(sorted(set(U)))
        if key in score_cache:
            return score_cache[key]
        
        U = list(key)
        if not U or 0 not in U:
            score_cache[key] = -1
            return -1
        if len(U) < 2:
            score_cache[key] = 0
            return 0
        max_u = max(U)
        if max_u == 0:
            score_cache[key] = 0
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
            score_cache[key] = -1
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            score_cache[key] = 0
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            score_cache[key] = 1.0
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        score_cache[key] = score
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
    # Extended to n=7 and M=25 to find better small sets
    max_m_exhaustive = 25
    max_n_exhaustive = 7
    
    for n in range(2, max_n_exhaustive + 1):
        range_m = range(1, max_m_exhaustive + 1)
        for combo in combinations(range_m, n - 1):
            cand = [0] + list(combo)
            if time.time() > deadline - 25:
                break
            try_set(cand)
        if time.time() > deadline - 25:
            break

    # Phase 2: Improved local search with multiple restarts
    if time.time() < deadline - 8:
        current = list(best_set)
        
        def local_search(start_set, max_iters=2000):
            nonlocal best_score, best_set
            curr = sorted(set(start_set))
            if not curr or 0 not in curr or len(curr) < 2:
                return curr
            
            curr_score = compute_score(curr)
            no_improve_count = 0
            max_no_improve = 50
            
            for it in range(max_iters):
                if time.time() > deadline - 8:
                    break
                
                improved = False
                
                # Try adding an element (expanded neighborhood)
                if len(curr) < 12:
                    max_cur = max(curr)
                    # Expand search range based on current max
                    search_limit = max_cur + max(5, max_cur // 2)
                    for new_el in range(1, search_limit + 1):
                        if new_el in curr:
                            continue
                        cand = sorted(curr + [new_el])
                        s = compute_score(cand)
                        if s > curr_score:
                            curr = cand
                            curr_score = s
                            improved = True
                            no_improve_count = 0
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                            break
                
                # Try replacing an element
                if len(curr) >= 2 and not improved:
                    for i in range(len(curr)):
                        if curr[i] == 0:
                            continue
                        max_cur = max(curr)
                        search_limit = max_cur + max(5, max_cur // 2)
                        for new_el in range(1, search_limit + 1):
                            if new_el in curr:
                                continue
                            cand = list(curr)
                            cand[i] = new_el
                            cand = sorted(set(cand))
                            if 0 not in cand or len(cand) < 2:
                                continue
                            s = compute_score(cand)
                            if s > curr_score:
                                curr = cand
                                curr_score = s
                                improved = True
                                no_improve_count = 0
                                if s > best_score:
                                    best_score = s
                                    best_set = list(cand)
                                break
                        if improved:
                            break
                
                # Try removing an element
                if len(curr) > 2 and not improved:
                    for i in range(len(curr)):
                        if curr[i] == 0:
                            continue
                        cand = [x for j, x in enumerate(curr) if j != i]
                        s = compute_score(cand)
                        if s > curr_score:
                            curr = cand
                            curr_score = s
                            improved = True
                            no_improve_count = 0
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                            break
                
                if improved:
                    continue
                
                # No improvement - check if we should stop
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
                
                # Random perturbation to escape local minima
                if random.random() < 0.15:
                    # Jump to a random set near current
                    n = random.randint(3, min(8, len(curr) + 2))
                    max_m = max(10, max(curr) + random.randint(0, 5))
                    if n - 1 > 0:
                        inner = random.sample(range(1, max_m + 1), min(n - 1, max_m))
                        cand = sorted([0] + inner)
                    else:
                        cand = [0]
                    if len(cand) >= 2:
                        s = compute_score(cand)
                        if s > curr_score:
                            curr = cand
                            curr_score = s
                            no_improve_count = 0
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
            
            return curr
        
        # Multiple restarts from different starting points
        restarts = 0
        while time.time() < deadline - 8 and restarts < 10:
            if restarts == 0:
                start = list(best_set)
            else:
                # Generate diverse starting points
                n = random.randint(3, 7)
                max_m = random.randint(8, 20)
                inner = random.sample(range(1, max_m + 1), min(n - 1, max_m))
                start = sorted([0] + inner)
            
            local_search(start)
            restarts += 1
        
        # Final polish: try all single-element changes from best_set
        if time.time() < deadline - 5:
            for _ in range(3):  # Multiple passes
                if time.time() > deadline - 5:
                    break
                changed = False
                max_cur = max(best_set)
                search_limit = max_cur + 10
                for new_el in range(1, search_limit + 1):
                    if new_el in best_set:
                        continue
                    cand = sorted(best_set + [new_el])
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        changed = True
                        break
                
                for i in range(len(best_set)):
                    if best_set[i] == 0:
                        continue
                    for new_el in range(1, search_limit + 1):
                        if new_el in best_set:
                            continue
                        cand = list(best_set)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) < 2:
                            continue
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            changed = True
                            break
                    if changed:
                        break
                
                for i in range(len(best_set)):
                    if best_set[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(best_set) if j != i]
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        changed = True
                        break
                
                if not changed:
                    break

    return list(best_set)
