import time
import math
import random

def compute_score(U):
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    sum_set = set()
    diff_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Start with a small set
    U = [0, 1, 3]
    U_set = set(U)
    
    # Compute initial sum and diff sets
    sum_set = set()
    diff_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    current_score = compute_score(U)
    
    # Local search: try adding elements
    max_element = 50  # Upper bound on elements to try
    best_U = list(U)
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        # Try adding each candidate
        for x in range(1, max_element + 1):
            if x in U_set:
                continue
            # Compute new sum and diff sets
            new_sum_set = set(sum_set)
            new_diff_set = set(diff_set)
            for u in U_set:
                new_sum_set.add(x + u)
                new_diff_set.add(x - u)
                new_diff_set.add(u - x)
            new_sum_set.add(2 * x)
            
            new_U_set = U_set | {x}
            new_max = max(new_U_set)
            if new_max == 0:
                continue
            ratio = len(new_diff_set) / len(new_sum_set)
            denom = math.log(2 * new_max + 1)
            if denom > 0:
                new_score = 1.0 + math.log(ratio) / denom
            else:
                new_score = 0.0
            
            if new_score > best_score:
                best_score = new_score
                best_U = list(new_U_set)
                U_set = new_U_set
                sum_set = new_sum_set
                diff_set = new_diff_set
                U = list(U_set)
                improved = True
        
        if not improved:
            # Try removing elements
            for x in list(U_set):
                if x == 0:
                    continue  # Must keep 0
                # Recompute from scratch (simpler)
                new_U_set = U_set - {x}
                if not new_U_set:
                    continue
                new_sum_set = set()
                new_diff_set = set()
                for a in new_U_set:
                    for b in new_U_set:
                        new_sum_set.add(a + b)
                        new_diff_set.add(a - b)
                new_max = max(new_U_set)
                if new_max == 0:
                    continue
                ratio = len(new_diff_set) / len(new_sum_set)
                denom = math.log(2 * new_max + 1)
                if denom > 0:
                    new_score = 1.0 + math.log(ratio) / denom
                else:
                    new_score = 0.0
                
                if new_score > best_score:
                    best_score = new_score
                    best_U = list(new_U_set)
                    U_set = new_U_set
                    sum_set = new_sum_set
                    diff_set = new_diff_set
                    U = list(U_set)
                    improved = True
        
        if not improved:
            # Try swapping: remove one, add another
            for rem in list(U_set):
                if rem == 0:
                    continue
                for add in range(1, max_element + 1):
                    if add in U_set:
                        continue
                    new_U_set = (U_set - {rem}) | {add}
                    new_sum_set = set()
                    new_diff_set = set()
                    for a in new_U_set:
                        for b in new_U_set:
                            new_sum_set.add(a + b)
                            new_diff_set.add(a - b)
                    new_max = max(new_U_set)
                    if new_max == 0:
                        continue
                    ratio = len(new_diff_set) / len(new_sum_set)
                    denom = math.log(2 * new_max + 1)
                    if denom > 0:
                        new_score = 1.0 + math.log(ratio) / denom
                    else:
                        new_score = 0.0
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_U = list(new_U_set)
                        U_set = new_U_set
                        sum_set = new_sum_set
                        diff_set = new_diff_set
                        U = list(U_set)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Increase max_element or break
            max_element += 10
            if max_element > 200:
                break
    
    return sorted(best_U)
