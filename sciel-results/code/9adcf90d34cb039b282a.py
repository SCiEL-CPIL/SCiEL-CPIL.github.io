import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    n = 256
    
    # Precompute conflicts: conflict[x][y] = True if x and y cannot both be in the set
    # x and y conflict if there exists z such that {x,y,z} is a 3-AP
    # For a 3-AP {a, b, c} with a < b < c: b = (a+c)/2, so a+c = 2b
    # Given two elements x, y, they conflict if:
    # 1. x + y is even and (x+y)//2 is in [1, n] (they are the two ends, middle exists)
    # 2. 2*x - y is in [1, n] and 2*x - y != x (y is one end, x is middle, other end exists)
    # 3. 2*y - x is in [1, n] and 2*y - x != y (x is one end, y is middle, other end exists)
    # Actually, simpler: x and y are in a 3-AP if there exists z in [1,n] such that
    # the set {x,y,z} forms an AP. This happens iff:
    # - x + y = 2z for some z in [1,n] (x,y are ends)
    # - x + z = 2y for some z in [1,n] (y is middle)
    # - y + z = 2x for some z in [1,n] (x is middle)
    # Which simplifies to:
    # - (x+y) is even and 1 <= (x+y)//2 <= n
    # - (x+z) = 2y => z = 2y - x, need 1 <= 2y-x <= n
    # - (y+z) = 2x => z = 2x - y, need 1 <= 2x-y <= n
    
    # Build conflict graph
    conflict = [[False] * (n+1) for _ in range(n+1)]
    for x in range(1, n+1):
        for y in range(x+1, n+1):
            # Check if x, y are part of any 3-AP
            # Case 1: x and y are ends, middle = (x+y)/2
            if (x + y) % 2 == 0:
                mid = (x + y) // 2
                if 1 <= mid <= n and mid != x and mid != y:
                    conflict[x][y] = True
                    conflict[y][x] = True
                    continue
            # Case 2: y is middle, so x, y, z with z = 2y - x
            z = 2*y - x
            if 1 <= z <= n and z != x and z != y:
                conflict[x][y] = True
                conflict[y][x] = True
                continue
            # Case 3: x is middle, so y, x, z with z = 2x - y
            z = 2*x - y
            if 1 <= z <= n and z != x and z != y:
                conflict[x][y] = True
                conflict[y][x] = True
                continue
    
    # Weights
    weights = [0.0] + [1.0/i for i in range(1, n+1)]
    
    # Greedy initialization: add elements in order of decreasing weight (increasing index)
    S = set()
    for x in range(1, n+1):
        valid = True
        for a in S:
            if conflict[x][a]:
                valid = False
                break
        if valid:
            S.add(x)
    
    current_score = sum(weights[x] for x in S)
    best_score = current_score
    best_S = set(S)
    
    # Local search with 1-swaps and 2-swaps
    all_nums = list(range(1, n+1))
    
    def is_valid_add(x, S):
        for a in S:
            if conflict[x][a]:
                return False
        return True
    
    def can_add_after_removal(y, S, x):
        # Check if y can be added to S \ {x}
        for a in S:
            if a == x:
                continue
            if conflict[y][a]:
                return False
        return True
    
    while time.time() < deadline:
        improved = False
        
        # Try to add any element
        candidates_add = [x for x in all_nums if x not in S]
        random.shuffle(candidates_add)
        
        for x in candidates_add:
            if is_valid_add(x, S):
                S.add(x)
                current_score += weights[x]
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
                improved = True
        
        # Try 1-swaps: remove x from S, add y not in S
        S_list = list(S)
        S_list.sort(key=lambda x: weights[x])  # Lowest weight first
        
        for x in S_list:
            if time.time() > deadline:
                break
            
            S.remove(x)
            score_without_x = current_score - weights[x]
            
            candidates_add = [y for y in all_nums if y not in S and y != x]
            candidates_add.sort(key=lambda y: weights[y], reverse=True)
            
            found_swap = False
            for y in candidates_add:
                if can_add_after_removal(y, S, x):
                    new_score = score_without_x + weights[y]
                    if new_score > current_score:
                        S.add(y)
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
                        improved = True
                        found_swap = True
                        break
            if not found_swap:
                S.add(x)
        
        # Try 2-swaps: remove two elements x1, x2 from S, add two elements y1, y2 not in S
        if time.time() < deadline:
            S_list = list(S)
            if len(S_list) >= 2:
                # Try removing pairs of low-weight elements
                S_list.sort(key=lambda x: weights[x])
                for i in range(min(len(S_list), 20)):
                    if time.time() > deadline:
                        break
                    x1 = S_list[i]
                    for j in range(i+1, min(len(S_list), i+15)):
                        x2 = S_list[j]
                        
                        S.remove(x1)
                        S.remove(x2)
                        score_without = current_score - weights[x1] - weights[x2]
                        
                        # Try to add two high-weight elements
                        candidates_add = [y for y in all_nums if y not in S and y != x1 and y != x2]
                        candidates_add.sort(key=lambda y: weights[y], reverse=True)
                        
                        # Try adding pairs from top candidates
                        found_2swap = False
                        for a_idx in range(min(len(candidates_add), 30)):
                            y1 = candidates_add[a_idx]
                            if not is_valid_add(y1, S):
                                continue
                            S.add(y1)
                            for b_idx in range(a_idx+1, min(len(candidates_add), a_idx+15)):
                                y2 = candidates_add[b_idx]
                                if is_valid_add(y2, S):
                                    new_score = score_without + weights[y1] + weights[y2]
                                    if new_score > current_score:
                                        S.add(y2)
                                        current_score = new_score
                                        if current_score > best_score:
                                            best_score = current_score
                                            best_S = set(S)
                                        improved = True
                                        found_2swap = True
                                        break
                            S.remove(y1)
                            if found_2swap:
                                break
                        
                        if not found_2swap:
                            S.add(x1)
                            S.add(x2)
        
        if not improved:
            # Random perturbation to escape local minima
            S_list = list(S)
            if len(S_list) >= 2:
                x1, x2 = random.sample(S_list, 2)
                S.remove(x1)
                S.remove(x2)
                candidates_add = [y for y in all_nums if y not in S and y != x1 and y != x2]
                if len(candidates_add) >= 2:
                    y1, y2 = random.sample(candidates_add, 2)
                    if is_valid_add(y1, S) and is_valid_add(y2, S):
                        S.add(y1)
                        S.add(y2)
                        current_score = sum(weights[x] for x in S)
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
                    else:
                        S.add(x1)
                        S.add(x2)
                else:
                    S.add(x1)
                    S.add(x2)
            elif len(S_list) == 1:
                x1 = S_list[0]
                S.remove(x1)
                candidates_add = [y for y in all_nums if y not in S and y != x1]
                if candidates_add:
                    y1 = random.choice(candidates_add)
                    if is_valid_add(y1, S):
                        S.add(y1)
                        current_score = sum(weights[x] for x in S)
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
                    else:
                        S.add(x1)
                else:
                    S.add(x1)
    
    return list(best_S)
