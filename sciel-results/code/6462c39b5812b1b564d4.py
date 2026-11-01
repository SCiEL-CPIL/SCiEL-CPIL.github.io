import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def get_score(U):
        if not U:
            return 0
        U = list(set(U))
        if 0 not in U:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
        
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 1:
            return 1.0
        return 1 + math.log(ratio) / denom

    # Initial seeds
    seeds = [
        [0, 1, 3],
        [0, 1, 2, 5],
        [0, 1, 2, 6],
        [0, 1, 4, 10],
        [0, 1, 2, 7],
        [0, 1, 2, 8],
        [0, 1, 3, 7],
        [0, 1, 4, 6],
        [0, 2, 5, 11]
    ]
    
    best_score = 0
    best_U = [0, 1, 3]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        U = list(set(seed))
        if 0 not in U:
            U.append(0)
        U.sort()
        
        current_score = get_score(U)
        if current_score > best_score:
            best_score = current_score
            best_U = U[:]
            
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding
            max_u = max(U)
            candidates_add = [max_u + 1, max_u + 2, max_u + 3, max_u + 5, max_u + 10, max_u + 20]
            random.shuffle(candidates_add)
            for x in candidates_add:
                if time.time() > deadline:
                    break
                new_U = U + [x]
                new_score = get_score(new_U)
                if new_score > current_score + 1e-9:
                    U = new_U
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_U = U[:]
                    break
            
            # Try removing
            if len(U) > 1:
                removable = [x for x in U if x != 0]
                random.shuffle(removable)
                for x in removable:
                    if time.time() > deadline:
                        break
                    new_U = [y for y in U if y != x]
                    new_score = get_score(new_U)
                    if new_score > current_score + 1e-9:
                        U = new_U
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_U = U[:]
                        break
            
            # Try swapping
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    random.shuffle(removable)
                    x = removable[0]
                    max_u = max(U)
                    candidates_swap = [max_u + 1, max_u + 2, max_u + 5, max_u + 10]
                    random.shuffle(candidates_swap)
                    for y in candidates_swap:
                        if time.time() > deadline:
                            break
                        new_U = [z for z in U if z != x] + [y]
                        new_score = get_score(new_U)
                        if new_score > current_score + 1e-9:
                            U = new_U
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_U = U[:]
                            break

    return best_U
