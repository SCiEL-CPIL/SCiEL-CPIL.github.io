import math
import random
import time

def score(U):
    if not U:
        return 0
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Known good Golomb rulers (n=3 to n=10)
    seeds = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 10, 12, 17, 20, 24, 28],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 31]
    ]
    
    best_set = seeds[0]
    best_score = score(best_set)
    
    # Find the best seed
    for s in seeds:
        sc = score(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)
            
    current_set = list(best_set)
    current_score = best_score
    
    # Hill climbing
    while time.time() < deadline:
        improved = False
        n = len(current_set)
        max_val = max(current_set) + 5
        
        if n <= 15:
            # Exhaustive local search
            # 1. Try shifting each element
            for i in range(n):
                if current_set[i] == 0:
                    continue
                old_val = current_set[i]
                for new_val in range(1, max_val + 1):
                    if new_val == old_val or new_val in current_set:
                        continue
                    neighbor = list(current_set)
                    neighbor[i] = new_val
                    neighbor.sort()
                    new_score = score(neighbor)
                    if new_score > current_score:
                        current_set = neighbor
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # 2. Try adding a new element
            if not improved:
                for new_val in range(1, max_val + 1):
                    if new_val in current_set:
                        continue
                    neighbor = list(current_set)
                    neighbor.append(new_val)
                    neighbor.sort()
                    new_score = score(neighbor)
                    if new_score > current_score:
                        current_set = neighbor
                        current_score = new_score
                        improved = True
                        break
            
            # 3. Try removing an element
            if not improved:
                for i in range(n):
                    if current_set[i] == 0:
                        continue
                    neighbor = list(current_set)
                    neighbor.pop(i)
                    new_score = score(neighbor)
                    if new_score > current_score:
                        current_set = neighbor
                        current_score = new_score
                        improved = True
                        break
        else:
            # Random moves for larger sets
            for _ in range(200):
                if time.time() > deadline:
                    break
                neighbor = list(current_set)
                move = random.choice(['add', 'remove', 'shift'])
                
                if move == 'add':
                    candidates = [i for i in range(1, max_val + 1) if i not in neighbor]
                    if candidates:
                        neighbor.append(random.choice(candidates))
                        neighbor.sort()
                elif move == 'remove':
                    if len(neighbor) > 1:
                        rem_idx = random.choice([i for i, x in enumerate(neighbor) if x != 0])
                        neighbor.pop(rem_idx)
                else:
                    if len(neighbor) > 1:
                        idx = random.choice([i for i, x in enumerate(neighbor) if x != 0])
                        new_val = random.randint(1, max_val)
                        if new_val not in neighbor:
                            neighbor[idx] = new_val
                            neighbor.sort()
                
                new_score = score(neighbor)
                if new_score > current_score:
                    current_set = neighbor
                    current_score = new_score
                    improved = True
                    break
        
        if current_score > best_score:
            best_set = current_set
            best_score = current_score

    return best_set
