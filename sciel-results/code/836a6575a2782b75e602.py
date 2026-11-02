import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    def calc_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        num = len(diffs) / len(sums)
        if num <= 1:
            return 0
        den = math.log(2 * max_u + 1)
        if den == 0:
            return 0
        return 1 + math.log(num) / den

    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 2, 3, 7],
        [0, 1, 2, 5],
        [0, 1, 2, 3, 4, 9],
        [0, 1, 2, 3, 4, 5, 11],
        [0, 1, 2, 3, 4, 5, 6, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 17],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19]
    ]
    
    best_set = None
    best_score = -1
    
    for c in candidates:
        s = calc_score(c)
        if s > best_score:
            best_score = s
            best_set = list(c)
            
    # Local Search
    max_u_limit = 30
    max_size = 12
    
    current_set = list(best_set)
    current_score = best_score
    
    # Simple hill climbing with random restarts
    iterations = 0
    max_iterations = 5000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        # Generate a neighbor
        neighbor = list(current_set)
        action = random.choice(['add', 'remove', 'replace'])
        
        if action == 'add' and len(neighbor) < max_size:
            x = random.randint(1, max_u_limit)
            if x not in neighbor:
                neighbor.append(x)
                neighbor.sort()
        elif action == 'remove' and len(neighbor) > 1:
            # Don't remove 0
            if 0 in neighbor:
                idx = random.choice([i for i, v in enumerate(neighbor) if v != 0])
            else:
                idx = random.randint(0, len(neighbor)-1)
            neighbor.pop(idx)
        elif action == 'replace' and len(neighbor) > 1:
            if 0 in neighbor:
                idx = random.choice([i for i, v in enumerate(neighbor) if v != 0])
            else:
                idx = random.randint(0, len(neighbor)-1)
            x = random.randint(1, max_u_limit)
            if x not in neighbor:
                neighbor[idx] = x
                neighbor.sort()
        
        if neighbor == current_set:
            continue
            
        new_score = calc_score(neighbor)
        
        if new_score > current_score:
            current_score = new_score
            current_set = neighbor
            if new_score > best_score:
                best_score = new_score
                best_set = list(current_set)
        else:
            # Simulated annealing style: accept worse with small prob?
            # For now, strict hill climbing to ensure stability
            pass
            
    return best_set
