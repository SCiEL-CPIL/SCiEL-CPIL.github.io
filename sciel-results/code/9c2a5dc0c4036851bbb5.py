import random
import time
import math

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    # Build position map
    edge_pos = {}
    for idx, (u, v) in enumerate(P['edge_order'], 1):
        if u > v:
            u, v = v, u
        edge_pos[(u, v)] = idx
    
    def get_pos(u, v):
        if u > v:
            u, v = v, u
        return edge_pos.get((u, v), 1000)

    def calc_cost(perm):
        max_pos = 0
        for i in range(n):
            u = perm[i]
            v = perm[(i + 1) % n]
            p = get_pos(u, v)
            if p > max_pos:
                max_pos = p
        return max_pos

    def local_search(perm):
        improved = True
        while improved:
            improved = False
            current_cost = calc_cost(perm)
            # 2-Opt
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    # Reverse segment i..j
                    new_perm = perm[:]
                    new_perm[i:j+1] = reversed(new_perm[i:j+1])
                    new_cost = calc_cost(new_perm)
                    if new_cost < current_cost:
                        perm = new_perm
                        current_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
            # Or-Opt (move segment of length 1 or 2)
            if not improved:
                for length in [1, 2]:
                    for i in range(n):
                        if i + length >= n:
                            continue
                        # Segment to move
                        seg = perm[i:i+length]
                        # Remove segment
                        new_perm = perm[:i] + perm[i+length:]
                        # Try inserting at all positions
                        for k in range(len(new_perm)):
                            if k == i or (i > 0 and k == i - 1):
                                continue
                            candidate = new_perm[:k] + seg + new_perm[k:]
                            # Check if it's the same as original
                            if candidate == perm:
                                continue
                            new_cost = calc_cost(candidate)
                            if new_cost < current_cost:
                                perm = candidate
                                current_cost = new_cost
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        return perm

    rng = random.Random(614)
    
    best_perm = list(range(n))
    rng.shuffle(best_perm)
    best_perm = local_search(best_perm)
    best_cost = calc_cost(best_perm)
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s margin
    
    # Iterated Local Search
    while time.time() - start_time < time_limit:
        # Perturb best solution
        current_perm = best_perm[:]
        # Apply 3-5 random 2-Opt moves
        num_perturbations = rng.randint(3, 5)
        for _ in range(num_perturbations):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            current_perm[i:j+1] = reversed(current_perm[i:j+1])
        
        # Local search
        improved_perm = local_search(current_perm)
        cost = calc_cost(improved_perm)
        
        if cost < best_cost:
            best_cost = cost
            best_perm = improved_perm
    
    return best_perm
# EVOLVE-BLOCK-END
