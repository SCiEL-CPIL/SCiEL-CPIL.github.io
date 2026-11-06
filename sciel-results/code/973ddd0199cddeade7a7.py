import time
import random
from itertools import combinations

def _compute_collision_count(S):
    """Count the number of sum collisions in set S.
    A collision is when two different pairs (i,j) and (k,l) have the same sum.
    Pairs are unordered, and we include diagonals (i,i).
    """
    n = len(S)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                sums[s].append((i, j))
            else:
                sums[s] = [(i, j)]
    
    # Count collisions: for each sum value, if there are k pairs with that sum,
    # the number of collision pairs is C(k, 2)
    collisions = 0
    for pairs in sums.values():
        k = len(pairs)
        if k > 1:
            collisions += k * (k - 1) // 2
    return collisions

def _max_sidon_size_bruteforce(S):
    """Find the maximum Sidon subset size by exhaustive search."""
    n = len(S)
    best = 0
    # Try all subsets from largest to smallest
    for size in range(n, best, -1):
        found = False
        for subset in combinations(range(n), size):
            S_sub = [S[i] for i in subset]
            # Check if Sidon
            sums = set()
            is_sidon = True
            for i in range(size):
                for j in range(i, size):
                    s = S_sub[i] + S_sub[j]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                best = size
                found = True
                break
        if found:
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for 90s limit
    
    best_solution = list(range(12))
    best_collision = _compute_collision_count(best_solution)
    
    # Local search with restarts
    random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Initialize: try different strategies
        if restart % 3 == 0:
            # Random
            S = random.sample(range(129), 12)
        elif restart % 3 == 1:
            # Arithmetic progression-like
            base = random.randint(0, 100)
            step = random.randint(1, 5)
            S = []
            for i in range(12):
                val = base + i * step + random.randint(-1, 1)
                if 0 <= val <= 128 and val not in S:
                    S.append(val)
            while len(S) < 12:
                val = random.randint(0, 128)
                if val not in S:
                    S.append(val)
            S = S[:12]
        else:
            # Clustered
            center = random.randint(10, 118)
            S = []
            for _ in range(12):
                val = center + random.randint(-8, 8)
                val = max(0, min(128, val))
                if val not in S:
                    S.append(val)
            while len(S) < 12:
                val = random.randint(0, 128)
                if val not in S:
                    S.append(val)
            S = S[:12]
        
        current_collision = _compute_collision_count(S)
        
        # Local search: try swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(i + 1, 12):
                    if time.time() > deadline:
                        break
                    for new_val in random.sample(range(129), min(10, 129)):
                        if new_val in S_list:
                            continue
                        S_list[j] = new_val
                        new_collision = _compute_collision_count(S_list)
                        if new_collision > current_collision:
                            current_collision = new_collision
                            S = list(S_list)
                            improved = True
                            break
                        S_list[j] = S[j]
        
        if current_collision > best_collision:
            best_collision = current_collision
            best_solution = list(S)
    
    # Final verification: ensure exactly 12 distinct integers in [0,128]
    best_solution = sorted(set(best_solution))
    while len(best_solution) < 12:
        for v in range(129):
            if v not in best_solution:
                best_solution.append(v)
                break
    best_solution = best_solution[:12]
    
    return best_solution
