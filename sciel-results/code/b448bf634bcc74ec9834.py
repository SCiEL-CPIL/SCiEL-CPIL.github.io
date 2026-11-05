import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a given subset has all unique pair sums including diagonals."""
    n = len(subset)
    sums = set()
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def exact_max_sidon_size(candidate_set):
    """Find the size of the largest Sidon subset by exhaustive search."""
    cs = sorted(candidate_set)
    n = len(cs)
    
    # Try from largest possible size down
    for k in range(n, 0, -1):
        for subset in combinations(cs, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Key insight: We want to MINIMIZE the largest Sidon subset.
    # A Sidon set requires all pair sums (including diagonals) to be unique.
    # To make it hard to find large Sidon subsets, we want many sum collisions.
    # 
    # Strategy: Use arithmetic progressions or structured sets where many pairs
    # have the same sum. For example, in {0,1,2,...,11}, the pair (0,3) and (1,2)
    # both sum to 3, creating collisions.
    #
    # The theoretical minimum: Can we get max Sidon size = 4?
    # A Sidon set of size 5 in [0,128] needs at least C(5,2)+5 = 15 distinct sums.
    # If we can force enough collisions, we might limit to size 4 or even 3.
    #
    # Let's try structured candidates and local search.
    
    # Initial candidates designed to maximize sum collisions
    candidates = [
        list(range(12)),  # Consecutive integers - many collisions
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],  # Two blocks
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],  # Two blocks
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],  # Tighter blocks
        [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32],  # Block structure
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # Even spaced
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17],  # Periodic blocks
        [0, 1, 3, 7, 12, 18, 25, 33, 42, 52, 63, 75],
        [0, 1, 4, 10, 20, 40, 80, 2, 5, 15, 35, 75],
    ]
    
    best_set = list(range(12))
    best_size = exact_max_sidon_size(best_set)
    
    for c in candidates:
        if len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        sz = exact_max_sidon_size(c)
        if sz < best_size:
            best_size = sz
            best_set = list(c)
            if best_size <= 3:
                break
    
    # Local search: try to create more sum collisions
    random.seed(42)
    current_set = list(best_set)
    current_size = best_size
    
    def try_create_collision(s):
        """Try to create a new sum collision by modifying one element."""
        s_set = set(s)
        n = len(s)
        
        # Find two pairs (a,b) and (c,d) where a+b = c+d would create a collision
        # We can adjust one element to create such a collision
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # Try to make s[i] + s[j] = s[k] + s[k] (diagonal)
                    target = 2 * s[k] - s[j]
                    if target != s[i] and 0 <= target <= 128 and target not in s_set:
                        new_s = list(s)
                        new_s[i] = target
                        new_sz = exact_max_sidon_size(new_s)
                        if new_sz < current_size:
                            return new_s, new_sz
                    
                    # Try to make s[i] + s[j] = s[k] + s[m] for another m
                    for m in range(k + 1, n):
                        if m == i or m == j:
                            continue
                        target = s[k] + s[m] - s[j]
                        if target != s[i] and 0 <= target <= 128 and target not in s_set:
                            new_s = list(s)
                            new_s[i] = target
                            new_sz = exact_max_sidon_size(new_s)
                            if new_sz < current_size:
                                return new_s, new_sz
        
        return None, current_size
    
    # Simulated annealing style local search
    while time.time() < deadline:
        improved = False
        
        # Try targeted collision creation
        new_s, new_sz = try_create_collision(current_set)
        if new_s is not None:
            current_set = new_s
            current_size = new_sz
            improved = True
            if current_size < best_size:
                best_size = current_size
                best_set = list(current_set)
                if best_size <= 3:
                    break
        
        # Random perturbation
        if random.random() < 0.3 or not improved:
            # Try random swap
            i, j = random.sample(range(12), 2)
            new_val = random.randint(0, 128)
            if new_val not in set(current_set):
                new_s = list(current_set)
                new_s[j] = new_val
                new_sz = exact_max_sidon_size(new_s)
                if new_sz <= current_size:
                    current_set = new_s
                    current_size = new_sz
                    if current_size < best_size:
                        best_size = current_size
                        best_set = list(current_set)
                        if best_size <= 3:
                            break
        
        # Random restart if stuck
        if random.random() < 0.1:
            current_set = random.sample(range(129), 12)
            current_size = exact_max_sidon_size(current_set)
            if current_size < best_size:
                best_size = current_size
                best_set = list(current_set)
                if best_size <= 3:
                    break
    
    return best_set
