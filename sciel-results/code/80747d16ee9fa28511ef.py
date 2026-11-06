"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute which sums are perfect squares
    max_sum = 2 * N  # 512
    is_square = [False] * (max_sum + 1)
    for k in range(1, int(math.isqrt(max_sum)) + 1):
        is_square[k * k] = True
    
    # Build conflict: for each number a, which numbers b (b >= a) conflict with a
    # Edge exists if a + b is a perfect square
    # Also self-conflict: 2a is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        # Self conflict
        if is_square[2 * a]:
            conflict[a].add(a)
        for b in range(a + 1, N + 1):
            if is_square[a + b]:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Greedy initial solution: process numbers in order of fewest conflicts, add if no conflict with current set
    degree = [len(conflict[i]) for i in range(1, N + 1)]
    order = sorted(range(1, N + 1), key=lambda x: (degree[x], x))
    
    current_set = set()
    for a in order:
        if a in conflict[a]:  # self-conflict, cannot include
            continue
        # Check if a conflicts with any in current_set
        ok = True
        for b in current_set:
            if b in conflict[a]:
                ok = False
                break
        if ok:
            current_set.add(a)
    
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Local search: try to swap or add/remove to improve
    # Represent set as a boolean array for fast checking
    def is_valid_set(s):
        for a in s:
            if 2 * a and is_square[2 * a]:
                return False
            for b in s:
                if b > a and is_square[a + b]:
                    return False
        return True
    
    def conflicts_with_set(a, s):
        """Check if adding a to s creates any conflict."""
        if is_square[2 * a]:
            return True
        for b in s:
            if b != a and is_square[a + b]:
                return True
        return False
    
    # Local search: try adding unselected elements, and try swapping
    for iteration in range(500):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding elements not in set
        candidates = [a for a in range(1, N + 1) if a not in current_set]
        random.shuffle(candidates)
        for a in candidates:
            if not conflicts_with_set(a, current_set):
                current_set.add(a)
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
        
        # Try removing one element and adding two others
        if time.time() > deadline:
            break
        for a in list(current_set):
            temp_set = current_set - {a}
            # Try to add up to 2 elements
            added = 0
            candidates = [b for b in range(1, N + 1) if b not in temp_set and b != a]
            random.shuffle(candidates)
            new_set = set(temp_set)
            for b in candidates:
                if len(new_set) + 2 <= len(current_set):
                    break
                if not conflicts_with_set(b, new_set):
                    new_set.add(b)
                    added += 1
            if len(new_set) > len(current_set):
                current_set = new_set
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
        
        if not improved:
            # Try a random restart with different greedy order
            if time.time() > deadline:
                break
            random.seed(iteration * 12345)
            order2 = list(range(1, N + 1))
            random.shuffle(order2)
            current_set = set()
            for a in order2:
                if is_square[2 * a]:
                    continue
                if not conflicts_with_set(a, current_set):
                    current_set.add(a)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
    
    return list(best_set)
# EVOLVE-BLOCK-END
