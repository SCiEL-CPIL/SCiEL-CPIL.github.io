import time
import math
import random
import sys
from collections import defaultdict

N = 128
sys.setrecursionlimit(10000)

# Precompute conflicts: for each x, conflicts[x] is a list of (u,v) pairs
# such that {x, u, v} is a forbidden triple (all three pairwise LCMs equal).
conflicts = [set() for _ in range(N + 1)]

def precompute_conflicts():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))

precompute_conflicts()

# Convert conflicts to lists for faster iteration
conflict_lists = [list(conflicts[i]) for i in range(N + 1)]

# Precompute bitmask representation of the set for fast membership
# bit i corresponds to element i+1 (1-indexed)
def set_to_mask(s):
    m = 0
    for x in s:
        m |= (1 << x)
    return m

def mask_to_set(m):
    s = set()
    for x in range(1, N + 1):
        if m & (1 << x):
            s.add(x)
    return s

def can_add_mask(x, mask):
    """Check if x can be added given the bitmask of current set."""
    for u, v in conflict_lists[x]:
        if (mask & (1 << u)) and (mask & (1 << v)):
            return False
    return True

def can_add(x, current_set):
    """Check if x can be added to current_set without creating a forbidden triple."""
    mask = set_to_mask(current_set)
    return can_add_mask(x, mask)

# Global pair conflict matrix
# pair_conflict[i][j] = True if there exists k such that {i,j,k} is a forbidden triple.
pair_conflict = [[False] * (N + 1) for _ in range(N + 1)]

def precompute_pair_conflicts():
    for a in range(1, N + 1):
        for (b, c) in conflicts[a]:
            pair_conflict[a][b] = True
            pair_conflict[a][c] = True
            pair_conflict[b][c] = True
            pair_conflict[b][a] = True
            pair_conflict[c][a] = True
            pair_conflict[c][b] = True

precompute_pair_conflicts()

def greedy_construct():
    """Greedy construction with random restarts."""
    best_s = set()
    for _ in range(300):
        order = list(range(1, N + 1))
        random.shuffle(order)
        cur = set()
        for x in order:
            if can_add(x, cur):
                cur.add(x)
        if len(cur) > len(best_s):
            best_s = cur
    return best_s

def local_search(s, deadline):
    """
    Improve set s using bitmask-based local search with 1-swap moves.
    """
    s = set(s)
    best_s = set(s)
    best_size = len(s)
    mask = set_to_mask(s)
    best_mask = mask
    
    while time.time() < deadline:
        improved = False
        
        # Try to add any element
        for x in range(1, N + 1):
            if not (mask & (1 << x)) and can_add_mask(x, mask):
                mask |= (1 << x)
                improved = True
                if bin(mask).count('1') > best_size:
                    best_size = bin(mask).count('1')
                    best_mask = mask
                break
        
        if improved:
            continue
        
        # If no element can be added, try 1-swap moves: remove one, add one
        candidates_to_remove = [x for x in range(1, N + 1) if mask & (1 << x)]
        random.shuffle(candidates_to_remove)
        
        found_improvement = False
        for x in candidates_to_remove:
            mask_without_x = mask & ~(1 << x)
            # Try to add any element to mask_without_x
            for y in range(1, N + 1):
                if not (mask_without_x & (1 << y)) and can_add_mask(y, mask_without_x):
                    new_mask = mask_without_x | (1 << y)
                    new_size = bin(new_mask).count('1')
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        mask = new_mask
                        found_improvement = True
                        break
            if found_improvement:
                break
        
        if not found_improvement:
            # Try 2-swap moves: remove two, add two
            candidates = [x for x in range(1, N + 1) if mask & (1 << x)]
            random.shuffle(candidates)
            found_2swap = False
            for i in range(min(len(candidates), 10)):
                for j in range(i + 1, min(len(candidates), 10)):
                    x1, x2 = candidates[i], candidates[j]
                    mask_without = mask & ~(1 << x1) & ~(1 << x2)
                    # Try to add two elements
                    added = 0
                    new_mask = mask_without
                    for y in range(1, N + 1):
                        if not (new_mask & (1 << y)) and can_add_mask(y, new_mask):
                            new_mask |= (1 << y)
                            added += 1
                            if added == 2:
                                break
                    if added == 2:
                        new_size = bin(new_mask).count('1')
                        if new_size > best_size:
                            best_size = new_size
                            best_mask = new_mask
                            mask = new_mask
                            found_2swap = True
                            break
                if found_2swap:
                    break
            
            if not found_2swap:
                break
    
    return mask_to_set(best_mask)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Greedy + Local Search to get a good lower bound
    initial_set = greedy_construct()
    initial_set = local_search(initial_set, deadline)
    
    best_size = len(initial_set)
    best_set = initial_set
    
    # 2. Try more local search iterations with different starting points
    if time.time() < deadline - 10.0:
        for _ in range(80):
            if time.time() > deadline - 5.0:
                break
            # Create a perturbed version of best_set
            perturbed = set(best_set)
            # Randomly remove 1-3 elements
            to_remove = random.sample(list(perturbed), min(random.randint(1, 3), len(perturbed)))
            for x in to_remove:
                perturbed.discard(x)
            
            # Try to add back elements
            for x in range(1, N + 1):
                if x not in perturbed and can_add(x, perturbed):
                    perturbed.add(x)
            
            # Local search from perturbed state
            improved = local_search(perturbed, deadline - 5.0)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    result = list(best_set)
    if not result:
        result = [1]
        
    return result

# Test locally
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            for k in range(j + 1, len(res)):
                a, b, c = res[i], res[j], res[k]
                if math.lcm(a, b) == math.lcm(a, c) == math.lcm(b, c):
                    valid = False
                    print(f"Invalid triple: {a}, {b}, {c}")
    print(f"Valid: {valid}")
