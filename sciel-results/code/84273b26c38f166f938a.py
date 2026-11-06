"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _can_add(S, x):
    """Check if x can be added to set S (list) while maintaining validity."""
    n = len(S)
    # Part 1: x must not divide any triple sum from S
    # Optimize: for each pair (i,j), compute (S[i]+S[j]) % x, then check if any S[k] % x completes it
    if n >= 3:
        # Build map: residue -> list of (i,j) pairs
        pair_res = {}
        for i in range(n):
            si = S[i] % x
            for j in range(i+1, n):
                r = (si + S[j]) % x
                if r not in pair_res:
                    pair_res[r] = []
                pair_res[r].append((i, j))
        
        for r, pairs in pair_res.items():
            need = (-r) % x
            for k in range(n):
                if S[k] % x == need:
                    for (pi, pj) in pairs:
                        if k != pi and k != pj:
                            return False
    
    # Part 2: for each a in S, a must not divide (x + b + c) for any pair (b,c) in S\{a}
    for idx_a in range(n):
        a = S[idx_a]
        target = (-x) % a
        # Check all pairs in S excluding idx_a
        for i in range(n):
            if i == idx_a:
                continue
            for j in range(i+1, n):
                if j == idx_a:
                    continue
                if (S[i] + S[j]) % a == target:
                    return False
    return True

def _greedy_descending():
    selected = []
    for x in range(192, 0, -1):
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_search(initial, deadline):
    current = list(initial)
    best = list(current)
    best_len = len(current)
    all_candidates = list(range(1, 193))
    
    while time.time() < deadline:
        improved = False
        
        # Try adding
        for _ in range(20):
            x = random.choice(all_candidates)
            if x not in current:
                if _can_add(current, x):
                    current.append(x)
                    improved = True
                    if len(current) > best_len:
                        best = list(current)
                        best_len = len(current)
                    break
        
        # Try swap
        if not improved and len(current) >= 2:
            for _ in range(40):
                i = random.randint(0, len(current) - 1)
                x = random.choice(all_candidates)
                if x not in current:
                    new_set = current[:i] + current[i+1:]
                    if _can_add(new_set, x):
                        current = new_set + [x]
                        improved = True
                        if len(current) > best_len:
                            best = list(current)
                            best_len = len(current)
                        break
        
        if not improved:
            break
    
    return best

def construct():
    deadline = time.time() + 55
    
    best = _greedy_descending()
    best_len = len(best)
    
    if time.time() < deadline - 5:
        result = _local_search(best, deadline - 2)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    while time.time() < deadline - 5:
        initial = random.sample(range(1, 193), 15)
        result = _local_search(initial, deadline - 2)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    return best

# EVOLVE-BLOCK-END
