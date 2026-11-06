"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

N = 128
# Precompute GCDs
gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        gcd_table[i][j] = gcd(i, j)
        gcd_table[j][i] = gcd(i, j)

def can_add(S, x):
    """
    Check if x can be added to set S without violating the condition.
    """
    groups = {}
    for a in S:
        g = gcd_table[x][a]
        if g not in groups:
            groups[g] = []
        groups[g].append(a)
    
    for g, members in groups.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                a = members[i]
                b = members[j]
                if gcd_table[a][b] == g:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = set()
    
    # Greedy initialization with random restarts
    for _ in range(100):
        if time.time() > deadline:
            break
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        current_set = set()
        for x in candidates:
            if can_add(current_set, x):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    # Local Search on best_set
    current_set = best_set.copy()
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add
        added = False
        for x in range(1, N + 1):
            if x not in current_set:
                if can_add(current_set, x):
                    current_set.add(x)
                    improved = True
                    added = True
                    break 
        if added:
            no_improve_count = 0
            continue
            
        # 2. If no add, try to swap
        escaped = False
        # Try a few random swaps to escape local optima
        for _ in range(20):
            if time.time() > deadline:
                break
            if len(current_set) < 2:
                break
            a = random.choice(list(current_set))
            b = random.randint(1, N)
            if b in current_set:
                continue
            
            new_set = current_set.copy()
            new_set.remove(a)
            new_set.add(b)
            
            # new_set is valid because it's a subset of a valid set?
            # No, current_set is valid. Removing a element keeps it valid. Adding b might break it.
            # We must check if new_set is valid.
            # Wait, can_add checks if b can be added to current_set.
            # Here we remove a first.
            # So we need to check if b can be added to (current_set - {a}).
            
            if can_add(new_set - {b}, b): # new_set already has b, so check b against rest
                 # Actually can_add(S, x) checks x against S.
                 # So can_add(new_set - {b}, b) is correct.
                 pass
            else:
                 continue
            
            # If valid, try to add more
            add_possible = False
            for y in range(1, N + 1):
                if y not in new_set:
                    if can_add(new_set, y):
                        new_set.add(y)
                        add_possible = True
                        break
            
            if add_possible and len(new_set) > len(current_set):
                current_set = new_set
                improved = True
                escaped = True
                break
            elif add_possible and len(new_set) == len(current_set):
                # Accept to explore
                current_set = new_set
                improved = True
                escaped = True
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                break
        else:
            no_improve_count = 0
            
    return list(best_set) if len(best_set) > len(current_set) else list(current_set)

# EVOLVE-BLOCK-END
