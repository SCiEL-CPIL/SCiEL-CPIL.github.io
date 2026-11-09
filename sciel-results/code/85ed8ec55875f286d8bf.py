import time
from math import gcd
import random

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    # 1. Precompute bad triples
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(b,c) == gcd(c,a)
    # We store for each x, the list of pairs (u, v) such that {x, u, v} is a bad triple.
    # u < v, and x is not u or v.
    trips_by_node = {i: [] for i in range(1, 129)}
    
    # Precompute gcd table for speed
    gcd_table = [[0] * 129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Find all bad triples
    # Optimization: Iterate a, b, c with a < b < c
    for a in range(1, 129):
        for b in range(a + 1, 129):
            gab = gcd_table[a][b]
            for c in range(b + 1, 129):
                gbc = gcd_table[b][c]
                if gab != gbc:
                    continue
                gca = gcd_table[a][c]
                if gca == gab:
                    # Bad triple (a, b, c)
                    trips_by_node[a].append((b, c))
                    trips_by_node[b].append((a, c))
                    trips_by_node[c].append((a, b))
    
    # 2. Greedy Initial Solution
    # Sort numbers by number of conflicts (len of trips_by_node)
    # Fewer conflicts = safer to add early?
    # Actually, numbers with few bad triples are "easier" to fit.
    order = sorted(range(1, 129), key=lambda x: len(trips_by_node[x]))
    
    def can_add(x, S_set):
        for u, v in trips_by_node[x]:
            if u in S_set and v in S_set:
                return False
        return True
        
    S = set()
    for x in order:
        if can_add(x, S):
            S.add(x)
            
    best_S = set(S)
    best_size = len(S)
    
    # 3. Simulated Annealing
    # State: S (set)
    # Move: Add, Remove, or Swap
    # Energy: -len(S) (we minimize energy, so maximize size)
    # Actually, let's just track size.
    
    rng = random.Random(42)
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    current_S = set(S)
    current_size = len(current_S)
    
    while T > min_T and time.time() < deadline - 5:
        # Choose move type
        move_type = rng.random()
        if move_type < 0.4:
            # Add
            candidates = [x for x in range(1, 129) if x not in current_S]
            if not candidates:
                T *= cooling_rate
                continue
            x = rng.choice(candidates)
            if can_add(x, current_S):
                current_S.add(x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = set(current_S)
            else:
                # Try to add anyway? No, must stay valid.
                # Maybe try to remove something that conflicts?
                # For now, just skip or try swap.
                pass
        elif move_type < 0.6:
            # Remove
            if not current_S:
                T *= cooling_rate
                continue
            x = rng.choice(list(current_S))
            current_S.remove(x)
            current_size -= 1
        else:
            # Swap
            if not current_S:
                T *= cooling_rate
                continue
            x = rng.choice(list(current_S))
            candidates = [y for y in range(1, 129) if y not in current_S]
            if not candidates:
                T *= cooling_rate
                continue
            y = rng.choice(candidates)
            
            # Check if swapping x for y is valid
            # Temporarily remove x
            current_S.remove(x)
            if can_add(y, current_S):
                current_S.add(y)
                # Size stays same
                if current_size > best_size:
                    best_size = current_size
                    best_S = set(current_S)
            else:
                # Swap not valid, revert
                current_S.add(x)
        
        T *= cooling_rate
    
    # 4. Final Local Search (Hill Climbing) to polish
    # Try to add any remaining element
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for x in range(1, 129):
            if x not in best_S:
                if can_add(x, best_S):
                    best_S.add(x)
                    changed = True
    
    return list(best_S)
