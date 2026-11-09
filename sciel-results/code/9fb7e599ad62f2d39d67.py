import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Precompute bad pairs for each x
    # bad_pairs[x] is a set of frozenset({a,b}) or tuple (a,b) with a<b
    bad_pairs = [set() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for a in range(1, n + 1):
            if a == x: continue
            for b in range(a + 1, n + 1):
                if b == x: continue
                if gcd_table[a][b] == gcd_table[b][x] == gcd_table[x][a]:
                    # Store pair (a,b) in a canonical form
                    if a < b:
                        bad_pairs[x].add((a, b))
                    else:
                        bad_pairs[x].add((b, a))

    def can_add(x, S_pairs):
        # S_pairs is a set of tuples (a,b) with a<b representing pairs in S
        # Check if any bad pair for x is in S_pairs
        for p in bad_pairs[x]:
            if p in S_pairs:
                return False
        return True

    # Helper to update S_pairs
    def add_to_set(S, S_pairs, x):
        S.add(x)
        for y in S:
            if y == x: continue
            if x < y:
                S_pairs.add((x, y))
            else:
                S_pairs.add((y, x))
                
    def remove_from_set(S, S_pairs, x):
        S.remove(x)
        for y in S:
            if x < y:
                S_pairs.remove((x, y))
            else:
                S_pairs.remove((y, x))

    def is_valid_swap(S, S_pairs, z, x, y):
        # Simulate removal of z
        # S' = S - {z}
        # Pairs to remove from S_pairs: (z, w) for w in S, w != z
        # We don't actually modify S_pairs, we just check.
        
        # Check x against S - {z}
        # Bad pairs for x that are in S - {z}
        # A bad pair (a,b) is in S - {z} if (a,b) in S_pairs AND z not in (a,b)
        
        # This logic is slightly tricky with the set of pairs.
        # Alternative: Just construct the new S and S_pairs?
        # No, that's expensive.
        
        # Let's stick to the O(N^2) check for swaps, but use the optimized can_add for single additions.
        # The swap check is:
        # 1. For each pair (a,b) in S - {z}, check if (a,b,x) is bad OR (a,b,y) is bad.
        # 2. For each a in S - {z}, check if (x,y,a) is bad.
        
        # We can precompute list of pairs in S.
        # Since N is small, we can just iterate.
        
        S_list = list(S)
        S_minus_z = [v for v in S_list if v != z]
        
        # Check x
        for i in range(len(S_minus_z)):
            a = S_minus_z[i]
            for j in range(i+1, len(S_minus_z)):
                b = S_minus_z[j]
                if gcd_table[a][b] == gcd_table[b][x] == gcd_table[x][a]:
                    return False
                    
        # Check y
        for i in range(len(S_minus_z)):
            a = S_minus_z[i]
            for j in range(i+1, len(S_minus_z)):
                b = S_minus_z[j]
                if gcd_table[a][b] == gcd_table[b][y] == gcd_table[y][a]:
                    return False

        # Check x, y, a
        for a in S_minus_z:
            if gcd_table[x][y] == gcd_table[y][a] == gcd_table[a][x]:
                return False
                
        return True

    # Initialization
    best_S = set()
    best_S_pairs = set()
    
    for _ in range(5):
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        S = set()
        S_pairs = set()
        for x in candidates:
            if can_add(x, S_pairs):
                add_to_set(S, S_pairs, x)
        if len(S) > len(best_S):
            best_S = S
            best_S_pairs = S_pairs

    # Local Search
    while time.time() < deadline:
        improved = False
        
        # 1. Try single addition
        remaining = [x for x in range(1, n+1) if x not in best_S]
        random.shuffle(remaining)
        for x in remaining:
            if can_add(x, best_S_pairs):
                add_to_set(best_S, best_S_pairs, x)
                improved = True
                break
        
        if improved:
            continue

        # 2. Try 1-for-2 swaps
        # Sample random swaps
        swaps_tried = 0
        limit = 5000
        while swaps_tried < limit and time.time() < deadline:
            if len(best_S) == 0: break
            z = random.choice(list(best_S))
            rem_list = [x for x in range(1, n+1) if x not in best_S]
            if len(rem_list) < 2:
                break
            x, y = random.sample(rem_list, 2)
            
            if is_valid_swap(best_S, best_S_pairs, z, x, y):
                remove_from_set(best_S, best_S_pairs, z)
                add_to_set(best_S, best_S_pairs, x)
                add_to_set(best_S, best_S_pairs, y)
                improved = True
                break
            
            swaps_tried += 1
        
        if improved:
            continue
        
        # 3. Try 1-for-1 swaps
        swaps_tried = 0
        limit = 2000
        while swaps_tried < limit and time.time() < deadline:
            if len(best_S) == 0: break
            z = random.choice(list(best_S))
            rem_list = [x for x in range(1, n+1) if x not in best_S]
            if not rem_list:
                break
            x = random.choice(rem_list)
            
            # Check if x can be added to S - {z}
            # We can simulate removal
            # Or just use can_add with a temporary set of pairs?
            # Easier: Check is_valid_swap but with y=None?
            # No, let's just use the O(N^2) check for single swap too, or reuse logic.
            
            # Check if (S - {z}) U {x} is valid
            # This is equivalent to: x does not form bad triple with any pair in S - {z}
            S_list = list(best_S)
            S_minus_z = [v for v in S_list if v != z]
            
            valid = True
            for i in range(len(S_minus_z)):
                a = S_minus_z[i]
                for j in range(i+1, len(S_minus_z)):
                    b = S_minus_z[j]
                    if gcd_table[a][b] == gcd_table[b][x] == gcd_table[x][a]:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                remove_from_set(best_S, best_S_pairs, z)
                add_to_set(best_S, best_S_pairs, x)
                improved = True
                break
            
            swaps_tried += 1
        
        if improved:
            continue
        
        # If stuck, do a random perturbation: remove 3 random elements
        # This allows escaping deep local minima
        if len(best_S) > 5:
            to_remove = random.sample(list(best_S), 3)
            for z in to_remove:
                remove_from_set(best_S, best_S_pairs, z)
            # Don't mark as improved, just let the loop try to add back

    return list(best_S)
