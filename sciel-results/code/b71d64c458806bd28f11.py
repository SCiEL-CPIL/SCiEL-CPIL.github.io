"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples and build pair_to_third mapping
    bad_triples = []
    pair_to_third = {}  # (a,b) -> set of c such that {a,b,c} is a bad triple
    x_conflict_pairs = {x: [] for x in range(1, n+1)}
    elem_triple_count = [0]*(n+1)
    
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                lac = lcm_table[a][c]
                if lac != lab:
                    continue
                lbc = lcm_table[b][c]
                if lbc == lab:
                    bad_triples.append((a, b, c))
                    # Add to pair_to_third
                    for x, y in [(a,b), (a,c), (b,c)]:
                        key = (min(x,y), max(x,y))
                        if key not in pair_to_third:
                            pair_to_third[key] = set()
                        # The third element
                        z = a + b + c - x - y
                        pair_to_third[key].add(z)
                    # Add to x_conflict_pairs
                    x_conflict_pairs[a].append((b, c))
                    x_conflict_pairs[b].append((a, c))
                    x_conflict_pairs[c].append((a, b))
                    elem_triple_count[a] += 1
                    elem_triple_count[b] += 1
                    elem_triple_count[c] += 1
    
    if time.time() > deadline:
        return list(range(1, n+1))
    
    def can_add(S_set, x):
        """Check if adding x to S_set creates any bad triple using pair_to_third."""
        for y in S_set:
            key = (min(x, y), max(x, y))
            if key in pair_to_third:
                for z in pair_to_third[key]:
                    if z in S_set:
                        return False
        return True
    
    def try_add(S_list, S_set, x):
        """Try to add x to the set. Returns True if successful."""
        if x in S_set:
            return False
        if can_add(S_set, x):
            S_list.append(x)
            S_set.add(x)
            return True
        return False
    
    # Strategy: Multiple restarts with different orderings, then local search
    
    random.seed(42)
    best_set = None
    best_size = 0
    
    # Generate multiple candidate orderings
    orderings = []
    
    # Ordering 1: large range first
    orderings.append(list(range(65, 129)))
    
    # Ordering 2: large range + some smaller
    orderings.append(list(range(65, 129)) + list(range(33, 65)))
    
    # Ordering 3: natural order
    orderings.append(list(range(1, n+1)))
    
    # Ordering 4: sorted by descending triple count (most constrained first)
    orderings.append(sorted(range(1, n+1), key=lambda x: -elem_triple_count[x]))
    
    # Ordering 5: sorted by ascending triple count (least constrained first)
    orderings.append(sorted(range(1, n+1), key=lambda x: elem_triple_count[x]))
    
    # Ordering 6: sorted by value descending
    orderings.append(list(range(n, 0, -1)))
    
    # Ordering 7-12: random shuffles
    for _ in range(6):
        order = list(range(1, n+1))
        random.shuffle(order)
        orderings.append(order)
    
    # Ordering 13: large range shuffled + small range shuffled
    large = list(range(65, 129))
    small = list(range(1, 65))
    random.shuffle(large)
    random.shuffle(small)
    orderings.append(large + small)
    
    # Ordering 14: alternate large and small
    large = list(range(65, 129))
    small = list(range(1, 65))
    random.shuffle(large)
    random.shuffle(small)
    interleaved = []
    i, j = 0, 0
    while i < len(large) or j < len(small):
        if i < len(large):
            interleaved.append(large[i])
            i += 1
        if j < len(small):
            interleaved.append(small[j])
            j += 1
    orderings.append(interleaved)
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        # Greedy construction
        S_list = []
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            try_add(S_list, S_set, x)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = list(S_list)
        
        # Local search: 1-swap moves
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_set = set(best_set)
            
            # Try 1-swap: remove x from S, add y not in S
            for x in list(best_set):
                if time.time() > deadline:
                    break
                new_S_set = S_set - {x}
                for y in range(1, n+1):
                    if y in S_set:
                        continue
                    if can_add(new_S_set, y):
                        # Check if this improves (it won't since same size, but might enable future improvements)
                        # Actually for 1-swap we just want to try different configurations
                        # Only accept if it allows adding more later, so let's do a greedy add after swap
                        new_set = [z for z in best_set if z != x] + [y]
                        new_S_set2 = set(new_set)
                        # Try to greedily add more
                        added = 0
                        for z in range(1, n+1):
                            if time.time() > deadline:
                                break
                            if z not in new_S_set2 and can_add(new_S_set2, z):
                                new_set.append(z)
                                new_S_set2.add(z)
                                added += 1
                        if len(new_set) > best_size:
                            best_set = new_set
                            best_size = len(new_set)
                            improved = True
                            break
                if improved:
                    break
        
        # Try 2-swap (kick): remove x1, x2 from S, add y1, y2 not in S
        if time.time() < deadline:
            improved2 = True
            while improved2 and time.time() < deadline:
                improved2 = False
                S_set = set(best_set)
                
                for i in range(len(best_set)):
                    if time.time() > deadline:
                        break
                    x1 = best_set[i]
                    for j in range(i+1, len(best_set)):
                        if time.time() > deadline:
                            break
                        x2 = best_set[j]
                        new_S_set = S_set - {x1, x2}
                        
                        # Find two elements to add
                        added = []
                        for y in range(1, n+1):
                            if y in S_set or y in added:
                                continue
                            if can_add(new_S_set | set(added), y):
                                added.append(y)
                                if len(added) == 2:
                                    break
                        
                        if len(added) == 2:
                            new_set = [z for z in best_set if z != x1 and z != x2] + added
                            # Try to greedily add more
                            new_S_set2 = set(new_set)
                            for z in range(1, n+1):
                                if time.time() > deadline:
                                    break
                                if z not in new_S_set2 and can_add(new_S_set2, z):
                                    new_set.append(z)
                                    new_S_set2.add(z)
                            if len(new_set) > best_size:
                                best_set = new_set
                                best_size = len(new_set)
                                improved2 = True
                                break
                    if improved2:
                        break
        
        # Also try adding elements greedily to the current best
        if time.time() < deadline:
            S_set = set(best_set)
            for x in range(1, n+1):
                if time.time() > deadline:
                    break
                if x not in S_set and can_add(S_set, x):
                    best_set.append(x)
                    S_set.add(x)
                    best_size = len(best_set)
    
    # Final: ensure we have at least one element
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
