"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    
    # Precompute gcd table for [1,128]
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute bad_triples[x] = set of (a,b) with a<b, a!=x, b!=x, 
    # such that gcd(a,b)==gcd(a,x)==gcd(b,x)
    bad_triples = [set() for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            g_ab = gcd_table[a][b]
            for c in range(1, n+1):
                if c == a or c == b:
                    continue
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    bad_triples[c].add((a, b))
                    if a < b:
                        bad_triples[a].add((b, c) if b < c else (c, b))
                        bad_triples[b].add((a, c) if a < c else (c, a))
                    else:
                        bad_triples[a].add((b, c) if b < c else (c, b))
                        bad_triples[b].add((a, c) if a < c else (c, a))
    
    # For fast checking, convert to lists
    bad_triples_list = [list(s) for s in bad_triples]
    
    def can_add(S, x):
        """Check if x can be added to set S without creating a bad triple."""
        for (a, b) in bad_triples_list[x]:
            if a in S and b in S:
                return False
        return True
    
    # Greedy construction with a given order
    def greedy_build(order):
        current = set()
        for x in order:
            if can_add(current, x):
                current.add(x)
        return current
    
    # Local search
    def local_search(S, max_iters=1000):
        S = set(S)
        current_size = len(S)
        best_S = set(S)
        best_size = current_size
        
        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Try to add elements
            added = False
            remaining = [x for x in range(1, n+1) if x not in S]
            random.shuffle(remaining)
            for x in remaining:
                if can_add(S, x):
                    S.add(x)
                    current_size += 1
                    added = True
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                    break
            
            if added:
                continue
            
            # Try 1-swap: remove one, try to add two or more
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            for u in S_list:
                if time.time() > deadline - 3:
                    return best_S
                S.remove(u)
                new_elements = []
                remaining = [x for x in range(1, n+1) if x not in S]
                random.shuffle(remaining)
                for x in remaining:
                    if can_add(S, x):
                        S.add(x)
                        new_elements.append(x)
                if len(new_elements) > 1:
                    current_size += len(new_elements) - 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                    improved = True
                    break
                else:
                    # Revert
                    for ne in new_elements:
                        S.discard(ne)
                    S.add(u)
            if improved:
                continue
            
            # Try 2-swap: remove two, try to add three or more
            improved2 = False
            S_list = list(S)
            random.shuffle(S_list)
            for i in range(len(S_list)):
                if time.time() > deadline - 3:
                    return best_S
                u = S_list[i]
                for j in range(i+1, len(S_list)):
                    v = S_list[j]
                    S.discard(u)
                    S.discard(v)
                    new_elements = []
                    remaining = [x for x in range(1, n+1) if x not in S]
                    random.shuffle(remaining)
                    for x in remaining:
                        if can_add(S, x):
                            S.add(x)
                            new_elements.append(x)
                    if len(new_elements) > 2:
                        current_size += len(new_elements) - 2
                        if current_size > best_size:
                            best_size = current_size
                            best_S = set(S)
                        improved2 = True
                        break
                    else:
                        for ne in new_elements:
                            S.discard(ne)
                        S.add(u)
                        S.add(v)
                    if improved2:
                        break
                if improved2:
                    break
            
            if not improved2:
                break
        
        return best_S
    
    # Multiple restarts
    best_set = set()
    best_size = 0
    
    rng = random.Random(42)
    
    for restart in range(50000):
        if time.time() > deadline - 5:
            break
        
        candidates = list(range(1, n+1))
        rng.shuffle(candidates)
        
        current = greedy_build(candidates)
        improved = local_search(current, max_iters=100)
        
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = set(improved)
            if best_size >= 110:
                break
    
    if time.time() < deadline - 5 and best_set:
        final = local_search(best_set, max_iters=500)
        if len(final) > best_size:
            best_set = set(final)
            best_size = len(final)
    
    if not best_set:
        return [1]
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
