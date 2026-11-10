"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    # Map element to list of indices of bad triples containing it
    elem_to_triples = {i: [] for i in range(1, n + 1)}
    for idx, t in enumerate(bad_triples):
        for x in t:
            elem_to_triples[x].append(idx)
            
    def can_add(x, current_set):
        """Check if x can be added to current_set without creating a bad triple."""
        for t_idx in elem_to_triples[x]:
            t = bad_triples[t_idx]
            others = [y for y in t if y != x]
            if others[0] in current_set and others[1] in current_set:
                return False
        return True

    best_size = 0
    best_set = set()
    
    # Number of restarts
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Greedy Hitting Set: remove elements to eliminate all bad triples
        in_set = [True] * (n + 1)
        current_set = set(range(1, n + 1))
        
        # Active triples: initially all
        active_triples = set(range(len(bad_triples)))
        
        removed = []
        
        while active_triples:
            if time.time() > deadline:
                break
            
            # Recompute counts from active_triples
            counts = defaultdict(int)
            for t_idx in active_triples:
                a, b, c = bad_triples[t_idx]
                if in_set[a]: counts[a] += 1
                if in_set[b]: counts[b] += 1
                if in_set[c]: counts[c] += 1
                
            if not counts:
                break
                
            max_count = max(counts.values())
            candidates = [x for x, c in counts.items() if c == max_count]
            
            # Tie-breaking: random
            x = random.choice(candidates)
            
            # Remove x
            in_set[x] = False
            current_set.remove(x)
            removed.append(x)
            
            # Update active_triples
            to_remove = set()
            for t_idx in elem_to_triples[x]:
                if t_idx in active_triples:
                    to_remove.add(t_idx)
            active_triples.difference_update(to_remove)
        
        # current_set is now valid.
        
        # Local Search Phase 1: Add back removed elements
        for _ in range(3): # 3 passes of add-back
            if time.time() > deadline:
                break
            random.shuffle(removed)
            for x in removed[:]:
                if time.time() > deadline:
                    break
                if in_set[x]:
                    continue
                if can_add(x, current_set):
                    current_set.add(x)
                    in_set[x] = True
                    removed.remove(x)
                    
        # Local Search Phase 2: 1-for-1 swap local search
        # Try to swap an element out to bring in a better one
        for _ in range(3): # 3 passes of swap
            if time.time() > deadline:
                break
            improved = False
            # Try swapping each element in current_set with each removed element
            for x_out in list(current_set):
                if time.time() > deadline:
                    break
                for x_in in removed:
                    if time.time() > deadline:
                        break
                    # Check if we can remove x_out and add x_in
                    # First, check if x_in can be added to (current_set - {x_out})
                    temp_set = current_set - {x_out}
                    if can_add(x_in, temp_set):
                        # Perform the swap
                        current_set.remove(x_out)
                        in_set[x_out] = False
                        current_set.add(x_in)
                        in_set[x_in] = True
                        removed.remove(x_in)
                        removed.append(x_out)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    if not best_set:
        return [1]
        
    return list(best_set)
# EVOLVE-BLOCK-END
