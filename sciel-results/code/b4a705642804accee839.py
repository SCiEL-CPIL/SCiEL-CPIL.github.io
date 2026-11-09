import time
import math
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute LCMs
    lcms = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcms[i][j] = i * j // math.gcd(i, j)
    
    # Precompute bad triples conflicts
    # conflict[i][j] is a bitset of k such that {i, j, k} is a bad triple
    # We only need i < j < k or i < k < j etc? 
    # Let's store for i < j, the set of k > j? No, k can be anywhere.
    # But in the search, we process elements in some order.
    # Let's just store all k != i, k != j.
    
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    
    # Iterate all triples
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if lcms[a][b] == lcms[a][c] == lcms[b][c]:
                    # {a, b, c} is a bad triple
                    # For pair (a, b), c is forbidden
                    conflict[a][b] |= (1 << c)
                    conflict[b][a] |= (1 << c)
                    # For pair (a, c), b is forbidden
                    conflict[a][c] |= (1 << b)
                    conflict[c][a] |= (1 << b)
                    # For pair (b, c), a is forbidden
                    conflict[b][c] |= (1 << a)
                    conflict[c][b] |= (1 << a)
                    
    # Greedy initial solution to set lower bound
    # Try a few random orderings or just natural order
    def greedy(order):
        selected = []
        available_mask = (1 << (N + 1)) - 2  # bits 1..N set
        # We need to check if adding x is valid given selected
        # A set S is valid if no bad triple is fully contained in S.
        # Greedy: add x if it doesn't form a bad triple with any pair in S.
        # This requires checking all pairs in S.
        
        # Better greedy:
        # Start with all numbers.
        # Repeatedly remove the number that is part of the most bad triples with remaining numbers?
        # Or just simple: iterate 1..128, add if valid.
        
        S = []
        for x in order:
            if not (1 << x) & available_mask:
                continue
            # Check if x can be added to S
            # x conflicts with S if there exist a, b in S such that {a, b, x} is bad.
            # This is equivalent to: for all a in S, (conflict[a][x] & S_mask) != 0?
            # No, conflict[a][x] is the set of c such that {a, x, c} is bad.
            # So if (conflict[a][x] & S_mask) is not empty, then there is a c in S forming a bad triple with a and x.
            # Wait, if c is in S, and a is in S, and x is being added, then {a, x, c} is a bad triple.
            # So yes, if for any a in S, (conflict[a][x] & S_mask) is non-zero, we cannot add x.
            
            can_add = True
            S_mask = 0
            for s in S:
                S_mask |= (1 << s)
                if conflict[a][x] & S_mask: # Wait, a is the loop var? No, a is element in S.
                    pass
            
            # Correct check:
            S_mask = 0
            for s in S:
                S_mask |= (1 << s)
            
            # For each s in S, if there exists t in S such that {s, t, x} is bad, then x cannot be added.
            # {s, t, x} bad implies t is in conflict[s][x].
            # So if (conflict[s][x] & S_mask) is non-zero, then such t exists.
            for s in S:
                if conflict[s][x] & S_mask:
                    can_add = False
                    break
            
            if can_add:
                S.append(x)
        return S

    # Initial greedy
    best_set = greedy(list(range(1, N + 1)))
    best_size = len(best_set)
    
    # Branch and Bound
    # State: current_selected (list), available_mask (int)
    # To speed up, we can pass the mask of selected elements too?
    # The check "can add x" requires iterating over selected.
    # If we store selected as a list, it's O(k) per check.
    
    def solve(selected, available_mask, depth):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > deadline:
            return
        
        if not available_mask:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
            return
            
        # Upper bound
        if len(selected) + available_mask.bit_count() <= best_size:
            return
            
        # Choose next element to branch on
        # Heuristic: pick element with highest conflict degree in available
        # Or just pick the first one. Let's try picking one that reduces available the most.
        # Actually, picking an element with many conflicts in `available` is good because
        # adding it will remove many from available (if we were to add it), 
        # but we are branching on "include" vs "exclude".
        
        # Let's just pick the lowest set bit in available_mask for simplicity and determinism.
        # Or random? Random might help avoid bad local structures.
        # Let's try lowest set bit first.
        
        # Get list of available indices
        # This is slow. Let's iterate bits.
        
        # Optimization: Precompute list of available indices?
        # Or just pick the first one.
        
        # Let's pick the first available element
        # Find first set bit
        first_bit = available_mask & -available_mask
        x = first_bit.bit_length() - 1
        
        # Branch 1: Include x
        new_selected = selected + [x]
        
        # Update available_mask
        # Remove x
        new_avail = available_mask & ~(1 << x)
        
        # Remove conflicts with new x
        # For each s in selected, remove conflict[s][x] from new_avail
        # Note: conflict[s][x] contains elements that form bad triples with s and x.
        # If any of those are in new_avail, they can no longer be added.
        # Also, we must ensure that x doesn't form a bad triple with any pair in selected.
        # We checked this in the "can_add" logic? 
        # In B&B, we assume the state is valid.
        # So before branching, we should ensure x is compatible with selected.
        # But we are branching on ALL available elements.
        # Wait, if x is in available_mask, it means it is compatible with selected?
        # Yes, if we maintain invariant: available_mask only contains elements compatible with selected.
        
        # Check compatibility of x with selected:
        # If there exist a, b in selected such that {a, b, x} is bad, x should not be in available.
        # How to maintain this invariant?
        # When we add y to selected, we remove from available any z such that {y, s, z} is bad for any s in selected.
        # This is what we did in the "Include x" step above:
        # "For each s in selected, remove conflict[s][x] from new_avail"
        # This removes z if {s, x, z} is bad.
        # But what if {s1, s2, x} is bad?
        # Then x should have been removed from available when s2 was added (if s1 was already there).
        # So the invariant holds IF we correctly update available when adding elements.
        
        # So, if x is in available_mask, it is safe to add.
        
        # Update new_avail
        for s in selected:
            new_avail &= ~conflict[s][x]
            
        solve(new_selected, new_avail, depth + 1)
        
        if time.time() - start_time > deadline:
            return
        
        # Branch 2: Exclude x
        solve(selected, new_avail, depth + 1) # Wait, new_avail is modified?
        # No, we should pass available_mask & ~(1<<x)
        solve(selected, available_mask & ~(1 << x), depth + 1)
        
    # Start search
    # Initial available is all numbers
    initial_avail = (1 << (N + 1)) - 2
    solve([], initial_avail, 0)
    
    return best_set

# EVOLVE-BLOCK-END
