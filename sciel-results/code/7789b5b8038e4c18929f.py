"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # 85s safety margin for 90s limit
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Precompute all bad triples: (a, b, c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.add(frozenset([a, b, c]))
    
    # Precompute: for each x, the list of bad triples containing x
    bad_triples_by_elem = {x: [] for x in range(1, N + 1)}
    for bt in bad_triples:
        for x in bt:
            bad_triples_by_elem[x].append(bt)
    
    # Precompute conflict pairs: for each x, which elements y form a "witness" pair
    # i.e., there exists z such that {x,y,z} is a bad triple
    # This helps us quickly check if adding x would create a conflict with existing S
    conflict_pairs = {x: set() for x in range(1, N + 1)}
    for bt in bad_triples:
        elems = list(bt)
        for i in range(3):
            for j in range(i+1, 3):
                conflict_pairs[elems[i]].add(elems[j])
                conflict_pairs[elems[j]].add(elems[i])
    
    def can_add_fast(x, S):
        """Check if x can be added to set S without creating a bad triple.
        Uses conflict_pairs for faster checking."""
        for y in conflict_pairs[x]:
            if y in S:
                # Need to check if there's a z in S such that {x,y,z} is bad
                for bt in bad_triples_by_elem[x]:
                    if y in bt:
                        z = [v for v in bt if v != x and v != y][0]
                        if z in S:
                            return False
        return True
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        for bt in bad_triples_by_elem[x]:
            other = [y for y in bt if y != x]
            if other[0] in S and other[1] in S:
                return False
        return True
    
    # Count conflicts for each element (fewer conflicts = easier to add)
    conflict_count = {x: len(conflict_pairs[x]) for x in range(1, N + 1)}
    
    def fill_set(S):
        """Greedily add all possible elements to S."""
        improved = True
        while improved:
            improved = False
            for x in elements:
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
        return S
    
    def local_search(S, max_swaps=50):
        """Try 1-out 1-in swap moves to improve the set."""
        S = set(S)
        for _ in range(max_swaps):
            if time.time() > deadline - 3:
                break
            if not S:
                break
            improved = False
            # Try removing each element and see if we can add more
            candidates_to_remove = list(S)
            random.shuffle(candidates_to_remove)
            for to_remove in candidates_to_remove:
                S.remove(to_remove)
                best_size_after = len(S)
                best_S_after = set(S)
                
                # Try adding each candidate
                for x in elements:
                    if x != to_remove and x not in S and can_add(x, S):
                        S.add(x)
                        S_copy = set(S)
                        fill_set(S_copy)
                        if len(S_copy) > best_size_after:
                            best_size_after = len(S_copy)
                            best_S_after = set(S_copy)
                        S.remove(x)
                
                if best_size_after > len(S):
                    S = best_S_after
                    improved = True
                    break
                else:
                    S.add(to_remove)
            
            if not improved:
                break
        
        return S
    
    best = set()
    best_size = 0
    
    # Strategy 1: Start from odd numbers (structurally avoids many conflicts)
    odds = set(x for x in elements if x % 2 == 1)
    fill_set(odds)
    odds = local_search(odds, max_swaps=30)
    if len(odds) > best_size:
        best_size = len(odds)
        best = set(odds)
    
    # Strategy 2: Start from elements with fewest conflicts
    sorted_by_conflicts = sorted(elements, key=lambda x: conflict_count[x])
    S2 = set()
    for x in sorted_by_conflicts:
        if can_add(x, S2):
            S2.add(x)
    fill_set(S2)
    S2 = local_search(S2, max_swaps=30)
    if len(S2) > best_size:
        best_size = len(S2)
        best = set(S2)
    
    # Multiple randomized greedy passes with local search
    num_starts = 300
    for start in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Randomize the order with bias toward low-conflict elements
        order = elements[:]
        random.shuffle(order)
        
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        
        # Fill
        fill_set(S)
        
        # Local search: try to improve
        if time.time() > deadline - 5:
            if len(S) > best_size:
                best_size = len(S)
                best = set(S)
            break
        
        S = local_search(S, max_swaps=40)
        
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Final: try to add any remaining elements
    if time.time() < deadline - 2:
        for x in elements:
            if x not in best and can_add(x, best):
                best.add(x)
    
    if not best:
        best = {1}
    
    return list(best)
# EVOLVE-BLOCK-END
