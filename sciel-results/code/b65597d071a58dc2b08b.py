"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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
    
    # Find all forbidden triples: {a,b,c} where lcm(a,b)==lcm(a,c)==lcm(b,c)
    # For efficiency, group by LCM value
    # For each L, find all pairs (a,b) with lcm(a,b)=L, then check triples within that group
    from collections import defaultdict
    lcm_pairs = defaultdict(list)
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            L = lcm_table[i][j]
            lcm_pairs[L].append((i, j))
    
    # For each L, for each pair of pairs sharing an element, check if they form a triple
    # Actually, for a triple {a,b,c} to have all pairwise LCMs = L, 
    # we need lcm(a,b)=L, lcm(a,c)=L, lcm(b,c)=L
    # So for each L, collect all numbers that appear in some pair with LCM L
    # Then check all triples from that set
    
    forbidden = set()
    for L, pairs in lcm_pairs.items():
        if len(pairs) < 2:
            continue
        # Get all elements involved
        elems = set()
        for a, b in pairs:
            elems.add(a)
            elems.add(b)
        elems = list(elems)
        if len(elems) < 3:
            continue
        # Check all triples from elems
        for tri in combinations(elems, 3):
            a, b, c = tri
            if (lcm_table[a][b] == L and lcm_table[a][c] == L and lcm_table[b][c] == L):
                forbidden.add((a, b, c))
    
    # Now we have the set of forbidden triples. 
    # We need the largest subset with no forbidden triple.
    # Use greedy: sort by number of conflicts, pick greedily, then local search
    
    # Count how many forbidden triples each element is in
    conflict_count = [0]*(n+1)
    for tri in forbidden:
        for x in tri:
            conflict_count[x] += 1
    
    # Greedy: pick elements in order of fewest conflicts first
    elements = list(range(1, n+1))
    elements.sort(key=lambda x: conflict_count[x])
    
    selected = []
    selected_set = set()
    for x in elements:
        if time.time() > deadline:
            break
        # Check if adding x creates a forbidden triple with any two in selected
        ok = True
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                a, b = selected[i], selected[j]
                if (a, b, x) in forbidden or (a, x, b) in forbidden or (b, a, x) in forbidden:
                    # Normalize: check if {a,b,x} is in forbidden
                    t = tuple(sorted((a, b, x)))
                    if t in forbidden:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            selected.append(x)
            selected_set.add(x)
    
    # Local search: try to swap elements to improve
    # Try adding elements not in selected by removing one selected element
    remaining = [x for x in range(1, n+1) if x not in selected_set]
    
    def is_valid_subset(s):
        s_list = list(s)
        for tri in combinations(s_list, 3):
            if tri in forbidden:
                return False
        return True
    
    # Try to add one element at a time by removing a conflicting element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in remaining[:]:
            if x in selected_set:
                continue
            # Find which selected elements conflict with x
            conflicts = []
            for a in selected:
                for b in selected:
                    if a >= b:
                        continue
                    t = tuple(sorted((a, b, x)))
                    if t in forbidden:
                        conflicts.append((a, b))
            
            if not conflicts:
                # Can add x directly
                selected.append(x)
                selected_set.add(x)
                remaining.remove(x)
                improved = True
                break
            
            # Try removing one element that's in a conflict to free up x
            # For each conflict pair, try removing one of them
            tried = set()
            for a, b in conflicts:
                for rem in [a, b]:
                    if rem in tried:
                        continue
                    tried.add(rem)
                    # Check if we can remove rem and add x
                    new_selected = [y for y in selected if y != rem]
                    new_set = set(new_selected)
                    new_set.add(x)
                    # Check validity of new_set
                    valid = True
                    for tri in combinations(sorted(new_set), 3):
                        if tri in forbidden:
                            valid = False
                            break
                    if valid:
                        selected = new_selected
                        selected_set = new_set
                        remaining.remove(x)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # Final local search: try swapping one element for another
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in remaining[:]:
                if x in selected_set:
                    continue
                for y in list(selected):
                    if y == x:
                        continue
                    new_selected = [z for z in selected if z != y]
                    new_selected.append(x)
                    new_set = set(new_selected)
                    # Check validity
                    valid = True
                    for tri in combinations(sorted(new_set), 3):
                        if tri in forbidden:
                            valid = False
                            break
                    if valid and len(new_selected) > len(selected):
                        selected = new_selected
                        selected_set = new_set
                        remaining.remove(x)
                        improved = True
                        break
                if improved:
                    break
    
    if not selected:
        return [1]
    
    return sorted(selected)
# EVOLVE-BLOCK-END
