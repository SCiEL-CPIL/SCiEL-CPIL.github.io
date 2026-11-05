import time
from itertools import combinations
from math import lcm
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM for all pairs
    lcm_pair = {}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_pair[(i, j)] = lcm(i, j)
    
    # Find all bad triples (a, b, c) where a < b < c and
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    triple_to_elems = {}  # map from frozenset to triple for quick lookup
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_pair[(a, b)]
            for c in range(b + 1, n + 1):
                lac = lcm_pair[(a, c)]
                if lac != lab:
                    continue
                lbc = lcm_pair[(b, c)]
                if lbc == lab:
                    t = (a, b, c)
                    bad_triples.append(t)
    
    if time.time() > deadline:
        return numbers
    
    # Build incidence: element -> list of triple indices it belongs to
    elem_to_triples = defaultdict(list)
    for idx, t in enumerate(bad_triples):
        for x in t:
            elem_to_triples[x].append(idx)
    
    # Use a greedy algorithm with local search
    # Start with all elements, then iteratively remove the element
    # that breaks the most remaining bad triples
    
    active_triples = [True] * len(bad_triples)
    counts = {x: 0 for x in numbers}
    for idx, t in enumerate(bad_triples):
        for x in t:
            counts[x] += 1
    
    removed = set()
    
    while True:
        if time.time() > deadline:
            break
        
        # Find active triple count
        active_count = sum(1 for at in active_triples if at)
        if active_count == 0:
            break
        
        # Find the element (not yet removed) with the highest count
        best_elem = None
        best_count = -1
        for x in numbers:
            if x not in removed and counts[x] > best_count:
                best_count = counts[x]
                best_elem = x
        
        if best_elem is None or best_count <= 0:
            break
        
        # Remove best_elem
        removed.add(best_elem)
        
        # Update: for each active triple containing best_elem, mark it inactive
        # and decrement counts for the other two elements
        for idx in elem_to_triples[best_elem]:
            if active_triples[idx]:
                active_triples[idx] = False
                t = bad_triples[idx]
                for x in t:
                    if x != best_elem:
                        counts[x] -= 1
    
    result = [x for x in numbers if x not in removed]
    
    # Local search: try to add back removed elements if possible
    # For each removed element, check if adding it back creates any new bad triple
    # among currently present elements
    
    current_set = set(result)
    
    # Try adding back removed elements in order of smallest value (or some heuristic)
    for x in sorted(removed):
        if time.time() > deadline:
            break
        
        # Check if adding x to current_set creates any bad triple
        # A bad triple involving x would be (x, y, z) where y, z are in current_set
        # and lcm(x,y) == lcm(x,z) == lcm(y,z)
        
        can_add = True
        # We need to check all pairs (y, z) in current_set
        # This could be O(|current_set|^2) which is at most 128^2 = 16384
        # For each pair, check if lcm(x,y) == lcm(x,z) == lcm(y,z)
        
        # Optimization: only check pairs where lcm(x,y) == lcm(x,z)
        # Group elements in current_set by lcm(x, element)
        
        groups = defaultdict(list)
        for y in current_set:
            ly = lcm(x, y)
            groups[ly].append(y)
        
        for L, group in groups.items():
            if len(group) >= 2:
                # Check all pairs in group
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        y = group[i]
                        z = group[j]
                        if lcm(y, z) == L:
                            can_add = False
                            break
                    if not can_add:
                        break
            if not can_add:
                break
        
        if can_add:
            current_set.add(x)
            result = list(current_set)
    
    # Ensure non-empty
    if not result:
        return [1]
    
    return result
