"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 90s limit
    n = 256
    base_order = list(range(1, n + 1))
    
    # Precompute for speed
    all_elements = set(base_order)
    
    def _is_valid_sidon(S):
        """Check if S is a Sidon set (all pair sums a+b with a<=b are unique)."""
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def _get_sums(S):
        """Return the set of all pair sums a+b with a<=b for Sidon set S."""
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                sums.add(a + b)
        return sums

    def _get_diffs(S):
        """Return the set of all nonzero differences a-b for a,b in S, a!=b."""
        diffs = set()
        for a in S:
            for b in S:
                if a != b:
                    diffs.add(a - b)
        return diffs

    def _greedy_construct(order, deadline, prefer_balance=True):
        """
        Greedy construction: iterate through order, for each element try to add
        to set A or set B (whichever allows it), maintaining Sidon + disjoint diffs.
        """
        sets = [[], []]
        sums = [set(), set()]
        diffs = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            placed = False
            # Try set 0 first, then set 1 (or balance based on sizes)
            if prefer_balance:
                sides = [0, 1] if len(sets[0]) <= len(sets[1]) else [1, 0]
            else:
                sides = [0, 1]
            
            for side in sides:
                S = sets[side]
                other_diffs = diffs[1 - side]
                
                # Check Sidon: new pair sums must be unique
                valid = True
                for a in S:
                    s = a + x
                    if s in sums[side]:
                        valid = False
                        break
                if valid and (x + x) in sums[side]:
                    valid = False
                if not valid:
                    continue
                
                # Check difference disjointness
                conflict = False
                for a in S:
                    if a != x:
                        d1 = x - a
                        d2 = a - x
                        if d1 in other_diffs or d2 in other_diffs:
                            conflict = True
                            break
                if conflict:
                    continue
                
                # Accept x into sets[side]
                sets[side].append(x)
                for a in S:
                    sums[side].add(a + x)
                    if a != x:
                        diffs[side].add(x - a)
                        diffs[side].add(a - x)
                sums[side].add(x + x)
                placed = True
                break
        
        return sets

    def _local_search(sets, deadline):
        """
        Local search: try swapping elements out and re-adding different ones.
        Also try simple additions of currently blocked elements after removals.
        """
        sets = [list(sets[0]), list(sets[1])]
        best_size = len(sets[0]) + len(sets[1])
        improved = True
        
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            # Try removing one element from each set and re-optimizing
            for side in (0, 1):
                if not sets[side]:
                    continue
                for idx in range(len(sets[side])):
                    if time.time() > deadline - 3.0:
                        break
                    
                    removed = sets[side].pop(idx)
                    
                    # Try to add back other elements in a greedy fashion
                    candidates = [x for x in base_order 
                                  if x not in sets[0] and x not in sets[1]]
                    random.shuffle(candidates)
                    
                    new_sets = [list(sets[0]), list(sets[1])]
                    new_sums = [_get_sums(new_sets[0]), _get_sums(new_sets[1])]
                    new_diffs = [_get_diffs(new_sets[0]), _get_diffs(new_sets[1])]
                    
                    for x in candidates:
                        if time.time() > deadline - 3.0:
                            break
                        placed = False
                        for s in (0, 1):
                            S = new_sets[s]
                            other_diffs = new_diffs[1 - s]
                            
                            valid = True
                            for a in S:
                                if a + x in new_sums[s]:
                                    valid = False
                                    break
                            if valid and (x + x) in new_sums[s]:
                                valid = False
                            if not valid:
                                continue
                            
                            conflict = False
                            for a in S:
                                if a != x:
                                    if (x - a) in other_diffs or (a - x) in other_diffs:
                                        conflict = True
                                        break
                            if conflict:
                                continue
                            
                            new_sets[s].append(x)
                            for a in S:
                                new_sums[s].add(a + x)
                                if a != x:
                                    new_diffs[s].add(x - a)
                                    new_diffs[s].add(a - x)
                            new_sums[s].add(x + x)
                            placed = True
                            break
                    
                    new_size = len(new_sets[0]) + len(new_sets[1])
                    if new_size > best_size:
                        sets = new_sets
                        best_size = new_size
                        improved = True
                        break
                    else:
                        sets[side].insert(idx, removed)
                if improved:
                    break
        
        return sets

    # Phase 1: Multiple greedy constructions with different orderings
    best_sets = None
    best_size = 0
    
    # Deterministic orders
    for order in [base_order, base_order[::-1], 
                  list(range(n, 0, -2)) + list(range(n-1, 0, -2)),  # evens desc then odds desc
                  list(range(1, n+1, 2)) + list(range(2, n+1, 2))]:  # odds asc then evens asc
        if time.time() > deadline - 10.0:
            break
        sets = _greedy_construct(order, deadline, prefer_balance=True)
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = (list(sets[0]), list(sets[1]))
    
    # Random orders
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 15.0:
            break
        order = base_order[:]
        random.shuffle(order)
        sets = _greedy_construct(order, deadline, prefer_balance=True)
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = (list(sets[0]), list(sets[1]))
    
    # Phase 2: Local search polish
    if best_sets and time.time() < deadline - 5.0:
        improved = _local_search(best_sets, deadline)
        size = len(improved[0]) + len(improved[1])
        if size > best_size:
            best_size = size
            best_sets = (list(improved[0]), list(improved[1]))
    
    # Ensure nonempty
    if best_sets is None or (not best_sets[0] and not best_sets[1]):
        best_sets = ([1], [2])
    elif not best_sets[0]:
        best_sets = ([1], list(best_sets[1]))
    elif not best_sets[1]:
        best_sets = (list(best_sets[0]), [1])
    
    return best_sets
# EVOLVE-BLOCK-END
