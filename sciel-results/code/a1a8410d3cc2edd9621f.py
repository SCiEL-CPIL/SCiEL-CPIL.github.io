"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 55.0  # Safety margin before 90s limit
    best_sets = ([], [])
    best_size = 0
    
    def _greedy_one_ordering(order, deadline):
        """
        Single forward pass over the given ordering of candidates.
        For each candidate x, try to add it to set 0 first, then set 1.
        Maintain incremental sums and diffs for O(1)-ish checks.
        """
        sets = [[], []]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            placed = False
            for side in (0, 1):
                S = sets[side]
                other_diffs = set_diffs[1 - side]
                
                # Check Sidon condition: all new pair sums a+x (a in S) and x+x
                # must not collide with existing sums in set_sums[side]
                new_sums_valid = True
                for a in S:
                    s = a + x
                    if s in set_sums[side]:
                        new_sums_valid = False
                        break
                if new_sums_valid:
                    if (x + x) in set_sums[side]:
                        new_sums_valid = False
                
                if not new_sums_valid:
                    continue
                
                # Check difference disjointness: new diffs must not intersect other set's diffs
                new_diffs = set()
                conflict = False
                for a in S:
                    d1 = x - a
                    d2 = a - x
                    if d1 in other_diffs or d2 in other_diffs:
                        conflict = True
                        break
                    new_diffs.add(d1)
                    new_diffs.add(d2)
                
                if conflict:
                    continue
                
                # Accept x into sets[side]
                sets[side].append(x)
                for a in S:
                    set_sums[side].add(a + x)
                    set_diffs[side].add(x - a)
                    set_diffs[side].add(a - x)
                set_sums[side].add(x + x)
                placed = True
                break  # Only place in one set per candidate
        
        return sets
    
    base_order = list(range(1, 257))
    
    # First try natural (ascending) order - deterministic baseline
    sets = _greedy_one_ordering(base_order, deadline)
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = (list(sets[0]), list(sets[1]))
    
    # Try descending order
    sets = _greedy_one_ordering(base_order[::-1], deadline)
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = (list(sets[0]), list(sets[1]))
    
    # Try multiple random permutations
    random.seed(42)
    while time.time() < deadline - 2.0:
        random_order = base_order[:]
        random.shuffle(random_order)
        
        sets = _greedy_one_ordering(random_order, deadline)
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = (list(sets[0]), list(sets[1]))
    
    # Ensure nonempty
    if not best_sets[0] and not best_sets[1]:
        best_sets = ([1], [2])
    elif not best_sets[0]:
        best_sets[0] = [1]
    elif not best_sets[1]:
        best_sets[1] = [1]
    
    return best_sets
# EVOLVE-BLOCK-END
