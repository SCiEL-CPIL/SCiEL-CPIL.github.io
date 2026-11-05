"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 55.0  # Safety margin before 90s limit
    best_sets = ([], [])
    best_size = 0

    def _can_add(S, sums, diffs, other_diffs, x):
        """Check if x can be added to set S."""
        # Sidon: all pair sums must be unique
        for a in S:
            s = a + x
            if s in sums:
                return False
        if (x + x) in sums:
            return False
        # Difference disjointness
        for a in S:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
        return True

    def _add(S, sums, diffs, x):
        """Add x to set S, updating sums and diffs."""
        S.append(x)
        for a in S[:-1]:
            sums.add(a + x)
            diffs.add(x - a)
            diffs.add(a - x)
        sums.add(x + x)

    def _rebuild_side(sets, set_sums, set_diffs, side):
        """Rebuild sums and diffs for one side from scratch."""
        S = sets[side]
        set_sums[side] = set()
        set_diffs[side] = set()
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                if i <= j:
                    set_sums[side].add(a + b)
            for b in S:
                if a != b:
                    set_diffs[side].add(a - b)
                    set_diffs[side].add(b - a)

    def _greedy_one_ordering(order, deadline, side_bias=None):
        """
        Single forward pass over the given ordering.
        side_bias: if 'first', always try set 0 first; if 'second', try set 1 first;
                   if None, try set 0 first (default).
        """
        sets = [[], []]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Determine order of sides to try
            if side_bias == 'second':
                sides = (1, 0)
            else:
                sides = (0, 1)
            
            for side in sides:
                S = sets[side]
                other_diffs = set_diffs[1 - side]
                
                if _can_add(S, set_sums[side], set_diffs[side], other_diffs, x):
                    _add(S, set_sums[side], set_diffs[side], x)
                    break
        
        return sets

    def _local_search(sets, deadline, max_iter=500):
        """
        Local search: try to improve by removing elements and re-adding,
        or by swapping elements between sets.
        """
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        for side in (0, 1):
            _rebuild_side(sets, set_sums, set_diffs, side)
        
        current_size = len(sets[0]) + len(sets[1])
        all_candidates = list(range(1, 257))
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding a random unplaced element
            unplaced = [x for x in all_candidates if x not in sets[0] and x not in sets[1]]
            if unplaced:
                random.shuffle(unplaced)
                for x in unplaced[:30]:
                    for side in (0, 1):
                        S = sets[side]
                        other_diffs = set_diffs[1 - side]
                        if _can_add(S, set_sums[side], set_diffs[side], other_diffs, x):
                            _add(S, set_sums[side], set_diffs[side], x)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element from one set and re-adding it to the other
            if not improved and current_size > 4:
                for side in (0, 1):
                    if len(sets[side]) > 1:
                        idx = random.randint(0, len(sets[side]) - 1)
                        x = sets[side][idx]
                        other_side = 1 - side
                        
                        # Check if x can be added to the other set
                        S_other = sets[other_side]
                        if _can_add(S_other, set_sums[other_side], set_diffs[other_side], set_diffs[side], x):
                            # Remove x from current set
                            sets[side].pop(idx)
                            # Rebuild sums and diffs for current set
                            _rebuild_side(sets, set_sums, set_diffs, side)
                            # Add x to other set
                            _add(S_other, set_sums[other_side], set_diffs[other_side], x)
                            improved = True
                            break
            
            # Try swapping two elements between sets
            if not improved and len(sets[0]) > 1 and len(sets[1]) > 1:
                idx0 = random.randint(0, len(sets[0]) - 1)
                idx1 = random.randint(0, len(sets[1]) - 1)
                x0 = sets[0][idx0]
                x1 = sets[1][idx1]
                
                # Check if x1 can replace x0 in set 0 and x0 can replace x1 in set 1
                # First, check if x1 can be added to set 0 (after removing x0)
                # This is complex, so we just check if x1 can be added to set 0 as-is
                # and x0 can be added to set 1 as-is
                if _can_add(sets[0], set_sums[0], set_diffs[0], set_diffs[1], x1) and \
                   _can_add(sets[1], set_sums[1], set_diffs[1], set_diffs[0], x0):
                    # Perform the swap
                    sets[0].pop(idx0)
                    sets[1].pop(idx1)
                    _rebuild_side(sets, set_sums, set_diffs, 0)
                    _rebuild_side(sets, set_sums, set_diffs, 1)
                    _add(sets[0], set_sums[0], set_diffs[0], x1)
                    _add(sets[1], set_sums[1], set_diffs[1], x0)
                    improved = True
            
            if not improved:
                break
        
        return sets

    def _polish(sets, deadline):
        """Final high-precision polish pass."""
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        for side in (0, 1):
            _rebuild_side(sets, set_sums, set_diffs, side)
        
        all_candidates = list(range(1, 257))
        improved = True
        polish_iter = 0
        
        while improved and time.time() < deadline and polish_iter < 100:
            polish_iter += 1
            improved = False
            
            # Try adding any unplaced element
            unplaced = [x for x in all_candidates if x not in sets[0] and x not in sets[1]]
            random.shuffle(unplaced)
            for x in unplaced:
                for side in (0, 1):
                    S = sets[side]
                    other_diffs = set_diffs[1 - side]
                    if _can_add(S, set_sums[side], set_diffs[side], other_diffs, x):
                        _add(S, set_sums[side], set_diffs[side], x)
                        improved = True
                        break
                if improved:
                    break
            
            # Try moving elements between sets
            if not improved:
                for side in (0, 1):
                    if len(sets[side]) > 1:
                        other_side = 1 - side
                        for idx in range(len(sets[side])):
                            x = sets[side][idx]
                            if _can_add(sets[other_side], set_sums[other_side], set_diffs[other_side], set_diffs[side], x):
                                sets[side].pop(idx)
                                _rebuild_side(sets, set_sums, set_diffs, side)
                                _add(sets[other_side], set_sums[other_side], set_diffs[other_side], x)
                                improved = True
                                break
                        if improved:
                            break
        
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
    
    # Try with side_bias='second'
    sets = _greedy_one_ordering(base_order, deadline, side_bias='second')
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = (list(sets[0]), list(sets[1]))
    
    # Try multiple random permutations
    random.seed(42)
    while time.time() < deadline - 8.0:
        random_order = base_order[:]
        random.shuffle(random_order)
        
        sets = _greedy_one_ordering(random_order, deadline)
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = (list(sets[0]), list(sets[1]))
    
    # Apply local search to the best solution
    if best_size > 0 and time.time() < deadline - 5.0:
        polished = _local_search([list(best_sets[0]), list(best_sets[1])], deadline - 3.0, max_iter=500)
        size = len(polished[0]) + len(polished[1])
        if size > best_size:
            best_size = size
            best_sets = (list(polished[0]), list(polished[1]))
    
    # Final polish pass
    if best_size > 0 and time.time() < deadline - 1.0:
        polished = _polish([list(best_sets[0]), list(best_sets[1])], deadline)
        size = len(polished[0]) + len(polished[1])
        if size > best_size:
            best_size = size
            best_sets = (list(polished[0]), list(polished[1]))
    
    # Ensure nonempty
    if not best_sets[0] and not best_sets[1]:
        best_sets = ([1], [2])
    elif not best_sets[0]:
        best_sets[0] = [1]
    elif not best_sets[1]:
        best_sets[1] = [1]
    
    return best_sets
# EVOLVE-BLOCK-END
