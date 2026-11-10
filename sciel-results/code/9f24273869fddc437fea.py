"""Improved program for two Sidon sets with disjoint difference sets."""
import time
import random
from itertools import combinations, combinations_with_replacement

def _is_sidon(arr):
    """Check if arr is a Sidon set (all pair sums a+b with a<=b are unique)."""
    n = len(arr)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(arr):
    """Return set of nonzero differences |b-a| for a != b in arr."""
    d = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            d.add(abs(arr[j] - arr[i]))
    return d

def _greedy_construct(order, deadline, current_best_size, balanced=False):
    """
    Greedily build two Sidon sets with disjoint difference sets.
    order: list of candidate values to try
    balanced: if True, alternate which side to try first
    Returns (A, B) or None if deadline exceeded.
    """
    sets = [[], []]
    diff_sets = [set(), set()]
    sum_sets = [set(), set()]
    step = 0
    
    for x in order:
        if time.time() > deadline:
            return None
        
        # Determine side order: balanced alternates, otherwise prefer 0
        if balanced:
            side_order = (step % 2, 1 - step % 2)
        else:
            side_order = (0, 1)
        
        for side in side_order:
            if len(sets[side]) >= 64:
                continue
            # Check Sidon condition for this set
            ok = True
            for a in sets[side]:
                s = a + x
                if s in sum_sets[side]:
                    ok = False
                    break
            if ok:
                s = x + x
                if s in sum_sets[side]:
                    ok = False
            if not ok:
                continue
            # Check difference set disjointness
            new_diffs = set()
            for a in sets[side]:
                d = abs(x - a)
                if d == 0:
                    continue
                new_diffs.add(d)
                if d in diff_sets[1 - side]:
                    ok = False
                    break
            if not ok:
                continue
            
            # Tentatively add
            sets[side].append(x)
            for a in sets[side]:
                sum_sets[side].add(a + x)
            sum_sets[side].add(x + x)
            diff_sets[side] = diff_sets[side] | new_diffs
            
            # Check if this improves combined size
            combined = len(sets[0]) + len(sets[1])
            if combined > current_best_size[0]:
                current_best_size[0] = combined
                current_best_size[1] = (list(sets[0]), list(sets[1]))
            
            step += 1
            break  # Only add to one side per candidate
    
    return current_best_size[1] if current_best_size[1] else None

def _local_improve(A, B, deadline, current_best):
    """
    Try to improve by adding elements and swapping between sets.
    Returns improved (A, B) or original.
    """
    A = list(A)
    B = list(B)
    best_A, best_B = list(A), list(B)
    best_size = len(A) + len(B)
    
    # Precompute sum sets and diff sets for fast checking
    def _sum_set(S):
        return set(a + b for a in S for b in S if a <= b)
    
    def _can_add(x, S, sum_set, other_diffs):
        """Check if x can be added to S given current sum_set and other_diffs."""
        if len(S) >= 64:
            return False
        # Check Sidon: all pair sums with x must be new
        for a in S:
            if (a + x) in sum_set:
                return False
        if (x + x) in sum_set:
            return False
        # Check difference disjointness
        new_diffs = set()
        for a in S:
            d = abs(x - a)
            if d == 0:
                continue
            new_diffs.add(d)
            if d in other_diffs:
                return False
        return True
    
    # Try adding elements not in either set
    used = set(A) | set(B)
    candidates = [x for x in range(1, 257) if x not in used]
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        for side in (0, 1):
            S = best_A if side == 0 else best_B
            O = best_B if side == 0 else best_A
            sum_set = _sum_set(S)
            other_diffs = _diffs(O)
            
            if _can_add(x, S, sum_set, other_diffs):
                new_S = S + [x]
                new_size = len(new_S) + len(O)
                if new_size > best_size:
                    if _is_sidon(new_S) and _is_sidon(O):
                        d1 = _diffs(new_S)
                        d2 = _diffs(O)
                        if not (d1 & d2):
                            if side == 0:
                                best_A = new_S
                            else:
                                best_B = new_S
                            best_size = new_size
    
    # Try swap-based improvement: move element from larger set to smaller
    # This can free up difference slots
    for _ in range(3):  # A few passes
        if time.time() > deadline:
            break
        
        improved = False
        for side in (0, 1):
            if len(best_A) + len(best_B) <= best_size:
                break
            S = best_A if side == 0 else best_B
            O = best_B if side == 0 else best_A
            
            if len(S) <= len(O):
                continue
            
            # Try removing an element from S and adding it to O
            for i, elem in enumerate(S):
                if time.time() > deadline:
                    break
                
                new_S = S[:i] + S[i+1:]
                new_O = O + [elem]
                
                if not _is_sidon(new_S) or not _is_sidon(new_O):
                    continue
                
                d1 = _diffs(new_S)
                d2 = _diffs(new_O)
                if d1 & d2:
                    continue
                
                # Check if we can then add something new
                new_size = len(new_S) + len(new_O)
                if new_size >= best_size:
                    # Try to add a new element to the smaller set
                    used_new = set(new_S) | set(new_O)
                    for x in range(1, 257):
                        if x in used_new:
                            continue
                        for s2 in (0, 1):
                            S2 = new_S if s2 == 0 else new_O
                            O2 = new_O if s2 == 0 else new_S
                            if len(S2) >= 64:
                                continue
                            sum_set2 = _sum_set(S2)
                            other_diffs2 = _diffs(O2)
                            if _can_add(x, S2, sum_set2, other_diffs2):
                                final_S2 = S2 + [x]
                                final_size = len(final_S2) + len(O2)
                                if final_size > best_size:
                                    if _is_sidon(final_S2) and _is_sidon(O2):
                                        fd1 = _diffs(final_S2)
                                        fd2 = _diffs(O2)
                                        if not (fd1 & fd2):
                                            if s2 == 0:
                                                best_A = final_S2
                                                best_B = O2
                                            else:
                                                best_A = O2
                                                best_B = final_S2
                                            best_size = final_size
                                            improved = True
                                            break
                        if improved:
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return (best_A, best_B)

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s limit
    
    best_A, best_B = [], []
    best_size = 0
    
    # Multiple random restarts with different orderings
    for restart in range(60):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if restart % 4 == 0:
            order = list(range(1, 257))
            random.shuffle(order)
            balanced = False
        elif restart % 4 == 1:
            # Smaller first bias
            order = list(range(1, 257))
            random.shuffle(order)
            order.sort(key=lambda x: (x % 16, random.random()))
            balanced = False
        elif restart % 4 == 2:
            # Balanced: alternate sides
            order = list(range(1, 257))
            random.shuffle(order)
            balanced = True
        else:
            # Reverse order bias
            order = list(range(256, 0, -1))
            random.shuffle(order)
            balanced = (restart % 2 == 0)
        
        result = _greedy_construct(order, deadline, [best_size, (best_A, best_B)], balanced=balanced)
        if result is not None:
            A, B = result
            if len(A) + len(B) > best_size:
                best_A, best_B = A, B
                best_size = len(A) + len(B)
        
        # Local improvement
        if best_size > 0 and time.time() < deadline - 8:
            improved = _local_improve(best_A, best_B, deadline, [best_size, (best_A, best_B)])
            if improved is not None:
                A, B = improved
                if len(A) + len(B) > best_size:
                    best_A, best_B = A, B
                    best_size = len(A) + len(B)
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
