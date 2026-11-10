"""Improved program for reciprocal Schur triple avoidance."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_conflicts(N=180):
    """
    Precompute the conflict graph: conflict[i] is the set of j such that
    i and j cannot both be in S because there exists some k in [1,N]
    with 1/k = 1/i + 1/j (i, j, k all distinct).
    
    Also precompute for each element, which pairs it forms conflicts with.
    """
    # conflict[i] = set of j that conflict with i (i.e., there exists k with 1/k=1/i+1/j)
    conflict = [set() for _ in range(N + 1)]
    
    for k in range(1, N + 1):
        k2 = k * k
        # Find all pairs (b, c) with b < c, b != k, c != k, b,c in [1,N]
        # such that (b-k)(c-k) = k^2
        for d in range(1, int(k2 ** 0.5) + 1):
            if k2 % d != 0:
                continue
            for dd in (d, k2 // d):
                if dd == 0:
                    continue
                # Positive case: b = k+dd, c = k + k2/dd
                b1 = k + dd
                c1 = k + k2 // dd
                if b1 != c1 and 1 <= b1 <= N and 1 <= c1 <= N and b1 != k and c1 != k:
                    if b1 < c1:
                        conflict[b1].add(c1)
                        conflict[c1].add(b1)
                    else:
                        conflict[c1].add(b1)
                        conflict[b1].add(c1)
                
                # Negative case: b = k-dd, c = k - k2/dd
                b2 = k - dd
                c2 = k - k2 // dd
                if b2 != c2 and 1 <= b2 <= N and 1 <= c2 <= N and b2 != k and c2 != k:
                    if b2 < c2:
                        conflict[b2].add(c2)
                        conflict[c2].add(b2)
                    else:
                        conflict[c2].add(b2)
                        conflict[b2].add(c2)
    
    return conflict

def _is_valid_set(S, conflict):
    """Check if S is a valid independent set in the conflict graph."""
    S_list = sorted(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            if S_list[j] in conflict[S_list[i]]:
                return False
    return True

def _count_conflicts_in_set(S, conflict):
    """Count the number of conflicting pairs in S."""
    count = 0
    S_list = sorted(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            if S_list[j] in conflict[S_list[i]]:
                count += 1
    return count

def _can_add(S, elem, conflict):
    """Check if adding elem to S creates any conflict."""
    for x in S:
        if elem in conflict[x]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180
    
    # Precompute conflict graph
    conflict = _precompute_conflicts(N)
    
    all_elements = set(range(1, N + 1))
    rng = random.Random(12345)
    
    best_set = None
    best_size = 0
    
    # Strategy: Multiple random restarts with greedy + local search
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Random initialization: start with a random subset, then greedily add
        # Or start with upper half and add randomly
        if restart == 0:
            # Deterministic start: upper half
            S = set(range(91, N + 1))
        elif restart == 1:
            # Upper half + some lower
            S = set(range(91, N + 1))
            for x in range(1, 91):
                if rng.random() < 0.3:
                    if _can_add(S, x, conflict):
                        S.add(x)
        else:
            # Random greedy: shuffle and add if possible
            S = set()
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            for x in candidates:
                if _can_add(S, x, conflict):
                    S.add(x)
        
        # Greedy addition: try to add any missing element
        missing = list(all_elements - S)
        rng.shuffle(missing)
        for elem in missing:
            if time.time() > deadline - 5:
                break
            if _can_add(S, elem, conflict):
                S.add(elem)
        
        # Local search: try swaps (remove one, add one or two)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try single additions (should already be done, but double-check)
            missing = list(all_elements - S)
            rng.shuffle(missing)
            for elem in missing:
                if time.time() > deadline - 5:
                    break
                if _can_add(S, elem, conflict):
                    S.add(elem)
                    improved = True
            
            # Try 1-for-1 swaps: remove x, add y (net 0 change in size, but may enable future additions)
            # More useful: try to find a y that can be added after removing x
            # Or: remove x, add y, then see if we can add more
            S_list = list(S)
            rng.shuffle(S_list)
            for x in S_list:
                if time.time() > deadline - 5:
                    break
                # Try removing x and adding various y
                S_without_x = S - {x}
                # Find y not in S that can be added to S_without_x
                candidates_y = [y for y in all_elements if y != x and y not in S_without_x]
                # Sort by "degree" - fewer conflicts first
                candidates_y.sort(key=lambda y: len(conflict[y] & S_without_x))
                for y in candidates_y:
                    if time.time() > deadline - 5:
                        break
                    if _can_add(S_without_x, y, conflict):
                        # After adding y, try to add more
                        S_new = S_without_x | {y}
                        # Try adding more
                        more_missing = [z for z in all_elements if z not in S_new and z != x]
                        more_missing.sort(key=lambda z: len(conflict[z] & S_new))
                        added_any = False
                        for z in more_missing:
                            if _can_add(S_new, z, conflict):
                                S_new.add(z)
                                added_any = True
                        if len(S_new) > len(S):
                            S = S_new
                            improved = True
                            break
                    if time.time() > deadline - 5:
                        break
            
            # Try 1-for-2 swaps: remove x, add y and z
            S_list = list(S)
            rng.shuffle(S_list)
            for x in S_list:
                if time.time() > deadline - 5:
                    break
                S_without_x = S - {x}
                candidates_y = [y for y in all_elements if y != x and y not in S_without_x]
                candidates_y.sort(key=lambda y: len(conflict[y] & S_without_x))
                for y in candidates_y[:20]:  # Limit to top 20 to save time
                    if time.time() > deadline - 5:
                        break
                    if _can_add(S_without_x, y, conflict):
                        S_with_y = S_without_x | {y}
                        candidates_z = [z for z in all_elements if z not in S_with_y and z != x and z != y]
                        candidates_z.sort(key=lambda z: len(conflict[z] & S_with_y))
                        for z in candidates_z[:10]:  # Limit to top 10
                            if _can_add(S_with_y, z, conflict):
                                S_new = S_with_y | {z}
                                # Try adding more
                                more_missing = [w for w in all_elements if w not in S_new]
                                more_missing.sort(key=lambda w: len(conflict[w] & S_new))
                                for w in more_missing:
                                    if _can_add(S_new, w, conflict):
                                        S_new.add(w)
                                if len(S_new) > len(S):
                                    S = S_new
                                    improved = True
                                    break
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Final verification
    if best_set is None:
        best_set = set(range(91, N + 1))
    
    return list(best_set)

# EVOLVE-BLOCK-END
