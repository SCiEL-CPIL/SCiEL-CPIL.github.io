"""Improved program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random

def _can_add_to_set(x, aset, sums_set, diffs_set, other_diffs):
    """Check if x can be added to aset maintaining Sidon and disjoint diffs."""
    # Sidon check: x+x and x+a for all a in aset must not be in sums_set
    if (x + x) in sums_set:
        return False
    for a in aset:
        if (x + a) in sums_set:
            return False
    # Diff disjointness check
    for a in aset:
        d = x - a
        if d in other_diffs:
            return False
        d2 = a - x
        if d2 in other_diffs:
            return False
    return True

def _add_to_set(x, aset, sums_set, diffs_set):
    """Add x to aset, updating sums and diffs."""
    aset.append(x)
    sums_set.add(x + x)
    for a in aset:
        if a != x:
            sums_set.add(x + a)
            diffs_set.add(x - a)
            diffs_set.add(a - x)

def _greedy_with_backtrack(order, deadline, rng):
    """
    Greedy construction with occasional backtracking.
    Tries to assign each element to A or B, preferring the set with fewer elements
    to keep them balanced.
    """
    A = []
    B = []
    A_sums = set()
    B_sums = set()
    A_diffs = set()
    B_diffs = set()
    
    # Track assignments for potential backtracking
    assignments = []  # (x, which_set) for each element tried
    
    for x in order:
        if time.time() > deadline:
            break
        
        # Try A first if B is larger, else B first (balance)
        if len(A) <= len(B):
            try_A_first = True
        else:
            try_A_first = False
        
        placed = False
        
        if try_A_first:
            if _can_add_to_set(x, A, A_sums, A_diffs, B_diffs):
                _add_to_set(x, A, A_sums, A_diffs)
                assignments.append((x, 'A'))
                placed = True
            elif _can_add_to_set(x, B, B_sums, B_diffs, A_diffs):
                _add_to_set(x, B, B_sums, B_diffs)
                assignments.append((x, 'B'))
                placed = True
        else:
            if _can_add_to_set(x, B, B_sums, B_diffs, A_diffs):
                _add_to_set(x, B, B_sums, B_diffs)
                assignments.append((x, 'B'))
                placed = True
            elif _can_add_to_set(x, A, A_sums, A_diffs, B_diffs):
                _add_to_set(x, A, A_sums, A_diffs)
                assignments.append((x, 'A'))
                placed = True
        
        # If not placed, we could try backtracking, but for speed we skip
        # In a more advanced version, we'd backtrack here
    
    return A, B, A_sums, A_diffs, B_sums, B_diffs

def _post_greedy_optimize(A, B, A_sums, A_diffs, B_sums, B_diffs, deadline):
    """
    After greedy, try to add more elements and do local swaps.
    """
    A = list(A)
    B = list(B)
    A_sums = set(A_sums)
    B_sums = set(B_sums)
    A_diffs = set(A_diffs)
    B_diffs = set(B_diffs)
    
    best_size = len(A) + len(B)
    best_A, best_B = A[:], B[:]
    best_As, best_Bs = set(A_sums), set(B_sums)
    best_Ad, best_Bd = set(A_diffs), set(B_diffs)
    
    # Try to add unused elements
    used = set(A) | set(B)
    candidates = [x for x in range(1, 257) if x not in used]
    rng = random.Random()
    rng.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        # Try adding to A
        if _can_add_to_set(x, A, A_sums, A_diffs, B_diffs):
            _add_to_set(x, A, A_sums, A_diffs)
            if len(A) + len(B) > best_size:
                best_size = len(A) + len(B)
                best_A, best_B = A[:], B[:]
                best_As, best_Bs = set(A_sums), set(B_sums)
                best_Ad, best_Bd = set(A_diffs), set(B_diffs)
            continue
        
        # Try adding to B
        if _can_add_to_set(x, B, B_sums, B_diffs, A_diffs):
            _add_to_set(x, B, B_sums, B_diffs)
            if len(A) + len(B) > best_size:
                best_size = len(A) + len(B)
                best_A, best_B = A[:], B[:]
                best_As, best_Bs = set(A_sums), set(B_sums)
                best_Ad, best_Bd = set(A_diffs), set(B_diffs)
    
    # Local search: try swapping one element out and adding another in
    # For each element in A, try removing it and adding a candidate
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try improving A
        for i, a in enumerate(list(A)):
            if time.time() > deadline:
                break
            # Remove a from A
            A_temp = A[:i] + A[i+1:]
            A_sums_temp = set()
            A_diffs_temp = set()
            for u in A_temp:
                A_sums_temp.add(u + u)
                for v in A_temp:
                    if v != u:
                        A_sums_temp.add(u + v)
                        A_diffs_temp.add(u - v)
                        A_diffs_temp.add(v - u)
            
            # Try adding candidates to A_temp
            for x in range(1, 257):
                if x in set(A_temp) | set(B):
                    continue
                if _can_add_to_set(x, A_temp, A_sums_temp, A_diffs_temp, B_diffs):
                    A_new = A_temp + [x]
                    A_sums_new = set(A_sums_temp)
                    A_diffs_new = set(A_diffs_temp)
                    A_sums_new.add(x + x)
                    for u in A_temp:
                        A_sums_new.add(x + u)
                        A_diffs_new.add(x - u)
                        A_diffs_new.add(u - x)
                    
                    if len(A_new) + len(B) > best_size:
                        best_size = len(A_new) + len(B)
                        best_A, best_B = A_new[:], B[:]
                        best_As, best_Bs = A_sums_new, set(B_sums)
                        best_Ad, best_Bd = A_diffs_new, set(B_diffs)
                        A, B = A_new, B
                        A_sums, B_sums = A_sums_new, B_sums
                        A_diffs, B_diffs = A_diffs_new, B_diffs
                        improved = True
                        break
        
        # Try improving B similarly
        for i, b in enumerate(list(B)):
            if time.time() > deadline:
                break
            B_temp = B[:i] + B[i+1:]
            B_sums_temp = set()
            B_diffs_temp = set()
            for u in B_temp:
                B_sums_temp.add(u + u)
                for v in B_temp:
                    if v != u:
                        B_sums_temp.add(u + v)
                        B_diffs_temp.add(u - v)
                        B_diffs_temp.add(v - u)
            
            for x in range(1, 257):
                if x in set(A) | set(B_temp):
                    continue
                if _can_add_to_set(x, B_temp, B_sums_temp, B_diffs_temp, A_diffs):
                    B_new = B_temp + [x]
                    B_sums_new = set(B_sums_temp)
                    B_diffs_new = set(B_diffs_temp)
                    B_sums_new.add(x + x)
                    for u in B_temp:
                        B_sums_new.add(x + u)
                        B_diffs_new.add(x - u)
                        B_diffs_new.add(u - x)
                    
                    if len(A) + len(B_new) > best_size:
                        best_size = len(A) + len(B_new)
                        best_A, best_B = A[:], B_new[:]
                        best_As, best_Bs = set(A_sums), B_sums_new
                        best_Ad, best_Bd = set(A_diffs), B_diffs_new
                        A, B = A, B_new
                        A_sums, B_sums = A_sums, B_sums_new
                        A_diffs, B_diffs = A_diffs, B_diffs_new
                        improved = True
                        break
    
    return best_A, best_B

def _validate_final(A, B):
    """Final validation of Sidon and disjoint diffs."""
    if not A or not B:
        return False
    
    # Check Sidon for A
    sums_A = set()
    for i, a in enumerate(A):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums_A:
                return False
            sums_A.add(s)
    
    # Check Sidon for B
    sums_B = set()
    for i, b in enumerate(B):
        for j in range(i, len(B)):
            s = B[i] + B[j]
            if s in sums_B:
                return False
            sums_B.add(s)
    
    # Check disjoint diffs
    diffs_A = set()
    for a1 in A:
        for a2 in A:
            if a1 != a2:
                diffs_A.add(a1 - a2)
    
    diffs_B = set()
    for b1 in B:
        for b2 in B:
            if b1 != b2:
                diffs_B.add(b1 - b2)
    
    if diffs_A & diffs_B:
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 90s limit
    n = 256
    
    best_A = [1]
    best_B = [2]
    best_size = 2
    
    rng = random.Random(12345)
    
    # Multiple strategies with different orderings
    strategies = []
    
    # Strategy 1: Random orderings
    for _ in range(15):
        order = list(range(1, n + 1))
        rng.shuffle(order)
        strategies.append(('random', order))
    
    # Strategy 2: Sorted ascending
    strategies.append(('asc', list(range(1, n + 1))))
    
    # Strategy 3: Sorted descending
    strategies.append(('desc', list(range(n, 0, -1))))
    
    # Strategy 4: Interleaved (odd then even)
    odd = [x for x in range(1, n + 1) if x % 2 == 1]
    even = [x for x in range(1, n + 1) if x % 2 == 0]
    interleaved = []
    for i in range(len(odd)):
        interleaved.append(odd[i])
        if i < len(even):
            interleaved.append(even[i])
    strategies.append(('interleave', interleaved))
    
    # Strategy 5: Large to small with random perturbation
    for _ in range(5):
        order = list(range(n, 0, -1))
        rng.shuffle(order)
        strategies.append(('rand_desc', order))
    
    for strat_name, order in strategies:
        if time.time() > deadline:
            break
        
        A, B, A_sums, A_diffs, B_sums, B_diffs = _greedy_with_backtrack(order, deadline, rng)
        size = len(A) + len(B)
        
        # Post-greedy optimization
        A_opt, B_opt = _post_greedy_optimize(A, B, A_sums, A_diffs, B_sums, B_diffs, deadline)
        size_opt = len(A_opt) + len(B_opt)
        
        if size_opt > best_size:
            best_size = size_opt
            best_A, best_B = A_opt[:], B_opt[:]
        
        # Check if we've hit the theoretical max (64+64=128)
        if best_size >= 120:
            break
    
    # Final validation
    if not _validate_final(best_A, best_B):
        # Fallback
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
