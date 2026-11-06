import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Cache result if available
    if hasattr(construct, '_precomputed') and construct._precomputed is not None:
        sol = construct._precomputed
        if _check_solution(sol):
            return sol

    # Search for minimal M
    # Lower bound estimate: 
    # For k elements, max diffs is k(k-1)/2.
    # We need 30 diffs. k=8 -> 28 diffs (not enough). k=9 -> 36 diffs.
    # So k must be at least 9.
    # For k=9, Sidon set in [0, M] implies M >= k(k-1) - 1 = 71? 
    # Actually, the bound is M >= k(k-1) - 1 is for perfect difference sets?
    # No, Sidon set in Z_N has size <= sqrt(N). Here we are in [0, M].
    # A known small Sidon set covering 1..30:
    # Let's search M from 50 to 100.
    
    best_solution = None
    
    # Try M from lower bound up
    # A reasonable lower bound for M covering 1..30 with k=9 is around 60-70.
    # Let's start from 55 to be safe and efficient.
    for M in range(55, 101):
        if time.time() > deadline - 5:
            break
            
        sol = _solve_for_M(M, deadline=deadline)
        
        if sol is not None:
            if _check_solution(sol):
                best_solution = sol
                break
    
    if best_solution is None:
        # Fallback: use a known valid set, even if not optimal
        # This set is a Sidon set. We need to verify it covers 1..30.
        # Example: {0, 1, 4, 10, 12, 17, 21, 25, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120}
        # This is an arithmetic progression with step 4? No, 0,1,4...
        # Let's just use a greedy construction as a fallback.
        best_solution = _greedy_construct()
        
    construct._precomputed = best_solution
    return best_solution

def _check_solution(A):
    if not A or 0 not in A:
        return False
    if len(A) > 32:
        return False
    diffs = set()
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            d = A[j] - A[i]
            if d in diffs:
                return False
            diffs.add(d)
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def _solve_for_M(M, deadline):
    """
    Find a Sidon set A subset of {0..M} containing 0, covering 1..30.
    """
    initial_A = [0]
    initial_used = set()
    initial_missing = set(range(1, 31))
    
    def search(A, used, missing):
        if time.time() > deadline:
            return None
            
        if not missing:
            return A[:]
            
        if len(A) > 32:
            return None
            
        # Heuristic: If we can't cover missing diffs with remaining elements, prune
        # Max new diffs from adding one element x is len(A)
        # But we need specific diffs.
        
        candidates = []
        for x in range(1, M + 1):
            if x in A:
                continue
                
            new_diffs = set()
            valid = True
            for a in A:
                d = abs(x - a)
                if d in used:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Count how many missing diffs this x covers
            coverage = 0
            for d in new_diffs:
                if d in missing:
                    coverage += 1
            
            # We prefer candidates that cover more missing diffs
            # Tie-break: smaller x? or larger x? 
            # Smaller x keeps the set compact, which might help with future constraints?
            # Or larger x might leave room?
            # Let's try smaller x first to keep M tight? But M is fixed.
            # Actually, we want to fill the gaps.
            candidates.append((coverage, x, new_diffs))
            
        if not candidates:
            return None
            
        # Sort by coverage (desc), then x (asc)
        candidates.sort(key=lambda item: (-item[0], item[1]))
        
        # Limit branching factor to avoid explosion
        # If coverage is 0, we are wasting a slot.
        # Only try top few candidates.
        max_tries = 20 if len(candidates) > 20 else len(candidates)
        
        for coverage, x, new_diffs in candidates[:max_tries]:
            new_A = A + [x]
            new_A.sort()
            new_used = set(used)
            new_used.update(new_diffs)
            
            new_missing = set(missing)
            for d in new_diffs:
                if d in new_missing:
                    new_missing.remove(d)
            
            res = search(new_A, new_used, new_missing)
            if res is not None:
                return res
                
        return None

    return search(initial_A, initial_used, initial_missing)

def _greedy_construct():
    a = [0]
    used = set()
    for _ in range(32):
        best_x = None
        best_cov = -1
        for x in range(1, 1000):
            if x in a:
                continue
            new_diffs = set()
            valid = True
            for ai in a:
                diff = abs(x - ai)
                if diff in used or diff in new_diffs:
                    valid = False
                    break
                new_diffs.add(diff)
            if valid:
                cov = sum(1 for d in new_diffs if d <= 30 and d not in used)
                if cov > best_cov:
                    best_cov = cov
                    best_x = x
                    best_diffs = new_diffs
        if best_x is None:
            break
        a.append(best_x)
        a.sort()
        used.update(best_diffs)
        if all(d in used for d in range(1, 31)):
            break
    return a
