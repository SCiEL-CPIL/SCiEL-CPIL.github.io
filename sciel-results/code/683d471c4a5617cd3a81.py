import math
import time
import random
from itertools import combinations

def _score(U):
    U = sorted(list(set(U)))
    if 0 not in U:
        return 0.0
    if len(U) < 2:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    # Calculate sumset and difference set
    plus_set = set()
    minus_set = set()
    for a in U:
        for b in U:
            plus_set.add(a + b)
            minus_set.add(a - b)
            
    n_plus = len(plus_set)
    n_minus = len(minus_set)
    
    if n_plus == 0:
        return 0.0
        
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
        
    try:
        val = 1.0 + math.log(ratio) / math.log(2 * mx + 1)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return val

def _is_sidon(U):
    """Check if U is a Sidon set (all sums a_i+a_j with i<=j are distinct)."""
    sums = set()
    n = len(U)
    for i in range(n):
        for j in range(i, n):
            s = U[i] + U[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _local_search(U, max_iter=1000, deadline=None):
    """
    Try to improve the set U by moving one element to a smaller value
    while maintaining the Sidon property (or just improving score if relaxed).
    We prioritize reducing max(U).
    """
    current_U = sorted(U)
    current_score = _score(current_U)
    best_U = list(current_U)
    best_score = current_score
    
    n = len(current_U)
    
    # We want to minimize max(U) for a given structure, or improve ratio.
    # Try to move each element (except 0) to a smaller position.
    # We only consider positions < current max to try to shrink the span.
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        # Try moving each element i (i > 0)
        for i in range(1, n):
            original_val = current_U[i]
            # Try all possible new values less than original_val
            # To be efficient, we can just try values from 0 to original_val-1
            # But we must keep the set sorted and unique.
            # A better approach: try replacing current_U[i] with x in range(0, original_val)
            # and check if the new set is valid and better.
            
            # Optimization: Only try x that are not in current_U (excluding index i)
            existing = set(current_U)
            existing.remove(original_val)
            
            # We want to reduce max. So we focus on x < original_val.
            # If i is the max element, reducing it is very good.
            # If i is not max, reducing it might not change max, but might improve ratio?
            # Usually, smaller sets are denser, which might increase collisions (bad for Sidon).
            # So we primarily seek to reduce the MAX element.
            
            if original_val != max(current_U):
                # Moving non-max element: only do if it allows max to be reduced later?
                # For now, focus on moving the max element to a smaller value.
                continue
            
            # Move the max element
            for x in range(0, original_val):
                if x in existing:
                    continue
                
                new_U = list(current_U)
                new_U[i] = x
                new_U.sort()
                
                # Check Sidon property
                if not _is_sidon(new_U):
                    continue
                
                new_score = _score(new_U)
                if new_score > best_score + 1e-9:
                    best_score = new_score
                    best_U = new_U
                    current_U = new_U
                    improved = True
                    break # Restart loop with new current_U
        
        if not improved:
            break
            
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # List of known good Golomb rulers / Sidon sets
    # These are optimal or near-optimal for minimizing span for a given n
    candidates = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 28],
        [0, 1, 4, 10, 18, 24, 28, 34],
        [0, 1, 4, 10, 18, 24, 28, 34, 43],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65, 77],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65, 77, 88],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65, 77, 88, 100],
        [0, 1, 4, 10, 18, 24, 28, 34, 43, 54, 65, 77, 88, 100, 112],
    ]
    
    best_U = candidates[0]
    best_score = 0.0
    
    for U in candidates:
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
            
        # Try to improve this candidate via local search
        if time.time() < deadline - 5:
            improved_U, improved_score = _local_search(U, max_iter=200, deadline=deadline)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
                
    # Also try a few random searches if time permits
    if time.time() < deadline - 10:
        for n in range(4, 12):
            # Generate a random Sidon-like set
            # Start with greedy construction
            U = [0]
            last = 0
            for _ in range(n - 1):
                found = False
                # Try to add the smallest possible number
                for x in range(last + 1, last + 100):
                    new_U = U + [x]
                    if _is_sidon(new_U):
                        U = new_U
                        last = x
                        found = True
                        break
                if not found:
                    break
            
            if len(U) == n:
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_U = list(U)
                    
                if time.time() < deadline - 5:
                    improved_U, improved_score = _local_search(U, max_iter=100, deadline=deadline)
                    if improved_score > best_score:
                        best_score = improved_score
                        best_U = improved_U
                        
    return best_U
