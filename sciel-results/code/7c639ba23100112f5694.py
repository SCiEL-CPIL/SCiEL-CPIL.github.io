import time
import random
import sys

START_TIME = time.time()
DEADLINE = 85.0  # safety margin before 90s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_validity(A):
    """Check if set A has cardinality-separated subset sums.
    For each cardinality k, track achievable sums. All sets must be disjoint."""
    n = len(A)
    if n == 0:
        return True
    
    # sums_by_k[k] = set of achievable sums using exactly k elements
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    
    for a in A:
        # Update in reverse to avoid using a twice
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                for s in sums_by_k[k]:
                    sums_by_k[k + 1].add(s + a)
    
    # Check pairwise disjointness
    seen = set()
    for k in range(n + 1):
        if sums_by_k[k] & seen:
            return False
        seen |= sums_by_k[k]
    return True

def check_validity_fast(A, current_sums_by_k):
    """Fast check assuming we already have sums_by_k for the current set.
    Returns (is_valid, new_sums_by_k) if we add element a to the set."""
    n = len(A)
    a = A[-1]
    
    new_sums_by_k = [set(s) for s in current_sums_by_k]
    for k in range(n - 1, -1, -1):
        if current_sums_by_k[k]:
            for s in current_sums_by_k[k]:
                new_sums_by_k[k + 1].add(s + a)
    
    seen = set()
    for k in range(n + 1):
        if new_sums_by_k[k] & seen:
            return False, None
        seen |= new_sums_by_k[k]
    return True, new_sums_by_k

def greedy_construct():
    """Greedy: try to add elements in ascending order."""
    A = []
    sums_by_k = [set() for _ in range(1)]
    sums_by_k[0].add(0)
    
    for a in range(1, 257):
        if is_done():
            break
        A.append(a)
        valid, new_sums_by_k = check_validity_fast(A, sums_by_k)
        if valid:
            sums_by_k = new_sums_by_k
        else:
            A.pop()
    
    return A

def greedy_construct_descending():
    """Greedy with descending order."""
    A = []
    sums_by_k = [set() for _ in range(1)]
    sums_by_k[0].add(0)
    
    for a in range(256, 0, -1):
        if is_done():
            break
        A.append(a)
        valid, new_sums_by_k = check_validity_fast(A, sums_by_k)
        if valid:
            sums_by_k = new_sums_by_k
        else:
            A.pop()
    
    return A

def local_search_improve(A):
    """Try to improve A by adding elements and swapping."""
    A = list(A)
    sums_by_k = [set() for _ in range(len(A) + 1)]
    sums_by_k[0].add(0)
    for a in A:
        for k in range(len(A) - 1, -1, -1):
            if sums_by_k[k]:
                for s in sums_by_k[k]:
                    sums_by_k[k + 1].add(s + a)
    
    improved = True
    while improved and not is_done():
        improved = False
        
        # Try to add a new element
        for add_val in range(1, 257):
            if add_val in A:
                continue
            A.append(add_val)
            valid, new_sums_by_k = check_validity_fast(A, sums_by_k)
            if valid:
                sums_by_k = new_sums_by_k
                improved = True
                break
            else:
                A.pop()
        
        if is_done():
            break
        
        # Try swapping elements
        for i in range(len(A)):
            if is_done():
                break
            old_val = A[i]
            for new_val in range(1, 257):
                if new_val in A:
                    continue
                A[i] = new_val
                # Rebuild sums_by_k from scratch for simplicity
                test_A = list(A)
                if check_validity(test_A):
                    # Try to add another element after swap
                    for add_val in range(1, 257):
                        if add_val in A:
                            continue
                        A.append(add_val)
                        if check_validity(A):
                            improved = True
                            break
                        else:
                            A.pop()
                    if improved:
                        break
                A[i] = old_val
            if improved:
                break
    
    return A

def random_search(num_trials=50):
    """Try random subsets and see which are valid."""
    best = []
    
    for trial in range(num_trials):
        if is_done():
            break
        
        size = random.randint(5, 20)
        A = random.sample(range(1, 257), size)
        
        if check_validity(A):
            if len(A) > len(best):
                best = list(A)
    
    return best

def construct():
    """Main entry point: try multiple strategies and return the best valid set."""
    best = []
    
    # Strategy 1: Greedy ascending
    A1 = greedy_construct()
    if len(A1) > len(best):
        best = list(A1)
    
    if is_done():
        return [int(x) for x in best]
    
    # Strategy 2: Greedy descending
    A2 = greedy_construct_descending()
    if len(A2) > len(best):
        best = list(A2)
    
    if is_done():
        return [int(x) for x in best]
    
    # Strategy 3: Local search improvement on best so far
    A3 = local_search_improve(best)
    if len(A3) > len(best):
        best = list(A3)
    
    if is_done():
        return [int(x) for x in best]
    
    # Strategy 4: Random search
    A4 = random_search(30)
    if len(A4) > len(best):
        best = list(A4)
    
    if is_done():
        return [int(x) for x in best]
    
    # Strategy 5: Local search on the best found
    A5 = local_search_improve(best)
    if len(A5) > len(best):
        best = list(A5)
    
    # Ensure we return a valid result
    if not check_validity(best):
        best = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return [int(x) for x in best]
