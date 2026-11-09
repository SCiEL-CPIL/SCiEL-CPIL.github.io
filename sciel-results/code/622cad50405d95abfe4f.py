import time
import random

def _check_validity(A):
    """Check if set A satisfies cardinality-separated subset sums."""
    n = len(A)
    if n == 0:
        return True
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0] = {0}
    for a in A:
        # Iterate backwards to avoid using the same element multiple times in one step
        # But we are building the set incrementally, so we update based on previous state
        # Actually, the standard DP for subset sums:
        # sums_by_k[k] contains all sums of subsets of size k from the processed elements.
        # When adding 'a', new sums of size k+1 are old sums of size k + a.
        for k in range(n - 1, -1, -1):
            for s in list(sums_by_k[k]):
                sums_by_k[k + 1].add(s + a)
    
    seen = {}
    for k in range(n + 1):
        for s in sums_by_k[k]:
            if s in seen:
                return False
            seen[s] = k
    return True

def _min_separation(A):
    """Calculate the minimum difference between min(Sum_{k+1}) and max(Sum_k).
    Returns -1 if invalid (collision)."""
    n = len(A)
    if n == 0:
        return float('inf')
    
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0] = {0}
    for a in A:
        for k in range(n - 1, -1, -1):
            for s in list(sums_by_k[k]):
                sums_by_k[k + 1].add(s + a)
                
    min_sep = float('inf')
    for k in range(n):
        if not sums_by_k[k] or not sums_by_k[k+1]:
            continue
        max_k = max(sums_by_k[k])
        min_k1 = min(sums_by_k[k+1])
        sep = min_k1 - max_k
        if sep <= 0:
            return -1
        if sep < min_sep:
            min_sep = sep
    return min_sep

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    best_A = []
    best_sep = -1
    
    # 1. Initial Greedy Construction
    # Try ascending order
    A = []
    for candidate in range(1, 257):
        if time.time() - start_time > deadline:
            break
        new_A = A + [candidate]
        if _check_validity(new_A):
            A = new_A
    
    best_A = A[:]
    best_sep = _min_separation(A)
    
    # 2. Local Search to improve separation and allow more elements
    improved = True
    while improved and time.time() - start_time < deadline - 5:
        improved = False
        n = len(A)
        
        # Try to add an element
        added = False
        for candidate in range(1, 257):
            if candidate in A:
                continue
            new_A = A + [candidate]
            if _check_validity(new_A):
                A = new_A
                best_A = A[:]
                best_sep = _min_separation(A)
                added = True
                improved = True
                break
        
        if added:
            continue
            
        # If no element could be added, try swaps to improve separation
        current_sep = _min_separation(A)
        if current_sep == -1:
            # Should not happen if we maintain validity, but just in case
            break
            
        best_swap = None
        best_swap_sep = current_sep
        
        # To speed up, we can limit the number of swap attempts or use random sampling
        # But n is small (~30), so n * 256 swaps is ~7680 checks.
        # Each check is O(n * Sum). Sum is ~4000. n=30. 30*4000 = 120k ops.
        # 7680 * 120k is too slow.
        # We need a faster validity check or fewer swaps.
        
        # Optimization: Only try swaps that seem promising?
        # Or just try a subset of swaps.
        
        # Let's try all swaps but with a fast preliminary check?
        # Or just accept any valid swap that improves separation.
        
        # To make it faster, we can check validity only for a random subset of swaps
        # or limit the candidates.
        
        # Let's try swapping A[i] with c.
        # We can iterate c from 1 to 256.
        
        found_improvement = False
        for i in range(len(A)):
            if time.time() - start_time > deadline - 2:
                break
            a_i = A[i]
            for c in range(1, 257):
                if c == a_i or c in A:
                    continue
                new_A = A[:i] + [c] + A[i+1:]
                # Fast check: does the new element keep the range sane?
                # Just do the full check for now, but maybe we can prune?
                # If c is very small, it might reduce max sums.
                # If c is very large, it might increase min sums? No, min sums use smallest.
                
                sep = _min_separation(new_A)
                if sep > best_swap_sep:
                    best_swap_sep = sep
                    best_swap = (i, c)
                    found_improvement = True
                    # We don't break immediately because we might find a better one,
                    # but for speed, we can break if improvement is significant?
                    # Or just take the first improvement to keep it fast.
                    # Let's take the first improvement to save time.
                    break
            if found_improvement:
                break
                
        if found_improvement:
            i, c = best_swap
            A = A[:i] + [c] + A[i+1:]
            improved = True
            
    return [int(x) for x in best_A]
