import time
import random
import sys
import numpy as np

def _all_triple_sums(lst):
    """Compute all unordered triple sums for a sorted list."""
    n = len(lst)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(lst[i] + lst[j] + lst[k])
    return sums

def _all_pair_sums(lst):
    """Compute all unordered pair sums for a sorted list."""
    n = len(lst)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(lst[i] + lst[j])
    return sums

def _is_valid_triples(lst):
    """Check if all unordered triple sums are distinct."""
    if not lst:
        return True
    sums = _all_triple_sums(lst)
    return len(sums) == len(set(sums)) # len(sums) is already unique count, so this is always true if we use a set.
    # Actually, _all_triple_sums returns a set. So len(sums) is the count of unique sums.
    # The total number of triples is comb(len(lst)+2, 3).
    # If len(sums) == total_triples, it is valid.
    n = len(lst)
    total = (n+2)*(n+1)*n//6
    return len(sums) == total

def _can_add(x, S, T, P):
    """
    Check if x can be added to S.
    S: list of current elements
    T: set of current triple sums
    P: set of current pair sums
    """
    # Generate new sums involving x
    new_sums = []
    new_sums.append(3 * x)
    
    for s in S:
        new_sums.append(2 * x + s)
        new_sums.append(x + 2 * s)
        
    for p in P:
        new_sums.append(x + p)
        
    # Check internal collisions
    if len(new_sums) != len(set(new_sums)):
        return False
        
    # Check collision with existing T
    for ns in new_sums:
        if ns in T:
            return False
            
    return True

def _add_element(x, S, T, P):
    """
    Add x to S, updating T and P.
    Assumes _can_add(x, S, T, P) is True.
    """
    S.append(x)
    S.sort()
    
    # Update T
    T.add(3 * x)
    for s in S:
        if s == x:
            continue
        T.add(2 * x + s)
        T.add(x + 2 * s)
        
    for p in P:
        T.add(x + p)
        
    # Update P
    for s in S:
        if s == x:
            continue
        P.add(x + s)
        
    # Note: The pair (x,x) sum 2x is not in P? 
    # P is defined as sums of pairs from S. 
    # In _all_pair_sums, i ranges 0..n, j ranges i..n. So (s,s) is included.
    # So we must add 2*x to P.
    P.add(2 * x)
    
    return S, T, P

def _remove_element(x, S, T, P):
    """
    Remove x from S, updating T and P.
    """
    S.remove(x)
    S.sort()
    
    # We must rebuild T and P because removing elements invalidates many sums.
    # For small n (< 50), rebuilding is O(n^3) which is ~125k ops. 
    # This is fast enough for a few hundred iterations.
    T = _all_triple_sums(S)
    P = _all_pair_sums(S)
    
    return S, T, P

def _greedy_construct(step=1, rng=None):
    S = []
    T = set()
    P = set()
    
    candidates = list(range(1, 4097))
    if rng:
        rng.shuffle(candidates)
        
    for x in candidates:
        if _can_add(x, S, T, P):
            S, T, P = _add_element(x, S, T, P)
            
    return S, T, P

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    best_set = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Phase 1: Multiple Greedy Initializations
    # 1. Deterministic greedy
    S, T, P = _greedy_construct()
    if len(S) > best_size:
        best_set = S
        best_size = len(S)
        
    # 2. Randomized greedy (shuffled candidates)
    for _ in range(10):
        if time.time() > deadline - 10: break
        S, T, P = _greedy_construct(rng=rng)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            
    # Phase 2: Local Search with Swaps and Expansions
    current_S = list(best_set)
    current_T = _all_triple_sums(current_S)
    current_P = _all_pair_sums(current_S)
    
    # Pre-compute all candidates
    all_candidates = list(range(1, 4097))
    
    iterations = 0
    while time.time() < deadline - 2:
        iterations += 1
        
        # Try to add any element
        added = False
        for x in all_candidates:
            if x in current_S:
                continue
            if _can_add(x, current_S, current_T, current_P):
                current_S, current_T, current_P = _add_element(x, current_S, current_T, current_P)
                added = True
                if len(current_S) > best_size:
                    best_set = list(current_S)
                    best_size = len(current_S)
                break # Restart search after addition? Or continue? 
                # Restarting is safer to maintain state consistency if we don't update carefully.
                # But _add_element updates state. So we can continue trying to add more.
                # However, the list of candidates is fixed. 
                # Let's just break and restart the outer loop to re-evaluate all candidates.
        
        if added:
            continue
            
        # If no addition possible, try a swap (remove 1, add 1)
        # Or try "remove 1, add 2"
        
        if len(current_S) > 1:
            # Pick a random element to remove
            # Heuristic: Remove an element that is "small" or "large"? 
            # Random is fine for diversity.
            idx = rng.randint(0, len(current_S) - 1)
            y = current_S[idx]
            
            # Remove y
            temp_S, temp_T, temp_P = _remove_element(y, current_S, current_T, current_P)
            
            # Try to add one element
            swap_success = False
            for x in all_candidates:
                if x == y or x in temp_S:
                    continue
                if _can_add(x, temp_S, temp_T, temp_P):
                    temp_S, temp_T, temp_P = _add_element(x, temp_S, temp_T, temp_P)
                    swap_success = True
                    break
                    
            if swap_success:
                # Accept the swap
                current_S = temp_S
                current_T = temp_T
                current_P = temp_P
                # Size is same, but structure changed. 
                # Check if we can now add more?
                # The outer loop will handle addition.
                
            else:
                # Try to add TWO elements
                # Remove y, then try to add x1, then x2
                # This is expensive: O(N^2) checks.
                # Only do this occasionally or if stuck for a while.
                if iterations % 5 == 0:
                    temp_S2, temp_T2, temp_P2 = _remove_element(y, current_S, current_T, current_P)
                    
                    found_pair = False
                    # To speed up, limit the search space for x1, x2
                    # Just iterate through all candidates for x1
                    for x1 in all_candidates:
                        if x1 == y or x1 in temp_S2:
                            continue
                        if _can_add(x1, temp_S2, temp_T2, temp_P2):
                            temp_S3, temp_T3, temp_P3 = _add_element(x1, temp_S2, temp_T2, temp_P2)
                            
                            # Now try to add x2 to temp_S3
                            for x2 in all_candidates:
                                if x2 == y or x2 == x1 or x2 in temp_S3:
                                    continue
                                if _can_add(x2, temp_S3, temp_T3, temp_P3):
                                    temp_S3, temp_T3, temp_P3 = _add_element(x2, temp_S3, temp_T3, temp_P3)
                                    # Success! Size increased by 1
                                    current_S = temp_S3
                                    current_T = temp_T3
                                    current_P = temp_P3
                                    found_pair = True
                                    if len(current_S) > best_size:
                                        best_set = list(current_S)
                                        best_size = len(current_S)
                                    break
                            if found_pair:
                                break
                    if found_pair:
                        continue # Restart outer loop
                    
        # If nothing worked, maybe restart from best_set to avoid drift?
        # Or just continue.
        
    # Final Validation
    if best_set and not _is_valid_triples(best_set):
        # Fallback
        S, T, P = _greedy_construct()
        best_set = S
        
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}")
    print(f"Set: {result[:20]}...")
    print(f"Valid: {_is_valid_triples(result)}")
