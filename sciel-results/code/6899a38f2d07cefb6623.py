# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement
import sys

def _check_sidon_set(S):
    """Check if S is a Sidon set (pair sums with replacement are unique)."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _diff_set(S):
    """Return the set of absolute nonzero differences for S."""
    diffs = set()
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            d = abs(S[i] - S[j])
            if d > 0:
                diffs.add(d)
    return diffs

def _try_add_to_sidon(S, x, diff_set):
    """
    Try to add x to Sidon set S.
    S must be sorted. x must not be in S.
    Check: new pair sums are unique, and new differences don't conflict with diff_set.
    Returns (new_S, new_diff_set) if successful, else (None, None).
    """
    if x in S:
        return None, None
    
    # Check pair sums
    sums = set()
    n = len(S)
    # Existing sums
    for i in range(n):
        for j in range(i, n):
            sums.add(S[i] + S[j])
    
    # New sums involving x
    for i in range(n):
        s = x + S[i]
        if s in sums:
            return None, None
        sums.add(s)
    # x + x
    sx = x + x
    if sx in sums:
        return None, None
    
    # Check differences
    new_diffs = set()
    for i in range(n):
        d = abs(x - S[i])
        if d > 0:
            if d in diff_set:
                return None, None
            new_diffs.add(d)
    
    # Merge diffs
    combined_diffs = diff_set | new_diffs
    
    # Build new sorted list
    new_S = list(S)
    # Insert x in sorted order
    lo, hi = 0, len(new_S)
    while lo < hi:
        mid = (lo + hi) // 2
        if new_S[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    new_S.insert(lo, x)
    
    return new_S, combined_diffs

def _greedy_increasing(max_n=256, max_size=64, rng=None, seed=0):
    """
    Greedy construction processing numbers in increasing order.
    This tends to produce better Sidon sets because smaller numbers
    have more room to avoid collisions.
    """
    if rng is None:
        rng = random.Random(seed)
    
    A = []
    B = []
    diff_A = set()
    diff_B = set()
    sums_A = set()
    sums_B = set()
    
    for x in range(1, max_n + 1):
        # Try to add to A first (in increasing order, this is natural)
        can_add_A = True
        if x in A:
            can_add_A = False
        else:
            # Check sums
            for a in A:
                if x + a in sums_A:
                    can_add_A = False
                    break
            if can_add_A and 2*x in sums_A:
                can_add_A = False
            # Check diffs
            if can_add_A:
                for a in A:
                    d = abs(x - a)
                    if d > 0 and d in diff_A:
                        can_add_A = False
                        break
        
        if can_add_A and len(A) < max_size:
            A.append(x)
            for a in A[:-1]:
                sums_A.add(x + a)
            sums_A.add(2*x)
            for a in A[:-1]:
                d = abs(x - a)
                if d > 0:
                    diff_A.add(d)
            continue
        
        # Try to add to B
        can_add_B = True
        if x in B:
            can_add_B = False
        else:
            for b in B:
                if x + b in sums_B:
                    can_add_B = False
                    break
            if can_add_B and 2*x in sums_B:
                can_add_B = False
            if can_add_B:
                for b in B:
                    d = abs(x - b)
                    if d > 0 and d in diff_B:
                        can_add_B = False
                        break
        
        if can_add_B and len(B) < max_size:
            B.append(x)
            for b in B[:-1]:
                sums_B.add(x + b)
            sums_B.add(2*x)
            for b in B[:-1]:
                d = abs(x - b)
                if d > 0:
                    diff_B.add(d)
    
    return A, B

def _greedy_with_swap(max_n=256, max_size=64, rng=None, seed=0):
    """
    Greedy construction with occasional swaps to improve packing.
    """
    if rng is None:
        rng = random.Random(seed)
    
    A = []
    B = []
    diff_A = set()
    diff_B = set()
    sums_A = set()
    sums_B = set()
    
    for x in range(1, max_n + 1):
        # Try A first
        can_add_A = True
        if x in A:
            can_add_A = False
        else:
            for a in A:
                if x + a in sums_A:
                    can_add_A = False
                    break
            if can_add_A and 2*x in sums_A:
                can_add_A = False
            if can_add_A:
                for a in A:
                    d = abs(x - a)
                    if d > 0 and d in diff_A:
                        can_add_A = False
                        break
        
        if can_add_A and len(A) < max_size:
            A.append(x)
            for a in A[:-1]:
                sums_A.add(x + a)
            sums_A.add(2*x)
            for a in A[:-1]:
                d = abs(x - a)
                if d > 0:
                    diff_A.add(d)
            continue
        
        # Try B
        can_add_B = True
        if x in B:
            can_add_B = False
        else:
            for b in B:
                if x + b in sums_B:
                    can_add_B = False
                    break
            if can_add_B and 2*x in sums_B:
                can_add_B = False
            if can_add_B:
                for b in B:
                    d = abs(x - b)
                    if d > 0 and d in diff_B:
                        can_add_B = False
                        break
        
        if can_add_B and len(B) < max_size:
            B.append(x)
            for b in B[:-1]:
                sums_B.add(x + b)
            sums_B.add(2*x)
            for b in B[:-1]:
                d = abs(x - b)
                if d > 0:
                    diff_B.add(d)
    
    return A, B

def _local_search_swap(A, B, max_n=256, max_size=64, rng=None, seed=0, time_budget=3.0):
    """
    Local search: try to swap elements between A and B, or replace elements
    to free up space for adding more elements.
    """
    if rng is None:
        rng = random.Random(seed)
    
    start_time = time.time()
    A = list(A)
    B = list(B)
    
    best_score = len(A) + len(B)
    best_A, best_B = list(A), list(B)
    
    def _rebuild_sums_diffs(S):
        sums = set()
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        for i in range(n):
            for j in range(i+1, n):
                d = abs(S[i] - S[j])
                if d > 0:
                    diffs.add(d)
        return sums, diffs
    
    sums_A, diff_A = _rebuild_sums_diffs(A)
    sums_B, diff_B = _rebuild_sums_diffs(B)
    
    while time.time() - start_time < time_budget:
        # Try swapping an element from A with a new element
        if rng.random() < 0.5 and len(A) > 0:
            idx = rng.randint(0, len(A)-1)
            old_x = A[idx]
            # Try to find a replacement
            for _ in range(50):
                new_x = rng.randint(1, max_n)
                if new_x == old_x or new_x in A:
                    continue
                # Remove old_x, try to add new_x
                temp_A = A[:idx] + A[idx+1:]
                temp_sums_A, temp_diff_A = _rebuild_sums_diffs(temp_A)
                
                can_add = True
                for a in temp_A:
                    if new_x + a in temp_sums_A:
                        can_add = False
                        break
                if can_add and 2*new_x in temp_sums_A:
                    can_add = False
                if can_add:
                    for a in temp_A:
                        d = abs(new_x - a)
                        if d > 0 and d in temp_diff_A:
                            can_add = False
                            break
                
                if can_add and len(temp_A) + 1 <= max_size:
                    # Check diff disjointness with B
                    new_diffs = set()
                    for a in temp_A:
                        d = abs(new_x - a)
                        if d > 0:
                            new_diffs.add(d)
                    if not (new_diffs & diff_B):
                        # Valid swap
                        new_A = temp_A + [new_x]
                        new_A.sort()
                        new_sums_A, new_diff_A = _rebuild_sums_diffs(new_A)
                        if not (new_diff_A & diff_B):
                            score = len(new_A) + len(B)
                            if score > best_score:
                                best_score = score
                                best_A, best_B = list(new_A), list(B)
                                A, B = list(new_A), list(B)
                                sums_A, diff_A = new_sums_A, new_diff_A
                            break
        
        # Try adding to A
        else:
            for _ in range(20):
                x = rng.randint(1, max_n)
                if x in A:
                    continue
                can_add = True
                for a in A:
                    if x + a in sums_A:
                        can_add = False
                        break
                if can_add and 2*x in sums_A:
                    can_add = False
                if can_add:
                    for a in A:
                        d = abs(x - a)
                        if d > 0 and d in diff_A:
                            can_add = False
                            break
                if can_add and len(A) < max_size:
                    A.append(x)
                    A.sort()
                    sums_A, diff_A = _rebuild_sums_diffs(A)
                    score = len(A) + len(B)
                    if score > best_score:
                        best_score = score
                        best_A, best_B = list(A), list(B)
                    break
    
    return best_A, best_B, best_score

def _validate(A, B, max_n=256):
    """
    Validate that A and B are valid Sidon subsets of [1, max_n]
    with disjoint nonzero difference sets.
    Returns (is_valid, score).
    """
    if not A or not B:
        return False, 0
    
    if len(A) > 64 or len(B) > 64:
        return False, 0
    
    # Check elements are in [1, max_n]
    for x in A:
        if x < 1 or x > max_n:
            return False, 0
    for x in B:
        if x < 1 or x > max_n:
            return False, 0
    
    # Check Sidon property
    if not _check_sidon_set(A):
        return False, 0
    if not _check_sidon_set(B):
        return False, 0
    
    # Check disjoint differences
    dA = _diff_set(A)
    dB = _diff_set(B)
    if dA & dB:
        return False, 0
    
    return True, len(A) + len(B)

def construct():
    """
    Main function: generate many varied candidate answers and keep the best one found.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    best_A, best_B = [], []
    best_score = 0
    
    # Strategy 1: Greedy increasing order (most promising)
    A, B = _greedy_increasing(max_n=256, max_size=64, seed=0)
    valid, score = _validate(A, B)
    if valid and score > best_score:
        best_score = score
        best_A, best_B = list(A), list(B)
    
    # Try different seeds for variation
    for seed in range(1, 15):
        if time.time() > deadline - 30.0:
            break
        
        A, B = _greedy_increasing(max_n=256, max_size=64, seed=seed)
        valid, score = _validate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
    
    # Strategy 2: Greedy with swap
    for seed in range(5):
        if time.time() > deadline - 20.0:
            break
        
        A, B = _greedy_with_swap(max_n=256, max_size=64, seed=seed)
        valid, score = _validate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
    
    # Local search on best solution
    if time.time() < deadline - 10.0 and best_score > 0:
        for ls_seed in range(3):
            if time.time() > deadline - 5.0:
                break
            
            remaining_time = deadline - 5.0 - time.time()
            if remaining_time <= 0:
                break
            
            A_ls, B_ls, score_ls = _local_search_swap(
                best_A, best_B, max_n=256, max_size=64,
                seed=ls_seed, time_budget=min(remaining_time, 8.0)
            )
            
            valid, score = _validate(A_ls, B_ls)
            if valid and score > best_score:
                best_score = score
                best_A, best_B = list(A_ls), list(B_ls)
    
    # Ensure we have a valid fallback
    if not best_A or not best_B:
        A = [1]
        B = [3]
        valid, score = _validate(A, B)
        if valid:
            best_A, best_B = A, B
            best_score = score
        else:
            best_A, best_B = [1], [3]
            best_score = 2
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
