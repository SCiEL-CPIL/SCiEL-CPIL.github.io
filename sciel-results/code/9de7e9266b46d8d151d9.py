"""Optimized program: MODEL-AND-OPTIMIZE for two Sidon sets with disjoint difference sets."""
import time
import random
from itertools import combinations_with_replacement, combinations
import sys

def _is_sidon(s):
    """Check if set s is Sidon (all pair sums a+b with a<=b are unique)."""
    if not s:
        return True
    sums = set()
    for i, a in enumerate(s):
        for b in s[i:]:
            s_val = a + b
            if s_val in sums:
                return False
            sums.add(s_val)
    return True

def _diffs(s):
    """Return set of nonzero absolute differences."""
    if len(s) < 2:
        return set()
    d = set()
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            d.add(lst[j] - lst[i])
    return d

def _validate(A, B):
    """Validate both sets are Sidon and have disjoint nonzero differences. Return True if valid."""
    if not A or not B:
        return False
    if not _is_sidon(A):
        return False
    if not _is_sidon(B):
        return False
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False
    return True

def _can_add_to_sidon(x, current, sums_set, max_size):
    """Check if x can be added to current Sidon set. Returns (can_add, new_sums)."""
    if len(current) >= max_size:
        return False, sums_set
    new_sums = set(sums_set)
    for a in current:
        s_val = a + x
        if s_val in new_sums:
            return False, sums_set
        new_sums.add(s_val)
    if x + x in new_sums:
        return False, sums_set
    new_sums.add(x + x)
    return True, new_sums

def _diff_conflict(x, current, other_diffs):
    """Check if adding x creates diff conflict with other_diffs."""
    for a in current:
        d = abs(x - a)
        if d != 0 and d in other_diffs:
            return True
    return False

def _add_diffs(x, current, diffs_set):
    """Add diffs from x to existing elements."""
    for a in current:
        d = abs(x - a)
        if d != 0:
            diffs_set.add(d)

def _greedy_construct(n=256, max_size=64, rng=None, order=None):
    """Greedy construction: try to build two Sidon sets with disjoint diffs."""
    if rng is None:
        rng = random.Random()
    if order is None:
        order = list(range(1, n+1))
        rng.shuffle(order)
    
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()
    
    for x in order:
        # Try adding to A
        canA, new_sumsA = _can_add_to_sidon(x, A, sumsA, max_size)
        if canA and not _diff_conflict(x, A, diffsB):
            A.append(x)
            sumsA = new_sumsA
            _add_diffs(x, A[:-1], diffsA)
            continue
        
        # Try adding to B
        canB, new_sumsB = _can_add_to_sidon(x, B, sumsB, max_size)
        if canB and not _diff_conflict(x, B, diffsA):
            B.append(x)
            sumsB = new_sumsB
            _add_diffs(x, B[:-1], diffsB)
    
    return A, B

def _local_search(A, B, n=256, max_size=64, rng=None, max_iters=5000, deadline=None):
    """Try to improve by adding elements or swapping with incremental checks."""
    if rng is None:
        rng = random.Random()
    
    best_score = len(A) + len(B)
    best_A, best_B = list(A), list(B)
    
    # Precompute current state
    sumsA = set()
    for i, a in enumerate(A):
        for b in A[i:]:
            sumsA.add(a + b)
    sumsB = set()
    for i, b in enumerate(B):
        for c in B[i:]:
            sumsB.add(b + c)
    diffsA = _diffs(A)
    diffsB = _diffs(B)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        
        # Try adding ALL possible elements to A (exhaustive within budget)
        if len(A) < max_size:
            candidates = [x for x in range(1, n+1) if x not in A]
            rng.shuffle(candidates)
            for x in candidates:
                if deadline and time.time() > deadline:
                    break
                canA, new_sumsA = _can_add_to_sidon(x, A, sumsA, max_size)
                if canA and not _diff_conflict(x, A, diffsB):
                    A.append(x)
                    sumsA = new_sumsA
                    _add_diffs(x, A[:-1], diffsA)
                    best_A = list(A)
                    best_score = max(best_score, len(A) + len(B))
                    improved = True
                    # After adding, re-check remaining candidates with updated state
                    break  # Re-evaluate after one addition
        
        # Try adding ALL possible elements to B
        if len(B) < max_size:
            candidates = [x for x in range(1, n+1) if x not in B]
            rng.shuffle(candidates)
            for x in candidates:
                if deadline and time.time() > deadline:
                    break
                canB, new_sumsB = _can_add_to_sidon(x, B, sumsB, max_size)
                if canB and not _diff_conflict(x, B, diffsA):
                    B.append(x)
                    sumsB = new_sumsB
                    _add_diffs(x, B[:-1], diffsB)
                    best_B = list(B)
                    best_score = max(best_score, len(A) + len(B))
                    improved = True
                    break
        
        # If no additions possible, try swap moves
        if not improved:
            # Try removing from A and adding different element
            if len(A) > 1:
                idx = rng.randrange(len(A))
                removed = A[idx]
                A_new = A[:idx] + A[idx+1:]
                # Recompute sums and diffs for A_new
                new_sumsA = set()
                for i, a in enumerate(A_new):
                    for b in A_new[i:]:
                        new_sumsA.add(a + b)
                new_diffsA = _diffs(A_new)
                if not (new_diffsA & diffsB):
                    A = A_new
                    sumsA = new_sumsA
                    diffsA = new_diffsA
                    best_A = list(A)
                    best_score = max(best_score, len(A) + len(B))
            
            # Try removing from B and adding different element
            if len(B) > 1:
                idx = rng.randrange(len(B))
                removed = B[idx]
                B_new = B[:idx] + B[idx+1:]
                new_sumsB = set()
                for i, b in enumerate(B_new):
                    for c in B_new[i:]:
                        new_sumsB.add(b + c)
                new_diffsB = _diffs(B_new)
                if not (diffsA & new_diffsB):
                    B = B_new
                    sumsB = new_sumsB
                    diffsB = new_diffsB
                    best_B = list(B)
                    best_score = max(best_score, len(A) + len(B))
    
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin from 90s limit
    
    best_A, best_B = [], []
    best_score = 0
    
    # Strategy 1: Multiple greedy runs with different random orderings
    # Use more restarts since greedy is now faster with incremental checks
    num_greedy = 200
    for seed in range(num_greedy):
        if time.time() > deadline - 15:  # Reserve 15s for local search
            break
        rng = random.Random(seed)
        order = list(range(1, 257))
        rng.shuffle(order)
        A, B = _greedy_construct(256, 64, rng, order)
        score = len(A) + len(B)
        if _validate(A, B) and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
    
    # Strategy 2: Try structured orderings
    for order_type in ['asc', 'desc', 'interleaved']:
        if time.time() > deadline - 15:
            break
        if order_type == 'asc':
            order = list(range(1, 257))
        elif order_type == 'desc':
            order = list(range(256, 0, -1))
        else:
            order = []
            for i in range(128):
                order.append(i+1)
                order.append(256-i)
        
        rng = random.Random(12345)
        A, B = _greedy_construct(256, 64, rng, order)
        score = len(A) + len(B)
        if _validate(A, B) and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
    
    # Strategy 3: Local search improvement on best found
    if time.time() < deadline - 2 and best_score > 0:
        A, B = _local_search(best_A, best_B, 256, 64, random.Random(42), 
                             max_iters=50000, deadline=deadline - 1)
        if _validate(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A, best_B = list(A), list(B)
    
    # Fallback: ensure we return valid nonempty sets
    if not best_A or not best_B:
        # Simple fallback
        best_A, best_B = _greedy_construct(256, 64, random.Random(0), list(range(1, 257)))
        if not _validate(best_A, best_B) or not best_A or not best_B:
            best_A = [1, 2, 4, 8]
            best_B = [3, 5, 7, 11]
    
    return [best_A, best_B]
