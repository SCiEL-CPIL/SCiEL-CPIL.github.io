"""Rewritten program for two Sidon sets with disjoint difference sets.
Uses a randomized search approach: generate many varied candidate answers
and keep the best one found within the time budget."""

import time
import random
import sys
from itertools import combinations, combinations_with_replacement

def _check_sidon_unique_sums(els):
    """Check that all pair sums a+b (a<=b) are unique."""
    sums = set()
    n = len(els)
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_nonzero_diffs(els):
    """Get set of nonzero differences (positive and negative)."""
    diffs = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = els[j] - els[i]
                if d != 0:
                    diffs.add(d)
    return diffs

def _is_valid_pair(A, B):
    """Check that A and B are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon_unique_sums(A):
        return False
    if not _check_sidon_unique_sums(B):
        return False
    diffsA = _get_nonzero_diffs(A)
    diffsB = _get_nonzero_diffs(B)
    if diffsA & diffsB:
        return False
    return True

def _greedy_build_A(B, n=256, max_size=64, rng=None):
    """Greedily build A given B, ensuring Sidon property and disjoint diffs with B."""
    if rng is None:
        rng = random.Random()
    
    B_diffs = _get_nonzero_diffs(B)
    A = []
    used_sums = set()
    A_diffs = set()  # Track A's own differences to avoid internal conflicts
    
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(A) >= max_size:
            break
        # Check Sidon property with new element
        new_sums = set()
        valid = True
        for a in A:
            s = a + x
            if s in used_sums or s in new_sums:
                valid = False
                break
            new_sums.add(s)
        if not valid:
            continue
        # Check diagonal sum x+x
        s2 = 2 * x
        if s2 in used_sums or s2 in new_sums:
            continue
        new_sums.add(s2)
        # Check difference disjointness with B
        for a in A:
            d1 = a - x
            d2 = x - a
            if d1 != 0 and d1 in B_diffs:
                valid = False
                break
            if d2 != 0 and d2 in B_diffs:
                valid = False
                break
        if not valid:
            continue
        # Check that new differences don't conflict with A's own differences
        for a in A:
            d1 = a - x
            d2 = x - a
            if d1 != 0 and d1 in A_diffs:
                valid = False
                break
            if d2 != 0 and d2 in A_diffs:
                valid = False
                break
        if not valid:
            continue
        A.append(x)
        used_sums |= new_sums
        # Add A's own differences to the set
        for a in A[:-1]:
            d = a - x
            if d != 0:
                A_diffs.add(d)
                A_diffs.add(-d)
    
    return A

def _greedy_build_B(A, n=256, max_size=64, rng=None):
    """Greedily build B given A, ensuring Sidon property and disjoint diffs with A."""
    if rng is None:
        rng = random.Random()
    
    A_diffs = _get_nonzero_diffs(A)
    B = []
    used_sums = set()
    B_diffs = set()  # Track B's own differences to avoid internal conflicts
    
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(B) >= max_size:
            break
        new_sums = set()
        valid = True
        for b in B:
            s = b + x
            if s in used_sums or s in new_sums:
                valid = False
                break
            new_sums.add(s)
        if not valid:
            continue
        s2 = 2 * x
        if s2 in used_sums or s2 in new_sums:
            continue
        new_sums.add(s2)
        for b in B:
            d1 = b - x
            d2 = x - b
            if d1 != 0 and d1 in A_diffs:
                valid = False
                break
            if d2 != 0 and d2 in A_diffs:
                valid = False
                break
        if not valid:
            continue
        # Check that new differences don't conflict with B's own differences
        for b in B:
            d1 = b - x
            d2 = x - b
            if d1 != 0 and d1 in B_diffs:
                valid = False
                break
            if d2 != 0 and d2 in B_diffs:
                valid = False
                break
        if not valid:
            continue
        B.append(x)
        used_sums |= new_sums
        # Add B's own differences to the set
        for b in B[:-1]:
            d = b - x
            if d != 0:
                B_diffs.add(d)
                B_diffs.add(-d)
    
    return B

def _local_search_improve(A, B, n=256, max_iters=100, rng=None):
    """Try to improve by greedily adding elements that fit."""
    if rng is None:
        rng = random.Random()
    best_A = list(A)
    best_B = list(B)
    best_size = len(A) + len(B)
    
    # Precompute difference sets for efficiency
    A_diffs = _get_nonzero_diffs(best_A)
    B_diffs = _get_nonzero_diffs(best_B)
    
    for _ in range(max_iters):
        improved = False
        
        # Try adding elements to A: scan all candidates, pick the first valid one
        if len(best_A) < 64:
            A_sums = set()
            for i in range(len(best_A)):
                for j in range(i, len(best_A)):
                    A_sums.add(best_A[i] + best_A[j])
            
            # Try all candidates in random order
            candidates = list(range(1, n + 1))
            rng.shuffle(candidates)
            for x in candidates:
                if x in best_A:
                    continue
                # Check Sidon
                new_sums = set()
                valid = True
                for a in best_A:
                    s = a + x
                    if s in A_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    continue
                s2 = 2 * x
                if s2 in A_sums or s2 in new_sums:
                    continue
                # Check diff disjointness with B
                for a in best_A:
                    d1 = a - x
                    d2 = x - a
                    if d1 != 0 and d1 in B_diffs:
                        valid = False
                        break
                    if d2 != 0 and d2 in B_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                # Check that new differences don't conflict with A's own differences
                for a in best_A:
                    d1 = a - x
                    d2 = x - a
                    if d1 != 0 and d1 in A_diffs:
                        valid = False
                        break
                    if d2 != 0 and d2 in A_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                best_A.append(x)
                best_A.sort()
                # Update A_diffs
                for a in best_A[:-1]:
                    d = a - x
                    if d != 0:
                        A_diffs.add(d)
                        A_diffs.add(-d)
                improved = True
                break
        
        # Try adding elements to B
        if len(best_B) < 64:
            B_sums = set()
            for i in range(len(best_B)):
                for j in range(i, len(best_B)):
                    B_sums.add(best_B[i] + best_B[j])
            
            candidates = list(range(1, n + 1))
            rng.shuffle(candidates)
            for x in candidates:
                if x in best_B:
                    continue
                new_sums = set()
                valid = True
                for b in best_B:
                    s = b + x
                    if s in B_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    continue
                s2 = 2 * x
                if s2 in B_sums or s2 in new_sums:
                    continue
                for b in best_B:
                    d1 = b - x
                    d2 = x - b
                    if d1 != 0 and d1 in A_diffs:
                        valid = False
                        break
                    if d2 != 0 and d2 in A_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                # Check that new differences don't conflict with B's own differences
                for b in best_B:
                    d1 = b - x
                    d2 = x - b
                    if d1 != 0 and d1 in B_diffs:
                        valid = False
                        break
                    if d2 != 0 and d2 in B_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                best_B.append(x)
                best_B.sort()
                # Update B_diffs
                for b in best_B[:-1]:
                    d = b - x
                    if d != 0:
                        B_diffs.add(d)
                        B_diffs.add(-d)
                improved = True
                break
        
        if not improved:
            break
    
    return best_A, best_B

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    n = 256
    max_size = 64
    
    best_A = [1]
    best_B = [2]
    best_score = 2  # |A| + |B|
    
    rng = random.Random()
    
    # Multiple restarts with different strategies
    for restart in range(20000):
        if time.time() > deadline:
            break
        
        # Strategy 1: Random seed element for A, then greedy B, then greedy A
        if restart % 3 == 0:
            seed_a = rng.randint(1, n)
            A_init = [seed_a]
            B = _greedy_build_B(A_init, n, max_size, rng)
            A = _greedy_build_A(B, n, max_size, rng)
        # Strategy 2: Random seed for B, then greedy A, then greedy B
        elif restart % 3 == 1:
            seed_b = rng.randint(1, n)
            B_init = [seed_b]
            A = _greedy_build_A(B_init, n, max_size, rng)
            B = _greedy_build_B(A, n, max_size, rng)
        # Strategy 3: Random small sets, then extend
        else:
            size_a = rng.randint(1, 5)
            size_b = rng.randint(1, 5)
            A_init = sorted(rng.sample(range(1, n + 1), size_a))
            B_init = sorted(rng.sample(range(1, n + 1), size_b))
            if not _is_valid_pair(A_init, B_init):
                continue
            B = _greedy_build_B(A_init, n, max_size, rng)
            A = _greedy_build_A(B, n, max_size, rng)
        
        if not A:
            A = [1]
        if not B:
            B = [2]
        
        # Verify validity
        if _is_valid_pair(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
        
        # Local search improvement - use more iterations when close to deadline
        if time.time() < deadline - 2:
            remaining = deadline - time.time()
            iters = max(5, int(remaining * 2))
            A_ls, B_ls = _local_search_improve(A, B, n, max_iters=iters, rng=rng)
            if _is_valid_pair(A_ls, B_ls):
                score_ls = len(A_ls) + len(B_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_A = list(A_ls)
                    best_B = list(B_ls)
    
    # Final check
    if not _is_valid_pair(best_A, best_B):
        # Fallback: simple valid pair
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
