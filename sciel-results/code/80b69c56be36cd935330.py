# EVOLVE-BLOCK-START
import time
import random
import sys

def _check_sidon(S):
    """Check Sidon property: all pair sums (with replacement) are unique."""
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(S):
    """Return set of absolute nonzero differences."""
    d = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(S[i] - S[j])
            if diff > 0:
                d.add(diff)
    return d

def _can_add_to_sidon(S, x, used_sums):
    """Check if x can be added to Sidon set S without violating Sidon property.
    S is sorted. Returns (True, new_used_sums) or (False, None)."""
    if x in S:
        return False, None
    new_sums = set(used_sums)
    for a in S:
        s = x + a
        if s in new_sums:
            return False, None
        new_sums.add(s)
    sx = x + x
    if sx in new_sums:
        return False, None
    new_sums.add(sx)
    return True, new_sums

def _can_add_diff(S, x, diff_set):
    """Check if differences from x to elements in S don't conflict with diff_set."""
    for a in S:
        d = abs(x - a)
        if d > 0 and d in diff_set:
            return False
    return True

def _greedy_structured(max_n=256, max_size=64):
    """
    Structured greedy: iterate 1..max_n in order, try to add to smaller set first.
    This balances the two sets and exploits the ordered nature.
    """
    A = []
    B = []
    diff_A = set()
    diff_B = set()
    sums_A = set()
    sums_B = set()
    
    for x in range(1, max_n + 1):
        # Try to add to smaller set first
        if len(A) <= len(B):
            # Try A first
            if len(A) < max_size:
                ok, new_sums = _can_add_to_sidon(A, x, sums_A)
                if ok and _can_add_diff(A, x, diff_A):
                    # Also check diff doesn't conflict with B's diffs
                    if not (diff_A & diff_B):
                        A.append(x)
                        A.sort()
                        sums_A = new_sums
                        for a in A:
                            if a != x:
                                d = abs(x - a)
                                if d > 0:
                                    diff_A.add(d)
                        continue
            # Try B
            if len(B) < max_size:
                ok, new_sums = _can_add_to_sidon(B, x, sums_B)
                if ok and _can_add_diff(B, x, diff_B):
                    B.append(x)
                    B.sort()
                    sums_B = new_sums
                    for b in B:
                        if b != x:
                            d = abs(x - b)
                            if d > 0:
                                diff_B.add(d)
                    continue
        else:
            # Try B first
            if len(B) < max_size:
                ok, new_sums = _can_add_to_sidon(B, x, sums_B)
                if ok and _can_add_diff(B, x, diff_B):
                    B.append(x)
                    B.sort()
                    sums_B = new_sums
                    for b in B:
                        if b != x:
                            d = abs(x - b)
                            if d > 0:
                                diff_B.add(d)
                        continue
            # Try A
            if len(A) < max_size:
                ok, new_sums = _can_add_to_sidon(A, x, sums_A)
                if ok and _can_add_diff(A, x, diff_A):
                    A.append(x)
                    A.sort()
                    sums_A = new_sums
                    for a in A:
                        if a != x:
                            d = abs(x - a)
                            if d > 0:
                                diff_A.add(d)
                    continue
    
    return A, B

def _validate(A, B, max_n=256):
    """Validate solution. Returns (is_valid, score)."""
    if not A or not B:
        return False, 0
    if len(A) > 64 or len(B) > 64:
        return False, 0
    for x in A:
        if x < 1 or x > max_n:
            return False, 0
    for x in B:
        if x < 1 or x > max_n:
            return False, 0
    if not _check_sidon(A):
        return False, 0
    if not _check_sidon(B):
        return False, 0
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False, 0
    return True, len(A) + len(B)

def _local_search_swap(A, B, max_n=256, max_size=64, rng=None, time_budget=10.0):
    """
    Local search: try swapping elements in A or B with unused numbers to improve.
    """
    if rng is None:
        rng = random.Random(0)
    
    start = time.time()
    A = list(A)
    B = list(B)
    best_A, best_B = list(A), list(B)
    best_score = len(A) + len(B)
    
    used_numbers = set(A) | set(B)
    available = [x for x in range(1, max_n + 1) if x not in used_numbers]
    
    while time.time() - start < time_budget:
        # Try to add a random available number to A
        if available and len(A) < max_size:
            x = rng.choice(available)
            A_new = A + [x]
            A_new.sort()
            if _check_sidon(A_new):
                dA_new = _diffs(A_new)
                dB = _diffs(B)
                if not (dA_new & dB):
                    A = A_new
                    best_A = list(A)
                    best_score = len(A) + len(B)
                    if x in available:
                        available.remove(x)
        
        # Try to add a random available number to B
        if available and len(B) < max_size:
            x = rng.choice(available)
            B_new = B + [x]
            B_new.sort()
            if _check_sidon(B_new):
                dB_new = _diffs(B_new)
                dA = _diffs(A)
                if not (dB_new & dA):
                    B = B_new
                    best_B = list(B)
                    best_score = len(A) + len(B)
                    if x in available:
                        available.remove(x)
        
        # Try swap: remove an element from A, try to add a different one
        if A and available:
            i = rng.randrange(len(A))
            old = A[i]
            A_tmp = A[:i] + A[i+1:]
            x = rng.choice(available)
            if x != old:
                A_new = A_tmp + [x]
                A_new.sort()
                if _check_sidon(A_new):
                    dA_new = _diffs(A_new)
                    dB = _diffs(B)
                    if not (dA_new & dB):
                        A = A_new
                        best_A = list(A)
                        best_score = len(A) + len(B)
                        if old in available:
                            available.append(old)
                        if x in available:
                            available.remove(x)
        
        # Try swap: remove an element from B, try to add a different one
        if B and available:
            i = rng.randrange(len(B))
            old = B[i]
            B_tmp = B[:i] + B[i+1:]
            x = rng.choice(available)
            if x != old:
                B_new = B_tmp + [x]
                B_new.sort()
                if _check_sidon(B_new):
                    dB_new = _diffs(B_new)
                    dA = _diffs(A)
                    if not (dB_new & dA):
                        B = B_new
                        best_B = list(B)
                        best_score = len(A) + len(B)
                        if old in available:
                            available.append(old)
                        if x in available:
                            available.remove(x)
    
    return best_A, best_B, best_score

def construct():
    """
    Main function: structured greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Phase 1: Structured greedy
    A, B = _greedy_structured(256, 64)
    valid, score = _validate(A, B)
    if not valid:
        # Fallback: simple valid pair
        A, B = [1], [3]
        valid, score = _validate(A, B)
    
    best_A, best_B = list(A), list(B)
    best_score = score if valid else 0
    
    # Phase 2: Local search with multiple seeds
    if time.time() < deadline - 5.0:
        for ls_seed in range(10):
            if time.time() > deadline - 2.0:
                break
            remaining = deadline - 2.0 - time.time()
            if remaining <= 1.0:
                break
            
            rng = random.Random(ls_seed)
            A_ls, B_ls, score_ls = _local_search_swap(
                best_A, best_B, max_n=256, max_size=64,
                rng=rng, time_budget=min(remaining, 5.0)
            )
            
            valid, score = _validate(A_ls, B_ls)
            if valid and score > best_score:
                best_score = score
                best_A, best_B = list(A_ls), list(B_ls)
    
    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
