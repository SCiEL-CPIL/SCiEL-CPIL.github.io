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

def _construct_from_partition(pool_A, pool_B, max_n=256, max_size=64, time_budget=5.0, rng=None):
    """
    Given a partition of differences into pool_A and pool_B, greedily construct
    two Sidon sets where each set only uses differences from its pool.
    We add elements in decreasing order to maximize spread.
    """
    if rng is None:
        rng = random.Random(0)
    
    start = time.time()
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    used_diffs_A = set()
    used_diffs_B = set()
    
    # Process elements in decreasing order
    for x in range(max_n, 0, -1):
        if time.time() - start > time_budget:
            break
        
        # Try adding to A
        if len(A) < max_size:
            ok = True
            new_diffs = set()
            for a in A:
                s = x + a
                if s in sums_A:
                    ok = False
                    break
                d = abs(x - a)
                if d > 0:
                    if d not in pool_A or d in used_diffs_A:
                        ok = False
                        break
                    new_diffs.add(d)
            if ok:
                sx = x + x
                if sx in sums_A:
                    ok = False
            if ok:
                A.append(x)
                for a in A:
                    if a != x:
                        sums_A.add(x + a)
                sums_A.add(x + x)
                used_diffs_A |= new_diffs
                continue
        
        # Try adding to B
        if len(B) < max_size:
            ok = True
            new_diffs = set()
            for b in B:
                s = x + b
                if s in sums_B:
                    ok = False
                    break
                d = abs(x - b)
                if d > 0:
                    if d not in pool_B or d in used_diffs_B:
                        ok = False
                        break
                    new_diffs.add(d)
            if ok:
                sx = x + x
                if sx in sums_B:
                    ok = False
            if ok:
                B.append(x)
                for b in B:
                    if b != x:
                        sums_B.add(x + b)
                sums_B.add(x + x)
                used_diffs_B |= new_diffs
    
    return A, B

def _construct_from_partition_increasing(pool_A, pool_B, max_n=256, max_size=64, time_budget=5.0, rng=None):
    """Same as above but in increasing order."""
    if rng is None:
        rng = random.Random(0)
    
    start = time.time()
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    used_diffs_A = set()
    used_diffs_B = set()
    
    for x in range(1, max_n + 1):
        if time.time() - start > time_budget:
            break
        
        # Try adding to A
        if len(A) < max_size:
            ok = True
            new_diffs = set()
            for a in A:
                s = x + a
                if s in sums_A:
                    ok = False
                    break
                d = abs(x - a)
                if d > 0:
                    if d not in pool_A or d in used_diffs_A:
                        ok = False
                        break
                    new_diffs.add(d)
            if ok:
                sx = x + x
                if sx in sums_A:
                    ok = False
            if ok:
                A.append(x)
                for a in A:
                    if a != x:
                        sums_A.add(x + a)
                sums_A.add(x + x)
                used_diffs_A |= new_diffs
                continue
        
        # Try adding to B
        if len(B) < max_size:
            ok = True
            new_diffs = set()
            for b in B:
                s = x + b
                if s in sums_B:
                    ok = False
                    break
                d = abs(x - b)
                if d > 0:
                    if d not in pool_B or d in used_diffs_B:
                        ok = False
                        break
                    new_diffs.add(d)
            if ok:
                sx = x + x
                if sx in sums_B:
                    ok = False
            if ok:
                B.append(x)
                for b in B:
                    if b != x:
                        sums_B.add(x + b)
                sums_B.add(x + x)
                used_diffs_B |= new_diffs
    
    return A, B

def _local_search_improve(A, B, max_n=256, max_size=64, time_budget=3.0, rng=None):
    """
    Local search: try to add elements or swap elements to increase total size.
    """
    if rng is None:
        rng = random.Random(0)
    
    start = time.time()
    A = list(A)
    B = list(B)
    best_score = len(A) + len(B)
    
    all_diffs_A = _diffs(A)
    all_diffs_B = _diffs(B)
    
    while time.time() - start < time_budget:
        improved = False
        
        # Try adding a new element to A or B
        candidates = [x for x in range(1, max_n + 1) if x not in set(A) and x not in set(B)]
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start > time_budget:
                break
            
            # Try adding to A
            if len(A) < max_size:
                new_A = A + [x]
                if _check_sidon(new_A):
                    new_diffs = _diffs(new_A)
                    if not (new_diffs & all_diffs_B):
                        A = new_A
                        all_diffs_A = new_diffs
                        best_score = len(A) + len(B)
                        improved = True
                        break
            
            # Try adding to B
            if len(B) < max_size:
                new_B = B + [x]
                if _check_sidon(new_B):
                    new_diffs = _diffs(new_B)
                    if not (new_diffs & all_diffs_A):
                        B = new_B
                        all_diffs_B = new_diffs
                        best_score = len(A) + len(B)
                        improved = True
                        break
        
        if not improved:
            # Try swaps: remove an element from A and add a different one
            if len(A) > 1:
                idx = rng.randint(0, len(A) - 1)
                removed = A.pop(idx)
                all_diffs_A = _diffs(A)
                
                for x in range(1, max_n + 1):
                    if x in set(A) or x in set(B):
                        continue
                    new_A = A + [x]
                    if _check_sidon(new_A):
                        new_diffs = _diffs(new_A)
                        if not (new_diffs & all_diffs_B):
                            A = new_A
                            all_diffs_A = new_diffs
                            improved = True
                            break
                
                if not improved:
                    A.append(removed)
                    all_diffs_A = _diffs(A)
            
            # Try swaps for B
            if len(B) > 1:
                idx = rng.randint(0, len(B) - 1)
                removed = B.pop(idx)
                all_diffs_B = _diffs(B)
                
                for x in range(1, max_n + 1):
                    if x in set(A) or x in set(B):
                        continue
                    new_B = B + [x]
                    if _check_sidon(new_B):
                        new_diffs = _diffs(new_B)
                        if not (new_diffs & all_diffs_A):
                            B = new_B
                            all_diffs_B = new_diffs
                            improved = True
                            break
                
                if not improved:
                    B.append(removed)
                    all_diffs_B = _diffs(B)
        
        if not improved:
            break
    
    return A, B

def construct():
    """
    Main: difference-set partitioning + greedy construction + local search.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    best_A, best_B = [1], [3]
    best_score = 2
    
    # Generate multiple random partitions of differences
    num_partitions = 30
    
    for p in range(num_partitions):
        if time.time() > deadline - 5.0:
            break
        
        remaining = deadline - 5.0 - time.time()
        if remaining <= 2.0:
            break
        
        rng = random.Random(p * 1000 + 42)
        
        # Create a random partition: each difference goes to A, B, or free
        pool_A = set()
        pool_B = set()
        
        # Bias: give more differences to the smaller set
        for d in range(1, 256):
            r = rng.random()
            if r < 0.4:
                pool_A.add(d)
            elif r < 0.8:
                pool_B.add(d)
            # else: free (neither set can use it)
        
        # Ensure both pools are non-empty
        if not pool_A:
            pool_A.add(1)
        if not pool_B:
            pool_B.add(2)
        
        # Try both decreasing and increasing order
        for order_type in ['decreasing', 'increasing']:
            if time.time() > deadline - 3.0:
                break
            
            t_budget = min(remaining, 3.0)
            
            if order_type == 'decreasing':
                A, B = _construct_from_partition(pool_A, pool_B, 256, 64, t_budget, rng)
            else:
                A, B = _construct_from_partition_increasing(pool_A, pool_B, 256, 64, t_budget, rng)
            
            if not A or not B:
                continue
            
            # Local search improvement
            if time.time() < deadline - 3.0:
                ls_budget = min(remaining - (time.time() - start_time), 2.0)
                if ls_budget > 0.5:
                    A, B = _local_search_improve(A, B, 256, 64, ls_budget, rng)
            
            valid, score = _validate(A, B)
            if valid and score > best_score:
                best_score = score
                best_A, best_B = list(A), list(B)
        
        # Early termination if we hit the theoretical max
        if best_score >= 64:
            break
    
    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
