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

def _can_add_to_sidon(S, sums, x):
    """Check if x can be added to Sidon set S without violating sum uniqueness."""
    for a in S:
        if x + a in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _add_to_sidon(S, sums, x):
    """Add x to Sidon set S, update sums. Returns new sums set."""
    new_sums = set(sums)
    for a in S:
        new_sums.add(x + a)
    new_sums.add(2 * x)
    return new_sums

def _backtrack_construct(max_n=256, max_size=64, time_limit=50.0, rng=None, diff_pool=None):
    """
    Constraint-propagation backtracking search.
    Assign each integer 1..max_n to A, B, or neither.
    Uses forward checking to prune infeasible branches.
    """
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    
    # diff_pool: dict mapping d -> 'A', 'B', or None (unused)
    # If None, we'll try all differences freely
    if diff_pool is None:
        diff_pool = {}
    
    # State
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    used_diffs_A = set()
    used_diffs_B = set()
    assigned = [0] * (max_n + 1)  # 0=neither, 1=A, 2=B
    
    best_A = []
    best_B = []
    best_score = 0
    
    # Precompute potential conflicts for quick checking
    def check_diff_available(x, pool, used_diffs, other_used_diffs):
        """Check if all diffs from x to elements in pool are available."""
        for a in pool:
            d = abs(x - a)
            if d > 0:
                if d in used_diffs:
                    return False
                if d in other_used_diffs:
                    return False
                # Check diff pool assignment
                if d in diff_pool:
                    assigned_pool = diff_pool[d]
                    if assigned_pool == 'A' and pool is B:
                        return False
                    if assigned_pool == 'B' and pool is A:
                        return False
        return True
    
    def backtrack(idx):
        nonlocal best_A, best_B, best_score
        nonlocal A, B, sums_A, sums_B, used_diffs_A, used_diffs_B, assigned
        
        if time.time() - start_time > time_limit:
            return
        
        if idx > max_n:
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
            return
        
        x = idx
        
        # Option 1: assign to A
        if len(A) < max_size:
            if _can_add_to_sidon(A, sums_A, x):
                # Check differences
                if check_diff_available(x, A, used_diffs_A, used_diffs_B):
                    # Try it
                    A.append(x)
                    new_sums_A = _add_to_sidon(A, sums_A, x)
                    new_diffs_A = set(used_diffs_A)
                    for a in A[:-1]:
                        d = abs(x - a)
                        if d > 0:
                            new_diffs_A.add(d)
                    assigned[x] = 1
                    
                    backtrack(idx + 1)
                    
                    A.pop()
                    used_diffs_A = set(new_diffs_A)  # restore
                    sums_A = set()
                    for i in range(len(A)):
                        for j in range(i, len(A)):
                            sums_A.add(A[i] + A[j])
                    assigned[x] = 0
        
        # Option 2: assign to B
        if len(B) < max_size:
            if _can_add_to_sidon(B, sums_B, x):
                if check_diff_available(x, B, used_diffs_B, used_diffs_A):
                    B.append(x)
                    new_sums_B = _add_to_sidon(B, sums_B, x)
                    new_diffs_B = set(used_diffs_B)
                    for b in B[:-1]:
                        d = abs(x - b)
                        if d > 0:
                            new_diffs_B.add(d)
                    assigned[x] = 2
                    
                    backtrack(idx + 1)
                    
                    B.pop()
                    used_diffs_B = set(new_diffs_B)
                    sums_B = set()
                    for i in range(len(B)):
                        for j in range(i, len(B)):
                            sums_B.add(B[i] + B[j])
                    assigned[x] = 0
        
        # Option 3: skip x
        backtrack(idx + 1)
    
    backtrack(1)
    return best_A, best_B

def _greedy_construct(max_n=256, max_size=64, rng=None, diff_pool=None):
    """
    Fast greedy construction as fallback.
    """
    if rng is None:
        rng = random.Random()
    
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    used_diffs_A = set()
    used_diffs_B = set()
    
    order = list(range(1, max_n + 1))
    rng.shuffle(order)
    
    for x in order:
        # Try A
        if len(A) < max_size and _can_add_to_sidon(A, sums_A, x):
            ok = True
            for a in A:
                d = abs(x - a)
                if d > 0 and (d in used_diffs_A or d in used_diffs_B):
                    ok = False
                    break
            if ok:
                A.append(x)
                sums_A = _add_to_sidon(A, sums_A, x)
                for a in A[:-1]:
                    d = abs(x - a)
                    if d > 0:
                        used_diffs_A.add(d)
                continue
        
        # Try B
        if len(B) < max_size and _can_add_to_sidon(B, sums_B, x):
            ok = True
            for b in B:
                d = abs(x - b)
                if d > 0 and (d in used_diffs_B or d in used_diffs_A):
                    ok = False
                    break
            if ok:
                B.append(x)
                sums_B = _add_to_sidon(B, sums_B, x)
                for b in B[:-1]:
                    d = abs(x - b)
                    if d > 0:
                        used_diffs_B.add(d)
    
    return A, B

def _local_search_improve(A, B, max_n=256, max_size=64, time_budget=5.0, rng=None):
    """
    Local search: try adding elements, swapping elements to improve size.
    """
    if rng is None:
        rng = random.Random()
    
    start_ls = time.time()
    A = list(A)
    B = list(B)
    sums_A = set()
    sums_B = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            sums_A.add(A[i] + A[j])
    for i in range(len(B)):
        for j in range(i, len(B)):
            sums_B.add(B[i] + B[j])
    used_diffs_A = _diffs(A)
    used_diffs_B = _diffs(B)
    all_A = set(A)
    all_B = set(B)
    
    improved = True
    while improved and time.time() - start_ls < time_budget:
        improved = False
        
        # Try adding to A or B
        candidates = [x for x in range(1, max_n + 1) if x not in all_A and x not in all_B]
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_ls > time_budget:
                break
            
            # Try A
            if len(A) < max_size and _can_add_to_sidon(A, sums_A, x):
                ok = True
                for a in A:
                    d = abs(x - a)
                    if d > 0 and (d in used_diffs_A or d in used_diffs_B):
                        ok = False
                        break
                if ok:
                    A.append(x)
                    sums_A = _add_to_sidon(A, sums_A, x)
                    for a in A[:-1]:
                        d = abs(x - a)
                        if d > 0:
                            used_diffs_A.add(d)
                    all_A.add(x)
                    improved = True
                    break
            
            # Try B
            if len(B) < max_size and _can_add_to_sidon(B, sums_B, x):
                ok = True
                for b in B:
                    d = abs(x - b)
                    if d > 0 and (d in used_diffs_B or d in used_diffs_A):
                        ok = False
                        break
                if ok:
                    B.append(x)
                    sums_B = _add_to_sidon(B, sums_B, x)
                    for b in B[:-1]:
                        d = abs(x - b)
                        if d > 0:
                            used_diffs_B.add(d)
                    all_B.add(x)
                    improved = True
                    break
        
        if not improved:
            # Try swaps: remove from one, add to other
            if len(A) > 1:
                idx = rng.randint(0, len(A) - 1)
                removed = A[idx]
                A_temp = A[:idx] + A[idx+1:]
                sums_temp = set()
                for i in range(len(A_temp)):
                    for j in range(i, len(A_temp)):
                        sums_temp.add(A_temp[i] + A_temp[j])
                used_diffs_temp = _diffs(A_temp)
                all_A_temp = set(A_temp)
                
                for x in range(1, max_n + 1):
                    if x in all_A_temp or x in all_B:
                        continue
                    if len(B) < max_size and _can_add_to_sidon(B, sums_B, x):
                        ok = True
                        for b in B:
                            d = abs(x - b)
                            if d > 0 and (d in used_diffs_B or d in used_diffs_temp):
                                ok = False
                                break
                        if ok:
                            B.append(x)
                            sums_B = _add_to_sidon(B, sums_B, x)
                            for b in B[:-1]:
                                d = abs(x - b)
                                if d > 0:
                                    used_diffs_B.add(d)
                            all_B.add(x)
                            A = A_temp
                            sums_A = sums_temp
                            used_diffs_A = used_diffs_temp
                            all_A = all_A_temp
                            improved = True
                            break
                
                if not improved:
                    A = A_temp + [removed]
                    A.sort()
                    all_A = set(A)
                    sums_A = set()
                    for i in range(len(A)):
                        for j in range(i, len(A)):
                            sums_A.add(A[i] + A[j])
                    used_diffs_A = _diffs(A)
            
            if len(B) > 1 and not improved:
                idx = rng.randint(0, len(B) - 1)
                removed = B[idx]
                B_temp = B[:idx] + B[idx+1:]
                sums_temp = set()
                for i in range(len(B_temp)):
                    for j in range(i, len(B_temp)):
                        sums_temp.add(B_temp[i] + B_temp[j])
                used_diffs_temp = _diffs(B_temp)
                all_B_temp = set(B_temp)
                
                for x in range(1, max_n + 1):
                    if x in all_A or x in all_B_temp:
                        continue
                    if len(A) < max_size and _can_add_to_sidon(A, sums_A, x):
                        ok = True
                        for a in A:
                            d = abs(x - a)
                            if d > 0 and (d in used_diffs_A or d in used_diffs_temp):
                                ok = False
                                break
                        if ok:
                            A.append(x)
                            sums_A = _add_to_sidon(A, sums_A, x)
                            for a in A[:-1]:
                                d = abs(x - a)
                                if d > 0:
                                    used_diffs_A.add(d)
                            all_A.add(x)
                            B = B_temp
                            sums_B = sums_temp
                            used_diffs_B = used_diffs_temp
                            all_B = all_B_temp
                            improved = True
                            break
                
                if not improved:
                    B = B_temp + [removed]
                    B.sort()
                    all_B = set(B)
                    sums_B = set()
                    for i in range(len(B)):
                        for j in range(i, len(B)):
                            sums_B.add(B[i] + B[j])
                    used_diffs_B = _diffs(B)
    
    return A, B

def construct():
    """
    Main entry point. Uses constraint-propagation backtracking with
    difference-pool pre-partitioning, plus greedy fallback and local search.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    best_A, best_B = [1], [3]
    best_score = 2
    
    max_n = 256
    max_size = 64
    
    # Strategy 1: Backtracking with random difference pools
    for rnd in range(10):
        if time.time() > deadline - 10.0:
            break
        
        rng = random.Random(rnd * 104729 + 7)
        
        # Create random difference pool assignment
        diff_pool = {}
        for d in range(1, max_n):
            r = rng.random()
            if r < 0.40:
                diff_pool[d] = 'A'
            elif r < 0.80:
                diff_pool[d] = 'B'
            # else: unused
        
        # Run backtracking with time limit
        bt_time = min(15.0, deadline - 10.0 - (time.time() - start_time))
        if bt_time > 2.0:
            A_bt, B_bt = _backtrack_construct(max_n, max_size, bt_time, rng, diff_pool)
            if A_bt and B_bt:
                valid, score = _validate(A_bt, B_bt)
                if valid and score > best_score:
                    best_score = score
                    best_A, best_B = list(A_bt), list(B_bt)
        
        if best_score >= 50:
            break
    
    # Strategy 2: Greedy with multiple random orders
    for rnd in range(20):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(rnd * 31 + 999)
        A_g, B_g = _greedy_construct(max_n, max_size, rng)
        if A_g and B_g:
            # Local search
            ls_budget = min(3.0, deadline - 5.0 - (time.time() - start_time))
            if ls_budget > 0.5:
                A_g, B_g = _local_search_improve(A_g, B_g, max_n, max_size, ls_budget, rng)
            
            valid, score = _validate(A_g, B_g)
            if valid and score > best_score:
                best_score = score
                best_A, best_B = list(A_g), list(B_g)
        
        if best_score >= 55:
            break
    
    # Strategy 3: Final local search on best found
    if time.time() < deadline - 2.0:
        final_budget = min(5.0, deadline - 2.0 - (time.time() - start_time))
        if final_budget > 1.0:
            rng_final = random.Random(42)
            A_final, B_final = _local_search_improve(best_A, best_B, max_n, max_size, final_budget, rng_final)
            valid, score = _validate(A_final, B_final)
            if valid and score > best_score:
                best_score = score
                best_A, best_B = list(A_final), list(B_final)
    
    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
