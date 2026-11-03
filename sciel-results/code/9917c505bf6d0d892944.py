"""Combined program: numpy-accelerated scoring + Golomb rulers + aggressive local search."""
import time
import math
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score for a set U containing 0 using numpy for speed."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u <= 0:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / denom
    return score


def _golomb_ruler(n):
    """Generate a Golomb ruler of order n."""
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99],
        16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115],
        17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132],
        18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151],
        19: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171],
        20: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 191],
    }
    if n in known:
        return known[n]
    # Greedy construction for larger n
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler


def _exhaustive_small(max_size=6, max_val=30):
    """Exhaustive search over small sets to find good configurations."""
    best_set = [0]
    best_score = 0.0
    candidates = list(range(1, max_val + 1))
    
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(candidates, size):
            U = [0] + list(combo)
            s = _compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set, best_score


def _local_search_add(U, max_iterations=10000, time_limit=20.0):
    """Local search focusing on adding new elements."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
    U = sorted(U)
    max_u = U[-1]
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    random.seed(42)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        margin = max(10, max_u // 3)
        new_val = random.randint(1, max_u + margin)
        if new_val in U:
            continue
        
        new_U = sorted(U + [new_val])
        new_score = _compute_score(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
            U = new_U
            max_u = U[-1]
    
    return best_U, best_score


def _local_search_swap(U, max_iterations=10000, time_limit=20.0):
    """Local search with swap operations."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
    U = sorted(U)
    n = len(U)
    max_u = U[-1]
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    random.seed(123)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        if n > 1:
            i = random.randint(1, n - 1)
            old_val = U[i]
            margin = max(10, max_u // 3)
            new_val = random.randint(1, max_u + margin)
            if new_val != old_val and new_val not in U:
                new_U = list(U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                new_score = _compute_score(new_U)
                
                if new_score > best_score:
                    best_score = new_score
                    best_U = new_U
                    U = new_U
                    max_u = U[-1]
    
    return best_U, best_score


def _local_search_remove(U, max_iterations=5000, time_limit=15.0):
    """Local search with removal operations."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
    U = sorted(U)
    n = len(U)
    max_u = U[-1]
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    random.seed(77)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        if n > 1:
            i = random.randint(1, n - 1)
            new_U = list(U)
            new_U.pop(i)
            new_U = sorted(new_U)
            new_score = _compute_score(new_U)
            
            if new_score > best_score:
                best_score = new_score
                best_U = new_U
                U = new_U
                n = len(U)
                max_u = U[-1] if U else 0
    
    return best_U, best_score


def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    best_set = [0, 1, 3]
    best_score = _compute_score(best_set)
    
    # Phase 1: Exhaustive search for small sets
    if time.time() < deadline - 60:
        ex_set, ex_score = _exhaustive_small(max_size=6, max_val=25)
        if ex_score > best_score:
            best_score = ex_score
            best_set = ex_set
    
    # Phase 2: Golomb rulers (strong structured candidates)
    if time.time() < deadline - 50:
        for n in range(3, 25):
            if time.time() >= deadline - 50:
                break
            try:
                U = _golomb_ruler(n)
                if not U or 0 not in U:
                    continue
                U = sorted(U)
                score = _compute_score(U)
                if score > best_score:
                    best_score = score
                    best_set = U
            except Exception:
                continue
    
    # Phase 3: Other structured constructions
    if time.time() < deadline - 40:
        structured_candidates = []
        
        # Geometric progressions
        for ratio in range(2, 8):
            for terms in range(3, 15):
                gp = [0]
                val = 1
                for _ in range(terms):
                    gp.append(val)
                    val *= ratio
                structured_candidates.append(gp)
        
        # Powers of 2
        for n in range(2, 15):
            structured_candidates.append([0] + [2**i for i in range(n)])
        
        # Sparse arithmetic progressions
        for step in range(2, 10):
            for length in range(3, 12):
                structured_candidates.append([0] + [step * i for i in range(1, length)])
        
        # Mixed constructions
        structured_candidates.append([0, 1, 2, 5, 10, 20, 40])
        structured_candidates.append([0, 1, 3, 7, 15, 31, 63])
        structured_candidates.append([0, 1, 2, 4, 8, 16, 32, 64])
        structured_candidates.append([0, 1, 3, 6, 10, 15, 21])
        
        for cand in structured_candidates:
            if time.time() >= deadline - 40:
                break
            s = _compute_score(cand)
            if s > best_score:
                best_score = s
                best_set = cand
    
    # Phase 4: Random sets
    if time.time() < deadline - 30:
        random.seed(42)
        for trial in range(30):
            if time.time() >= deadline - 30:
                break
            n = random.randint(3, 20)
            max_val = random.randint(10, 200)
            U = [0] + random.sample(range(1, max_val + 1), n - 1)
            U = sorted(U)
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_set = U
    
    # Phase 5: Local search - add
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 20
        if remaining > 5:
            ls_set, ls_score = _local_search_add(best_set, max_iterations=20000, time_limit=min(remaining, 30))
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set
    
    # Phase 6: Local search - swap
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 15
        if remaining > 5:
            ls_set, ls_score = _local_search_swap(best_set, max_iterations=20000, time_limit=min(remaining, 25))
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set
    
    # Phase 7: Local search - remove
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 10
        if remaining > 5:
            ls_set, ls_score = _local_search_remove(best_set, max_iterations=10000, time_limit=min(remaining, 15))
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set
    
    # Phase 8: Final polish - combined local search with perturbations
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        for _ in range(3):
            if time.time() >= deadline - 5:
                break
            U = list(best_set)
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                U.pop(idx)
            max_u = max(U) if U else 0
            new_val = random.randint(1, max_u + max(10, max_u // 3))
            if new_val not in U:
                U.append(new_val)
            
            sub_remaining = deadline - time.time() - 5
            if sub_remaining > 3:
                U, score = _local_search_add(U, max_iterations=5000, time_limit=min(sub_remaining, 10))
                if score > best_score:
                    best_score = score
                    best_set = U
    
    # Ensure 0 is in the set
    if 0 not in best_set:
        best_set.append(0)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
