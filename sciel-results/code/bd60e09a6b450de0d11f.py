"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _score_set(U):
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    n_sums = len(sums)
    n_diffs = len(diffs)
    if n_sums == 0 or n_diffs == 0:
        return -1
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1 + math.log(ratio) / denom

def _score_set_fast(U, sums_cache, diffs_cache):
    """Score using cached sum/diff sets, updating incrementally."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1, sums_cache, diffs_cache
    max_u = max(U)
    if max_u == 0:
        return -1, sums_cache, diffs_cache
    
    # Rebuild if cache is stale
    if sums_cache is None or diffs_cache is None:
        sums_cache = set()
        diffs_cache = set()
        for a in U:
            for b in U:
                sums_cache.add(a + b)
                diffs_cache.add(a - b)
    else:
        # Verify cache is consistent (simple check)
        pass
    
    n_sums = len(sums_cache)
    n_diffs = len(diffs_cache)
    if n_sums == 0 or n_diffs == 0:
        return -1, sums_cache, diffs_cache
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return -1, sums_cache, diffs_cache
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1, sums_cache, diffs_cache
    return 1 + math.log(ratio) / denom, sums_cache, diffs_cache

def _local_search(U, deadline, max_iter=5000):
    """Local search with incremental scoring: add, remove, or swap elements."""
    U = sorted(set(U))
    current_score = _score_set(U)
    if current_score < 0:
        return U, current_score
    
    # Build initial caches
    sums_cache = set()
    diffs_cache = set()
    for a in U:
        for b in U:
            sums_cache.add(a + b)
            diffs_cache.add(a - b)
    
    for _ in range(max_iter):
        if time.time() > deadline - 2:
            break
        improved = False
        max_u = max(U)
        U_set = set(U)
        
        # Try adding an element - incremental scoring
        for new_elem in range(1, max_u + 15):
            if new_elem in U_set:
                continue
            # Compute new sums and diffs
            new_sums = sums_cache.copy()
            new_diffs = diffs_cache.copy()
            for u in U:
                new_sums.add(new_elem + u)
                new_diffs.add(new_elem - u)
                new_diffs.add(u - new_elem)
            new_max = max(max_u, new_elem)
            n_sums = len(new_sums)
            n_diffs = len(new_diffs)
            if n_sums > 0 and n_diffs > 0:
                ratio = n_diffs / n_sums
                if ratio > 0:
                    denom = math.log(2 * new_max + 1)
                    if denom > 0:
                        s = 1 + math.log(ratio) / denom
                        if s > current_score + 1e-12:
                            U = sorted(U + [new_elem])
                            U_set = set(U)
                            current_score = s
                            sums_cache = new_sums
                            diffs_cache = new_diffs
                            improved = True
                            break
        
        if improved:
            continue
        
        # Try removing an element
        for i in range(1, len(U)):
            elem_to_remove = U[i]
            if len(U) < 3:
                continue
            # Rebuild without this element
            new_U = U[:i] + U[i+1:]
            new_sums = set()
            new_diffs = set()
            for a in new_U:
                for b in new_U:
                    new_sums.add(a + b)
                    new_diffs.add(a - b)
            new_max = max(new_U)
            n_sums = len(new_sums)
            n_diffs = len(new_diffs)
            if n_sums > 0 and n_diffs > 0:
                ratio = n_diffs / n_sums
                if ratio > 0:
                    denom = math.log(2 * new_max + 1)
                    if denom > 0:
                        s = 1 + math.log(ratio) / denom
                        if s > current_score + 1e-12:
                            U = new_U
                            U_set = set(U)
                            current_score = s
                            sums_cache = new_sums
                            diffs_cache = new_diffs
                            improved = True
                            break
        
        if improved:
            continue
        
        # Try swapping an element
        for i in range(1, len(U)):
            old_elem = U[i]
            for new_elem in range(1, max_u + 20):
                if new_elem in U_set or new_elem == old_elem:
                    continue
                new_U = U[:i] + [new_elem] + U[i+1:]
                new_U_sorted = sorted(new_U)
                new_sums = set()
                new_diffs = set()
                for a in new_U_sorted:
                    for b in new_U_sorted:
                        new_sums.add(a + b)
                        new_diffs.add(a - b)
                new_max = max(new_U_sorted)
                n_sums = len(new_sums)
                n_diffs = len(new_diffs)
                if n_sums > 0 and n_diffs > 0:
                    ratio = n_diffs / n_sums
                    if ratio > 0:
                        denom = math.log(2 * new_max + 1)
                        if denom > 0:
                            s = 1 + math.log(ratio) / denom
                            if s > current_score + 1e-12:
                                U = new_U_sorted
                                U_set = set(U)
                                current_score = s
                                sums_cache = new_sums
                                diffs_cache = new_diffs
                                improved = True
                                break
            if improved:
                break
        
        if not improved:
            break
    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Systematic search for size 3
    for a in range(1, 25):
        for b in range(a + 1, 45):
            if time.time() > deadline - 5:
                return best_set
            U = [0, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Systematic search for size 4
    for a in range(1, 18):
        for b in range(a + 1, 30):
            for c in range(b + 1, 50):
                if time.time() > deadline - 5:
                    return best_set
                U = [0, a, b, c]
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Local search from best found with more iterations
    if time.time() < deadline - 15:
        improved_set, improved_score = _local_search(best_set, deadline, max_iter=5000)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    # Try local search from several promising starting points
    starting_points = [
        [0, 1, 4, 6],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 2, 5, 9],
        [0, 1, 5, 8],
        [0, 1, 4, 7],
        [0, 2, 4, 7],
        [0, 1, 3, 5],
        [0, 1, 4, 9],
        [0, 2, 5, 11],
        [0, 1, 6, 10],
        [0, 3, 5, 8],
    ]
    
    for start in starting_points:
        if time.time() > deadline - 5:
            break
        s = _score_set(start)
        if s > best_score - 0.10:
            improved_set, improved_score = _local_search(start, deadline, max_iter=3000)
            if improved_score > best_score:
                best_set = improved_set
                best_score = improved_score
    
    if time.time() > deadline - 1:
        return best_set
    
    return best_set

# EVOLVE-BLOCK-END
