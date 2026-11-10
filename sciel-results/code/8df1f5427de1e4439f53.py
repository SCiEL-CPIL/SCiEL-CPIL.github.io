"""Combined B2[2] set construction with optimized local search and insertion."""
import time
import random
from collections import defaultdict

def _greedy_build(candidates, counts_init=None):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    chosen = []
    counts = defaultdict(int) if counts_init is None else dict(counts_init)
    chosen_set = set()
    for x in candidates:
        if x in chosen_set:
            continue
        ok = True
        new_sums = []
        for y in chosen:
            s = x + y
            new_sums.append(s)
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            new_sums.append(s2)
            if counts[s2] + 1 > 2:
                ok = False
        if ok:
            for s in new_sums:
                counts[s] += 1
            chosen.append(x)
            chosen_set.add(x)
    return chosen, counts

def _is_valid_b2(chosen):
    """Verify B2[2] property."""
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    return all(v <= 2 for v in counts.values())

def _try_insert(chosen, counts, y, chosen_set):
    """Try to insert y into chosen. Returns True if successful."""
    if y in chosen_set:
        return False
    ok = True
    for z in chosen:
        s = y + z
        if counts.get(s, 0) + 1 > 2:
            ok = False
            break
    if ok:
        s2 = 2 * y
        if counts.get(s2, 0) + 1 > 2:
            ok = False
    if ok:
        for z in chosen:
            s = y + z
            counts[s] = counts.get(s, 0) + 1
        counts[2 * y] = counts.get(2 * y, 0) + 1
        chosen.append(y)
        chosen_set.add(y)
        return True
    return False

def _local_search(chosen, time_limit=15.0):
    """Local search with insertion and swap operations."""
    chosen = sorted(chosen)
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    chosen_set = set(chosen)
    
    start_time = time.time()
    improved = True
    stall_count = 0
    max_stall = 50
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
        
        # Phase 1: Try to insert any candidate that fits (increases size)
        # Compute conflict scores for all non-chosen candidates
        conflict = {}
        for y in range(1, 1001):
            if y in chosen_set:
                continue
            conf = 0
            for z in chosen:
                s = y + z
                if counts.get(s, 0) >= 2:
                    conf += 1
            if conf == 0:
                if counts.get(2 * y, 0) >= 2:
                    conf = 1
            conflict[y] = conf
        
        # Try inserting low-conflict candidates first
        candidates_sorted = sorted(conflict.keys(), key=lambda y: (conflict[y], y))
        for y in candidates_sorted:
            if time.time() - start_time > time_limit:
                break
            if conflict[y] <= 1:
                if _try_insert(chosen, counts, y, chosen_set):
                    improved = True
                    stall_count = 0
                    break
        
        if improved:
            continue
        
        # Phase 2: Try swaps (remove x, add y) to enable future insertions
        # Focus on removing elements that block the most insertions
        chosen_sorted = sorted(chosen)
        for i in range(len(chosen_sorted)):
            if time.time() - start_time > time_limit:
                break
            x = chosen_sorted[i]
            
            # Temporarily remove x
            for j in range(len(chosen)):
                s = x + chosen[j]
                counts[s] -= 1
            counts[2 * x] -= 1
            chosen_set.discard(x)
            
            # Try to add a promising candidate
            best_y = None
            best_conf = 999
            for y in range(1, 1001):
                if y in chosen_set:
                    continue
                conf = 0
                for z in chosen:
                    s = y + z
                    if counts.get(s, 0) >= 2:
                        conf += 1
                if conf == 0 and counts.get(2 * y, 0) >= 2:
                    conf = 1
                if conf < best_conf:
                    best_conf = conf
                    best_y = y
                if conf == 0:
                    break
            
            if best_y is not None and best_conf <= 2:
                if _try_insert(chosen, counts, best_y, chosen_set):
                    improved = True
                    stall_count = 0
                    break
            else:
                # Restore x
                for j in range(len(chosen)):
                    s = x + chosen[j]
                    counts[s] += 1
                counts[2 * x] += 1
                chosen_set.add(x)
        
        if improved:
            continue
        
        # Phase 3: Try 2-for-1 swaps (remove 2 elements, add 1 better one)
        # This can sometimes unlock insertions
        if len(chosen) >= 2:
            for i in range(len(chosen)):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, len(chosen)):
                    if time.time() - start_time > time_limit:
                        break
                    x1 = chosen[i]
                    x2 = chosen[j]
                    
                    # Remove x1 and x2
                    for k in range(len(chosen)):
                        s = x1 + chosen[k]
                        counts[s] -= 1
                    counts[2 * x1] -= 1
                    for k in range(len(chosen)):
                        if k != i:
                            s = x2 + chosen[k]
                            counts[s] -= 1
                    counts[2 * x2] -= 1
                    chosen_set.discard(x1)
                    chosen_set.discard(x2)
                    
                    # Try to add two new candidates
                    added = 0
                    for y in range(1, 1001):
                        if time.time() - start_time > time_limit:
                            break
                        if y in chosen_set:
                            continue
                        if _try_insert(chosen, counts, y, chosen_set):
                            added += 1
                            if added >= 2:
                                break
                    
                    if added >= 2:
                        improved = True
                        stall_count = 0
                        break
                    else:
                        # Restore x1 and x2
                        for k in range(len(chosen)):
                            s = x1 + chosen[k]
                            counts[s] += 1
                        counts[2 * x1] += 1
                        for k in range(len(chosen)):
                            if x2 not in chosen:
                                s = x2 + chosen[k]
                                counts[s] += 1
                        counts[2 * x2] += 1
                        chosen_set.add(x1)
                        chosen_set.add(x2)
            
            if improved:
                continue
        
        # No improvement this iteration
        stall_count += 1
    
    return sorted(chosen)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    # Strategy 1: Standard greedy (increasing order) - fast baseline
    candidates = list(range(1, 1001))
    chosen1, _ = _greedy_build(candidates)
    
    best = chosen1[:]
    best_size = len(best)
    
    # Strategy 2: Try several random orderings
    rng = random.Random(42)
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        cand = list(range(1, 1001))
        
        # Different mixing strategies
        if restart % 3 == 0:
            rng.shuffle(cand)
        elif restart % 3 == 1:
            # Partial shuffle: keep first third sorted, shuffle rest
            third = len(cand) // 3
            rng.shuffle(cand[third:])
        else:
            # Random with bias toward smaller numbers
            weights = [1001 - x for x in cand]
            selected = rng.choices(range(1, 1001), weights=weights, k=1000)
            # Ensure uniqueness by deduping
            seen = set()
            unique_cand = []
            for x in selected:
                if x not in seen:
                    seen.add(x)
                    unique_cand.append(x)
            cand = unique_cand + [x for x in range(1, 1001) if x not in seen]
        
        chosen_r, _ = _greedy_build(cand)
        if len(chosen_r) > best_size:
            best = chosen_r[:]
            best_size = len(best)
    
    # Local search on the best solution
    if time.time() < deadline - 3.0:
        ls_time = min(20.0, deadline - time.time() - 2.0)
        if ls_time > 1.0:
            best = _local_search(best, time_limit=ls_time)
            best_size = len(best)
    
    # Verify validity
    if not _is_valid_b2(best):
        best = chosen1[:]
    
    return best
