"""Optimized program: bitmask-accelerated Sidon checks + randomized greedy + local search."""
import random
import time
import sys

N = 256
MAX_SUM = 2 * N  # 512

def _is_sidon(sel):
    """Check if sel is a Sidon set (all unordered sums a+b with a<=b distinct)."""
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            b = sel[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _build_sum_mask(sel):
    """Build a bitmask of all unordered sums a+b with a<=b from sel."""
    mask = 0
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            mask |= (1 << s)
    return mask

def _is_maximal(sel):
    """Check if sel is inclusion-maximal: no missing element can be added."""
    sel_set = set(sel)
    # Precompute sum mask
    sum_mask = _build_sum_mask(sel)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x preserves Sidon property
        ok = True
        new_sum_mask = 0
        for a in sel:
            s = a + x
            if (sum_mask >> s) & 1 or (new_sum_mask >> s) & 1:
                ok = False
                break
            new_sum_mask |= (1 << s)
        if not ok:
            continue
        # Check 2*x not in sums and not in new_sums
        if (sum_mask >> (2 * x)) & 1 or (new_sum_mask >> (2 * x)) & 1:
            ok = False
        if ok:
            return False  # Can add x, so not maximal
    return True

def _count_blockable(sel):
    """Count how many missing elements are blocked (cannot be added)."""
    sel_set = set(sel)
    sum_mask = _build_sum_mask(sel)
    
    blocked = 0
    total_missing = N - len(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added
        can_add = True
        new_sum_mask = 0
        for a in sel:
            s = a + x
            if (sum_mask >> s) & 1 or (new_sum_mask >> s) & 1:
                can_add = False
                break
            new_sum_mask |= (1 << s)
        if can_add:
            if (sum_mask >> (2 * x)) & 1 or (new_sum_mask >> (2 * x)) & 1:
                can_add = False
        if not can_add:
            blocked += 1
    return blocked, total_missing

def _greedy_with_blocking(random_seed=42, time_budget=5.0):
    """Randomized greedy: at each step, add the element that blocks the most remaining."""
    rng = random.Random(random_seed)
    start = time.time()
    
    selected = []
    sum_mask = 0
    remaining = set(range(1, N + 1))
    
    while remaining:
        if time.time() - start > time_budget:
            break
        
        best_elem = None
        best_block_count = -1
        
        # Sample a subset if too many remaining
        candidates = list(remaining)
        if len(candidates) > 40:
            candidates = rng.sample(candidates, 40)
        
        for x in candidates:
            # Check if x can be added
            new_sum_mask = 0
            ok = True
            for a in selected:
                s = a + x
                if (sum_mask >> s) & 1 or (new_sum_mask >> s) & 1:
                    ok = False
                    break
                new_sum_mask |= (1 << s)
            if not ok:
                continue
            if (sum_mask >> (2 * x)) & 1 or (new_sum_mask >> (2 * x)) & 1:
                continue
            
            # Count how many remaining elements become blocked by adding x
            blocked = 0
            all_sum_mask = sum_mask | new_sum_mask | (1 << (2 * x))
            for y in remaining:
                if y == x:
                    continue
                # Check if y can still be added after adding x
                y_blocked = False
                for a in selected:
                    if (all_sum_mask >> (a + y)) & 1:
                        y_blocked = True
                        break
                if not y_blocked:
                    if (all_sum_mask >> (2 * y)) & 1:
                        y_blocked = True
                if y_blocked:
                    blocked += 1
            
            if blocked > best_block_count or \
               (blocked == best_block_count and (best_elem is None or x < best_elem)):
                best_block_count = blocked
                best_elem = x
        
        if best_elem is None:
            break
        
        # Add best_elem
        selected.append(best_elem)
        for a in selected[:-1]:
            sum_mask |= (1 << (a + best_elem))
        sum_mask |= (1 << (2 * best_elem))
        remaining.discard(best_elem)
        
        # Early termination: if already maximal and small enough, stop
        if len(selected) >= 8:
            # Quick check if maximal
            if _is_maximal(selected):
                break
    
    return selected

def _local_search(sel, rng, time_budget=10.0):
    """Try to improve (reduce size) by removing elements and checking maximality."""
    start = time.time()
    best = sorted(sel)
    best_len = len(best)
    
    for iteration in range(500):
        if time.time() - start > time_budget:
            break
        if best_len <= 7:
            break
        
        cur = sorted(best)
        
        # Try removing each element and check if the result is still maximal
        improvement_found = False
        for i in range(len(cur)):
            if time.time() - start > time_budget:
                break
            x_remove = cur[i]
            new_sel = [x for x in cur if x != x_remove]
            
            # Check if still maximal (subset of Sidon is always Sidon)
            if _is_maximal(new_sel):
                if len(new_sel) < best_len:
                    best = new_sel
                    best_len = len(new_sel)
                    improvement_found = True
                    break  # Found improvement, restart with smaller set
        
        if not improvement_found:
            # Try swap: remove one, add another (same size, may enable future reductions)
            cur = sorted(best)
            for i in range(len(cur)):
                if time.time() - start > time_budget:
                    break
                x_remove = cur[i]
                new_sel = [x for x in cur if x != x_remove]
                new_sel_set = set(new_sel)
                
                # Recompute sum mask without x_remove
                new_sum_mask = _build_sum_mask(new_sel)
                
                # Try adding random candidates
                candidates = list(set(range(1, N + 1)) - new_sel_set)
                rng.shuffle(candidates)
                
                for x_add in candidates[:30]:
                    # Check if x_add can be added
                    ok = True
                    tmp_sum_mask = new_sum_mask
                    for a in new_sel:
                        s = a + x_add
                        if (tmp_sum_mask >> s) & 1:
                            ok = False
                            break
                        tmp_sum_mask |= (1 << s)
                    if not ok:
                        continue
                    if (tmp_sum_mask >> (2 * x_add)) & 1:
                        continue
                    
                    new_final = sorted(new_sel + [x_add])
                    # Check if this new set is maximal
                    if _is_maximal(new_final):
                        # Same size, but might be a better configuration
                        # Try to see if we can reduce from here
                        for j in range(len(new_final)):
                            x_remove2 = new_final[j]
                            smaller = [x for x in new_final if x != x_remove2]
                            if _is_maximal(smaller):
                                if len(smaller) < best_len:
                                    best = smaller
                                    best_len = len(smaller)
                                    improvement_found = True
                                    break
                        if improvement_found:
                            break
                if improvement_found:
                    break
    
    return best

def construct():
    deadline = time.time() + 55  # Safety margin before 60s kill
    
    best_set = None
    best_len = N + 1
    
    # Phase 1: Multiple randomized greedy runs (faster with bitmask)
    for seed in range(200):
        if time.time() > deadline - 15:
            break
        sel = _greedy_with_blocking(random_seed=seed, time_budget=2.5)
        if sel and _is_sidon(sel):
            if _is_maximal(sel):
                if len(sel) < best_len:
                    best_len = len(sel)
                    best_set = sel
                    if best_len <= 8:
                        break
    
    # Also try the original ascending greedy as baseline
    selected, sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    if _is_maximal(selected):
        if len(selected) < best_len:
            best_len = len(selected)
            best_set = selected
    
    if best_set is None:
        # Fallback: return the ascending greedy (always valid)
        return selected
    
    # Phase 2: Local search on best
    if time.time() < deadline - 5:
        rng = random.Random(12345)
        improved = _local_search(best_set, rng, time_budget=min(20, deadline - time.time() - 2))
        if improved and _is_sidon(improved) and _is_maximal(improved):
            if len(improved) < best_len:
                best_set = improved
                best_len = len(improved)
    
    # Phase 3: Additional local search with different random seed
    if time.time() < deadline - 3 and best_len > 8:
        rng2 = random.Random(99999)
        improved2 = _local_search(best_set, rng2, time_budget=min(10, deadline - time.time() - 1))
        if improved2 and _is_sidon(improved2) and _is_maximal(improved2):
            if len(improved2) < best_len:
                best_set = improved2
                best_len = len(improved2)
    
    # Phase 4: More greedy runs with different parameters if time remains
    if time.time() < deadline - 5 and best_len > 10:
        for seed in range(200, 400):
            if time.time() > deadline - 5:
                break
            sel = _greedy_with_blocking(random_seed=seed, time_budget=2.0)
            if sel and _is_sidon(sel):
                if _is_maximal(sel):
                    if len(sel) < best_len:
                        best_len = len(sel)
                        best_set = sel
    
    # Final verification
    if best_set and _is_sidon(best_set) and _is_maximal(best_set):
        return best_set
    
    # Absolute fallback
    return selected
