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

def _is_maximal_fast(sel):
    """Fast check if sel is inclusion-maximal using precomputed conflict info."""
    sel_set = set(sel)
    sum_mask = _build_sum_mask(sel)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # x can be added if for all a in sel, (a+x) not in sum_mask, AND 2x not in sum_mask
        # (No internal collisions possible since x+a == x+b implies a==b, and x+a==2x implies a==x which is not in sel)
        blocked = False
        for a in sel:
            s = a + x
            if (sum_mask >> s) & 1:
                blocked = True
                break
        if blocked:
            continue
        if (sum_mask >> (2 * x)) & 1:
            continue
        # x can be added, so not maximal
        return False
    return True

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
            ok = True
            for a in selected:
                s = a + x
                if (sum_mask >> s) & 1:
                    ok = False
                    break
            if not ok:
                continue
            if (sum_mask >> (2 * x)) & 1:
                continue
            
            # Count how many remaining elements become blocked by adding x
            new_sums = set()
            for a in selected:
                new_sums.add(a + x)
            new_sums.add(2 * x)
            all_sum_mask = sum_mask
            for s in new_sums:
                all_sum_mask |= (1 << s)
            
            blocked = 0
            for y in remaining:
                if y == x:
                    continue
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
            if _is_maximal_fast(selected):
                break
    
    return selected

def _local_search_aggressive(sel, rng, time_budget=15.0):
    """Aggressive local search: try removals and swaps to find smaller maximal sets."""
    start = time.time()
    best = sorted(sel)
    best_len = len(best)
    
    for iteration in range(2000):
        if time.time() - start > time_budget:
            break
        if best_len <= 7:
            break
        
        cur = sorted(best)
        improvement_found = False
        
        # Strategy 1: Try removing each element
        for i in range(len(cur)):
            if time.time() - start > time_budget:
                break
            x_remove = cur[i]
            new_sel = [x for x in cur if x != x_remove]
            
            if _is_maximal_fast(new_sel):
                if len(new_sel) < best_len:
                    best = new_sel
                    best_len = len(new_sel)
                    improvement_found = True
                    break
        
        if improvement_found:
            continue
        
        # Strategy 2: Try swap - remove one, add another, then try to reduce
        cur = sorted(best)
        for i in range(len(cur)):
            if time.time() - start > time_budget:
                break
            x_remove = cur[i]
            new_sel = [x for x in cur if x != x_remove]
            new_sel_set = set(new_sel)
            
            candidates = list(set(range(1, N + 1)) - new_sel_set)
            rng.shuffle(candidates)
            
            for x_add in candidates[:20]:
                if time.time() - start > time_budget:
                    break
                ok = True
                sum_mask_tmp = _build_sum_mask(new_sel)
                for a in new_sel:
                    s = a + x_add
                    if (sum_mask_tmp >> s) & 1:
                        ok = False
                        break
                if not ok:
                    continue
                if (sum_mask_tmp >> (2 * x_add)) & 1:
                    continue
                
                new_final = sorted(new_sel + [x_add])
                
                for j in range(len(new_final)):
                    x_remove2 = new_final[j]
                    smaller = [x for x in new_final if x != x_remove2]
                    if _is_maximal_fast(smaller):
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
    
    # Phase 1: Multiple randomized greedy runs
    for seed in range(300):
        if time.time() > deadline - 12:
            break
        sel = _greedy_with_blocking(random_seed=seed, time_budget=2.0)
        if sel and _is_sidon(sel):
            if _is_maximal_fast(sel):
                if len(sel) < best_len:
                    best_len = len(sel)
                    best_set = sel
                    if best_len <= 10:
                        break
    
    # Also try the original ascending greedy as baseline
    selected, sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    if _is_maximal_fast(selected):
        if len(selected) < best_len:
            best_len = len(selected)
            best_set = selected
    
    if best_set is None:
        return selected
    
    # Phase 2: Aggressive local search
    if time.time() < deadline - 3:
        rng = random.Random(12345)
        improved = _local_search_aggressive(best_set, rng, time_budget=min(25, deadline - time.time() - 1))
        if improved and _is_sidon(improved) and _is_maximal_fast(improved):
            if len(improved) < best_len:
                best_set = improved
                best_len = len(improved)
    
    # Phase 3: More greedy runs if time remains and we haven't found small enough
    if time.time() < deadline - 5 and best_len > 12:
        for seed in range(300, 500):
            if time.time() > deadline - 5:
                break
            sel = _greedy_with_blocking(random_seed=seed, time_budget=1.5)
            if sel and _is_sidon(sel):
                if _is_maximal_fast(sel):
                    if len(sel) < best_len:
                        best_len = len(sel)
                        best_set = sel
                        if best_len <= 10:
                            break
                    if time.time() < deadline - 8:
                        rng2 = random.Random(seed * 7 + 1)
                        improved2 = _local_search_aggressive(sel, rng2, time_budget=min(5, deadline - time.time() - 3))
                        if improved2 and _is_sidon(improved2) and _is_maximal_fast(improved2):
                            if len(improved2) < best_len:
                                best_len = len(improved2)
                                best_set = improved2
    
    # Final verification
    if best_set and _is_sidon(best_set) and _is_maximal_fast(best_set):
        return best_set
    
    return selected
