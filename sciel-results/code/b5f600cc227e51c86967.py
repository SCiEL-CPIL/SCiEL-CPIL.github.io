"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations
import numpy as np

N = 192
MAX_K = 48
DEADLINE_MARGIN = 5.0  # seconds to keep before the 90s kill


def _is_valid_set(selected):
    """Full validation of a set. Used only for final check."""
    sel = list(selected)
    n = len(sel)
    if n < 4:
        return True
    for i in range(n):
        a = sel[i]
        others = [v for v in sel if v != a]
        if len(others) >= 3:
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
    return True

def _valid_addition_fast(x, selected_list):
    """
    Check if adding x to selected_list keeps the set valid.
    Uses vectorized NumPy operations for speed.
    """
    sel = np.array(selected_list, dtype=np.int64)
    n = len(sel)
    
    # Condition 1: x must not divide sum of any 3 distinct others in sel
    if n >= 3:
        # Generate all triples of indices
        idx = np.arange(n)
        i_arr, j_arr, k_arr = np.meshgrid(idx, idx, idx, indexing='ij')
        mask = (i_arr < j_arr) & (j_arr < k_arr)
        i_flat, j_flat, k_flat = i_arr[mask], j_arr[mask], k_arr[mask]
        
        # Compute sums of triples
        triple_sums = sel[i_flat] + sel[j_flat] + sel[k_flat]
        
        # Check if any triple sum is divisible by x
        if np.any(triple_sums % x == 0):
            return False
    
    # Condition 2: For each a in sel, (x + u + v) % a != 0
    # where u, v are in sel \ {a}
    for idx_a, a in enumerate(sel):
        # Get all other elements
        others_mask = np.ones(n, dtype=bool)
        others_mask[idx_a] = False
        others = sel[others_mask]
        m = len(others)
        
        if m < 2:
            continue
            
        # Generate all pairs of distinct others
        oidx = np.arange(m)
        oi_arr, oj_arr = np.meshgrid(oidx, oidx, indexing='ij')
        omask = oi_arr < oj_arr
        oi_flat, oj_flat = oi_arr[omask], oj_arr[omask]
        
        # Compute (x + u + v) for all pairs
        pair_sums = x + others[oi_flat] + others[oj_flat]
        
        # Check if any pair sum plus x is divisible by a
        if np.any(pair_sums % a == 0):
            return False
    
    return True

def _greedy_removal(pool_start, pool_end):
    """Start with all elements in [pool_start, pool_end], remove those causing violations."""
    selected = list(range(pool_start, pool_end + 1))
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            a = selected[i]
            others = [v for v in selected if v != a]
            if len(others) >= 3:
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        selected.pop(i)
                        changed = True
                        break
            if changed:
                break
    return selected

def _local_search(selected, time_limit):
    """
    Improved local search with vectorized validation and more aggressive moves.
    """
    selected = list(selected)
    selected_set = set(selected)
    candidates = [x for x in range(1, N + 1) if x not in selected_set]
    candidates.sort(reverse=True)
    
    deadline = time.time() + time_limit
    best_size = len(selected)
    best_selected = list(selected)
    last_improve_time = time.time()
    stuck_counter = 0
    
    while time.time() < deadline:
        improved = False
        stuck_counter += 1
        
        # Step 1: Try to add any candidate
        for x in candidates:
            if time.time() >= deadline:
                break
            if x in selected_set:
                continue
            if _valid_addition_fast(x, selected):
                selected.append(x)
                selected_set.add(x)
                improved = True
                last_improve_time = time.time()
                stuck_counter = 0
                candidates.remove(x)
                break
        
        if improved:
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = list(selected)
            continue
        
        # Step 2: Try Swap (Remove one, Add one) - try multiple times
        if stuck_counter > 5:
            swap_done = False
            for _ in range(20):
                if time.time() >= deadline:
                    break
                if len(selected) == 0 or not candidates:
                    break
                idx = random.randint(0, len(selected) - 1)
                to_remove = selected[idx]
                sel_without = selected[:idx] + selected[idx+1:]
                sel_without_set = set(sel_without)
                
                x = random.choice(candidates)
                if x in sel_without_set:
                    continue
                if _valid_addition_fast(x, sel_without):
                    selected = sel_without + [x]
                    selected_set = sel_without_set | {x}
                    candidates.remove(x)
                    if to_remove not in candidates:
                        candidates.append(to_remove)
                    swap_done = True
                    stuck_counter = 0
                    break
            
            if swap_done:
                last_improve_time = time.time()
                continue
        
        # Step 3: Try Remove-1-Add-2
        if not improved and stuck_counter > 10:
            for i in range(len(selected)):
                if time.time() >= deadline:
                    break
                a = selected[i]
                sel_without_a = selected[:i] + selected[i+1:]
                sel_without_a_set = set(sel_without_a)
                
                added = []
                for x in candidates:
                    if time.time() >= deadline:
                        break
                    if x in sel_without_a_set or x == a:
                        continue
                    if _valid_addition_fast(x, sel_without_a):
                        temp_sel = sel_without_a + [x]
                        temp_set = sel_without_a_set | {x}
                        
                        for y in candidates:
                            if time.time() >= deadline:
                                break
                            if y in temp_set or y == a or y == x:
                                continue
                            if _valid_addition_fast(y, temp_sel):
                                added = [x, y]
                                break
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    selected = sel_without_a + added
                    selected_set = sel_without_a_set | set(added)
                    for x in added:
                        if x in candidates:
                            candidates.remove(x)
                    if a not in candidates:
                        candidates.append(a)
                    
                    improved = True
                    last_improve_time = time.time()
                    stuck_counter = 0
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_selected = list(selected)
                    break
        
        # Step 4: Try Swap 2-2 (Remove two, Add two)
        if not improved and stuck_counter > 15:
            swap2_done = False
            for _ in range(10):
                if time.time() >= deadline:
                    break
                if len(selected) < 2 or len(candidates) < 2:
                    break
                idx1 = random.randint(0, len(selected) - 1)
                idx2 = random.randint(0, len(selected) - 2)
                if idx2 >= idx1:
                    idx2 += 1
                
                to_remove1 = selected[idx1]
                to_remove2 = selected[idx2]
                sel_without = selected[:idx1] + selected[idx1+1:idx2] + selected[idx2+1:]
                sel_without_set = set(sel_without)
                
                # Try to add two candidates
                c1, c2 = random.sample(candidates, 2)
                if _valid_addition_fast(c1, sel_without):
                    temp_sel = sel_without + [c1]
                    temp_set = sel_without_set | {c1}
                    if _valid_addition_fast(c2, temp_sel):
                        selected = temp_sel + [c2]
                        selected_set = temp_set | {c2}
                        for c in [c1, c2]:
                            if c in candidates:
                                candidates.remove(c)
                        for r in [to_remove1, to_remove2]:
                            if r not in candidates:
                                candidates.append(r)
                        swap2_done = True
                        stuck_counter = 0
                        break
            
            if swap2_done:
                last_improve_time = time.time()
                continue
        
        # If stuck for a long time, shuffle candidates to try different order
        if not improved and stuck_counter > 20:
            random.shuffle(candidates)
            last_improve_time = time.time()
            stuck_counter = 0

    return best_selected

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 60s limit, keep 5s margin
    
    # Phase 1: Greedy removal from dense upper pool
    best = _greedy_removal(120, 192)
    best_size = len(best)
    
    # Phase 2: Try different pool starts if time permits
    if time.time() < deadline - 20:
        for ps in [100, 80, 60, 140]:
            if time.time() >= deadline - 20:
                break
            candidate = _greedy_removal(ps, 192)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
    
    # Phase 3: Local search to add more elements
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        best = _local_search(best, remaining)
        best_size = len(best)
        
    # Final validation
    if not _is_valid_set(best):
        best = _greedy_removal(144, 192)
        
    return best
# EVOLVE-BLOCK-END
