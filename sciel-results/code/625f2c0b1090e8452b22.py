"""Combined B&B MWIS solver with tighter bounds and enhanced local search."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from math import gcd, lcm
from fractions import Fraction

def construct():
    deadline = time.time() + 85  # 85s safety margin before 90s kill
    t0 = time.time()
    
    n_max = 256
    small = list(range(2, 129))  # 2..128
    large = list(range(129, 257))  # 129..256
    n_small = len(small)
    n_large = len(large)
    
    small_idx = {v: i for i, v in enumerate(small)}
    large_idx = {v: i for i, v in enumerate(large)}
    
    # Small weights
    sw = [Fraction(1, v) for v in small]
    # Large weights
    lw = [Fraction(1, v) for v in large]
    
    # Small-small conflict: lcm(small[i], small[j]) <= 256
    ss_conflict = [0] * n_small
    for i in range(n_small):
        for j in range(i + 1, n_small):
            if lcm(small[i], small[j]) <= 256:
                ss_conflict[i] |= (1 << j)
                ss_conflict[j] |= (1 << i)
    
    # Small-large conflict: small[i] divides large[j]
    sl_conflict = [0] * n_small
    for i, s in enumerate(small):
        for j, L in enumerate(large):
            if L % s == 0:
                sl_conflict[i] |= (1 << j)
    
    # Float weights for fast bound computation
    sw_float = [float(w) for w in sw]
    lw_float = [float(w) for w in lw]
    
    # Suffix small weights (float) - sorted by weight descending for tighter bounds
    # Reorder small vertices by weight descending (smaller number = larger weight)
    # Actually small is already 2..128, so weight 1/v is already descending
    suffix_sw_float = [0.0] * (n_small + 1)
    for i in range(n_small - 1, -1, -1):
        suffix_sw_float[i] = suffix_sw_float[i + 1] + sw_float[i]
    
    total_lw_float = sum(lw_float)
    
    # Precompute: for each small vertex, the float sum of large weights it excludes
    sl_excl_float = [0.0] * n_small
    for i in range(n_small):
        for j in range(n_large):
            if (sl_conflict[i] >> j) & 1:
                sl_excl_float[i] += lw_float[j]
    
    # Greedy seed
    def greedy_small():
        selected = []
        for s in small:
            if all(lcm(s, y) > 256 for y in selected):
                selected.append(s)
        return selected
    
    greedy_sel = greedy_small()
    greedy_mask = 0
    greedy_excluded = 0
    for s in greedy_sel:
        i = small_idx[s]
        greedy_mask |= (1 << i)
        greedy_excluded |= sl_conflict[i]
    
    def compute_exact_score(mask, excluded):
        s = Fraction(0)
        for i in range(n_small):
            if (mask >> i) & 1:
                s += sw[i]
        for j in range(n_large):
            if not (excluded >> j) & 1:
                s += lw[j]
        return s
    
    best_score = compute_exact_score(greedy_mask, greedy_excluded)
    best_small_mask = greedy_mask
    best_large_excluded = greedy_excluded
    
    # Also try: no smalls, all larges
    all_large_score = Fraction(0)
    for j in range(n_large):
        all_large_score += lw[j]
    if all_large_score > best_score:
        best_score = all_large_score
        best_small_mask = 0
        best_large_excluded = 0
    
    # Try several different greedy orderings for better initial solutions
    random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        # Weighted random greedy
        remaining = list(range(n_small))
        random.shuffle(remaining)
        sel_mask = 0
        sel_excl = 0
        for i in remaining:
            if (sel_mask & ss_conflict[i]) == 0:
                sel_mask |= (1 << i)
                sel_excl |= sl_conflict[i]
        sc = compute_exact_score(sel_mask, sel_excl)
        if sc > best_score:
            best_score = sc
            best_small_mask = sel_mask
            best_large_excluded = sel_excl
    
    # Branch and bound with tighter bounds
    sys.setrecursionlimit(10000)
    
    def solve_bnb():
        nonlocal best_score, best_small_mask, best_large_excluded, deadline
        
        def dfs(pos, selected_mask, excluded_large, current_score, current_float, avail_large_float):
            nonlocal best_score, best_small_mask, best_large_excluded, deadline
            
            if time.time() > deadline:
                return
            
            if pos == n_small:
                # Compute exact score
                large_score = Fraction(0)
                for j in range(n_large):
                    if not (excluded_large >> j) & 1:
                        large_score += lw[j]
                total = current_score + large_score
                if total > best_score:
                    best_score = total
                    best_small_mask = selected_mask
                    best_large_excluded = excluded_large
                return
            
            # Tight upper bound: current + remaining smalls + available larges
            # Additional bound: for remaining smalls, use fractional knapsack style
            # Sort remaining by weight/size ratio (here all size=1, so just weight)
            # Since weights are already sorted descending, suffix bound is valid
            if current_float + suffix_sw_float[pos] + avail_large_float <= float(best_score) + 1e-12:
                return
            
            # Option 1: Include vertex pos (if compatible)
            if (selected_mask & ss_conflict[pos]) == 0:
                new_mask = selected_mask | (1 << pos)
                new_excluded = excluded_large | sl_conflict[pos]
                new_score = current_score + sw[pos]
                new_float = current_float + sw_float[pos]
                
                # Compute accurate new available large float
                new_excl_only = sl_conflict[pos] & ~excluded_large
                new_excl_sum = 0.0
                for j in range(n_large):
                    if (new_excl_only >> j) & 1:
                        new_excl_sum += lw_float[j]
                new_avail_large = avail_large_float - new_excl_sum
                
                dfs(pos + 1, new_mask, new_excluded, new_score, new_float, new_avail_large)
            
            # Option 2: Exclude vertex pos
            dfs(pos + 1, selected_mask, excluded_large, current_score, current_float, avail_large_float)
        
        dfs(0, 0, 0, Fraction(0), 0.0, total_lw_float)
    
    solve_bnb()
    
    # Enhanced local search with 1-in, 1-out, 1-for-1, and 2-for-1 exchange moves
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding each non-selected small
            for i in range(n_small):
                if (best_small_mask >> i) & 1:
                    continue
                if (best_small_mask & ss_conflict[i]) != 0:
                    continue
                new_mask = best_small_mask | (1 << i)
                new_excl = best_large_excluded | sl_conflict[i]
                new_score = compute_exact_score(new_mask, new_excl)
                if new_score > best_score:
                    best_score = new_score
                    best_small_mask = new_mask
                    best_large_excluded = new_excl
                    improved = True
            
            # Try removing each selected small
            for i in range(n_small):
                if not (best_small_mask >> i) & 1:
                    continue
                new_mask = best_small_mask & ~(1 << i)
                new_excl = 0
                for k in range(n_small):
                    if (new_mask >> k) & 1:
                        new_excl |= sl_conflict[k]
                new_score = compute_exact_score(new_mask, new_excl)
                if new_score > best_score:
                    best_score = new_score
                    best_small_mask = new_mask
                    best_large_excluded = new_excl
                    improved = True
            
            # Try 1-in-1-out exchange
            for i_out in range(n_small):
                if not (best_small_mask >> i_out) & 1:
                    continue
                for i_in in range(n_small):
                    if (best_small_mask >> i_in) & 1:
                        continue
                    temp_mask = best_small_mask & ~(1 << i_out)
                    if (temp_mask & ss_conflict[i_in]) != 0:
                        continue
                    new_mask = temp_mask | (1 << i_in)
                    new_excl = 0
                    for k in range(n_small):
                        if (new_mask >> k) & 1:
                            new_excl |= sl_conflict[k]
                    new_score = compute_exact_score(new_mask, new_excl)
                    if new_score > best_score:
                        best_score = new_score
                        best_small_mask = new_mask
                        best_large_excluded = new_excl
                        improved = True
                        break
                if improved:
                    break
            
            # Try 2-for-1 exchange: remove two smalls, add one
            if not improved:
                sel_list = [i for i in range(n_small) if (best_small_mask >> i) & 1]
                for i_out1 in sel_list:
                    for i_out2 in sel_list:
                        if i_out2 <= i_out1:
                            continue
                        temp_mask = best_small_mask & ~(1 << i_out1) & ~(1 << i_out2)
                        for i_in in range(n_small):
                            if (best_small_mask >> i_in) & 1:
                                continue
                            if (temp_mask & ss_conflict[i_in]) != 0:
                                continue
                            new_mask = temp_mask | (1 << i_in)
                            new_excl = 0
                            for k in range(n_small):
                                if (new_mask >> k) & 1:
                                    new_excl |= sl_conflict[k]
                            new_score = compute_exact_score(new_mask, new_excl)
                            if new_score > best_score:
                                best_score = new_score
                                best_small_mask = new_mask
                                best_large_excluded = new_excl
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
    
    # Build final set
    result = []
    for i in range(n_small):
        if (best_small_mask >> i) & 1:
            result.append(small[i])
    for j in range(n_large):
        if not (best_large_excluded >> j) & 1:
            result.append(large[j])
    
    if not result:
        return [2]
    
    # Verify constraint
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if lcm(result[i], result[j]) <= 256:
                return greedy_sel
    
    return result
# EVOLVE-BLOCK-END
