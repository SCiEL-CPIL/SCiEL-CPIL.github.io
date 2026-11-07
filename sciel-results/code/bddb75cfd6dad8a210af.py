import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 12
    m = 40
    total = 1 << n
    
    # Precompute comparability as bitmasks (symmetric)
    comp = [0] * total
    for i in range(total):
        for j in range(i + 1, total):
            if (i & j) == i or (i & j) == j:
                comp[i] |= (1 << j)
                comp[j] |= (1 << i)
    
    def score_of(sel, sel_mask):
        s = 0
        for x in sel:
            s += (comp[x] & sel_mask).bit_count()
        return s // 2
    
    def make_initial(seed):
        rng = random.Random(seed)
        initial = []
        initial.append(0)
        for i in range(n):
            initial.append(1 << i)
        for i in range(1, n):
            initial.append((1 << 0) | (1 << i))
        for i in range(2, n):
            initial.append((1 << 0) | (1 << 1) | (1 << i))
        for i in range(3, n):
            initial.append((1 << 0) | (1 << 1) | (1 << 2) | (1 << i))
        seen = set()
        selected = []
        for mask in initial:
            if mask not in seen:
                seen.add(mask)
                selected.append(mask)
        if len(selected) >= m:
            rng.shuffle(selected)
            selected = selected[:m]
        else:
            all_masks = list(range(total))
            rng.shuffle(all_masks)
            for mask in all_masks:
                if mask not in seen:
                    seen.add(mask)
                    selected.append(mask)
                    if len(selected) == m:
                        break
        return selected
    
    best_score = -1
    best_sel = None
    
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        selected = make_initial(r)
        sel_mask = 0
        for x in selected:
            sel_mask |= (1 << x)
        current = score_of(selected, sel_mask)
        best_local = current
        best_local_sel = list(selected)
        
        # Simulated annealing with single swaps
        temperature = 10.0
        cooling = 0.9995
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(m):
                if time.time() >= deadline - 2.0:
                    break
                old = selected[idx]
                sel_wo = sel_mask & ~(1 << old)
                loss = (comp[old] & sel_wo).bit_count()
                
                new = random.randrange(total)
                if (sel_mask >> new) & 1:
                    continue
                gain = (comp[new] & sel_wo).bit_count()
                delta = gain - loss
                
                accept = False
                if delta >= 0:
                    accept = True
                else:
                    ratio = delta / max(temperature, 0.1)
                    if ratio > -50:
                        accept = random.random() < pow(2.718281828, ratio)
                
                if accept:
                    selected[idx] = new
                    sel_mask = (sel_mask & ~(1 << old)) | (1 << new)
                    current += delta
                    if current > best_local:
                        best_local = current
                        best_local_sel = list(selected)
                    improved = True
                    break
            temperature *= cooling
        
        if best_local > best_score:
            best_score = best_local
            best_sel = best_local_sel
    
    # Final greedy polish with vectorized candidate ranking
    if best_sel is not None and time.time() < deadline - 1.0:
        selected = list(best_sel)
        sel_mask = 0
        for x in selected:
            sel_mask |= (1 << x)
        current = best_score
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for idx in range(m):
                if time.time() >= deadline - 0.5:
                    break
                old = selected[idx]
                sel_wo = sel_mask & ~(1 << old)
                loss = (comp[old] & sel_wo).bit_count()
                # Vectorized: compute gain for all candidates not in sel_mask
                all_idx = np.arange(total, dtype=np.int64)
                # comp rows as bitmasks; we need bit_count of (comp[new] & sel_wo)
                # Use numpy: for each new, count bits in comp[new] & sel_wo
                # Precompute comp as numpy array of uint64? total=4096, fits in uint64
                # But we already have comp as Python ints. Let's use a fast loop with early exit.
                best_gain = -1
                best_new = None
                # Try candidates in order of popcount (likely to be comparable)
                for new in range(total):
                    if (sel_mask >> new) & 1:
                        continue
                    gain = (comp[new] & sel_wo).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new
                if best_new is not None and best_gain > loss:
                    selected[idx] = best_new
                    sel_mask = (sel_mask & ~(1 << old)) | (1 << best_new)
                    current += best_gain - loss
                    improved = True
                    break
        
        if current > best_score:
            best_score = current
            best_sel = selected
    
    # 2-opt local search: try swapping two positions simultaneously
    if best_sel is not None and time.time() < deadline - 0.5:
        selected = list(best_sel)
        sel_mask = 0
        for x in selected:
            sel_mask |= (1 << x)
        current = best_score
        improved = True
        while improved and time.time() < deadline - 0.3:
            improved = False
            for i in range(m):
                if time.time() >= deadline - 0.3:
                    break
                for j in range(i + 1, m):
                    if time.time() >= deadline - 0.3:
                        break
                    old_i = selected[i]
                    old_j = selected[j]
                    # Remove both
                    sel_wo = sel_mask & ~(1 << old_i) & ~(1 << old_j)
                    loss = ((comp[old_i] & sel_wo).bit_count() +
                             (comp[old_j] & sel_wo).bit_count())
                    # Try swapping with random candidates
                    for attempt in range(20):
                        new_i = random.randrange(total)
                        new_j = random.randrange(total)
                        if new_i == new_j:
                            continue
                        if (sel_mask >> new_i) & 1 or (sel_mask >> new_j) & 1:
                            continue
                        if new_i == old_i or new_i == old_j:
                            continue
                        if new_j == old_i or new_j == old_j:
                            continue
                        gain = ((comp[new_i] & sel_wo).bit_count() +
                                (comp[new_j] & sel_wo).bit_count())
                        delta = gain - loss
                        if delta > 0:
                            selected[i] = new_i
                            selected[j] = new_j
                            sel_mask = (sel_mask & ~(1 << old_i) & ~(1 << old_j)) | (1 << new_i) | (1 << new_j)
                            current += delta
                            if current > best_score:
                                best_score = current
                                best_sel = list(selected)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if current > best_score:
            best_score = current
            best_sel = selected
    
    result = []
    for mask in best_sel:
        result.append([b for b in range(n) if (mask >> b) & 1])
    return [[int(x) for x in sl] for sl in result]
