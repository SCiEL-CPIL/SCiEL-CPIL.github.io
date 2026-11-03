import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute digits for all words
    digits = [None] * total_space
    for i in range(total_space):
        d = [0] * n
        val = i
        for j in range(n):
            d[j] = val % q
            val //= q
        digits[i] = tuple(d)

    # Precompute neighbor bitmasks
    neighbors = [0] * total_space
    if radius == 1:
        for i in range(total_space):
            m = 1 << i
            for pos in range(n):
                old_val = digits[i][pos]
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    diff = new_val - old_val
                    j = i + diff * powers[pos]
                    m |= (1 << j)
            neighbors[i] = m
    else:
        for i in range(total_space):
            m = 1 << i
            for j in range(total_space):
                dist = sum(1 for k in range(n) if digits[i][k] != digits[j][k])
                if dist <= radius:
                    m |= (1 << j)
            neighbors[i] = m

    full_mask = (1 << total_space) - 1

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def run_greedy():
        covered = 0
        selected = []
        selected_set = set()
        remaining = full_mask
        
        while remaining and time.time() < deadline - 10:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                gain = (neighbors[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            
            if best_word < 0:
                break
            
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= neighbors[best_word]
            remaining = full_mask & ~covered
            
        return selected, covered, selected_set

    def local_search(selected, covered, selected_set, time_limit):
        selected = selected[:]
        covered = covered
        selected_set = selected_set.copy()
        selected_mask = 0
        for w in selected:
            selected_mask |= (1 << w)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Phase 1: Deletion
            i = 0
            while i < len(selected):
                w = selected[i]
                if (covered & ~neighbors[w]) == full_mask:
                    covered &= ~neighbors[w]
                    selected.pop(i)
                    selected_set.remove(w)
                    selected_mask &= ~(1 << w)
                    improved = True
                else:
                    i += 1
            
            # Phase 2: 2-for-1 Swap
            n_sel = len(selected)
            if n_sel < 2:
                break
                
            for i_idx in range(n_sel):
                if time.time() > time_limit:
                    return selected, covered, selected_set, selected_mask
                u = selected[i_idx]
                for j_idx in range(i_idx + 1, n_sel):
                    v = selected[j_idx]
                    
                    holes = full_mask & ~(covered & ~neighbors[u] & ~neighbors[v])
                    
                    if holes == 0:
                        covered &= ~neighbors[u]
                        covered &= ~neighbors[v]
                        selected.pop(j_idx)
                        selected.pop(i_idx)
                        selected_set.remove(u)
                        selected_set.remove(v)
                        selected_mask &= ~(1 << u)
                        selected_mask &= ~(1 << v)
                        improved = True
                        break
                    
                    if holes.bit_count() > 15:
                        continue

                    h_indices = []
                    temp_holes = holes
                    while temp_holes:
                        lb = temp_holes & -temp_holes
                        idx = lb.bit_length() - 1
                        h_indices.append(idx)
                        temp_holes ^= lb
                    
                    if not h_indices:
                        continue
                        
                    cand_mask = neighbors[h_indices[0]]
                    for h in h_indices[1:]:
                        cand_mask &= neighbors[h]
                    
                    cand_mask &= ~selected_mask
                    
                    if cand_mask:
                        lb = cand_mask & -cand_mask
                        w_new = lb.bit_length() - 1
                        
                        covered &= ~neighbors[u]
                        covered &= ~neighbors[v]
                        covered |= neighbors[w_new]
                        
                        selected.pop(j_idx)
                        selected.pop(i_idx)
                        selected.append(w_new)
                        
                        selected_set.remove(u)
                        selected_set.remove(v)
                        selected_set.add(w_new)
                        
                        selected_mask &= ~(1 << u)
                        selected_mask &= ~(1 << v)
                        selected_mask |= (1 << w_new)
                        
                        improved = True
                        break

                if improved:
                    break

        return selected, covered, selected_set, selected_mask

    selected, covered, selected_set = run_greedy()
    
    best_selected = selected[:]
    best_size = len(selected)
    
    while time.time() < deadline - 5:
        selected, covered, selected_set, selected_mask = local_search(
            selected, covered, selected_set, deadline - 5
        )
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]
        
        if len(selected) > 0:
            u_idx = random.randrange(len(selected))
            u = selected[u_idx]
            
            while True:
                w_new = random.randint(0, total_space - 1)
                if w_new not in selected_set:
                    break
            
            new_covered = (covered & ~neighbors[u]) | neighbors[w_new]
            if new_covered == full_mask:
                covered = new_covered
                selected[u_idx] = w_new
                selected_set.remove(u)
                selected_set.add(w_new)
                selected_mask &= ~(1 << u)
                selected_mask |= (1 << w_new)
        
        if random.random() < 0.1:
             w_add = random.randint(0, total_space - 1)
             if w_add not in selected_set:
                 selected.append(w_add)
                 selected_set.add(w_add)
                 covered |= neighbors[w_add]
                 selected_mask |= (1 << w_add)

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
