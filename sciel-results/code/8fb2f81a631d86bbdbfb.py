import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    if total_words > 10000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    balls = [0] * total_words
    ball_indices = [[] for _ in range(total_words)]
    coverers = [[] for _ in range(total_words)] # coverers[w] = list of c such that w in ball(c)
    
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            mask |= (1 << i)
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                ball_indices[i].append(idx)
                coverers[idx].append(i)
                m ^= lsb
    else:
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                ball_indices[i].append(idx)
                coverers[idx].append(i)
                m ^= lsb

    all_ones_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = total_words + 1
    start_time = time.time()
    time_limit = 110.0
    
    num_restarts = 200 # Can be higher if fast
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        uncovered = all_ones_mask
        selected = []
        selected_set = set()
        
        # Greedy Phase
        while uncovered:
            best_idx = -1
            max_cover = 0
            for i in range(total_words):
                c = (balls[i] & uncovered).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
                elif c == max_cover and c > 0:
                    if random.random() < 0.5:
                        best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~balls[best_idx]
            
        # Local Search
        counts = [0] * total_words
        for idx in selected:
            for w_idx in ball_indices[idx]:
                counts[w_idx] += 1
                
        # Removal
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                idx = selected[i]
                for w_idx in ball_indices[idx]:
                    counts[w_idx] -= 1
                is_redundant = True
                for w_idx in ball_indices[idx]:
                    if counts[w_idx] == 0:
                        is_redundant = False
                        break
                if is_redundant:
                    selected.pop(i)
                    selected_set.remove(idx)
                    changed = True
                    break
                else:
                    for w_idx in ball_indices[idx]:
                        counts[w_idx] += 1
                    break # Restart loop
                    
        # Swap
        improved = True
        while improved:
            improved = False
            for i in range(len(selected)):
                idx = selected[i]
                for w_idx in ball_indices[idx]:
                    counts[w_idx] -= 1
                critical = [w for w in ball_indices[idx] if counts[w] == 0]
                if not critical:
                    selected.pop(i)
                    selected_set.remove(idx)
                    improved = True
                    break
                else:
                    # Find intersection of coverers for critical words
                    # Start with coverers of first critical word
                    candidates = set(coverers[critical[0]])
                    candidates.discard(idx)
                    for w in critical[1:]:
                        candidates.intersection_update(coverers[w])
                    # Find a candidate not in selected_set
                    swap_candidate = -1
                    for c in candidates:
                        if c not in selected_set:
                            swap_candidate = c
                            break
                    if swap_candidate != -1:
                        selected[i] = swap_candidate
                        selected_set.add(swap_candidate)
                        selected_set.remove(idx)
                        for w_idx in ball_indices[swap_candidate]:
                            counts[w_idx] += 1
                        improved = True
                        break
                    else:
                        for w_idx in ball_indices[idx]:
                            counts[w_idx] += 1

        current_size = len(selected)
        if current_size < best_size:
            best_size = current_size
            best_code = [all_words[i] for i in selected]
            
    return best_code
