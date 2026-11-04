import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute neighbors for each word (radius 1)
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    neighbors = []
    for i, c in enumerate(words):
        neigh = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors.append(neigh)
    
    neighbors_np = [np.array(neigh, dtype=np.int32) for neigh in neighbors]

    # Greedy Set Cover for initialization
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    while uncovered.any():
        best_idx = -1
        best_count = 0
        for j in range(N):
            count = np.sum(uncovered[neighbors_np[j]])
            if count > best_count:
                best_count = count
                best_idx = j
        if best_idx == -1:
            break
        selected.append(best_idx)
        uncovered[neighbors_np[best_idx]] = False
        if time.time() > deadline - 5.0:
            break
    
    # Compute coverage counts
    coverage_count = np.zeros(N, dtype=np.int32)
    for j in selected:
        coverage_count[neighbors_np[j]] += 1
    
    # Aggressive reduction: multiple passes
    def reduce_set(selected_list, cov, dl):
        changed = True
        passes = 0
        while changed and time.time() < dl:
            changed = False
            passes += 1
            for idx in range(len(selected_list)):
                j = selected_list[idx]
                if np.min(cov[neighbors_np[j]]) > 1:
                    cov[neighbors_np[j]] -= 1
                    selected_list.pop(idx)
                    changed = True
                    break
        return selected_list

    selected = reduce_set(selected, coverage_count, deadline)
    
    best_selected = list(selected)
    best_size = len(best_selected)
    best_cov = coverage_count.copy()
    
    random.seed(42)
    all_indices = list(range(N))
    selected_set = set(selected)
    
    # Local search: add one, remove many, accept if smaller
    stall_count = 0
    max_stall = 5000
    
    while time.time() < deadline - 3.0:
        if stall_count > max_stall:
            break
        
        candidates = []
        for _ in range(5):
            cand = random.choice(all_indices)
            if cand not in selected_set:
                candidates.append(cand)
        
        if not candidates:
            stall_count += 1
            continue
        
        best_cand = None
        best_score = -1
        for cand in candidates:
            low_cov_mask = (coverage_count[neighbors_np[cand]] <= 1)
            score = np.sum(low_cov_mask)
            if score > best_score:
                best_score = score
                best_cand = cand
        
        if best_cand is None:
            stall_count += 1
            continue
        
        new_selected = list(selected)
        new_selected.append(best_cand)
        new_cov = coverage_count.copy()
        new_cov[neighbors_np[best_cand]] += 1
        
        new_selected = reduce_set(new_selected, new_cov, deadline)
        
        if len(new_selected) < len(selected):
            selected = new_selected
            selected_set = set(selected)
            coverage_count = np.zeros(N, dtype=np.int32)
            for j in selected:
                coverage_count[neighbors_np[j]] += 1
            best_selected = list(selected)
            best_size = len(selected)
            best_cov = coverage_count.copy()
            stall_count = 0
        else:
            stall_count += 1
    
    # Final polish
    final_selected = list(best_selected)
    final_cov = best_cov.copy()
    final_selected = reduce_set(final_selected, final_cov, deadline)
    
    codewords = [words[i] for i in final_selected]
    return codewords
