import itertools
import random
import time
import sys
import numpy as np
from itertools import product

def construct(q, n, radius):
    if n == 0:
        return [[]]
    
    # For this specific task, q=3, n=7, radius=1.
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1()
    
    # Fallback for general radius (not optimized)
    # Greedy set cover
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    def dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
    
    neighbors = [[] for _ in range(len(words))]
    for i, w in enumerate(words):
        for j, v in enumerate(words):
            if dist(w, v) <= radius:
                neighbors[i].append(j)
        
    covered = [False] * len(words)
    code = []
    while not all(covered):
        best_idx = -1
        best_count = -1
        for i in range(len(words)):
            cnt = sum(1 for j in neighbors[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        code.append(words[best_idx])
        for j in neighbors[best_idx]:
            covered[j] = True
    return code

def _solve_ternary_7_1():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (evaluator kills at 150s, but we want buffer)
    
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    
    # Precompute neighbors (ball contents)
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        covered = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            orig_val = w_list[i]
            for val in range(q):
                if val != orig_val:
                    w_list[i] = val
                    covered.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig_val
        neighbors.append(covered)
    
    # Precompute coverers: coverers[point] = list of codewords that cover point
    coverers = [[] for _ in range(N)]
    for cw_idx, ball in enumerate(neighbors):
        for pt in ball:
            coverers[pt].append(cw_idx)
            
    # Precompute ball bitsets for fast greedy
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask
        
    all_mask = (1 << N) - 1
    
    # 1. Initial Construction: Greedy Set Cover using Bitsets
    def greedy_init(seed):
        random.seed(seed)
        current_cover_mask = 0
        codewords = []
        is_chosen = [False] * N
        
        while current_cover_mask != all_mask:
            if time.time() > deadline - 60:
                break
                
            best_i = -1
            best_count = 0
            
            for i in range(N):
                if is_chosen[i]:
                    continue
                new_bits = ball_bitsets[i] & ~current_cover_mask
                if new_bits:
                    count = new_bits.bit_count()
                    if count > best_count:
                        best_count = count
                        best_i = i
                        
            if best_i == -1:
                break # Should not happen
                
            codewords.append(best_i)
            is_chosen[best_i] = True
            current_cover_mask |= ball_bitsets[best_i]
            
        # Ensure validity
        if current_cover_mask != all_mask:
            while current_cover_mask != all_mask:
                i = random.randint(0, N-1)
                if not is_chosen[i]:
                    codewords.append(i)
                    is_chosen[i] = True
                current_cover_mask |= ball_bitsets[i]
                if time.time() > deadline - 10:
                    break
                    
        return codewords

    # Run multiple greedy restarts to find a good initial code
    best_greedy_code = None
    best_greedy_len = float('inf')
    
    # Time budget for greedy: 5s
    greedy_deadline = start_time + 5
    
    seed = 0
    while time.time() < greedy_deadline:
        code = greedy_init(seed)
        if len(code) < best_greedy_len:
            best_greedy_len = len(code)
            best_greedy_code = code
            if best_greedy_len <= 186:
                break
        seed += 1
        
    if best_greedy_code is None:
        best_greedy_code = greedy_init(0)
        
    # 2. Local Search Setup
    codewords = list(best_greedy_code)
    
    # Use numpy for cover counts
    cover_count = np.zeros(N, dtype=np.int16)
    for cw in codewords:
        cover_count[neighbors[cw]] += 1
        
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Helper: Try to remove redundant codewords
    def try_remove_redundant(codewords, cover_count):
        removed = 0
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 30:
                break
            cw = codewords[i]
            # Check if all points in ball(cw) have count > 1
            b = neighbors[cw]
            if np.min(cover_count[b]) > 1:
                cover_count[b] -= 1
                codewords.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    # Initial redundancy removal
    try_remove_redundant(codewords, cover_count)
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)
        
    # Local Search Loop
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2:
        improved = False
        
        # 1. Try Removals
        removed = try_remove_redundant(codewords, cover_count)
        if removed > 0:
            improved = True
            no_improvement_count = 0
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue # Restart loop after modification
        
        # 2. Try Swaps with Critical Word Heuristic
        if codewords:
            # Pick a random codeword to try swapping
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find critical points: points in ball(old_cw) with count == 1
            b_old = neighbors[old_cw]
            critical_points = []
            for pt in b_old:
                if cover_count[pt] == 1:
                    critical_points.append(pt)
            
            if not critical_points:
                # Should have been removed by redundancy check
                # But if it wasn't (e.g. time limit), force remove
                cover_count[b_old] -= 1
                codewords.pop(idx)
                improved = True
                no_improvement_count = 0
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                continue
            
            # Find candidates that cover ALL critical points
            # candidates = intersection of coverers[p] for p in critical_points
            # Start with coverers of the first critical point
            candidate_set = set(coverers[critical_points[0]])
            for p in critical_points[1:]:
                candidate_set.intersection_update(coverers[p])
            
            # Remove old_cw and existing codewords from candidates
            candidate_set.discard(old_cw)
            for cw in codewords:
                candidate_set.discard(cw)
                
            if candidate_set:
                # Pick the best candidate
                # Heuristic: Pick the one with the lowest "cost"
                # Cost could be the sum of cover_count in its ball (lower is better, less redundancy)
                best_new_cw = -1
                min_cost = float('inf')
                
                # Limit candidates to check to avoid O(N) scan if set is huge
                # But intersection of coverers is usually small.
                candidates_list = list(candidate_set)
                if len(candidates_list) > 20:
                    random.shuffle(candidates_list)
                    candidates_list = candidates_list[:20]
                    
                for c_new in candidates_list:
                    # Calculate cost: sum of cover counts in the ball
                    # Lower cost is better (less redundancy)
                    cost = np.sum(cover_count[neighbors[c_new]])
                    if cost < min_cost:
                        min_cost = cost
                        best_new_cw = c_new
                        
                if best_new_cw != -1:
                    # Perform swap
                    # Remove old_cw
                    cover_count[neighbors[old_cw]] -= 1
                    codewords.pop(idx)
                    # Add c_new
                    cover_count[neighbors[best_new_cw]] += 1
                    codewords.append(best_new_cw)
                    
                    improved = True
                    no_improvement_count = 0
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                    continue
            
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Restart with a new greedy code?
                # Or just break?
                # Let's try a few restarts
                if no_improvement_count > 200:
                    break
                
                # Restart with a new greedy code
                new_greedy = greedy_init(random.randint(0, 1000))
                if len(new_greedy) < len(codewords):
                    codewords = list(new_greedy)
                    cover_count = np.zeros(N, dtype=np.int16)
                    for cw in codewords:
                        cover_count[neighbors[cw]] += 1
                    no_improvement_count = 0
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
        
    # Final cleanup: remove any remaining redundancies
    try_remove_redundant(codewords, cover_count)
    if len(codewords) < best_size:
        best_codewords = list(codewords)
        
    return [all_words[c] for c in best_codewords]
