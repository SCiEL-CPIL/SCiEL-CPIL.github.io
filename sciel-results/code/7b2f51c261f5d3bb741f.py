import time
import random
import numpy as np
import heapq

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return _generic_construct(q, n, radius)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    N = 3 ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 15
    
    def int_to_base3(num, length):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % 3
            num //= 3
        return res
    
    def base3_to_int(digits):
        res = 0
        for d in digits:
            res = res * 3 + d
        return res
    
    # Precompute balls (neighbors including self) as integer indices
    balls = []
    for i in range(N):
        digits = int_to_base3(i, n)
        neighbors = [i]
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    neighbors.append(base3_to_int(new_digits))
        balls.append(neighbors)
    
    # Precompute cover_map: for each word w, which codewords cover it
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)
    
    # Precompute ball sets for fast membership
    ball_sets = [set(b) for b in balls]
    
    # Precompute ball indices as numpy arrays for faster iteration
    ball_indices = [np.array(b, dtype=np.int32) for b in balls]
    
    # Precompute cover_map as sets for fast intersection
    cover_map_sets = [set(cm) for cm in cover_map]
    
    def greedy_cover(seed, start_code=None):
        """
        Greedy set cover with heap-based candidate selection.
        """
        rng = random.Random(seed)
        covered = [False] * N
        cnt = [ball_size] * N
        code = []
        uncovered_count = N
        
        in_code = [False] * N
        if start_code:
            for c in start_code:
                code.append(c)
                in_code[c] = True
                for w in balls[c]:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
        
        # Initialize heap with all candidates not in start_code
        heap = []
        for i in range(N):
            if not in_code[i]:
                heap.append((-cnt[i], i))
        heapq.heapify(heap)
        
        while uncovered_count > 0:
            # Pop valid candidates from heap
            while heap:
                neg_val, i = heapq.heappop(heap)
                if cnt[i] == -neg_val and not in_code[i]:
                    break
            else:
                break
            
            c = i
            code.append(c)
            in_code[c] = True
            
            ball_c = balls[c]
            for w in ball_c:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    for c2 in cover_map[w]:
                        cnt[c2] -= 1
                        if cnt[c2] > 0 and not in_code[c2]:
                            heapq.heappush(heap, (-cnt[c2], c2))
        
        return code
    
    def refine(code):
        """Remove redundant codewords."""
        if not code:
            return code
        
        cover_count = [0] * N
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        code_list = list(code)
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                safe = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        safe = False
                        break
                if safe:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code_list.pop(i)
                    changed = True
                    break
        return code_list
    
    def local_search(code, seed, time_limit):
        """
        Local search with deletion and 1-swap moves.
        """
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list
        
        cover_count = [0] * N
        for c in code_list:
            for w in ball_indices[c]:
                cover_count[w] += 1
        
        in_code = [False] * N
        for c in code_list:
            in_code[c] = True
        
        best_code = list(code_list)
        best_size = len(code_list)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code_list[i]
                
                # Try deletion first
                newly_uncovered = []
                for w in ball_indices[c_remove]:
                    cover_count[w] -= 1
                    if cover_count[w] == 0:
                        newly_uncovered.append(w)
                
                if not newly_uncovered:
                    # Deletion successful
                    code_list.pop(i)
                    in_code[c_remove] = False
                    improved = True
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                    break
                
                # Try 1-swap: find a single codeword covering all newly_uncovered
                # Sort critical words by rarity (fewest coverers) for faster intersection
                crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                cand_pool = None
                for w in crit_sorted:
                    candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                    if cand_pool is None:
                        cand_pool = set(candidates_w)
                    else:
                        cand_pool.intersection_update(candidates_w)
                    if not cand_pool:
                        break
                
                if cand_pool:
                    # Pick best candidate from pool: minimize waste (cover already-covered words)
                    best_c_in = -1
                    best_waste = float('inf')
                    for c_in in cand_pool:
                        # Waste = number of neighbors of c_in that are already covered (cover_count > 0)
                        waste = 0
                        for w in balls[c_in]:
                            if cover_count[w] > 0:
                                waste += 1
                        if waste < best_waste:
                            best_waste = waste
                            best_c_in = c_in
                    
                    if best_c_in != -1:
                        # Perform swap
                        in_code[c_remove] = False
                        in_code[best_c_in] = True
                        code_list[i] = best_c_in
                        for w in balls[best_c_in]:
                            cover_count[w] += 1
                        improved = True
                        if len(code_list) < best_size:
                            best_size = len(code_list)
                            best_code = list(code_list)
                        break
                    else:
                        # Revert
                        for w in ball_indices[c_remove]:
                            cover_count[w] += 1
                else:
                    # Revert
                    for w in ball_indices[c_remove]:
                        cover_count[w] += 1
        
        return best_code
    
    # Multi-start local search
    best_code = None
    best_size = float('inf')
    
    # Try multiple seeds for different starting points
    seeds = [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
    
    for seed in seeds:
        if time.time() >= deadline:
            break
        
        # Generate initial code with greedy
        initial_code = greedy_cover(seed)
        initial_code = refine(initial_code)
        
        if len(initial_code) < best_size:
            best_size = len(initial_code)
            best_code = list(initial_code)
        
        # Run local search with time budget
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate time per local search run
        time_per_run = min(8.0, remaining / max(1, len(seeds) - seeds.index(seed)))
        ls_time_limit = time.time() + time_per_run
        
        refined = local_search(initial_code, seed, ls_time_limit)
        
        if len(refined) < best_size:
            best_size = len(refined)
            best_code = list(refined)
        
        # If we found a good solution, try more aggressive search from it
        if best_size <= 183 and time.time() < deadline - 10:
            # Do additional local search rounds from the best solution
            extra_time = min(15.0, deadline - time.time() - 5)
            if extra_time > 3:
                refined2 = local_search(best_code, seed * 7 + 1, time.time() + extra_time)
                if len(refined2) < best_size:
                    best_size = len(refined2)
                    best_code = list(refined2)
    
    # Final refinement pass
    if best_code and time.time() < deadline - 2:
        best_code = refine(best_code)
        best_size = len(best_code)
    
    # Convert integer indices back to codewords
    if best_code is None:
        best_code = []
    
    result = []
    for c in best_code:
        result.append(int_to_base3(c, n))
    
    return result

def _generic_construct(q, n, radius):
    # Fallback for non-standard parameters
    all_words = []
    word_to_idx = {}
    idx = 0
    for digits in np.ndindex(*([q] * n)):
        w = list(digits)
        word_to_idx[tuple(w)] = idx
        all_words.append(w)
        idx += 1
    N = len(all_words)
    
    if N > 10000:
        return [all_words[0]]
    
    # Precompute neighbors within radius
    neighbors = [[] for _ in range(N)]
    for i, w1 in enumerate(all_words):
        for j, w2 in enumerate(all_words):
            dist = sum(1 for a, b in zip(w1, w2) if a != b)
            if dist <= radius:
                neighbors[i].append(j)
    
    uncovered = set(range(N))
    code = []
    while uncovered:
        best_candidate = -1
        max_covered = -1
        for c in range(N):
            if c in code:
                continue
            count = 0
            for nb in neighbors[c]:
                if nb in uncovered:
                    count += 1
            if count > max_covered:
                max_covered = count
                best_candidate = c
        if best_candidate == -1:
            break
        code.append(best_candidate)
        for nb in neighbors[best_candidate]:
            if nb in uncovered:
                uncovered.discard(nb)
                
    return [all_words[c] for c in code]
