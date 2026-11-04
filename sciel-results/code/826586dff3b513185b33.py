import numpy as np
import time
import random
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    # 1. Generate all words and index mapping
    # For q=3, n=7, radius=1:
    # N = 3^7 = 2187
    # Each ball has 1 + n*(q-1) = 1 + 7*2 = 15 words.
    
    words = []
    word_to_idx = {}
    # Unroll for efficiency since q,n are small and fixed in practice?
    # No, keep generic as per interface, but optimize for the specific case.
    # However, the interface implies general q, n, radius.
    # But the task description says "Football pool problem for 7 matches...".
    # The evaluator passes q, n, radius.
    
    # Let's assume the specific case is dominant or we need to be efficient enough.
    # 2187 is very small. We can afford O(N^2) in Python if careful? 
    # 2187^2 is ~4.7M operations. In pure Python that's a second or two. 
    # But we have 120s.
    
    # Let's use a vectorized approach for initial coverage count to speed up greedy.
    
    # Generate all words
    from itertools import product
    words = list(product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors (indices of words within Hamming distance radius)
    # For radius 1:
    neighbors = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbors.append(neigh)
        
    # Convert to arrays for faster manipulation?
    # neighbors is a list of lists.
    # Let's create a "cover matrix" or use counts.
    
    # Strategy:
    # 1. Greedy Set Cover with randomized tie-breaking (Las Vegas/Greedy)
    # 2. Local Search (Deletions, Swaps, 2-exchanges)
    
    # Greedy Phase
    # We maintain a count of how many times each point is covered by the current code.
    # But for greedy, we just care about UNCOVERED points.
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()
    
    # Precompute neighbor lists as arrays for boolean indexing
    neighbor_arrays = [np.array(neigh) for neigh in neighbors]
    
    # Optimization: Instead of scanning all N words every time, 
    # we can maintain a list of candidates. But N is small.
    
    # To avoid bias, we can randomize the order of checking or add noise to gains.
    # Let's try a standard greedy first, then random restarts if time permits.
    
    def greedy_cover():
        uncovered_local = np.ones(N, dtype=bool)
        code_local = []
        code_set_local = set()
        
        while np.any(uncovered_local):
            if time.time() > deadline - 30:
                break
            
            best_idx = -1
            max_gain = 0
            # Randomization: pick randomly among the top few?
            # Or just standard max. Let's stick to standard max for determinism in the seed,
            # but the current seed already uses standard greedy.
            # To get a distinct structural change, let's implement a WEIGHTED greedy 
            # or a different heuristic.
            
            # Idea: Minimize "cost" per coverage? Cost is 1 for all.
            # Idea: Use a random tie-breaker to explore different greedy paths.
            
            for i in range(N):
                if i in code_set_local:
                    continue
                gain = np.sum(uncovered_local[neighbor_arrays[i]])
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            
            if max_gain == 0:
                break
            
            code_local.append(best_idx)
            code_set_local.add(best_idx)
            uncovered_local[neighbor_arrays[best_idx]] = False
            
        return code_local, code_set_local

    # Run Greedy
    code, code_set = greedy_cover()
    
    # Local Search Phase
    # We want to minimize len(code).
    # Counts array: counts[i] = number of codewords in code that cover point i.
    
    def build_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbor_arrays[idx]] += 1
        return counts
    
    counts = build_counts(code)
    
    def is_removable(c, counts):
        # If any point in neighbors[c] has count 1, it is NOT removable
        # counts[c] includes c itself.
        # So if min(counts[neighbors[c]]) > 1, it is removable.
        return np.min(counts[neighbor_arrays[c]]) > 1
    
    def try_remove_all(code, code_set, counts):
        changed = True
        while changed:
            changed = False
            for c in list(code):
                if is_removable(c, counts):
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1
                    changed = True
        return code, code_set, counts
    
    code, code_set, counts = try_remove_all(code, code_set, counts)
    
    # Local Search Loop
    # We perform swaps: replace one codeword with another to enable further removals.
    # Or directly try to find a "better" swap that reduces the total number of critical points.
    
    # Critical points for a codeword c are points p in neighbors[c] such that counts[p] == 1.
    # If we swap c with c_new, we must cover all critical points of c.
    
    # Optimization:
    # 1. Identify "critical" codewords (those that have at least one point covered only by them).
    # 2. For each critical codeword c, find candidate replacements c_new that cover c's critical points.
    # 3. Among candidates, pick the one that minimizes the number of NEW critical points created,
    #    or maximizes the number of redundant covers.
    
    # Let's structure the loop to try improvements.
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # 1. Deletion Pass
        to_remove = [c for c in code if is_removable(c, counts)]
        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
        
        # 2. Swap Pass
        # We iterate through codewords that are NOT removable.
        # Shuffling order helps escape local minima.
        candidates_for_swap = [c for c in code if not is_removable(c, counts)]
        random.shuffle(candidates_for_swap)
        
        for c in candidates_for_swap:
            if time.time() > deadline - 10:
                break
                
            # Check if c is still in code and critical (in case previous swaps changed counts)
            if c not in code_set:
                continue
            if is_removable(c, counts):
                # It became removable due to previous swaps
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
                improved = True
                continue
                
            # Find critical points of c
            crit_points = np.where(counts[neighbor_arrays[c]] == 1)[0]
            
            if len(crit_points) == 0:
                # Should have been removed
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
                improved = True
                continue
                
            # Find candidates c_new NOT in code_set that cover ALL crit_points
            # c_new must be such that all crit_points are in neighbors[c_new]
            
            # Efficient candidate filtering:
            # For each crit_point p, candidates must be in neighbors[p].
            # Intersect these sets.
            
            # We need neighbor_lists_by_point?
            # neighbors[i] is list of points covered by i.
            # We need reverse: point_to_coverers[p] = list of i such that p in neighbors[i].
            # This is expensive to build explicitly for N=2187 (2187 lists of size ~15).
            # But we can build it once.
            
            # Let's build point_to_coverers
            # Wait, we are inside the loop. We should precompute this outside.
            # I'll move point_to_coverers computation to before the loop.
            
        # Since I need point_to_coverers, I'll restructure.
    
    # Let's restart the local search with precomputed reverse indices.
    point_to_coverers = [[] for _ in range(N)]
    for i in range(N):
        for p in neighbors[i]:
            point_to_coverers[p].append(i)
    
    # Re-initialize for the main loop
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Deletion Pass
        to_remove = [c for c in code if is_removable(c, counts)]
        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
        
        # Swap Pass
        candidates_for_swap = [c for c in code if not is_removable(c, counts)]
        random.shuffle(candidates_for_swap)
        
        for c in candidates_for_swap:
            if time.time() > deadline - 10:
                break
                
            if c not in code_set:
                continue
            if is_removable(c, counts):
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
                improved = True
                continue
                
            # Find critical points of c
            crit_mask = counts[neighbor_arrays[c]] == 1
            crit_points = neighbor_arrays[c][crit_mask]
            
            if len(crit_points) == 0:
                continue
                
            # Candidates must cover ALL crit_points
            # Start with candidates for the first crit point
            candidate_set = set(point_to_coverers[crit_points[0]])
            
            for p in crit_points[1:]:
                candidate_set &= set(point_to_coverers[p])
                
            # Remove current code from candidates (and c itself)
            candidate_set -= code_set
            
            if not candidate_set:
                continue
                
            # Evaluate candidates
            # Score: Minimize new critical points, or maximize "safety"
            # A good swap reduces the number of critical points for other codewords
            # or creates a configuration where more codewords can be removed.
            
            best_cand = -1
            best_score = -1
            
            # Heuristic Score:
            # For each candidate w:
            # 1. Add w to code (counts += neighbors[w])
            # 2. Remove c from code (counts -= neighbors[c])
            # 3. Count how many points become critical (count == 1) in the new state?
            #    This is expensive to simulate fully.
            
            # Cheaper heuristic:
            # We want w to cover points that are currently "dangerous" (count == 1) for OTHERS?
            # No, we want w to make the overall covering more robust.
            
            # Let's use: Score = (Number of points in neighbors[w] with count > 1) - 
            #                    (Number of points in neighbors[c] with count == 2 and not in neighbors[w])
            # This is an approximation.
            
            # Actually, a simpler robust metric:
            # Try the swap temporarily and see if it allows immediate deletions?
            # That is costly.
            
            # Let's stick to: Minimize the sum of 1/(count-1) for points affected?
            # Or just: Maximize the overlap of w with redundant regions.
            
            # Let's try: Score = sum(1 for p in neighbors[w] if counts[p] > 1)
            # And Penalty = sum(1 for p in neighbors[c] if counts[p] == 2 and p not in neighbors[w])
            
            neigh_c_set = set(neighbors[c])
            
            for w in candidate_set:
                # Quick check: is w a good candidate?
                # It covers all crit_points.
                
                # Calculate potential score
                score = 0
                
                # Reward covering already well-covered points
                # Iterate neighbors[w]
                # Note: neighbors[w] is a list of indices.
                # counts is an array.
                
                # Optimization: use numpy
                # But candidate_set might be small.
                
                # Let's estimate:
                # Points p in neighbors[w] that are NOT in neighbors[c]:
                # These gain +1 coverage. If counts[p] == 1, they become safe (good).
                # If counts[p] == 0, error (shouldn't happen as c covers its neighbors? No, w is new).
                
                # Points p in neighbors[c] that are NOT in neighbors[w]:
                # These lose -1 coverage. If counts[p] == 2, they become critical (bad).
                
                # We can compute intersection size.
                
                # Since N is small, let's just loop.
                
                # We want to prioritize swaps that allow deletion of OTHER codewords.
                # This is hard to know in advance.
                
                # Let's use a random tie-breaker among candidates with score > 0.
                
                # Score:
                # +1 for each p in neighbors[w] \ neighbors[c] where counts[p] == 1 (makes them redundant)
                # -2 for each p in neighbors[c] \ neighbors[w] where counts[p] == 2 (makes them critical)
                
                # Precompute neighbor sets for faster check?
                # We have neighbor_arrays.
                
                # Let's just use the list membership for the small lists (size 15).
                # w_list = neighbors[w]
                # c_list = neighbors[c]
                
                # Convert to sets for O(1) lookup?
                # We already have neighbor_sets? No, we have neighbors (lists).
                # Let's create neighbor_sets for all nodes once?
                # neighbor_sets = [set(neigh) for neigh in neighbors]
                # This takes memory but speeds up swaps significantly.
                
                pass # Placeholder
                
            # Since I didn't implement the loop body fully, I need to finish it.
            
    # I realized I am running out of time in the thought process. I need to generate the final code.
    # Let's consolidate the logic into the final code block, ensuring efficiency.

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0
    
    from itertools import product
    words = list(product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors
    neighbors = []
    neighbor_sets = []
    point_to_coverers = [[] for _ in range(N)]
    
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbors.append(neigh)
        neighbor_sets.append(set(neigh))
        
    for i, neigh in enumerate(neighbors):
        for p in neigh:
            point_to_coverers[p].append(i)
            
    # Greedy Cover
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()
    neighbor_arrays = [np.array(neigh) for neigh in neighbors]
    
    while np.any(uncovered) and time.time() < deadline - 30:
        best_idx = -1
        max_gain = 0
        for i in range(N):
            if i in code_set:
                continue
            gain = np.sum(uncovered[neighbor_arrays[i]])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        
        if max_gain == 0:
            break
            
        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbor_arrays[best_idx]] = False
        
    # Local Search
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbor_arrays[idx]] += 1
        
    def is_removable(c):
        return np.min(counts[neighbor_arrays[c]]) > 1
        
    def cleanup():
        nonlocal code, code_set, counts
        changed = True
        while changed:
            changed = False
            for c in list(code):
                if is_removable(c):
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1
                    changed = True

    cleanup()
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        cleanup()
        
        # Swap Pass
        critical_codewords = [c for c in code if not is_removable(c)]
        random.shuffle(critical_codewords)
        
        for c in critical_codewords:
            if time.time() > deadline - 5:
                break
            if c not in code_set:
                continue
            if is_removable(c):
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
                improved = True
                continue
                
            crit_mask = counts[neighbor_arrays[c]] == 1
            crit_points = neighbor_arrays[c][crit_mask]
            
            if len(crit_points) == 0:
                continue
                
            # Find candidates
            candidate_set = set(point_to_coverers[crit_points[0]])
            for p in crit_points[1:]:
                candidate_set &= set(point_to_coverers[p])
            candidate_set -= code_set
            
            if not candidate_set:
                continue
                
            best_w = -1
            best_score = -1
            neigh_c_set = neighbor_sets[c]
            
            for w in candidate_set:
                score = 0
                # Reward: points in neighbors[w] \ neighbors[c] with count 1
                # Penalty: points in neighbors[c] \ neighbors[w] with count 2
                
                # Approximate score
                # We can sample if candidate_set is large
                # For now, assume it's small.
                
                # Check all p in neighbors[w]
                for p in neighbors[w]:
                    if p not in neigh_c_set:
                        if counts[p] == 1:
                            score += 1
                            
                # Check all p in neighbors[c]
                neigh_w_set = neighbor_sets[w]
                for p in neighbors[c]:
                    if p not in neigh_w_set:
                        if counts[p] == 2:
                            score -= 2
                            
                if score > best_score:
                    best_score = score
                    best_w = w
                    
            if best_w != -1 and best_score > 0:
                # Perform swap
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
                
                code.append(best_w)
                code_set.add(best_w)
                counts[neighbor_arrays[best_w]] += 1
                
                improved = True
                cleanup()
                
    # Final cleanup
    cleanup()
    
    return [list(words[i]) for i in code]
