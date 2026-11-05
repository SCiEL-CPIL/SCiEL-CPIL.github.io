import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as numpy arrays
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(np.array(list(neigh_set), dtype=int))
        
    # Helper: Calculate coverage counts
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    # Helper: Reduce code by removing redundant words
    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    # 1. Fast Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0
    
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
            
    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # 2. Intensive Local Search with Perturbation
    # We try to escape local minima by:
    # a) 1-swap moves that enable reduction
    # b) Perturbation (remove k words, re-greedy) if stuck
    
    stuck_count = 0
    
    while time.time() - start_time < time_limit:
        # Reduce
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            stuck_count = 0
            
        if time.time() - start_time > time_limit:
            break
            
        # Try 1-swap moves
        improved = False
        indices = list(range(len(C)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            c = C[idx]
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                # Should have been reduced, force it
                C.remove(c)
                counts[neighbors[c]] -= 1
                improved = True
                break
                
            if len(unique_points) > 4:
                continue
                
            # Find candidates covering all unique points
            if len(unique_points) == 1:
                candidates_set = set(neighbors[unique_points[0]].tolist())
            else:
                candidates_set = set(neighbors[unique_points[0]].tolist())
                for u in unique_points[1:]:
                    candidates_set &= set(neighbors[u].tolist())
                    if not candidates_set:
                        break
            
            if not candidates_set:
                continue
                
            candidates_set.difference_update(C)
            if not candidates_set:
                continue
                
            # Score candidates: prefer those covering many points with count 1 (potential new redundancies)
            best_cand = -1
            best_score = -1
            cand_list = list(candidates_set)
            # Limit candidates to check for speed
            if len(cand_list) > 10:
                random.shuffle(cand_list)
                cand_list = cand_list[:10]
                
            for c_prime in cand_list:
                # Estimate score: number of points in neighbors[c_prime] with count 1 (excluding unique points of c?)
                # Actually, just count how many points it covers that are currently count 1.
                # This creates "overlap" which might allow removal of other words.
                score = np.sum(counts[neighbors[c_prime]] == 1)
                if score > best_score:
                    best_score = score
                    best_cand = c_prime
            
            if best_cand == -1:
                continue
                
            # Apply swap
            old_C = C.copy()
            old_counts = counts.copy()
            
            C.remove(c)
            C.append(best_cand)
            counts[neighbors[c]] -= 1
            counts[neighbors[best_cand]] += 1
            
            if np.any(counts < 1):
                C = old_C
                counts = old_counts
                continue
                
            new_C, new_counts = reduce_code(C, counts)
            
            if len(new_C) < len(old_C):
                C = new_C
                counts = new_counts
                improved = True
                break
            else:
                # Accept neutral swap to explore? 
                # Only accept if it changes the set of unique points significantly?
                # For now, accept to move away from local min, but don't count as "improved" for stuck_count reset unless it leads to reduction later.
                # Actually, accepting neutral swaps can lead to cycles. 
                # Let's only accept if it's a "good" neutral swap (e.g. increases potential for reduction).
                # Heuristic: Accept if best_score > 0 (it creates new redundancies)
                if best_score > 2:
                    C = new_C
                    counts = new_counts
                    improved = True
                    break
            
        if not improved:
            stuck_count += 1
            # If stuck for a while, perturb
            if stuck_count > 20:
                # Perturb: Remove 5-10 random words and re-greedy
                k = random.randint(5, 10)
                if len(C) > k + 10:
                    remove_indices = random.sample(range(len(C)), k)
                    for ri in sorted(remove_indices, reverse=True):
                        c = C[ri]
                        counts[neighbors[c]] -= 1
                        C.pop(ri)
                    
                    # Re-greedy to fill holes
                    uncovered = counts == 0
                    while np.any(uncovered):
                        best_idx = -1
                        best_count = -1
                        for i in range(N):
                            if i in C: continue
                            cnt = np.sum(uncovered[neighbors[i]])
                            if cnt > best_count:
                                best_count = cnt
                                best_idx = i
                        if best_count == 0:
                            # Should not happen if we removed words from a valid code? 
                            # Wait, removing words creates uncovered areas.
                            # But we need to ensure we don't pick words already in C.
                            break
                        C.append(best_idx)
                        counts[neighbors[best_idx]] += 1
                        uncovered[neighbors[best_idx]] = False
                    
                    C, counts = reduce_code(C, counts)
                    stuck_count = 0

    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
