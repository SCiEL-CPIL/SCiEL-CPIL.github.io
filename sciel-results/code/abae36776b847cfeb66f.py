import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors (balls of radius 1)
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= radius
    neighbors_list = []
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
                w_list[pos] = orig_val # Restore
        neighbors_list.append(np.array(neigh, dtype=np.int32))
        
    # Build adjacency matrix for vectorized greedy
    # M[i, j] = 1 if j is in neighbors_list[i]
    M = np.zeros((N, N), dtype=np.float32)
    for i, neigh in enumerate(neighbors_list):
        M[i, neigh] = 1.0

    # 3. Greedy Set Cover (Vectorized)
    uncovered = np.ones(N, dtype=np.float32) # Use float for matmul
    code = []
    code_set = set()
    
    while np.any(uncovered > 0.5):
        if time.time() > deadline - 30:
            break
            
        # Compute gains for all candidates
        gains = M @ uncovered
        # Mask out already selected
        for idx in code_set:
            gains[idx] = -1.0
            
        best_idx = np.argmax(gains)
        
        if gains[best_idx] <= 0:
            break
            
        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors_list[best_idx]] = 0.0
        
    # 4. Local Search
    # counts[i] is the number of codewords in code that cover point i
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1
        
    def is_removable(c):
        """Check if codeword c can be removed"""
        # If any point in neighbors[c] has count 1, it is NOT removable
        return np.all(counts[neighbors_list[c]] > 1)

    def get_critical_points(c):
        """Return list of points covered ONLY by codeword c"""
        neigh = neighbors_list[c]
        return neigh[counts[neigh] == 1]

    def apply_swap(c, w):
        """Swap c for w in code"""
        code.remove(c)
        code_set.remove(c)
        counts[neighbors_list[c]] -= 1
        
        code.append(w)
        code_set.add(w)
        counts[neighbors_list[w]] += 1
        
        # Immediate deletion pass
        to_remove = []
        for c2 in code:
            if is_removable(c2):
                to_remove.append(c2)
        for c2 in to_remove:
            code.remove(c2)
            code_set.remove(c2)
            counts[neighbors_list[c2]] -= 1

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # --- Deletion Pass ---
        to_remove = []
        for c in code:
            if is_removable(c):
                to_remove.append(c)
                
        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
            
        # --- Swap Pass ---
        current_code_copy = list(code)
        random.shuffle(current_code_copy)
        
        for c in current_code_copy:
            if time.time() > deadline - 5:
                break
            
            if c not in code_set:
                continue

            crit_points = get_critical_points(c)
            
            if len(crit_points) == 0:
                continue
                
            # Find candidates
            # Candidates must cover all points in crit_points
            candidate_mask = np.ones(N, dtype=bool)
            for p in crit_points:
                candidate_mask &= (M[:, p] > 0.5)
            
            # Exclude current code
            for idx in code_set:
                candidate_mask[idx] = False
            
            candidates = np.where(candidate_mask)[0]
            
            if len(candidates) == 0:
                continue
            
            # Evaluate candidates
            best_candidate = -1
            best_score = -np.inf
            
            neigh_c = neighbors_list[c]
            set_neigh_c = set(neigh_c)
            
            for w in candidates:
                neigh_w = neighbors_list[w]
                set_neigh_w = set(neigh_w)
                
                # Gain: Points in neigh_w with count 1 that are NOT in neigh_c
                gain = 0
                for p in neigh_w:
                    if counts[p] == 1 and p not in set_neigh_c:
                        gain += 1
                        
                # Penalty: Points in neigh_c with count 2 that are NOT in neigh_w
                penalty = 0
                for p in neigh_c:
                    if counts[p] == 2 and p not in set_neigh_w:
                        penalty += 1
                        
                score = gain - penalty
                
                if score > best_score:
                    best_score = score
                    best_candidate = w
                    
            if best_candidate != -1 and best_score > 0:
                # Perform swap
                apply_swap(c, best_candidate)
                improved = True

    # Final cleanup
    changed = True
    while changed:
        changed = False
        for c in list(code):
            if is_removable(c):
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                changed = True

    return [list(words[i]) for i in code]
