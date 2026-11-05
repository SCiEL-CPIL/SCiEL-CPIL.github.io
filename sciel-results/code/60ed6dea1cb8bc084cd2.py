import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Specific optimization for q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy for small N
        words = list(itertools.product(range(q), repeat=n))
        N = len(words)
        word_to_idx = {w: i for i, w in enumerate(words)}
        
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
            
        # Greedy
        uncovered = np.ones(N, dtype=bool)
        code = []
        while np.any(uncovered):
            best_idx = -1
            max_gain = 0
            for i in range(N):
                if i in code: continue
                gain = np.sum(uncovered[neighbors[i]])
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            if max_gain == 0: break
            code.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            
        return [list(words[i]) for i in code]

    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbors
    neighbors_list = []
    neighbor_sets = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbor_sets.append(set(nb))
        
    # Precompute coverers (inverse mapping) for fast gain calculation
    # coverers[j] is an array of i such that j is in neighbors_list[i]
    coverers = [np.array([], dtype=np.int32) for _ in range(N)]
    coverers_lists = [[] for _ in range(N)]
    for i in range(N):
        for j in neighbors_list[i]:
            coverers_lists[j].append(i)
    for j in range(N):
        coverers[j] = np.array(coverers_lists[j], dtype=np.int32)

    # ---------- Phase 1: Greedy Set-Cover ----------
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()

    while np.any(uncovered):
        if time.time() > deadline - 30:
            break

        # Compute gains vectorized
        gains = np.zeros(N, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            gains[coverers[idx]] += 1

        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]

        if max_gain == 0:
            break

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors_list[best_idx]] = False

    # ---------- Phase 2: Local Search with Kicks ----------
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1

    best_size = len(code)
    best_code = list(code)

    def greedy_repair(uncovered_indices):
        """Greedily add codewords to cover uncovered_indices."""
        if len(uncovered_indices) == 0:
            return
        temp_uncovered = set(uncovered_indices)
        
        while temp_uncovered:
            # Find candidate with max gain among those covering at least one uncovered point
            # We can restrict candidates to neighbors of uncovered points
            candidate_pool = set()
            for u in temp_uncovered:
                candidate_pool.update(neighbor_sets[u])
            
            # Filter out already chosen
            candidate_pool -= code_set
            
            if not candidate_pool:
                break
                
            best_c = -1
            best_gain = 0
            for c in candidate_pool:
                gain = 0
                for u in neighbors_list[c]:
                    if u in temp_uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    
            if best_c == -1 or best_gain == 0:
                break
                
            code.append(best_c)
            code_set.add(best_c)
            counts[neighbors_list[best_c]] += 1
            for u in neighbors_list[best_c]:
                temp_uncovered.discard(u)

    def try_delete():
        """Remove redundant codewords."""
        removed = False
        # Vectorized check: for each c in code, if min(counts[neighbors[c]]) > 1, remove
        to_remove = []
        for c in code:
            if np.min(counts[neighbors_list[c]]) > 1:
                to_remove.append(c)
        
        for c in to_remove:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1
            removed = True
        return removed

    def try_swap():
        """Try to swap codewords to improve redundancy/positioning."""
        improved = False
        # Shuffle code to avoid bias
        code_copy = list(code)
        random.shuffle(code_copy)
        
        for c in code_copy:
            if time.time() > deadline - 5:
                break
            if c not in code_set:
                continue
            
            # Check if removable
            if np.min(counts[neighbors_list[c]]) > 1:
                continue # Handled by try_delete
            
            # Get critical points
            crit_points = [i for i in neighbors_list[c] if counts[i] == 1]
            if not crit_points:
                continue
            
            # Find candidates covering all critical points
            candidate_set = set(neighbor_sets[crit_points[0]])
            for p in crit_points[1:]:
                candidate_set &= neighbor_sets[p]
            
            candidates = [x for x in candidate_set if x not in code_set]
            if not candidates:
                continue
            
            # Score candidates
            # We want to maximize the number of points in B(w) that have count 1 (creating redundancy)
            # This helps future deletions.
            best_w = -1
            best_score = -1
            
            for w in candidates:
                # Score: count of points in B(w) where counts == 1
                # These points will become count 2 after swap (if they were 1, and w covers them)
                # Wait, if w covers a point x where counts[x]==1, and we add w, counts[x] becomes 2.
                # This creates redundancy.
                score = int(np.sum(counts[neighbors_list[w]] == 1))
                
                # Penalty: if w covers points where counts == 0 (should not happen as w covers crit points which are 1)
                # Actually, we just want to maximize redundancy creation.
                
                if score > best_score:
                    best_score = score
                    best_w = w
                    
            if best_w != -1 and best_score > 0:
                # Perform swap
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                
                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors_list[best_w]] += 1
                
                improved = True
                # After swap, try delete immediately
                try_delete()
                
        return improved

    def kick_and_repair(kick_size):
        """Remove kick_size random codewords and repair."""
        if len(code) <= kick_size:
            return False
            
        kick_candidates = list(code)
        random.shuffle(kick_candidates)
        kicked = kick_candidates[:kick_size]
        
        for c in kicked:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1
            
        # Find uncovered
        uncovered_mask = (counts == 0)
        if np.any(uncovered_mask):
            uncovered_indices = np.where(uncovered_mask)[0]
            greedy_repair(uncovered_indices)
            try_delete()
            return True
        return False

    # Main Local Search Loop
    no_improve_count = 0
    kick_interval = 20 # Try kick every 20 no-improve steps
    
    while time.time() < deadline - 10:
        improved = False
        
        # 1. Delete
        if try_delete():
            improved = True
            no_improve_count = 0
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
                
        # 2. Swap
        if try_swap():
            improved = True
            no_improve_count = 0
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
                
        if not improved:
            no_improve_count += 1
            
            # 3. Kick if stuck
            if no_improve_count >= kick_interval:
                # Random kick size
                kick_size = random.randint(1, 3)
                kick_and_repair(kick_size)
                no_improve_count = 0
                kick_interval = random.randint(10, 30)
                
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
            
            # If stuck for a long time, do a bigger kick
            if no_improve_count > 50:
                kick_size = random.randint(5, 10)
                kick_and_repair(kick_size)
                no_improve_count = 0
                kick_interval = random.randint(10, 30)
                
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)

    # Return best found
    return [list(idx_to_tuple(i)) for i in best_code]
