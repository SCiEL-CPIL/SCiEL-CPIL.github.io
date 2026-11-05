import numpy as np
import time
import random
import sys
from collections import Counter

def construct(q, n, radius):
    # Handle non-target parameters with a generic greedy fallback
    if q != 3 or n != 8 or radius != 1:
        N = q ** n
        if N > 1000000:
            return [[0]*n]
        
        words = []
        for i in range(N):
            w = []
            x = i
            for _ in range(n):
                w.append(x % q)
                x //= q
            w.reverse()
            words.append(tuple(w))
        
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        neighbors = []
        for i, w in enumerate(words):
            neigh = set([i])
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    new_w = tuple(new_w)
                    neigh.add(word_to_idx[new_w])
            neighbors.append(list(neigh))
        
        uncovered = np.ones(N, dtype=bool)
        C = []
        start = time.time()
        limit = 100.0
        while np.any(uncovered):
            best_idx = -1
            best_cnt = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_cnt <= 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start > limit: break
            
        counts = np.zeros(N, dtype=int)
        for c in C:
            counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    
    # Generate all words
    words = []
    for i in range(N):
        w = []
        x = i
        for _ in range(8):
            w.append(x % 3)
            x //= 3
        w.reverse()
        words.append(tuple(w))
    
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors (17 per word: self + 16 neighbors)
    neighbors = np.zeros((N, 17), dtype=np.int32)
    for i, w in enumerate(words):
        neighbors[i, 0] = i
        cnt = 1
        for pos in range(8):
            for delta in [1, 2]:
                new_val = (w[pos] + delta) % 3
                new_w = list(w)
                new_w[pos] = new_val
                new_w = tuple(new_w)
                neighbors[i, cnt] = word_to_idx[new_w]
                cnt += 1
    
    # Precompute inverse: covering[p] = list of codeword indices c such that p is in neighbors[c]
    # This is the "dual hypergraph" representation
    covering = [[] for _ in range(N)]
    for c in range(N):
        for p in neighbors[c]:
            covering[p].append(c)
    # Convert to numpy arrays for speed
    covering_arrays = [np.array(covering[p], dtype=np.int32) for p in range(N)]
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    def calc_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    def reduce_code(code, counts):
        """Remove codewords where all neighbors have count >= 2."""
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed:
                break
        return current_code, current_counts

    def is_valid(code, counts):
        """Check if all points are covered (count >= 1)."""
        return np.all(counts >= 1)

    # 1. Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    # Track how many uncovered points each codeword covers
    # Use vectorized approach
    while np.any(uncovered):
        if time.time() - start_time > time_limit - 10:
            break
        # Compute coverage of each codeword over uncovered points
        uncovered_neigh_counts = np.zeros(N, dtype=np.int32)
        for c in range(N):
            uncovered_neigh_counts[c] = np.sum(uncovered[neighbors[c]])
        best_idx = int(np.argmax(uncovered_neigh_counts))
        best_cnt = uncovered_neigh_counts[best_idx]
        if best_cnt <= 0: break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    # Initial reduction
    counts = calc_counts(C)
    C, counts = reduce_code(C, counts)
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    C_set = set(C)
    
    # Local Search Loop
    while time.time() - start_time < time_limit - 5:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        C_set = set(C)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit - 5:
            break
            
        # 2. 1-Swap Pass
        improved = False
        indices = list(range(len(C)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit - 5:
                break
            c = C[idx]
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                C.remove(c)
                C_set.discard(c)
                counts[neighbors[c]] -= 1
                improved = True
                break
            
            if len(unique_points) > 3:
                continue
            
            # Find candidates c' that cover all unique_points using inverse covering
            if len(unique_points) == 1:
                candidates_set = set(covering_arrays[unique_points[0]].tolist())
            else:
                candidates_set = set(covering_arrays[unique_points[0]].tolist())
                for u in unique_points[1:]:
                    candidates_set &= set(covering_arrays[u].tolist())
                    if not candidates_set:
                        break
            
            if not candidates_set:
                continue
            
            candidates_set.difference_update(C_set)
            
            if not candidates_set:
                continue
            
            c_prime = random.choice(list(candidates_set))
            
            old_C = C.copy()
            old_C_set = C_set.copy()
            old_counts = counts.copy()
            
            C.remove(c)
            C_set.discard(c)
            C.append(c_prime)
            C_set.add(c_prime)
            
            counts[neighbors[c]] -= 1
            counts[neighbors[c_prime]] += 1
            
            if not is_valid(C, counts):
                C = old_C
                C_set = old_C_set
                counts = old_counts
                continue
            
            new_C, new_counts = reduce_code(C, counts)
            new_C_set = set(new_C)
            
            if len(new_C) < len(old_C):
                C = new_C
                C_set = new_C_set
                counts = new_counts
                if len(C) < best_size:
                    best_size = len(C)
                    best_C = C.copy()
                    best_counts = counts.copy()
                improved = True
                break
            else:
                C = old_C
                C_set = old_C_set
                counts = old_counts
        
        # 3. 2-Swap Pass (remove 2, add 2)
        if not improved and len(C) >= 2:
            improved2 = False
            # Sample pairs to try
            num_pairs = min(50, len(C) * (len(C)-1) // 2)
            tried_pairs = set()
            
            for _ in range(num_pairs):
                if time.time() - start_time > time_limit - 5:
                    break
                
                i1, i2 = random.sample(range(len(C)), 2)
                c1, c2 = C[i1], C[i2]
                pair_key = (min(c1, c2), max(c1, c2))
                if pair_key in tried_pairs:
                    continue
                tried_pairs.add(pair_key)
                
                # Find unique points only covered by c1 or c2 (not by any other codeword)
                # A point is "unique to {c1,c2}" if count == 1 and it's in neighbors[c1] or neighbors[c2]
                # OR count == 2 and it's in both neighbors[c1] and neighbors[c2]
                
                n1 = neighbors[c1]
                n2 = neighbors[c2]
                
                # Points that become uncovered if we remove both c1 and c2
                # These are points where count == 1 and in n1 or n2, or count == 2 and in both n1 and n2
                # More precisely: after removing c1 and c2, a point p is uncovered if:
                #   original count[p] - (1 if p in n1 else 0) - (1 if p in n2 else 0) < 1
                
                # Compute which points would be uncovered
                # For efficiency, check only points in n1 ∪ n2
                n1_set = set(n1.tolist())
                n2_set = set(n2.tolist())
                candidate_points = n1_set | n2_set
                
                uncovered_after_removal = []
                for p in candidate_points:
                    dec = (1 if p in n1_set else 0) + (1 if p in n2_set else 0)
                    if counts[p] - dec < 1:
                        uncovered_after_removal.append(p)
                
                if len(uncovered_after_removal) > 4:
                    continue
                
                if len(uncovered_after_removal) == 0:
                    # Both can be removed
                    old_C = C.copy()
                    old_C_set = C_set.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c1)
                    C.remove(c2)
                    C_set.discard(c1)
                    C_set.discard(c2)
                    counts[neighbors[c1]] -= 1
                    counts[neighbors[c2]] -= 1
                    
                    new_C, new_counts = reduce_code(C, counts)
                    if len(new_C) < len(old_C):
                        C = new_C
                        C_set = set(C)
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        improved2 = True
                        break
                    else:
                        C = old_C
                        C_set = old_C_set
                        counts = old_counts
                    continue
                
                # Need to find 2 codewords that cover all uncovered_after_removal points
                # and are not in C
                if len(uncovered_after_removal) == 1:
                    # Need at least one codeword covering this point
                    u0 = uncovered_after_removal[0]
                    cands0 = set(covering_arrays[u0].tolist())
                    cands0.difference_update(C_set)
                    cands0.discard(c1)
                    cands0.discard(c2)
                    if not cands0:
                        continue
                    # Try adding one or two codewords from cands0
                    for c_add in random.sample(list(cands0), min(5, len(cands0))):
                        old_C = C.copy()
                        old_C_set = C_set.copy()
                        old_counts = counts.copy()
                        
                        C.remove(c1)
                        C.remove(c2)
                        C_set.discard(c1)
                        C_set.discard(c2)
                        C.append(c_add)
                        C_set.add(c_add)
                        counts[neighbors[c1]] -= 1
                        counts[neighbors[c2]] -= 1
                        counts[neighbors[c_add]] += 1
                        
                        if not is_valid(C, counts):
                            C = old_C
                            C_set = old_C_set
                            counts = old_counts
                            continue
                        
                        new_C, new_counts = reduce_code(C, counts)
                        if len(new_C) < len(old_C):
                            C = new_C
                            C_set = set(C)
                            counts = new_counts
                            if len(C) < best_size:
                                best_size = len(C)
                                best_C = C.copy()
                                best_counts = counts.copy()
                            improved2 = True
                            break
                        else:
                            C = old_C
                            C_set = old_C_set
                            counts = old_counts
                    if improved2:
                        break
                
                elif len(uncovered_after_removal) <= 4:
                    # Find 2 codewords covering all uncovered points
                    # First codeword must cover at least one uncovered point
                    u0 = uncovered_after_removal[0]
                    cands0 = set(covering_arrays[u0].tolist())
                    cands0.difference_update(C_set)
                    cands0.discard(c1)
                    cands0.discard(c2)
                    
                    if not cands0:
                        continue
                    
                    for c_add1 in random.sample(list(cands0), min(10, len(cands0))):
                        # Check which uncovered points c_add1 covers
                        n_add1 = set(neighbors[c_add1].tolist())
                        remaining_uncovered = [p for p in uncovered_after_removal if p not in n_add1]
                        
                        if not remaining_uncovered:
                            # One codeword suffices
                            old_C = C.copy()
                            old_C_set = C_set.copy()
                            old_counts = counts.copy()
                            
                            C.remove(c1)
                            C.remove(c2)
                            C_set.discard(c1)
                            C_set.discard(c2)
                            C.append(c_add1)
                            C_set.add(c_add1)
                            counts[neighbors[c1]] -= 1
                            counts[neighbors[c2]] -= 1
                            counts[neighbors[c_add1]] += 1
                            
                            if not is_valid(C, counts):
                                C = old_C
                                C_set = old_C_set
                                counts = old_counts
                                continue
                            
                            new_C, new_counts = reduce_code(C, counts)
                            if len(new_C) < len(old_C):
                                C = new_C
                                C_set = set(C)
                                counts = new_counts
                                if len(C) < best_size:
                                    best_size = len(C)
                                    best_C = C.copy()
                                    best_counts = counts.copy()
                                improved2 = True
                                break
                            else:
                                C = old_C
                                C_set = old_C_set
                                counts = old_counts
                            if improved2:
                                break
                        else:
                            # Need second codeword to cover remaining_uncovered
                            u1 = remaining_uncovered[0]
                            cands1 = set(covering_arrays[u1].tolist())
                            for u_rem in remaining_uncovered[1:]:
                                cands1 &= set(covering_arrays[u_rem].tolist())
                                if not cands1:
                                    break
                            cands1.difference_update(C_set)
                            cands1.discard(c1)
                            cands1.discard(c2)
                            cands1.discard(c_add1)
                            
                            if not cands1:
                                continue
                            
                            for c_add2 in random.sample(list(cands1), min(5, len(cands1))):
                                old_C = C.copy()
                                old_C_set = C_set.copy()
                                old_counts = counts.copy()
                                
                                C.remove(c1)
                                C.remove(c2)
                                C_set.discard(c1)
                                C_set.discard(c2)
                                C.append(c_add1)
                                C.append(c_add2)
                                C_set.add(c_add1)
                                C_set.add(c_add2)
                                counts[neighbors[c1]] -= 1
                                counts[neighbors[c2]] -= 1
                                counts[neighbors[c_add1]] += 1
                                counts[neighbors[c_add2]] += 1
                                
                                if not is_valid(C, counts):
                                    C = old_C
                                    C_set = old_C_set
                                    counts = old_counts
                                    continue
                                
                                new_C, new_counts = reduce_code(C, counts)
                                if len(new_C) < len(old_C):
                                    C = new_C
                                    C_set = set(C)
                                    counts = new_counts
                                    if len(C) < best_size:
                                        best_size = len(C)
                                        best_C = C.copy()
                                        best_counts = counts.copy()
                                    improved2 = True
                                    break
                                else:
                                    C = old_C
                                    C_set = old_C_set
                                    counts = old_counts
                                if improved2:
                                    break
                        if improved2:
                            break
                    if improved2:
                        break
            
            if improved2:
                improved = True
        
        # 4. Random kick if no improvement
        if not improved:
            for _ in range(3):
                if time.time() - start_time > time_limit - 5:
                    break
                if not C:
                    break
                c_idx = random.randint(0, len(C)-1)
                c = C[c_idx]
                c_prime = neighbors[c][random.randint(0, 16)]
                if c_prime in C_set:
                    continue
                
                old_C = C.copy()
                old_C_set = C_set.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C_set.discard(c)
                C.append(c_prime)
                C_set.add(c_prime)
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                if not is_valid(C, counts):
                    C = old_C
                    C_set = old_C_set
                    counts = old_counts
                    continue
                
                new_C, new_counts = reduce_code(C, counts)
                if len(new_C) < len(old_C):
                    C = new_C
                    C_set = set(C)
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    break
                else:
                    C = old_C
                    C_set = old_C_set
                    counts = old_counts

    # Final Reduction
    best_C, _ = reduce_code(best_C, calc_counts(best_C))
    
    return [list(words[i]) for i in best_C]
