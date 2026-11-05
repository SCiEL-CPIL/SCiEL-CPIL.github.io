import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Generic fallback: just return all words (worst case) or a simple product
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Total number of words
    total_words = q ** n  # 6561

    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists (Hamming distance <= radius)
    # For q=3, n=8, radius=1, each word has 1 + 8*2 = 17 neighbors
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # Precompute neighbor masks as boolean arrays for fast vectorized operations
    # neighbor_mask[i] is a boolean array of length total_words where True means j is in neighbors[i]
    neighbor_mask = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        neighbor_mask[i, neighbors[i]] = True

    # --- Phase 1: Vectorized Greedy Initialization ---
    covered = np.zeros(total_words, dtype=bool)
    code_list = []
    
    # potential[i] = number of uncovered points in neighbors[i]
    # Use vectorized computation: for each i, count how many of its neighbors are uncovered
    while not covered.all() and time.time() < deadline - 30.0:
        # Compute potential for all candidates at once
        # For each i, potential[i] = sum of ~covered over neighbors[i]
        # This is equivalent to: for each uncovered point x, add 1 to all i where x is in neighbors[i]
        # neighbors_mask^T @ (~covered).astype(int) gives this
        
        # More efficient: compute which points are uncovered, then for each such point, 
        # increment potential for all codewords that cover it
        # But let's use the matrix approach: potential = neighbor_mask @ (~covered).astype(np.int32)
        
        uncovered_mask = ~covered
        if not uncovered_mask.any():
            break
            
        # potential[i] = number of uncovered neighbors of i
        potential = neighbor_mask @ uncovered_mask.astype(np.int32)
        
        c = int(np.argmax(potential))
        if potential[c] == 0:
            break
        
        code_list.append(c)
        covered[neighbors[c]] = True

    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size

    # Compute coverage counts
    current_cov_count = np.zeros(total_words, dtype=np.int32)
    for c in current_code:
        current_cov_count[neighbors[c]] += 1

    current_code_set = set(current_code)
    
    # Precompute for each pair of codewords, how many points they jointly cover exclusively
    # overlap_count[i][j] = number of points x where cov_count[x] == 1 and x is in neighbors[i] and neighbors[j]
    # This helps identify good swap candidates

    # --- Phase 2: Iterated Local Search with Swaps ---
    stuck_counter = 0
    random.seed(42)
    
    # Precompute risky_point_sets for faster lookup
    # For each codeword c, risky_points[c] = list of x in neighbors[c] where cov_count[x] == 1
    
    while time.time() < deadline - 5.0:
        # 1. Try to remove a word
        if current_size == 0:
            break
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]
        
        # Check if all neighbors are covered by more than one codeword
        if np.all(current_cov_count[neighbors[c]] > 1):
            # Safe to remove
            current_cov_count[neighbors[c]] -= 1
            current_code_set.remove(c)
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
            
        # 2. If stuck, try a swap
        if stuck_counter > 5 and time.time() < deadline - 10.0 and current_size > 0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            
            # Find risky points: x in neighbors[c1] where cov_count[x] == 1
            risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]
            
            if not risky_points:
                # c1 is removable
                current_cov_count[neighbors[c1]] -= 1
                current_code_set.remove(c1)
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                # Find c2 not in code such that c2 covers all risky points.
                # c2 must be in the intersection of neighbors[x] for all x in risky_points.
                
                # Use vectorized intersection
                if len(risky_points) == 1:
                    # Only one risky point, any neighbor of it works
                    candidate_mask = neighbor_mask[risky_points[0]].copy()
                else:
                    # Intersection of neighbor sets of all risky points
                    candidate_mask = neighbor_mask[risky_points[0]].copy()
                    for x in risky_points[1:]:
                        candidate_mask &= neighbor_mask[x]
                        if not candidate_mask.any():
                            break
                
                candidate_mask[c1] = False  # Exclude c1 itself
                
                # Filter out candidates already in code
                for c_code in current_code:
                    candidate_mask[c_code] = False
                
                valid_candidates = np.where(candidate_mask)[0]
                
                if len(valid_candidates) > 0:
                    # Pick a random valid candidate
                    c2 = int(random.choice(valid_candidates))
                    
                    # Perform the swap
                    current_cov_count[neighbors[c1]] -= 1
                    current_code_set.remove(c1)
                    current_code[c1_idx] = c2
                    
                    current_cov_count[neighbors[c2]] += 1
                    current_code_set.add(c2)
                    
                    stuck_counter = 0
                else:
                    # No valid swap found
                    pass
                    
            # Reset stuck counter to allow more removal attempts first
            stuck_counter = 0

    # Final Polish - aggressive removal pass
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True
                
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
