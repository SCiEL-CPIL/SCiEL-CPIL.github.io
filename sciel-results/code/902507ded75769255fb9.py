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
    start_time = time.time()
    deadline = start_time + 118.0
    
    # Precompute neighbors efficiently using vectorized NumPy
    idx_arr = np.arange(N, dtype=np.int64)
    digits = np.zeros((N, n), dtype=np.int64)
    tmp = idx_arr.copy()
    for pos in range(n - 1, -1, -1):
        digits[:, pos] = tmp % q
        tmp //= q
    
    # For each position pos and each new_val != orig, compute the neighbor index
    # delta = (new_val - orig_val) * q^(n-1-pos)
    q_powers = np.array([q ** (n - 1 - pos) for pos in range(n)], dtype=np.int64)
    
    neighbors = [None] * N
    neighbor_sets = [None] * N
    for pos in range(n):
        orig = digits[:, pos]  # (N,)
        power = q_powers[pos]
        for new_val in range(q):
            mask = orig != new_val
            if not mask.any():
                continue
            deltas = (new_val - orig) * power
            neigh_indices = idx_arr + deltas
            # Scatter into neighbors list
            for i in np.where(mask)[0]:
                if neighbors[i] is None:
                    neighbors[i] = [i]
                neighbors[i].append(int(neigh_indices[i]))
    
    # Initialize lists and sets
    for i in range(N):
        if neighbors[i] is None:
            neighbors[i] = [i]
        else:
            neighbors[i] = list(neighbors[i])
        neighbor_sets[i] = set(neighbors[i])
    
    # Precompute neighbor membership arrays for fast lookup
    # neighbor_member[i][x] = 1 if x is in neighbors[i], else 0
    # This is too memory intensive (6561*6561), so use a different approach:
    # For each point x, precompute which codewords cover it
    covered_by = [None] * N
    for i in range(N):
        for x in neighbors[i]:
            if covered_by[x] is None:
                covered_by[x] = []
            covered_by[x].append(i)
    
    # Also precompute as sets for fast intersection
    covered_by_sets = [set(covered_by[i]) if covered_by[i] else set() for i in range(N)]
    
    # Phase 1: Fast Greedy with count-array approach
    covered = np.zeros(N, dtype=bool)
    code_list = []
    count = np.full(N, 17, dtype=np.int32)  # Each word covers itself + 16 neighbors = 17
    
    while not covered.all() and time.time() < deadline - 40.0:
        c = int(np.argmax(count))
        if count[c] == 0:
            break
        code_list.append(c)
        for x in neighbors[c]:
            if not covered[x]:
                covered[x] = True
                for y in neighbors[x]:
                    count[y] -= 1
    
    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size
    
    # Initialize coverage counts
    current_cov_count = np.zeros(N, dtype=np.int32)
    for c in current_code:
        for x in neighbors[c]:
            current_cov_count[x] += 1
    
    random.seed(42)
    
    # Helper: try to remove a codeword if all its neighbors have count >= 2
    def try_remove(idx):
        nonlocal current_code, current_size, current_cov_count, stuck_counter
        c = current_code[idx]
        # Use precomputed neighbor set for fast min check
        n_set = neighbor_sets[c]
        removable = True
        for x in n_set:
            if current_cov_count[x] <= 1:
                removable = False
                break
        if removable:
            for x in neighbors[c]:
                current_cov_count[x] -= 1
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            return True
        return False
    
    # Helper: try a swap - remove c, add c' that covers all unique points of c
    def try_swap(c_idx):
        nonlocal current_code, current_size, current_cov_count, stuck_counter
        c = current_code[c_idx]
        risky_points = []
        n_set = neighbor_sets[c]
        for x in n_set:
            if current_cov_count[x] == 1:
                risky_points.append(x)
        
        if not risky_points:
            # Can just remove
            for x in neighbors[c]:
                current_cov_count[x] -= 1
            current_code.pop(c_idx)
            current_size -= 1
            stuck_counter = 0
            return True
        
        if len(risky_points) > 3:
            return False
        
        # Find candidates that cover all risky points using covered_by_sets
        candidate_set = covered_by_sets[risky_points[0]].copy()
        for x in risky_points[1:]:
            candidate_set &= covered_by_sets[x]
            if not candidate_set:
                break
        
        if not candidate_set:
            return False
        
        # Remove current codewords from candidates
        code_set = set(current_code)
        candidate_set -= code_set
        
        if not candidate_set:
            return False
        
        # Pick a candidate with good overlap
        candidates = list(candidate_set)[:20]
        best_cand = None
        best_score = -1
        for cand in candidates:
            overlap = 0
            for x in neighbors[cand]:
                if current_cov_count[x] > 0:
                    overlap += 1
            if overlap > best_score:
                best_score = overlap
                best_cand = cand
        
        if best_cand is None:
            return False
        
        # Apply swap
        for x in neighbors[c]:
            current_cov_count[x] -= 1
        for x in neighbors[best_cand]:
            current_cov_count[x] += 1
        current_code[c_idx] = best_cand
        stuck_counter = 0
        return True
    
    # Helper: try two-swap - remove two codewords, add one
    def try_two_swap():
        nonlocal current_code, current_size, current_cov_count, stuck_counter
        if current_size < 2:
            return False
        
        i1 = random.randint(0, current_size - 1)
        i2 = random.randint(0, current_size - 1)
        if i2 == i1:
            i2 = (i2 + 1) % current_size
        
        c1 = current_code[i1]
        c2 = current_code[i2]
        
        # Find points that would become uncovered if we remove both c1 and c2
        n1_set = neighbor_sets[c1]
        n2_set = neighbor_sets[c2]
        
        risky = set()
        for x in n1_set:
            if current_cov_count[x] == 1:
                risky.add(x)
        for x in n2_set:
            if current_cov_count[x] == 1:
                risky.add(x)
        
        # Also consider points covered by both c1 and c2 with count == 2
        for x in n1_set:
            if current_cov_count[x] == 2 and x in n2_set:
                risky.add(x)
        
        if len(risky) > 4:
            return False
        
        if not risky:
            # Can just remove both
            for x in neighbors[c1]:
                current_cov_count[x] -= 1
            for x in neighbors[c2]:
                current_cov_count[x] -= 1
            idxs = sorted([i1, i2], reverse=True)
            for idx in idxs:
                current_code.pop(idx)
            current_size -= 2
            stuck_counter = 0
            return True
        
        # Find candidates that cover all risky points using covered_by_sets
        risky_list = list(risky)
        candidate_set = covered_by_sets[risky_list[0]].copy()
        for x in risky_list[1:]:
            candidate_set &= covered_by_sets[x]
            if not candidate_set:
                break
        
        if not candidate_set:
            return False
        
        # Remove current codewords from candidates
        code_set = set(current_code)
        candidate_set -= code_set
        
        if not candidate_set:
            return False
        
        # Pick a candidate with good overlap
        candidates = list(candidate_set)[:20]
        best_cand = None
        best_score = -1
        for cand in candidates:
            overlap = 0
            for x in neighbors[cand]:
                if current_cov_count[x] > 0:
                    overlap += 1
            if overlap > best_score:
                best_score = overlap
                best_cand = cand
        
        if best_cand is None:
            return False
        
        # Apply: remove c1 and c2, add best_cand
        for x in neighbors[c1]:
            current_cov_count[x] -= 1
        for x in neighbors[c2]:
            current_cov_count[x] -= 1
        for x in neighbors[best_cand]:
            current_cov_count[x] += 1
        
        idxs = sorted([i1, i2], reverse=True)
        for idx in idxs:
            current_code.pop(idx)
        current_code.append(best_cand)
        current_size -= 1
        stuck_counter = 0
        return True
    
    # Helper: full reduction pass
    def full_reduction():
        nonlocal current_code, current_size, current_cov_count
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in current_code:
                n_set = neighbor_sets[c]
                removable = True
                for x in n_set:
                    if current_cov_count[x] <= 1:
                        removable = False
                        break
                if removable:
                    for x in neighbors[c]:
                        current_cov_count[x] -= 1
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                current_code = new_code
                current_size = len(current_code)
    
    # Phase 2: Local Search
    stuck_counter = 0
    
    # Helper: try 2-opt - replace two codewords with one that covers all their unique points
    def try_two_opt():
        nonlocal current_code, current_size, current_cov_count, stuck_counter
        if current_size < 2:
            return False
        
        # Pick two random codewords
        i1 = random.randint(0, current_size - 1)
        i2 = random.randint(0, current_size - 1)
        while i2 == i1:
            i2 = random.randint(0, current_size - 1)
        
        c1 = current_code[i1]
        c2 = current_code[i2]
        
        # Find points that would become uncovered if we remove both c1 and c2
        # A point x is risky if after removing c1 and c2, its coverage drops below 1
        # new_cov[x] = current_cov_count[x] - (1 if x in N(c1) else 0) - (1 if x in N(c2) else 0)
        # risky if new_cov[x] < 1, i.e., current_cov_count[x] < 1 + in_c1 + in_c2
        
        # Use precomputed neighbor sets for fast membership
        n1_set = neighbor_sets[c1]
        n2_set = neighbor_sets[c2]
        
        # Collect risky points by checking union of neighbors
        risky = []
        candidate_risky = n1_set | n2_set
        for x in candidate_risky:
            in_c1 = 1 if x in n1_set else 0
            in_c2 = 1 if x in n2_set else 0
            if current_cov_count[x] < 1 + in_c1 + in_c2:
                risky.append(x)
        
        if len(risky) > 5:
            return False
        
        if not risky:
            # Can just remove both
            for x in neighbors[c1]:
                current_cov_count[x] -= 1
            for x in neighbors[c2]:
                current_cov_count[x] -= 1
            idxs = sorted([i1, i2], reverse=True)
            for idx in idxs:
                current_code.pop(idx)
            current_size -= 2
            stuck_counter = 0
            return True
        
        # Find candidates that cover all risky points using precomputed covered_by_sets
        # A candidate c covers point x iff c is in covered_by[x]
        # We need c such that for all x in risky, c is in covered_by[x]
        # i.e., c is in intersection of covered_by[x] for x in risky
        risky_set = set(risky)
        
        # Start with covered_by of first risky point
        candidate_set = covered_by_sets[risky[0]].copy()
        for x in risky[1:]:
            candidate_set &= covered_by_sets[x]
            if not candidate_set:
                break
        
        if not candidate_set:
            return False
        
        # Remove current codewords from candidates
        code_set = set(current_code)
        candidate_set -= code_set
        
        if not candidate_set:
            return False
        
        # Pick a candidate with maximum overlap with existing coverage
        candidates = list(candidate_set)[:30]
        best_cand = None
        best_score = -1
        for cand in candidates:
            overlap = 0
            for x in neighbors[cand]:
                if current_cov_count[x] > 0:
                    overlap += 1
            if overlap > best_score:
                best_score = overlap
                best_cand = cand
        
        if best_cand is None:
            return False
        
        # Apply: remove c1 and c2, add best_cand
        for x in neighbors[c1]:
            current_cov_count[x] -= 1
        for x in neighbors[c2]:
            current_cov_count[x] -= 1
        for x in neighbors[best_cand]:
            current_cov_count[x] += 1
        
        idxs = sorted([i1, i2], reverse=True)
        for idx in idxs:
            current_code.pop(idx)
        current_code.append(best_cand)
        current_size -= 1
        stuck_counter = 0
        return True
    
    while time.time() < deadline - 5.0:
        if current_size == 0:
            break
        
        # First, try a full reduction
        old_size = current_size
        full_reduction()
        
        if current_size < best_size:
            best_size = current_size
            best_code = current_code[:]
        
        if current_size < old_size:
            stuck_counter = 0
            continue
        
        # Try random removals
        removed = False
        for _ in range(8):
            if time.time() > deadline - 5.0:
                break
            if current_size == 0:
                break
            idx = random.randint(0, current_size - 1)
            if try_remove(idx):
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
                removed = True
                break
        
        if removed:
            continue
        
        stuck_counter += 1
        
        # Try swaps if stuck
        if stuck_counter > 2 and time.time() < deadline - 10.0:
            swapped = False
            for _ in range(8):
                if time.time() > deadline - 10.0:
                    break
                if current_size == 0:
                    break
                c_idx = random.randint(0, current_size - 1)
                if try_swap(c_idx):
                    # Try reduction after swap
                    old_sz = current_size
                    full_reduction()
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                    swapped = True
                    break
            
            if swapped:
                continue
        
        # Try 2-opt (remove two, add one) if more stuck
        if stuck_counter > 6 and current_size >= 2 and time.time() < deadline - 15.0:
            two_opted = False
            for _ in range(8):
                if time.time() > deadline - 15.0:
                    break
                if try_two_opt():
                    # Try reduction after 2-opt
                    old_sz = current_size
                    full_reduction()
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                    two_opted = True
                    break
            
            if two_opted:
                continue
        
        # Try two-swaps if more stuck
        if stuck_counter > 12 and current_size >= 2 and time.time() < deadline - 15.0:
            two_swapped = False
            for _ in range(5):
                if time.time() > deadline - 15.0:
                    break
                if try_two_swap():
                    # Try reduction after two-swap
                    old_sz = current_size
                    full_reduction()
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                    two_swapped = True
                    break
            
            if two_swapped:
                continue
        
        # If completely stuck, try a random kick to escape local minimum
        if stuck_counter > 20 and time.time() < deadline - 10.0 and current_size > 0:
            c_idx = random.randint(0, current_size - 1)
            c = current_code[c_idx]
            cand = neighbors[c][random.randint(1, len(neighbors[c])-1)]  # Skip self
            if cand not in current_code:
                old_code = current_code[:]
                old_counts = current_cov_count.copy()
                
                # Apply kick
                for x in neighbors[c]:
                    current_cov_count[x] -= 1
                for x in neighbors[cand]:
                    current_cov_count[x] += 1
                current_code[c_idx] = cand
                
                # Check validity
                if np.any(current_cov_count < 1):
                    # Revert
                    current_code = old_code
                    current_cov_count = old_counts
                else:
                    # Try reduction
                    full_reduction()
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                    stuck_counter = 0
    
    # Final reduction
    current_code = best_code[:]
    current_size = len(current_code)
    current_cov_count = np.zeros(N, dtype=np.int32)
    for c in current_code:
        for x in neighbors[c]:
            current_cov_count[x] += 1
    full_reduction()
    
    all_words = list(itertools.product(range(3), repeat=8))
    return [list(all_words[i]) for i in current_code]
