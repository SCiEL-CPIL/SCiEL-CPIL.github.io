import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n          # 6561
    ball_size = 1 + n * (q - 1)    # 17
    half_n = n // 2                # 4

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (ball of radius 1) for each point
    neighbors = [None] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        nbrs = [idx]
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            nbrs.append(idx + delta)

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            nbrs.append(idx + delta)

        neighbors[idx] = nbrs

    # Vectorized balls matrix (6561, 17)
    balls = np.array(neighbors, dtype=np.int32)

    # Slice decomposition: split into two 4D subspaces
    # First 4 coords: indices 0..3, Last 4 coords: indices 4..7
    # A "slice" is defined by fixing the first 4 coordinates
    # There are 3^4 = 81 slices
    slice_size = q ** half_n  # 81 points per slice
    num_slices = q ** half_n  # 81 slices

    # Precompute slice assignment: for each point, which slice it belongs to
    # Slice index = first 4 coordinates interpreted as base-3 number
    slice_of_point = np.zeros(total_points, dtype=np.int32)
    for idx in range(total_points):
        # Extract first 4 digits
        s = 0
        temp = idx
        for i in range(half_n):
            s = s * q + (temp % q)
            temp //= q
        slice_of_point[idx] = s

    # For each slice, precompute the set of points in that slice
    slice_points = [list(range(total_points))] * num_slices
    for s in range(num_slices):
        slice_points[s] = []
    for idx in range(total_points):
        slice_points[slice_of_point[idx]].append(idx)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def slice_idx_to_word(idx):
        """Convert slice-local index (0..80) to 4-digit word"""
        w = [0] * half_n
        for i in range(half_n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_slice_idx(w):
        """Convert 4-digit word to slice-local index"""
        idx = 0
        for i in range(half_n - 1, -1, -1):
            idx = idx * q + w[i]
        return idx

    def greedy_set_cover(seed, deterministic=False, use_slices=False):
        """Greedy set cover with optional slice-aware candidate selection"""
        rng = np.random.default_rng(seed) if not deterministic else None
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        gain = np.full(total_points, ball_size, dtype=np.int32)
        
        if use_slices:
            # Track which slices still need coverage
            slice_covered = np.zeros(num_slices, dtype=bool)
            
            while not np.all(is_covered):
                max_gain = np.max(gain)
                candidates = np.where(gain == max_gain)[0]
                
                if len(candidates) > 1 and not deterministic:
                    # Prefer candidates from slices with less coverage
                    cand_slices = slice_of_point[candidates]
                    slice_priority = np.zeros(num_slices, dtype=np.int32)
                    for s in range(num_slices):
                        pts_in_slice = slice_points[s]
                        uncovered_in_slice = np.sum(~is_covered[pts_in_slice])
                        slice_priority[s] = uncovered_in_slice
                    
                    # Weight candidates by their slice's uncovered count
                    weights = slice_priority[cand_slices]
                    if weights.max() > 0:
                        # Pick from highest-priority slices
                        max_weight = weights.max()
                        high_priority = candidates[weights == max_weight]
                        best_idx = int(rng.choice(high_priority))
                    else:
                        best_idx = int(rng.choice(candidates))
                else:
                    best_idx = int(candidates[0])

                selected.append(best_idx)
                new_points = neighbors[best_idx]
                for p in new_points:
                    if not is_covered[p]:
                        is_covered[p] = True
                        gain[neighbors[p]] -= 1

        else:
            while not np.all(is_covered):
                max_gain = np.max(gain)
                candidates = np.where(gain == max_gain)[0]
                if len(candidates) > 1 and not deterministic:
                    best_idx = int(rng.choice(candidates))
                else:
                    best_idx = int(candidates[0])

                selected.append(best_idx)
                new_points = neighbors[best_idx]
                for p in new_points:
                    if not is_covered[p]:
                        is_covered[p] = True
                        gain[neighbors[p]] -= 1

        return selected

    def prune(selected_list, coverage_count=None):
        if not selected_list:
            return selected_list
        selected_list = list(selected_list)
        if coverage_count is None:
            coverage_count = np.zeros(total_points, dtype=np.int32)
            for idx in selected_list:
                coverage_count[balls[idx]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    selected_list.pop(i)
                    coverage_count[balls[idx]] -= 1
                    changed = True
                    break
        return selected_list

    def rebuild_coverage(selected_list):
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return coverage_count

    def local_search_walk(selected_indices, deadline):
        """Slice-aware local search: prioritize moves within the same slice"""
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        # Precompute for each selected codeword, which slice it's in
        while time.time() < deadline:
            # Deterministic removal sweep until fixed point
            removed_any = True
            while removed_any and time.time() < deadline:
                removed_any = False
                for i in range(len(selected_list)):
                    idx = selected_list[i]
                    if np.all(coverage_count[balls[idx]] > 1):
                        coverage_count[balls[idx]] -= 1
                        selected_list.pop(i)
                        selected_set.discard(idx)
                        removed_any = True
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        break

            if time.time() > deadline:
                break

            n_sel = len(selected_list)
            if n_sel == 0:
                break

            # Slice-aware swap: prefer moving within the same slice
            progress = False
            for _ in range(60):
                if time.time() > deadline:
                    break
                i = rng.randrange(n_sel)
                c = selected_list[i]
                c_slice = slice_of_point[c]
                
                # Get neighbors, sorted by whether they're in the same slice
                nbrs = list(neighbors[c])
                same_slice_nbrs = [nb for nb in nbrs if nb != c and slice_of_point[nb] == c_slice]
                diff_slice_nbrs = [nb for nb in nbrs if nb != c and slice_of_point[nb] != c_slice]
                
                # Try same-slice neighbors first
                rng.shuffle(same_slice_nbrs)
                rng.shuffle(diff_slice_nbrs)
                
                for c_prime in same_slice_nbrs + diff_slice_nbrs:
                    if c_prime in selected_set:
                        continue
                    # Tentatively swap
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1
                    if np.any(coverage_count[balls[c]] == 0):
                        # Invalid, revert
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        continue
                    # Valid swap
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    progress = True
                    break
                if progress:
                    break

            if progress:
                pruned = prune(selected_list, coverage_count)
                if len(pruned) < len(selected_list):
                    coverage_count = rebuild_coverage(pruned)
                    selected_list = pruned
                    selected_set = set(selected_list)
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    def polish(selected_indices, deadline):
        """High-precision polish with slice-aware moves and 2-opt style exchanges"""
        selected_list = list(selected_indices)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(7)

        for _ in range(300):
            if time.time() > deadline:
                break
            # Deterministic removal sweep
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    coverage_count[balls[idx]] -= 1
                    selected_list.pop(i)
                    selected_set.discard(idx)
                    changed = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                    break
            if changed:
                continue

            # Slice-aware neighbor swap
            n_sel = len(selected_list)
            if n_sel == 0:
                break
            i = rng.randrange(n_sel)
            c = selected_list[i]
            c_slice = slice_of_point[c]
            
            nbrs = list(neighbors[c])
            same_slice_nbrs = [nb for nb in nbrs if nb != c and slice_of_point[nb] == c_slice]
            diff_slice_nbrs = [nb for nb in nbrs if nb != c and slice_of_point[nb] != c_slice]
            
            rng.shuffle(same_slice_nbrs)
            rng.shuffle(diff_slice_nbrs)
            
            swapped = False
            for c_prime in same_slice_nbrs + diff_slice_nbrs:
                if c_prime in selected_set:
                    continue
                coverage_count[balls[c]] -= 1
                coverage_count[balls[c_prime]] += 1
                if np.any(coverage_count[balls[c]] == 0):
                    coverage_count[balls[c_prime]] -= 1
                    coverage_count[balls[c]] += 1
                    continue
                selected_list[i] = c_prime
                selected_set.discard(c)
                selected_set.add(c_prime)
                swapped = True
                break

            if swapped:
                pruned = prune(selected_list, coverage_count)
                if len(pruned) < len(selected_list):
                    coverage_count = rebuild_coverage(pruned)
                    selected_list = pruned
                    selected_set = set(selected_list)
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    def slice_rebalance(selected_list, deadline):
        """Try to reduce codewords in over-represented slices by moving to under-represented slices"""
        if time.time() > deadline:
            return selected_list
        
        selected_list = list(selected_list)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        
        # Count codewords per slice
        slice_counts = np.zeros(num_slices, dtype=np.int32)
        for idx in selected_list:
            slice_counts[slice_of_point[idx]] += 1
        
        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(123)
        
        # Try to move codewords from dense slices to sparse slices
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Find slice with most codewords and slice with fewest
            if len(selected_list) == 0:
                break
            
            max_slice = np.argmax(slice_counts)
            min_slice = np.argmin(slice_counts)
            
            if slice_counts[max_slice] <= slice_counts[min_slice] + 1:
                continue  # Already balanced
            
            # Pick a random codeword from the dense slice
            dense_cw = [idx for idx in selected_list if slice_of_point[idx] == max_slice]
            if not dense_cw:
                continue
            c = rng.choice(dense_cw)
            
            # Try to move it to the sparse slice
            # Find a point in the sparse slice that is a neighbor of c
            sparse_pts = slice_points[min_slice]
            moved = False
            for target in sparse_pts:
                if target in selected_set:
                    continue
                # Check if target is within distance 1 of c (i.e., in neighbors[c])
                if target not in neighbors[c]:
                    continue
                
                # Tentatively move
                coverage_count[balls[c]] -= 1
                coverage_count[balls[target]] += 1
                if np.any(coverage_count[balls[c]] == 0):
                    coverage_count[balls[target]] -= 1
                    coverage_count[balls[c]] += 1
                    continue
                
                # Valid move
                selected_list[selected_list.index(c)] = target
                selected_set.discard(c)
                selected_set.add(target)
                slice_counts[max_slice] -= 1
                slice_counts[min_slice] += 1
                moved = True
                break
            
            if moved:
                pruned = prune(selected_list, coverage_count)
                if len(pruned) < len(selected_list):
                    coverage_count = rebuild_coverage(pruned)
                    selected_list = pruned
                    selected_set = set(selected_list)
                    # Recompute slice counts
                    slice_counts = np.zeros(num_slices, dtype=np.int32)
                    for idx in selected_list:
                        slice_counts[slice_of_point[idx]] += 1
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
        
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0

    best_code = None
    best_len = float('inf')

    # Run multiple greedy instances with different strategies
    # Strategy 1: Standard greedy (deterministic)
    selected = greedy_set_cover(0, deterministic=True, use_slices=False)
    selected = prune(selected)
    if len(selected) < best_len:
        best_len = len(selected)
        best_code = selected

    # Strategy 2: Slice-aware greedy
    if time.time() < deadline - 30:
        selected = greedy_set_cover(1, deterministic=False, use_slices=True)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected

    # Strategy 3: More random greedy instances
    for seed in range(2, 8):
        if time.time() > deadline - 25:
            break
        selected = greedy_set_cover(seed, deterministic=False, use_slices=(seed % 2 == 0))
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected

    if best_code is None:
        best_code = prune(greedy_set_cover(0, deterministic=True))
        best_len = len(best_code)

    # Local search walk on the best code
    best_code = local_search_walk(best_code, deadline)

    # Slice rebalancing
    if time.time() < deadline - 15:
        best_code = slice_rebalance(best_code, deadline)

    # Final polish with restarts
    best_code = polish(best_code, deadline)

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
