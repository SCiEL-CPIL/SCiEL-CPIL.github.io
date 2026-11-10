import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq_norms = (grid ** 2).sum(axis=1)
    
    # Precompute: for each pair of points, store the 4x4 determinant contribution
    # Actually, let's precompute conflicts differently.
    # For a set S and candidate p, we need: for every 4-subset {q1,q2,q3,q4} of S,
    # det([q1-p, |q1|^2-|p|^2], [q2-p, |q2|^2-|p|^2], [q3-p, |q3|^2-|p|^2], [q4-p, |q4|^2-|p|^2]) != 0 (abs >= 0.5)
    
    # diff[i,j] = grid[j] - grid[i], shape (N, N, 3)
    diff = grid[np.newaxis, :, :] - grid[:, np.newaxis, :]
    sq_diff = sq_norms[np.newaxis, :] - sq_norms[:, np.newaxis]  # (N, N)
    
    def can_add_fast(selected, new_idx):
        """Check if new_idx can be added to selected (list of indices)."""
        k = len(selected)
        if k < 4:
            return True
        
        # Get vectors from new_idx to all selected points
        d_vecs = diff[new_idx, selected, :]  # (k, 3)
        d_sqs = sq_diff[new_idx, selected]   # (k,)
        
        # Check all 4-combinations
        # For speed, we'll batch process
        combos = list(itertools.combinations(range(k), 4))
        if not combos:
            return True
        
        # Build matrix: (num_combos, 4, 4)
        c = len(combos)
        mat = np.empty((c, 4, 4), dtype=np.float64)
        for i, (a, b, c_idx, d) in enumerate(combos):
            mat[i, 0] = np.append(d_vecs[a], d_sqs[a])
            mat[i, 1] = np.append(d_vecs[b], d_sqs[b])
            mat[i, 2] = np.append(d_vecs[c_idx], d_sqs[c_idx])
            mat[i, 3] = np.append(d_vecs[d], d_sqs[d])
        
        dets = np.linalg.det(mat)
        return not np.any(np.abs(dets) < 0.5)
    
    def greedy_construct(seed):
        """Greedy construction with random order."""
        np.random.seed(seed)
        order = np.random.permutation(N)
        selected = []
        selected_set = set()
        for idx in order:
            if can_add_fast(selected, idx):
                selected.append(idx)
                selected_set.add(idx)
        return selected
    
    def polish(selected, time_limit):
        """Try to add more points to selected set."""
        selected = list(selected)
        selected_set = set(selected)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try adding in a fixed order first
            for idx in range(N):
                if time.time() - start > time_limit:
                    break
                if idx not in selected_set:
                    if can_add_fast(selected, idx):
                        selected.append(idx)
                        selected_set.add(idx)
                        improved = True
        return selected
    
    def local_improve(selected, time_limit):
        """Local improvement: try to find addable points, use swaps if stuck."""
        selected = list(selected)
        selected_set = set(selected)
        start = time.time()
        
        while time.time() - start < time_limit:
            # Phase 1: Try to add any point
            added = False
            candidates = [i for i in range(N) if i not in selected_set]
            np.random.shuffle(candidates)
            for idx in candidates:
                if time.time() - start > time_limit:
                    break
                if can_add_fast(selected, idx):
                    selected.append(idx)
                    selected_set.add(idx)
                    added = True
                    break
            
            if added:
                continue
            
            # Phase 2: Try 1-swap
            swapped = False
            for _ in range(30):
                if time.time() - start > time_limit:
                    break
                out_idx = np.random.choice(selected)
                remaining = [i for i in selected if i != out_idx]
                remaining_set = selected_set - {out_idx}
                
                candidates = [i for i in range(N) if i not in remaining_set]
                np.random.shuffle(candidates)
                for in_idx in candidates[:100]:
                    if can_add_fast(remaining, in_idx):
                        selected = remaining + [in_idx]
                        selected_set = remaining_set | {in_idx}
                        swapped = True
                        break
                if swapped:
                    break
            
            # Phase 3: Try 1-for-2 swap (remove 1, add 2)
            if not swapped:
                found = False
                for _ in range(15):
                    if time.time() - start > time_limit:
                        break
                    out_idx = np.random.choice(selected)
                    remaining = [i for i in selected if i != out_idx]
                    remaining_set = selected_set - {out_idx}
                    
                    candidates = [i for i in range(N) if i not in remaining_set]
                    np.random.shuffle(candidates)
                    
                    for in1 in candidates[:30]:
                        if time.time() - start > time_limit:
                            break
                        if can_add_fast(remaining, in1):
                            temp = remaining + [in1]
                            temp_set = remaining_set | {in1}
                            candidates2 = [i for i in range(N) if i not in temp_set]
                            np.random.shuffle(candidates2)
                            for in2 in candidates2[:30]:
                                if can_add_fast(temp, in2):
                                    selected = temp + [in2]
                                    selected_set = temp_set | {in2}
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                
                if not found:
                    # Stuck, break
                    break
        
        return selected
    
    # Main search
    best_overall = []
    start_total = time.time()
    deadline = start_total + 570  # Safety margin of 60s
    
    # First, do several greedy constructions
    num_greedy = 20
    for i in range(num_greedy):
        if time.time() > deadline:
            break
        seed = i * 7 + 1
        result = greedy_construct(seed)
        if len(result) > len(best_overall):
            best_overall = result
    
    # Polish the best
    if best_overall:
        polished = polish(best_overall, 30)
        if len(polished) > len(best_overall):
            best_overall = polished
    
    # Now do local improvement with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        # Split remaining time among several local improvement runs
        num_ls = min(5, max(1, int(remaining_time / 60)))
        ls_time_each = min(remaining_time / num_ls, 120)
        
        for i in range(num_ls):
            if time.time() > deadline:
                break
            # Start from best_overall or a fresh greedy
            if i == 0:
                start_set = list(best_overall)
            else:
                start_set = greedy_construct(i * 13 + 100)
                if len(start_set) < len(best_overall):
                    start_set = list(best_overall)
            
            result = local_improve(start_set, ls_time_each)
            if len(result) > len(best_overall):
                best_overall = result
                # Quick polish
                polished = polish(best_overall, 10)
                if len(polished) > len(best_overall):
                    best_overall = polished
    
    # Final polish
    if best_overall and time.time() < deadline - 5:
        polished = polish(best_overall, 10)
        if len(polished) > len(best_overall):
            best_overall = polished
    
    if not best_overall:
        return np.empty((0, 3), dtype=np.int64)
    
    return grid[best_overall].astype(np.int64)
