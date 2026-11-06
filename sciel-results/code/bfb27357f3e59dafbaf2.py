import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Precompute grid points
    grid_points = [(x, y) for x in range(N) for y in range(N)]
    grid_array = np.array(grid_points, dtype=np.int64)
    
    best_points = []
    best_count = 0
    
    def run_greedy():
        S = []
        S_array = np.array([]).reshape(0, 2)
        D = np.array([]).reshape(0, 0)
        
        candidates_idx = np.arange(len(grid_points))
        random.shuffle(candidates_idx.tolist())
        
        while len(candidates_idx) > 0 and time.time() < deadline:
            if len(S) == 0:
                # Pick first point randomly
                idx = candidates_idx[0]
                p = grid_points[idx]
                S.append(p)
                S_array = np.array([p], dtype=np.int64)
                D = np.array([[0]], dtype=np.int64)
                candidates_idx = candidates_idx[1:]
                continue
            
            # Batch validation
            # D_G: (num_candidates, len(S))
            # S_array: (len(S), 2)
            
            # Compute squared distances
            # candidates: (C, 2)
            # S_array: (K, 2)
            # diff = candidates[:, np.newaxis, :] - S_array[np.newaxis, :, :]
            # d_sq = np.sum(diff**2, axis=2)
            
            cand_arr = grid_array[candidates_idx]
            
            # This might be memory heavy if C is large and K is large.
            # C ~ 4000, K ~ 50.
            # 4000 * 50 * 2 = 400,000 floats. Fine.
            
            diff = cand_arr[:, np.newaxis, :] - S_array[np.newaxis, :, :]
            d_sq = np.sum(diff**2, axis=2)
            
            # Check 1: Distances from candidate to S are distinct
            # Sort rows and check for duplicates
            # np.sort(d_sq, axis=1)
            # diffs = np.diff(sorted, axis=1)
            # has_dup = np.any(diffs == 0, axis=1)
            
            sorted_d = np.sort(d_sq, axis=1)
            diffs = np.diff(sorted_d, axis=1)
            has_dup = np.any(diffs == 0, axis=1)
            
            # Check 2: Conflict with existing pairwise distances
            # D is (K, K)
            # We need to check if d_sq[i, j] == D[j, k] for any k != j?
            # Actually, if d_sq[i, j] == D[j, k], then triangle (Candidate, S_j, S_k) is isosceles at S_j.
            # So for each candidate i, we check if any value in row i of d_sq is in row j of D (for the corresponding j).
            # d_sq[i, j] is distance from Cand_i to S_j.
            # We need to check if d_sq[i, j] is in D[j, :] (excluding D[j,j]=0).
            
            # Vectorized:
            # Expand d_sq to (C, K, 1)
            # Expand D to (1, K, K)
            # Compare
            # conflict_mask = (d_sq[:, :, np.newaxis] == D[np.newaxis, :, :])
            # This creates (C, K, K) boolean array.
            # 4000 * 50 * 50 = 10,000,000 booleans. 10 MB. OK.
            
            conflict_mask = (d_sq[:, :, np.newaxis] == D[np.newaxis, :, :])
            
            # If any conflict in row i, candidate i is invalid.
            has_conflict = np.any(conflict_mask, axis=(1, 2))
            
            # Valid candidates are those with no dup and no conflict
            valid_mask = ~(has_dup | has_conflict)
            
            if not np.any(valid_mask):
                break
            
            # Pick a random valid candidate
            valid_indices = np.where(valid_mask)[0]
            pick_idx = random.choice(valid_indices.tolist())
            
            p_idx = candidates_idx[pick_idx]
            p = grid_points[p_idx]
            
            S.append(p)
            S_array = np.vstack([S_array, np.array([p])])
            
            # Update D
            # New row/col
            # d_sq[pick_idx] is the distance vector from new point to old S
            new_d = d_sq[pick_idx]
            
            D_new = np.zeros((len(S), len(S)), dtype=np.int64)
            D_new[:-1, :-1] = D
            D_new[:-1, -1] = new_d
            D_new[-1, :-1] = new_d
            D_new[-1, -1] = 0
            D = D_new
            
            # Remove picked candidate from candidates
            # candidates_idx is a list/array. We need to remove the element at pick_idx.
            # Converting to list for removal is O(C).
            # We can just keep a list of indices and mark used ones?
            # Or use a set of used indices.
            # Since we shuffle initially, we can just pop from a list.
            # But we used numpy array for candidates_idx.
            # Let's switch to a list for candidates to make removal easier, or just filter.
            
            # Filter candidates_idx
            candidates_idx = np.delete(candidates_idx, pick_idx)
            
        return S

    # Run multiple restarts
    for trial in range(1000):
        if time.time() > deadline - 5:
            break
        
        pts = run_greedy()
        if len(pts) > best_count:
            best_points = pts
            best_count = len(pts)
            
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
