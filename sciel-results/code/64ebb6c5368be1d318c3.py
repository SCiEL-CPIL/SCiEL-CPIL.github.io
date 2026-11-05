import time
import numpy as np

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    N = 256
    
    # Precompute all 4-term APs (0-indexed)
    # a, a+d, a+2d, a+3d with a+3d <= 255
    # d can be at most (255-1)/3 = 84
    aps = []
    for d in range(1, N // 3 + 1):
        max_start = N - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            aps.append((i0, i0 + d, i0 + 2 * d, i0 + 3 * d))
    
    n_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (n_aps, 4)
    
    # Precompute for each position:
    # 1. The indices of APs it belongs to
    # 2. The slot (0,1,2,3) it occupies in each of those APs
    # 3. The indices of the OTHER 3 positions in each of those APs
    # This allows O(1) delta calculation per AP per position if vectorized properly.
    
    # pos_ap_idxs[pos] : array of AP indices containing pos
    # pos_slot[pos] : array of slots (0-3) pos occupies in those APs
    # pos_others[pos] : array of shape (num_aps_for_pos, 3) containing the other 3 indices
    
    pos_ap_idxs = [np.array([], dtype=np.int32) for _ in range(N)]
    pos_slot = [np.array([], dtype=np.int32) for _ in range(N)]
    pos_others = [np.array([], dtype=np.int32).reshape(0, 3) for _ in range(N)]
    
    # Build these structures
    # We can iterate over APs and append to the position lists
    # But for efficiency in the loop below, pre-building is fine since N=256 is small
    
    # Alternative faster build:
    # For each AP, update the 4 positions
    for ap_idx in range(n_aps):
        i0, i1, i2, i3 = aps_arr[ap_idx]
        
        # pos i0
        pos_ap_idxs[i0] = np.append(pos_ap_idxs[i0], ap_idx)
        pos_slot[i0] = np.append(pos_slot[i0], 0)
        others = np.array([i1, i2, i3], dtype=np.int32)
        pos_others[i0] = np.vstack([pos_others[i0], others]) if pos_others[i0].size > 0 else others.reshape(1,3)
        
        # pos i1
        pos_ap_idxs[i1] = np.append(pos_ap_idxs[i1], ap_idx)
        pos_slot[i1] = np.append(pos_slot[i1], 1)
        others = np.array([i0, i2, i3], dtype=np.int32)
        pos_others[i1] = np.vstack([pos_others[i1], others]) if pos_others[i1].size > 0 else others.reshape(1,3)
        
        # pos i2
        pos_ap_idxs[i2] = np.append(pos_ap_idxs[i2], ap_idx)
        pos_slot[i2] = np.append(pos_slot[i2], 2)
        others = np.array([i0, i1, i3], dtype=np.int32)
        pos_others[i2] = np.vstack([pos_others[i2], others]) if pos_others[i2].size > 0 else others.reshape(1,3)
        
        # pos i3
        pos_ap_idxs[i3] = np.append(pos_ap_idxs[i3], ap_idx)
        pos_slot[i3] = np.append(pos_ap_idxs[i3], ap_idx) # Typo check: should be pos_slot
        pos_slot[i3] = np.append(pos_slot[i3], 3)
        others = np.array([i0, i1, i2], dtype=np.int32)
        pos_others[i3] = np.vstack([pos_others[i3], others]) if pos_others[i3].size > 0 else others.reshape(1,3)

    # Fix typo in build above: pos_ap_idxs[i3] was appended twice? No, I wrote it once in the thought, but let's be careful in code.
    # Let's rewrite the build cleanly to avoid bugs.
    
    pos_ap_idxs = [np.array([], dtype=np.int32) for _ in range(N)]
    pos_others = [np.array([], dtype=np.int32).reshape(0, 3) for _ in range(N)]
    
    for ap_idx in range(n_aps):
        i0, i1, i2, i3 = aps_arr[ap_idx]
        
        pos_ap_idxs[i0] = np.append(pos_ap_idxs[i0], ap_idx)
        pos_others[i0] = np.vstack([pos_others[i0], [[i1, i2, i3]]])
        
        pos_ap_idxs[i1] = np.append(pos_ap_idxs[i1], ap_idx)
        pos_others[i1] = np.vstack([pos_others[i1], [[i0, i2, i3]]])
        
        pos_ap_idxs[i2] = np.append(pos_ap_idxs[i2], ap_idx)
        pos_others[i2] = np.vstack([pos_others[i2], [[i0, i1, i3]]])
        
        pos_ap_idxs[i3] = np.append(pos_ap_idxs[i3], ap_idx)
        pos_others[i3] = np.vstack([pos_others[i3], [[i0, i1, i2]]])

    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        c = coloring[aps_arr]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())
    
    def calc_all_deltas_fast(coloring):
        """
        Calculate delta for flipping each position.
        Delta is +1 if flip creates mono AP, -1 if flip breaks mono AP, 0 otherwise.
        
        For a position p in an AP with other positions o1, o2, o3:
        Let cp = coloring[p], co1, co2, co3 be colors of others.
        If co1 == co2 == co3:
            If cp == co1: The AP is currently mono. Flipping p makes it non-mono. Delta = -1.
            If cp != co1: The AP is not mono. Flipping p makes cp == co1 (since cp becomes 1-cp, and co1 is 1 or 0). 
                         Wait, if cp != co1, then 1-cp == co1. So the AP becomes mono. Delta = +1.
        Else:
            The AP is not mono, and cannot become mono by flipping just p (since the others are not all equal). Delta = 0.
        """
        deltas = np.zeros(N, dtype=np.int32)
        
        for p in range(N):
            others = pos_others[p]
            if others.size == 0:
                continue
            
            c_others = coloring[others] # shape (K, 3)
            
            # Check if the 3 others are all equal
            others_equal = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
            
            if not np.any(others_equal):
                continue
            
            c_p = coloring[p]
            
            # For APs where others are equal:
            # If c_p == c_others[0] (which is the common color), delta is -1
            # If c_p != c_others[0], delta is +1
            
            # We only care about the APs where others_equal is True
            # delta_contrib = -1 if c_p == c_others[:,0] else +1
            # But only for rows where others_equal is True.
            
            # Vectorized:
            # Create an array of deltas for all APs for this position
            # If others_equal:
            #    delta = -1 if c_p == c_others[:,0] else 1
            # Else:
            #    delta = 0
            
            # Since c_others[:,0] is the common color for those rows:
            is_match = (c_others[:, 0] == c_p)
            
            # delta_vals: +1 if not match, -1 if match
            delta_vals = np.where(is_match, -1, 1)
            
            # Mask with others_equal
            delta_vals = np.where(others_equal, delta_vals, 0)
            
            deltas[p] = np.sum(delta_vals)
            
        return deltas

    def local_search(coloring, current_count, max_sweeps=1000):
        """
        Greedy local search. Flip the position with the most negative delta.
        If multiple have same delta, pick one (or all? No, greedy is safer for correctness).
        """
        improved = True
        sweeps = 0
        while improved and sweeps < max_sweeps:
            if time.time() > deadline - 2:
                break
            
            deltas = calc_all_deltas_fast(coloring)
            min_delta = np.min(deltas)
            
            if min_delta >= 0:
                break
            
            # Find all positions with min_delta
            # Picking just one is standard greedy. Picking all might be faster but riskier.
            # Let's pick the first one found or all of them if we want to be aggressive.
            # Given the fast delta calc, we can try flipping ALL improving positions in a batch?
            # No, flipping one changes the deltas of neighbors.
            # Standard greedy: flip one.
            
            best_idx = int(np.argmin(deltas))
            coloring[best_idx] = 1 - coloring[best_idx]
            current_count += int(min_delta)
            
            improved = True
            sweeps += 1
            
        return coloring, current_count

    def batched_local_search(coloring, current_count, max_sweeps=100):
        """
        Flip all positions with negative delta simultaneously.
        This is a heuristic that can escape some local minima faster, but might overshoot.
        We recalculate count exactly after the batch.
        """
        for _ in range(max_sweeps):
            if time.time() > deadline - 2:
                break
            
            deltas = calc_all_deltas_fast(coloring)
            improving = np.where(deltas < 0)[0]
            
            if len(improving) == 0:
                break
            
            # Flip all
            for p in improving:
                coloring[p] = 1 - coloring[p]
            
            # Recalculate exact count
            current_count = count_mono(coloring)
            
        return coloring, current_count

    # --- Initialization Strategies ---
    
    def get_spectral_init():
        # Build pair interaction matrix
        # M[i,j] = number of APs containing both i and j
        M = np.zeros((N, N), dtype=np.int32)
        for ap in aps:
            i0, i1, i2, i3 = ap
            for i in range(4):
                for j in range(i+1, 4):
                    a, b = ap[i], ap[j]
                    M[a, b] += 1
                    M[b, a] += 1
        
        # Spectral clustering: use the second smallest eigenvector (Fiedler vector)
        # Or the largest? For min-cut like problems, Fiedler vector (2nd smallest) is often used.
        # Here we want to partition into 2 sets.
        try:
            eigvals, eigvecs = np.linalg.eigh(M)
            # eigh returns sorted ascending.
            # The smallest is 0 (if connected). The next is Fiedler.
            # Sometimes the largest eigenvector is better for independent set type problems?
            # Let's try Fiedler (index 1) and Largest (index -1).
            
            v_fiedler = eigvecs[:, 1]
            v_max = eigvecs[:, -1]
            
            # Threshold at median or 0
            init_fiedler = (v_fiedler > np.median(v_fiedler)).astype(np.int8)
            init_max = (v_max > np.median(v_max)).astype(np.int8)
            
            return [init_fiedler, init_max]
        except:
            return [np.random.randint(0, 2, N, dtype=np.int8)]

    def get_pattern_inits():
        inits = []
        # Thue-Morse
        tm = np.array([(bin(i).count('1') % 2) for i in range(N)], dtype=np.int8)
        inits.append(tm)
        inits.append(1 - tm)
        
        # Blocks
        for bs in [4, 8, 16, 32]:
            pat = np.array([(i // bs) % 2 for i in range(N)], dtype=np.int8)
            inits.append(pat)
            inits.append(1 - pat)
            
        # Stripes
        for per in [3, 5, 7]:
            pat = np.array([(i // per) % 2 for i in range(N)], dtype=np.int8)
            inits.append(pat)
        
        return inits

    # --- Main Search Loop ---
    
    best_coloring = None
    best_count = float('inf')
    
    # 1. Spectral Inits
    spec_inits = get_spectral_init()
    for init in spec_inits:
        coloring = init.copy()
        c = count_mono(coloring)
        if c < best_count:
            best_count = c
            best_coloring = coloring.copy()
        
        # Refine
        coloring, c = local_search(coloring, c)
        if c < best_count:
            best_count = c
            best_coloring = coloring.copy()
            
        # Batch refine
        if time.time() < deadline - 10:
            coloring_b = coloring.copy()
            c_b = c
            coloring_b, c_b = batched_local_search(coloring_b, c_b)
            if c_b < best_count:
                best_count = c_b
                best_coloring = coloring_b.copy()
                
    # 2. Pattern Inits
    pat_inits = get_pattern_inits()
    for init in pat_inits:
        if time.time() > deadline - 20:
            break
        coloring = init.copy()
        c = count_mono(coloring)
        if c < best_count:
            best_count = c
            best_coloring = coloring.copy()
        
        coloring, c = local_search(coloring, c)
        if c < best_count:
            best_count = c
            best_coloring = coloring.copy()
            
    # 3. Random Inits + Perturbation
    np.random.seed(42)
    
    # Run a few random restarts
    for i in range(10):
        if time.time() > deadline - 30:
            break
        coloring = np.random.randint(0, 2, N, dtype=np.int8)
        c = count_mono(coloring)
        
        coloring, c = local_search(coloring, c)
        if c < best_count:
            best_count = c
            best_coloring = coloring.copy()
            
    # 4. Perturbation Search (Simulated Annealing lite)
    # Start from best, perturb, re-search
    if best_coloring is not None:
        current_best = best_coloring.copy()
        current_best_count = best_count
        
        for i in range(50):
            if time.time() > deadline - 5:
                break
            
            # Perturb
            perturbed = current_best.copy()
            n_flips = np.random.randint(2, 8)
            flip_idxs = np.random.choice(N, n_flips, replace=False)
            perturbed[flip_idxs] = 1 - perturbed[flip_idxs]
            
            c = count_mono(perturbed)
            
            # Refine
            perturbed, c = local_search(perturbed, c)
            
            if c < current_best_count:
                current_best_count = c
                current_best = perturbed.copy()
                if c < best_count:
                    best_count = c
                    best_coloring = current_best.copy()
                    
    # Final polish
    if best_coloring is not None and time.time() < deadline - 2:
        coloring = best_coloring.copy()
        c = best_count
        coloring, c = local_search(coloring, c)
        if c < best_count:
            best_count = c
            best_coloring = coloring.copy()

    return [int(c) for c in best_coloring]

if __name__ == "__main__":
    # Test
    res = construct()
    assert len(res) == 256
    assert all(x in [0, 1] for x in res)
    print("Constructed successfully.")
