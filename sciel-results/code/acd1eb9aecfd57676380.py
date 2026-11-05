"""Rewritten program: structural change to spectral + greedy local search for 2-coloring minimizing mono 4-APs."""
import time
import numpy as np

def construct():
    deadline = time.time() + 115  # safety margin before 120s
    
    N = 256
    
    # Precompute all 4-term APs (0-indexed)
    aps = []
    for d in range(1, N // 3 + 1):
        max_start = N - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            aps.append((i0, i0 + d, i0 + 2 * d, i0 + 3 * d))
    
    n_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (n_aps, 4)
    
    # Precompute membership: for each position, which APs and slots
    # Use arrays for speed
    ap_of_pos = [[] for _ in range(N)]  # list of (ap_idx, slot)
    for ap_idx in range(n_aps):
        for slot in range(4):
            pos = aps_arr[ap_idx, slot]
            ap_of_pos[pos].append((ap_idx, slot))
    
    # Convert to numpy arrays per position for fast access
    pos_ap_indices = []
    pos_ap_slots = []
    for pos in range(N):
        if ap_of_pos[pos]:
            idxs = [x[0] for x in ap_of_pos[pos]]
            slots = [x[1] for x in ap_of_pos[pos]]
            pos_ap_indices.append(np.array(idxs, dtype=np.int32))
            pos_ap_slots.append(np.array(slots, dtype=np.int32))
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_ap_slots.append(np.array([], dtype=np.int32))
    
    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        c = coloring[aps_arr]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())
    
    def calc_delta(coloring, idx):
        """Change in mono count if we flip position idx."""
        if len(pos_ap_indices[idx]) == 0:
            return 0
        ap_idxs = pos_ap_indices[idx]
        slots = pos_ap_slots[idx]
        
        sub_aps = aps_arr[ap_idxs]
        sub_c = coloring[sub_aps].copy()
        
        # Current mono count
        cur_mono = (sub_c[:, 0] == sub_c[:, 1]) & (sub_c[:, 1] == sub_c[:, 2]) & (sub_c[:, 2] == sub_c[:, 3])
        cur_count = int(cur_mono.sum())
        
        # Flip the relevant slot in each AP
        for i, slot in enumerate(slots):
            sub_c[i, slot] = 1 - sub_c[i, slot]
        
        new_mono = (sub_c[:, 0] == sub_c[:, 1]) & (sub_c[:, 1] == sub_c[:, 2]) & (sub_c[:, 2] == sub_c[:, 3])
        new_count = int(new_mono.sum())
        
        return new_count - cur_count
    
    def calc_all_deltas(coloring):
        """Calculate delta for all positions at once (vectorized-ish)."""
        deltas = np.zeros(N, dtype=np.int32)
        for idx in range(N):
            if len(pos_ap_indices[idx]) == 0:
                continue
            ap_idxs = pos_ap_indices[idx]
            slots = pos_ap_slots[idx]
            
            sub_aps = aps_arr[ap_idxs]
            sub_c = coloring[sub_aps].copy()
            
            cur_mono = (sub_c[:, 0] == sub_c[:, 1]) & (sub_c[:, 1] == sub_c[:, 2]) & (sub_c[:, 2] == sub_c[:, 3])
            cur_count = cur_mono.sum()
            
            for i, slot in enumerate(slots):
                sub_c[i, slot] = 1 - sub_c[i, slot]
            
            new_mono = (sub_c[:, 0] == sub_c[:, 1]) & (sub_c[:, 1] == sub_c[:, 2]) & (sub_c[:, 2] == sub_c[:, 3])
            new_count = new_mono.sum()
            
            deltas[idx] = new_count - cur_count
        return deltas
    
    def greedy_local_search(coloring, current_count, max_iterations=200):
        """Greedy local search: repeatedly flip the best position until no improvement."""
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            deltas = calc_all_deltas(coloring)
            min_delta = deltas.min()
            if min_delta >= 0:
                break
            best_idx = int(deltas.argmin())
            coloring[best_idx] = 1 - coloring[best_idx]
            current_count += min_delta
        return coloring, current_count
    
    def simulated_annealing(coloring, current_count, start_temp=2.0, end_temp=0.01, iterations=5000):
        """Simulated annealing to escape local minima."""
        temp = start_temp
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
            # Decay temperature
            progress = it / iterations
            temp = start_temp * (end_temp / start_temp) ** progress
            
            idx = np.random.randint(N)
            delta = calc_delta(coloring, idx)
            
            if delta < 0 or (np.random.random() < np.exp(-delta / max(temp, 1e-10))):
                coloring[idx] = 1 - coloring[idx]
                current_count += delta
        
        return coloring, current_count
    
    # Build the adjacency/interaction matrix for spectral initialization
    # M[i,j] = number of 4-APs that contain both i and j
    # This is expensive to build fully, so we use a sparse approach
    # Actually, let's use a different spectral approach:
    # For each AP (a,b,c,d), the monochromatic condition is satisfied when
    # all four have the same sign. We can think of this as a 4-body interaction.
    # A simpler spectral approach: build a matrix where M[i,j] counts pairs in APs
    # and use that for initialization.
    
    # Build pair interaction matrix
    pair_count = np.zeros((N, N), dtype=np.int32)
    for ap in aps:
        for i in range(4):
            for j in range(i + 1, 4):
                pair_count[ap[i], ap[j]] += 1
                pair_count[ap[j], ap[i]] += 1
    
    # Spectral initialization using the pair interaction matrix
    # The idea: positions that frequently appear together in APs should tend
    # to have the same color (to avoid mono APs, we actually want them to differ
    # in some sense, but the spectral method gives a good starting point)
    
    # Use eigendecomposition of the pair_count matrix
    # The top eigenvector gives a soft assignment
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(pair_count)
        # Use the second eigenvector (index -2) for a balanced split
        # The top eigenvector is usually all positive
        v = eigenvectors[:, -2]
        # Threshold at median for balanced coloring
        median_val = np.median(v)
        coloring_spec = (v > median_val).astype(np.int32)
    except:
        coloring_spec = np.random.randint(0, 2, N)
    
    best_coloring = None
    best_count = float('inf')
    
    # Multiple initializations
    initializations = []
    
    # Spectral
    initializations.append(coloring_spec.copy())
    
    # Random
    np.random.seed(42)
    for _ in range(5):
        initializations.append(np.random.randint(0, 2, N))
    
    # Block patterns
    for block_size in [2, 3, 4, 5, 7, 8, 16]:
        coloring = np.array([(i // block_size) % 2 for i in range(N)])
        initializations.append(coloring)
    
    # Thue-Morse-like
    coloring = np.array([(bin(i).count('1') % 2) for i in range(N)])
    initializations.append(coloring)
    
    # Run local search from each initialization
    for init in initializations:
        if time.time() > deadline - 10:
            break
        
        coloring = init.copy()
        current_count = count_mono(coloring)
        
        if current_count < best_count:
            best_count = current_count
            best_coloring = coloring.copy()
        
        # Greedy local search
        coloring, current_count = greedy_local_search(coloring, current_count)
        
        if current_count < best_count:
            best_count = current_count
            best_coloring = coloring.copy()
        
        # Simulated annealing to escape local minima
        if time.time() < deadline - 15:
            coloring_sa = coloring.copy()
            count_sa = current_count
            coloring_sa, count_sa = simulated_annealing(coloring_sa, count_sa, 
                                                         start_temp=1.0, end_temp=0.01,
                                                         iterations=3000)
            # Greedy after SA
            coloring_sa, count_sa = greedy_local_search(coloring_sa, count_sa)
            
            if count_sa < best_count:
                best_count = count_sa
                best_coloring = coloring_sa.copy()
        
        # More aggressive: multiple SA rounds with perturbation
        if time.time() < deadline - 10:
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                perturbed = best_coloring.copy() if best_coloring is not None else coloring.copy()
                n_flips = np.random.randint(5, 15)
                flip_idx = np.random.choice(N, n_flips, replace=False)
                perturbed[flip_idx] = 1 - perturbed[flip_idx]
                count_p = count_mono(perturbed)
                
                perturbed, count_p = greedy_local_search(perturbed, count_p)
                
                if count_p < best_count:
                    best_count = count_p
                    best_coloring = perturbed.copy()
    
    # Final polish: one more greedy pass on the best
    if best_coloring is not None and time.time() < deadline - 5:
        best_coloring, best_count = greedy_local_search(best_coloring.copy(), best_count)
    
    return best_coloring.tolist()
