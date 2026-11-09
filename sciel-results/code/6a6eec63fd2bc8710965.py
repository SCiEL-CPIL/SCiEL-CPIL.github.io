import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def construct():
    N = 256
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Precompute all 4-APs
    aps = []
    pos_to_aps_indices = [[] for _ in range(N)]
    
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(N - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(aps)
            aps.append(ap)
            for pos in ap:
                pos_to_aps_indices[pos].append(idx)
                
    M = len(aps)
    aps_array = np.array(aps, dtype=np.int32)
    
    # Helper to count monochromatic APs
    def count_mono(colors):
        # Vectorized count
        # colors shape (N,)
        # aps_array shape (M, 4)
        c0 = colors[aps_array[:, 0]]
        c1 = colors[aps_array[:, 1]]
        c2 = colors[aps_array[:, 2]]
        c3 = colors[aps_array[:, 3]]
        return np.sum((c0 == c1) & (c1 == c2) & (c2 == c3))

    # Thue-Morse initialization
    def thue_morse(n):
        return bin(n).count('1') % 2

    best_colors = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    best_cost = count_mono(best_colors)
    
    # Vectorized get_delta
    def get_delta(colors, i):
        ap_indices = pos_to_aps_indices[i]
        if not ap_indices:
            return 0
        ap_indices = np.array(ap_indices, dtype=np.int32)
        ap_data = aps_array[ap_indices]
        colors_in_aps = colors[ap_data]
        
        old_c = colors[i]
        new_c = 1 - old_c
        
        # Count matches in the 4 positions
        # Note: colors_in_aps includes the position i itself
        match_old = (colors_in_aps == old_c).sum(axis=1)
        match_new = (colors_in_aps == new_c).sum(axis=1)
        
        # Exclude i from the count of old_c matches
        # i is old_c, so it contributes 1 to match_old
        # It contributes 0 to match_new
        n_old_others = match_old - 1
        n_new_others = match_new
        
        before_mono = (n_old_others == 3).sum()
        after_mono = (n_new_others == 3).sum()
        
        return int(after_mono - before_mono)

    def hill_climb(colors):
        colors = colors.copy()
        cost = count_mono(colors)
        improved = True
        # Limit iterations to prevent infinite loops if stuck in oscillation (though greedy should converge)
        max_iters = 1000
        iter_count = 0
        while improved and iter_count < max_iters:
            iter_count += 1
            improved = False
            best_delta = 0
            best_i = -1
            for i in range(N):
                d = get_delta(colors, i)
                if d < best_delta:
                    best_delta = d
                    best_i = i
            if best_delta < 0:
                colors[best_i] = 1 - colors[best_i]
                cost += best_delta
                improved = True
        return colors, cost

    # Run hill climb from Thue-Morse
    colors, cost = hill_climb(best_colors)
    if cost < best_cost:
        best_cost = cost
        best_colors = colors

    # Spectral Method Preparation
    # Construct incidence matrix A (N x M)
    # A[i, k] = 1 if i in AP k
    row_indices = []
    col_indices = []
    for i in range(N):
        for k in pos_to_aps_indices[i]:
            row_indices.append(i)
            col_indices.append(k)
    
    A = sp.csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(N, M))
    
    # Compute top singular vectors
    # We want the right singular vectors V
    # A = U S V^T
    # We want columns of V
    # Use svds for top k
    try:
        k_spec = 20
        if M > k_spec and N > k_spec:
            U, s, Vt = spla.svds(A, k=k_spec, return_singular_vectors='right')
            # Vt is k x N, rows are singular vectors
            V_top = Vt.T # N x k
        else:
            V_top = np.zeros((N, 0))
    except Exception:
        V_top = np.zeros((N, 0))
        
    def spectral_project(colors):
        if V_top.shape[1] == 0:
            return colors
        # Map colors to spins: 0 -> -1, 1 -> 1
        spins = 2 * colors.astype(np.float64) - 1
        # Project onto orthogonal complement of V_top
        # P_top = V_top V_top^T
        # s_proj = s - P_top s
        # s_proj = s - V_top (V_top^T s)
        proj = V_top @ spins
        s_proj = spins - V_top @ proj
        # Round to nearest integer, then map back to 0/1
        s_rounded = np.sign(s_proj)
        # Handle 0s by keeping original or random? 
        # If s_proj is 0, sign is 0. Let's map 0 to original color or random.
        # Better: map to 0 if < 0, 1 if >= 0? 
        # Let's just use sign, and if 0, use 1.
        s_rounded[s_rounded == 0] = 1
        new_colors = ((s_rounded + 1) / 2).astype(np.int8)
        return new_colors

    import random
    random.seed(42)
    
    current_colors = best_colors.copy()
    current_cost = best_cost
    
    while time.time() - start_time < time_limit - 5:
        # Choose strategy: Random Kick or Spectral Project
        if random.random() < 0.3 and V_top.shape[1] > 0:
            new_colors = spectral_project(current_colors)
            # Add small random noise to break symmetries if projection is degenerate
            # Flip 1-2 random bits
            for _ in range(random.randint(1, 2)):
                i = random.randint(0, N - 1)
                new_colors[i] = 1 - new_colors[i]
        else:
            k = random.randint(1, 5)
            new_colors = current_colors.copy()
            for _ in range(k):
                i = random.randint(0, N - 1)
                new_colors[i] = 1 - new_colors[i]
        
        res_colors, res_cost = hill_climb(new_colors)
        
        if res_cost < best_cost:
            best_cost = res_cost
            best_colors = res_colors
            current_colors = best_colors.copy()
            current_cost = best_cost

    return best_colors.tolist()
