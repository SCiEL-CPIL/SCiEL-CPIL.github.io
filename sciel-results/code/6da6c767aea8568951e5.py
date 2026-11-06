"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n = 256
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= 256
    # We'll use 0-indexed: positions 0..255
    # AP: (i, i+d, i+2d, i+3d) where i+3d < 256
    ap_list = []
    for d in range(1, (n + 2) // 3 + 1):  # d such that 3d < n
        for i in range(n - 3 * d):
            ap_list.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_aps = len(ap_list)
    
    # Precompute for each index, which APs it participates in
    # index_to_aps[j] = list of AP indices where position j is involved
    index_to_aps = [[] for _ in range(n)]
    for ap_idx, (a, b, c, d) in enumerate(ap_list):
        index_to_aps[a].append(ap_idx)
        index_to_aps[b].append(ap_idx)
        index_to_aps[c].append(ap_idx)
        index_to_aps[d].append(ap_idx)
    
    # AP position arrays for fast delta computation
    ap_a = np.array([x[0] for x in ap_list], dtype=np.int32)
    ap_b = np.array([x[1] for x in ap_list], dtype=np.int32)
    ap_c = np.array([x[2] for x in ap_list], dtype=np.int32)
    ap_d = np.array([x[3] for x in ap_list], dtype=np.int32)
    
    def count_mono(coloring):
        """Count monochromatic 4-term APs. coloring is a numpy array of 0/1."""
        # For each AP, check if all four positions have the same color
        mono = (coloring[ap_a] == coloring[ap_b]) & \
               (coloring[ap_b] == coloring[ap_c]) & \
               (coloring[ap_c] == coloring[ap_d])
        return int(np.sum(mono))
    
    def flip_delta(coloring, idx):
        """Compute the change in mono-4AP count if we flip position idx."""
        old_color = coloring[idx]
        new_color = 1 - old_color
        delta = 0
        for ap_idx in index_to_aps[idx]:
            a, b, c, d = ap_list[ap_idx]
            # Get colors of the four positions
            ca = coloring[a]
            cb = coloring[b]
            cc = coloring[c]
            cd = coloring[d]
            
            # Check if currently monochromatic
            was_mono = (ca == cb) and (cb == cc) and (cc == cd)
            
            # After flip, the color at idx becomes new_color
            # Determine the color at each position after flip
            na = new_color if a == idx else ca
            nb = new_color if b == idx else cb
            nc = new_color if c == idx else cc
            nd = new_color if d == idx else cd
            
            is_mono_now = (na == nb) and (nb == nc) and (nc == nd)
            
            if was_mono and not is_mono_now:
                delta -= 1
            elif not was_mono and is_mono_now:
                delta += 1
        
        return delta
    
    # Generate initial colorings
    def thue_morse():
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            colors[i] = bin(i).count('1') % 2
        return colors
    
    def block_pattern(block_size):
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            colors[i] = (i // block_size) % 2
        return colors
    
    def mod_pattern(m):
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            colors[i] = (i % m) % 2
        return colors
    
    # Multi-phase search: SA + greedy polish + final intensive polish
    best_coloring = None
    best_count = float('inf')
    
    # Phase 1: Try several structured initializations
    initializations = [
        thue_morse(),
        block_pattern(2),
        block_pattern(3),
        block_pattern(4),
        block_pattern(5),
        mod_pattern(3),
        mod_pattern(4),
        mod_pattern(5),
        np.random.randint(0, 2, n).astype(np.int8),
    ]
    
    for init in initializations:
        cnt = count_mono(init)
        if cnt < best_count:
            best_count = cnt
            best_coloring = init.copy()
    
    # Phase 2: Simulated Annealing from best initialization
    coloring = best_coloring.copy()
    current_count = best_count
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    while time.time() < deadline - 30:  # Leave 30s for polish phases
        if T < min_T:
            break
        
        idx = np.random.randint(0, n)
        d = flip_delta(coloring, idx)
        
        if d <= 0 or (np.random.random() < np.exp(-d / T)):
            coloring[idx] = 1 - coloring[idx]
            current_count += d
            if current_count < best_count:
                best_count = current_count
                best_coloring = coloring.copy()
        
        T *= cooling_rate
    
    # Phase 3: Greedy polish - repeatedly flip the best single flip until no improvement
    coloring = best_coloring.copy()
    current_count = best_count
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for idx in range(n):
            d = flip_delta(coloring, idx)
            if d < 0:
                coloring[idx] = 1 - coloring[idx]
                current_count += d
                improved = True
                if current_count < best_count:
                    best_count = current_count
                    best_coloring = coloring.copy()
    
    # Phase 4: Final intensive polish with multiple restarts of greedy
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        coloring = best_coloring.copy()
        current_count = best_count
        # Try random perturbations then greedy
        for _ in range(5):
            idx = np.random.randint(0, n)
            coloring[idx] = 1 - coloring[idx]
            current_count += flip_delta(coloring, idx)
            # Greedy polish
            for idx2 in range(n):
                d = flip_delta(coloring, idx2)
                if d < 0:
                    coloring[idx2] = 1 - coloring[idx2]
                    current_count += d
            if current_count < best_count:
                best_count = current_count
                best_coloring = coloring.copy()
    
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
