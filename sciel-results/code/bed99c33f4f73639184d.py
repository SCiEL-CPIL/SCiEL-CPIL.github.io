import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    # Using 0-indexed: positions 0..N-1
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # NumPy array for vectorized full evaluation
    ap_indices = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # For each position, list of AP indices containing it
    pos_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    # Precompute for each position: the AP indices and the other 3 positions
    # This allows fast delta computation
    pos_ap_data = []
    for pos in range(N):
        ap_idxs = pos_aps[pos]
        if not ap_idxs:
            pos_ap_data.append((np.array([], dtype=np.int32), np.array([], dtype=np.int32)))
            continue
        # For each AP containing pos, store the other 3 positions
        other_positions = []
        for ap_idx in ap_idxs:
            ap = aps[ap_idx]
            others = [ap[j] for j in range(4) if ap[j] != pos]
            other_positions.append(others)
        pos_ap_data.append((np.array(ap_idxs, dtype=np.int32), np.array(other_positions, dtype=np.int32)))
    
    def count_mono_np(colors):
        """Count monochromatic 4-term APs using NumPy vectorization."""
        c0 = colors[ap_indices[:, 0]]
        c1 = colors[ap_indices[:, 1]]
        c2 = colors[ap_indices[:, 2]]
        c3 = colors[ap_indices[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())
    
    def delta_for_flip_fast(colors, pos):
        """Compute change in mono count if we flip position pos, using precomputed data."""
        c = colors[pos]
        nc = 1 - c
        ap_idxs, other_pos = pos_ap_data[pos]
        if len(ap_idxs) == 0:
            return 0
        # Get colors of the other 3 positions in each AP
        # other_pos shape: (num_aps_containing_pos, 3)
        oc0 = colors[other_pos[:, 0]]
        oc1 = colors[other_pos[:, 1]]
        oc2 = colors[other_pos[:, 2]]
        # Check if all three others are equal
        all_equal = (oc0 == oc1) & (oc1 == oc2)
        # If they're equal to nc, flipping creates a mono AP (gain)
        # If they're equal to c, flipping breaks a mono AP (loss)
        gain = int(np.sum(all_equal & (oc0 == nc)))
        loss = int(np.sum(all_equal & (oc0 == c)))
        return gain - loss
    
    # Thue-Morse coloring
    def thue_morse(n):
        x = n
        c = 0
        while x:
            c ^= 1
            x >>= 1
        return c
    
    # Try multiple initial colorings and local search
    best_colors = None
    best_count = float('inf')
    
    initializations = []
    
    # Thue-Morse
    tm = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    initializations.append(tm)
    
    # Random initializations
    for _ in range(12):
        initializations.append(np.random.randint(0, 2, N, dtype=np.int8))
    
    # Alternating blocks of various sizes
    for block in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
        initializations.append(np.array([(i // block) % 2 for i in range(N)], dtype=np.int8))
    
    # Periodic patterns
    for period in [8, 12, 16, 24, 32, 48, 64]:
        for offset in [0, period // 2]:
            pattern = np.array([((i + offset) % period) < period // 2 for i in range(N)], dtype=np.int8)
            initializations.append(pattern)
    
    # Van der Corput / low-discrepancy style
    for multiplier in [3, 5, 7, 11, 13]:
        for period in [16, 32, 64]:
            pattern = np.array([((i * multiplier) % period) < period // 2 for i in range(N)], dtype=np.int8)
            initializations.append(pattern)
    
    # Binary representation based patterns
    for bit_pos in [0, 1, 2, 3, 4, 5]:
        initializations.append(np.array([(i >> bit_pos) & 1 for i in range(N)], dtype=np.int8))
    
    # XOR of multiple bits
    for b1 in range(6):
        for b2 in range(b1 + 1, 6):
            initializations.append(np.array([((i >> b1) ^ (i >> b2)) & 1 for i in range(N)], dtype=np.int8))
    
    for init_colors in initializations:
        if time.time() > deadline - 10:
            break
        
        colors = init_colors.copy()
        current_count = count_mono_np(colors)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
        
        # Local search with simulated annealing using fast delta
        T = 5.0
        cooling = 0.9998
        min_T = 0.001
        no_improve = 0
        max_no_improve = 5000
        steps = 0
        
        while T > min_T and time.time() < deadline - 5:
            pos = random.randint(0, N - 1)
            d = delta_for_flip_fast(colors, pos)
            
            if d < 0 or (d == 0 and random.random() < 0.1) or (d > 0 and random.random() < np.exp(-d / T)):
                colors[pos] = 1 - colors[pos]
                current_count += d
                steps += 1
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                steps += 1
            
            if no_improve > max_no_improve:
                # Restart with perturbation: flip a random subset
                n_perturb = random.randint(3, 15)
                perturb_positions = random.sample(range(N), n_perturb)
                for p in perturb_positions:
                    colors[p] = 1 - colors[p]
                current_count = count_mono_np(colors)
                T = max(T, 3.0)
                no_improve = 0
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
            
            T *= cooling
        
        # Final greedy pass: multiple sweeps
        for sweep in range(5):
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                order = list(range(N))
                random.shuffle(order)
                for pos in order:
                    d = delta_for_flip_fast(colors, pos)
                    if d < 0:
                        colors[pos] = 1 - colors[pos]
                        current_count += d
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_colors = colors.copy()
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
    
    # Final high-precision polish: try all single flips on best solution
    if best_colors is not None and time.time() < deadline - 1:
        colors = best_colors.copy()
        current_count = best_count
        for _ in range(3):
            improved = True
            while improved and time.time() < deadline - 0.5:
                improved = False
                order = list(range(N))
                random.shuffle(order)
                for pos in order:
                    d = delta_for_flip_fast(colors, pos)
                    if d < 0:
                        colors[pos] = 1 - colors[pos]
                        current_count += d
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_colors = colors.copy()
    
    # Convert to list of ints 0/1
    result = [int(c) for c in best_colors]
    return result
