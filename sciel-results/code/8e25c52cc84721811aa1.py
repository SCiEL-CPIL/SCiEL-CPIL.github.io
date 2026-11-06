import random
import math

def construct():
    N = 256
    
    # Precompute all 4-term APs
    aps = []
    # index -> list of AP indices
    index_to_aps = [[] for _ in range(N)]
    
    ap_id = 0
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            # a, a+d, a+2d, a+3d
            # Indices are 0-based in our array, but problem is 1..256
            # So index 0 corresponds to integer 1.
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            
            aps.append((i0, i1, i2, i3))
            index_to_aps[i0].append(ap_id)
            index_to_aps[i1].append(ap_id)
            index_to_aps[i2].append(ap_id)
            index_to_aps[i3].append(ap_id)
            ap_id += 1
            
    total_aps = len(aps)
    
    def count_mono(colors):
        count = 0
        for ap in aps:
            c0 = colors[ap[0]]
            if c0 == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count

    def delta_flip(colors, idxs):
        """
        Calculate change in monochromatic AP count if bits at idxs are flipped.
        idxs is a list of unique indices to flip.
        """
        affected_aps = set()
        for idx in idxs:
            affected_aps.update(index_to_aps[idx])
        
        if not affected_aps:
            return 0
            
        delta = 0
        # Set of indices being flipped for O(1) lookup
        flip_set = set(idxs)
        
        for ap_id in affected_aps:
            ap = aps[ap_id]
            
            # Get colors before
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            is_mono_before = (c0 == c1 == c2 == c3)
            
            # Get colors after
            # If index is in flip_set, flip the color
            nc0 = 1 - c0 if ap[0] in flip_set else c0
            nc1 = 1 - c1 if ap[1] in flip_set else c1
            nc2 = 1 - c2 if ap[2] in flip_set else c2
            nc3 = 1 - c3 if ap[3] in flip_set else c3
            
            is_mono_after = (nc0 == nc1 == nc2 == nc3)
            
            if is_mono_before and not is_mono_after:
                delta -= 1
            elif not is_mono_before and is_mono_after:
                delta += 1
                
        return delta

    best_colors = None
    best_count = float('inf')
    
    # Simulated Annealing Parameters
    num_restarts = 10
    steps_per_restart = 5000
    T0 = 1.0
    T_end = 0.01
    cooling_rate = (T_end / T0) ** (1.0 / steps_per_restart)
    
    base_seed = 42
    
    for restart in range(num_restarts):
        # Deterministic random seed for this restart
        random.seed(base_seed + restart)
        
        # Initialization
        if restart % 2 == 0:
            # Random
            colors = [random.randint(0, 1) for _ in range(N)]
        else:
            # Structured: alternating blocks of 4
            colors = [i // 4 % 2 for i in range(N)]
            
        current_count = count_mono(colors)
        current_temp = T0
        
        for step in range(steps_per_restart):
            # Generate move: flip 2 random distinct indices
            i1 = random.randint(0, N - 1)
            i2 = random.randint(0, N - 1)
            while i2 == i1:
                i2 = random.randint(0, N - 1)
            
            delta = delta_flip(colors, [i1, i2])
            
            if delta < 0:
                # Accept
                colors[i1] = 1 - colors[i1]
                colors[i2] = 1 - colors[i2]
                current_count += delta
            else:
                # Metropolis criterion
                if current_temp > 0:
                    prob = math.exp(-delta / current_temp)
                    if random.random() < prob:
                        colors[i1] = 1 - colors[i1]
                        colors[i2] = 1 - colors[i2]
                        current_count += delta
            
            current_temp *= cooling_rate
            
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
            
    return best_colors
