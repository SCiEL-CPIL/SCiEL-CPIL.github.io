import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # Store as 0-indexed tuples
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(aps)

    # For each index, precompute the list of AP indices it belongs to
    # and for each such AP, which other 3 indices are in it
    index_to_aps = [[] for _ in range(N)]
    for ap_idx, (i1, i2, i3, i4) in enumerate(aps):
        for idx in (i1, i2, i3, i4):
            others = [x for x in (i1, i2, i3, i4) if x != idx]
            index_to_aps[idx].append((ap_idx, others))

    def count_mono(color):
        """Count monochromatic 4-APs."""
        count = 0
        for i1, i2, i3, i4 in aps:
            c = color[i1]
            if c == color[i2] == color[i3] == color[i4]:
                count += 1
        return count

    def get_delta(color, idx, new_val):
        """Compute change in mono count if we flip color[idx] to new_val."""
        old_val = color[idx]
        if old_val == new_val:
            return 0
        delta = 0
        for ap_idx, others in index_to_aps[idx]:
            # Check if all 4 are monochromatic before and after
            # Before: color[idx] == old_val, others have their colors
            all_same_before = (old_val == color[others[0]] == color[others[1]] == color[others[2]])
            all_same_after = (new_val == color[others[0]] == color[others[1]] == color[others[2]])
            if all_same_before and not all_same_after:
                delta -= 1
            elif all_same_after and not all_same_before:
                delta += 1
        return delta

    # Generate good initial configurations
    def random_init():
        return np.array([random.randint(0, 1) for _ in range(N)])

    def block_init(block_size):
        return np.array([(i // block_size) % 2 for i in range(N)])

    def stripe_init(period):
        return np.array([(i % period) < period // 2 for i in range(N)]).astype(int)

    # Try several initial configurations and pick the best few
    candidates = []
    
    # Block patterns of various sizes
    for bs in [2, 3, 4, 5, 6, 7, 8]:
        c = block_init(bs)
        candidates.append(c)
    
    # Stripe patterns
    for p in [3, 5, 7, 9, 11]:
        c = stripe_init(p)
        candidates.append(c)
    
    # Random inits
    for _ in range(10):
        candidates.append(random_init())
    
    # Evaluate all candidates
    scored = []
    for c in candidates:
        c = np.array(c, dtype=np.int8)
        sc = count_mono(c)
        scored.append((sc, c))
    
    scored.sort(key=lambda x: x[0])
    
    # Take top 5 as starting points
    top_starts = [c.copy() for _, c in scored[:5]]
    best_score = scored[0][0]
    best_color = scored[0][1].copy()

    # Simulated annealing from each top start
    for start_color in top_starts:
        if time.time() > deadline - 5.0:
            break
        
        color = start_color.copy()
        current_score = count_mono(color)
        
        # SA parameters
        T = 3.0
        T_min = 0.01
        cooling = 0.997
        max_iter_per_restart = 200000
        
        rng = random.Random()
        
        for it in range(max_iter_per_restart):
            if time.time() > deadline - 3.0:
                break
            
            # Choose move: 1-bit flip (70%) or 2-bit flip (30%)
            if rng.random() < 0.7:
                idx = rng.randint(0, N - 1)
                new_val = 1 - color[idx]
                delta = get_delta(color, idx, new_val)
                
                # Accept or reject
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    color[idx] = new_val
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_color = color.copy()
            else:
                # 2-bit flip
                idx1 = rng.randint(0, N - 1)
                idx2 = rng.randint(0, N - 1)
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % N
                v1 = 1 - color[idx1]
                v2 = 1 - color[idx2]
                
                # Compute combined delta
                delta1 = get_delta(color, idx1, v1)
                # Temporarily set idx1 to compute delta for idx2
                color[idx1] = v1
                delta2 = get_delta(color, idx2, v2)
                total_delta = delta1 + delta2
                color[idx1] = 1 - v1  # restore
                
                if total_delta < 0 or rng.random() < np.exp(-total_delta / T):
                    color[idx1] = v1
                    color[idx2] = v2
                    current_score += total_delta
                    if current_score < best_score:
                        best_score = current_score
                        best_color = color.copy()
            
            # Cool
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Restart with slightly perturbed best if time allows
        if time.time() > deadline - 5.0:
            break

    # Final: try local search (hill climbing) from best_color
    if time.time() < deadline - 2.0:
        color = best_color.copy()
        current_score = count_mono(color)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for idx in range(N):
                new_val = 1 - color[idx]
                delta = get_delta(color, idx, new_val)
                if delta < 0:
                    color[idx] = new_val
                    current_score += delta
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_color = color.copy()

    # Return as list of ints
    return [int(x) for x in best_color]
# EVOLVE-BLOCK-END
