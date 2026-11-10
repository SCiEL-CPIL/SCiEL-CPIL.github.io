import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Precompute all 4-term arithmetic progressions
    # Store as indices 0..N-1
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    # Convert to numpy array for vectorized counting
    aps_arr = np.array(aps, dtype=np.int32)
    
    # 2. Precompute neighbors for fast delta calculation
    # neighbors[i] contains indices of other elements in APs with i
    # We need to know, for a flip at i, which APs are affected.
    # An AP is affected if it contains i.
    # For each i, store the other 3 indices of every AP containing i.
    neighbors = [[] for _ in range(N)]
    for i1, i2, i3, i4 in aps:
        neighbors[i1].append((i2, i3, i4))
        neighbors[i2].append((i1, i3, i4))
        neighbors[i3].append((i1, i2, i4))
        neighbors[i4].append((i1, i2, i3))
        
    # Convert to numpy arrays
    neighbors_np = []
    for i in range(N):
        if neighbors[i]:
            neighbors_np.append(np.array(neighbors[i], dtype=np.int32))
        else:
            neighbors_np.append(np.empty((0, 3), dtype=np.int32))

    def count_mono(colors):
        """Count monochromatic 4-APs using vectorized operations."""
        # colors is a 1D numpy array of 0s and 1s
        c1 = colors[aps_arr[:, 0]]
        c2 = colors[aps_arr[:, 1]]
        c3 = colors[aps_arr[:, 2]]
        c4 = colors[aps_arr[:, 3]]
        return np.sum((c1 == c2) & (c2 == c3) & (c3 == c4))

    best_score = float('inf')
    best_coloring = None

    # 3. Structured Initialization Strategy
    # Base pattern: blocks of two (00110011...)
    # This pattern is known to break many small-step APs.
    base_pattern = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8)
    
    # Additional base patterns for diversity
    patterns = [
        base_pattern,
        np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i % 4) < 2 for i in range(N)], dtype=np.int8),
        np.array([1 - ((i // 2) % 2) for i in range(N)], dtype=np.int8),
    ]

    # 4. Final Polish: Exact local search until no improving flip exists
    def polish(colors, score, time_limit=5.0):
        """Perform exact local search: only accept improving moves."""
        polish_start = time.time()
        improved = True
        while improved and time.time() < polish_start + time_limit:
            improved = False
            for i in range(N):
                if time.time() > polish_start + time_limit:
                    break
                old_c = colors[i]
                new_c = 1 - old_c
                
                others = neighbors_np[i]
                if len(others) == 0:
                    continue
                cols = colors[others]
                c0, c1, c2 = cols[:, 0], cols[:, 1], cols[:, 2]
                all_eq = (c0 == c1) & (c1 == c2)
                val = c0
                was_mono = all_eq & (val == old_c)
                becomes_mono = all_eq & (val == new_c)
                delta = np.sum(becomes_mono) - np.sum(was_mono)
                
                if delta < 0:
                    colors[i] = new_c
                    score += delta
                    improved = True
        return score

    # 5. Main search loop with multiple diverse initializations
    # Use a smaller time budget per restart to allow more restarts
    num_restarts = 0
    while time.time() < deadline - 10.0:  # Leave 10s for final polish
        # Choose a base pattern cyclically with random perturbation
        base_idx = num_restarts % len(patterns)
        colors = patterns[base_idx].copy()
        
        # Apply random perturbations
        num_perturb = random.randint(3, 12)
        perturb_indices = random.sample(range(N), num_perturb)
        for idx in perturb_indices:
            colors[idx] = 1 - colors[idx]
            
        current_score = count_mono(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = colors.copy()

        # Simulated Annealing
        T = 1.0
        alpha = 0.9995
        min_T = 0.02
        
        # Track stagnation for kicking
        stagnation_count = 0
        last_improve_score = current_score
        
        # Time limit for this SA run: use remaining time divided by expected restarts
        # But at least 5 seconds per run
        sa_deadline = min(deadline - 10.0, time.time() + max(5.0, (deadline - 10.0 - time.time()) / 3.0))
        
        while T > min_T and time.time() < sa_deadline:
            # Decide move type
            if stagnation_count > 50:
                use_kick = random.random() < 0.2
            else:
                use_kick = random.random() < 0.02

            if use_kick:
                # Structural kick: Flip a random set of bits
                kick_size = random.randint(2, 6)
                kick_indices = random.sample(range(N), kick_size)
                
                old_colors = colors[kick_indices].copy()
                new_colors = 1 - old_colors
                colors[kick_indices] = new_colors
                
                new_score = count_mono(colors)
                delta = new_score - current_score
                
                # Accept with probability exp(-delta/T)
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if current_score < last_improve_score:
                        last_improve_score = current_score
                        stagnation_count = 0
                        if current_score < best_score:
                            best_score = current_score
                            best_coloring = colors.copy()
                else:
                    # Revert
                    colors[kick_indices] = old_colors
                    stagnation_count += 1
            else:
                # Single Flip
                i = random.randint(0, N - 1)
                old_c = colors[i]
                new_c = 1 - old_c
                
                others = neighbors_np[i]
                if len(others) == 0:
                    delta = 0
                else:
                    cols = colors[others]
                    c0, c1, c2 = cols[:, 0], cols[:, 1], cols[:, 2]
                    all_eq = (c0 == c1) & (c1 == c2)
                    val = c0
                    was_mono = all_eq & (val == old_c)
                    becomes_mono = all_eq & (val == new_c)
                    delta = np.sum(becomes_mono) - np.sum(was_mono)
                    
                if delta < 0 or random.random() < np.exp(-delta / T):
                    colors[i] = new_c
                    current_score += delta
                    if current_score < last_improve_score:
                        last_improve_score = current_score
                        stagnation_count = 0
                        if current_score < best_score:
                            best_score = current_score
                            best_coloring = colors.copy()
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1

            T *= alpha

        num_restarts += 1

    # 6. Final high-precision polish pass on the best coloring found
    if best_coloring is not None:
        final_colors = best_coloring.copy()
        final_score = polish(final_colors, best_score, time_limit=8.0)
        if final_score < best_score:
            best_score = final_score
            best_coloring = final_colors

    return best_coloring.tolist()
# EVOLVE-BLOCK-END
