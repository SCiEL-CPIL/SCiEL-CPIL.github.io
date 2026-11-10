import random
import time
import numpy as np

N = 256

def _build_aps_np():
    """Build NumPy array of all 4-term arithmetic progressions in 0..N-1."""
    aps_list = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps_list.append([a, a + d, a + 2 * d, a + 3 * d])
    return np.array(aps_list, dtype=np.int32)

def _build_neighbors(aps_np):
    """Build neighbor lists and 'others' array for fast local search."""
    num_aps = aps_np.shape[0]
    neighbors = [[] for _ in range(N)]
    others = np.zeros((num_aps, 4, 3), dtype=np.int32)
    
    for idx in range(num_aps):
        a, b, c, d = aps_np[idx]
        # For position 0 (a), others are b, c, d
        neighbors[a].append((idx, 0))
        others[idx, 0] = [b, c, d]
        # For position 1 (b), others are a, c, d
        neighbors[b].append((idx, 1))
        others[idx, 1] = [a, c, d]
        # For position 2 (c), others are a, b, d
        neighbors[c].append((idx, 2))
        others[idx, 2] = [a, b, d]
        # For position 3 (d), others are a, b, c
        neighbors[d].append((idx, 3))
        others[idx, 3] = [a, b, c]
        
    return neighbors, others

def _evaluate_np(colors_np, aps_np):
    """Vectorized evaluation of monochromatic APs."""
    # colors_np is shape (N,)
    # aps_np is shape (num_aps, 4)
    # colors_np[aps_np] is shape (num_aps, 4)
    vals = colors_np[aps_np]
    # Check if all 4 are equal
    # Compare each column to the first column
    all_same = np.all(vals == vals[:, :1], axis=1)
    return np.sum(all_same)

def _local_search_fast(colors, aps_np, neighbors, others, max_passes=50):
    """Fast local search using pre-computed neighbors and NumPy for checks."""
    colors = list(colors)
    colors_np = np.array(colors, dtype=np.int8)
    
    # Initial state
    ap_mono = np.zeros(aps_np.shape[0], dtype=bool)
    vals = colors_np[aps_np]
    ap_mono = np.all(vals == vals[:, :1], axis=1)
    current_mono = np.sum(ap_mono)
    
    best_mono = current_mono
    best_colors = list(colors)
    
    for _ in range(max_passes):
        improved = False
        for i in range(N):
            new_color = 1 - colors[i]
            delta = 0
            
            # Check all APs containing i
            for ap_idx, pos in neighbors[i]:
                if ap_mono[ap_idx]:
                    # If it was mono, flipping one bit makes it non-mono
                    delta -= 1
                else:
                    # If it was not mono, check if flipping i makes it mono
                    # This requires the other 3 bits to be equal to new_color
                    other_indices = others[ap_idx, pos]
                    # colors_np is updated as we go? No, we use colors list for logic, 
                    # but we need to check current state.
                    # Since we are in a pass, colors[i] is still old.
                    # We need to check if colors[other_indices] are all new_color.
                    if np.all(colors_np[other_indices] == new_color):
                        delta += 1
                        
            if delta < 0:
                # Accept flip
                colors[i] = new_color
                colors_np[i] = new_color
                current_mono += delta
                
                # Update ap_mono for affected APs
                for ap_idx, pos in neighbors[i]:
                    # Re-evaluate this specific AP
                    # vals = colors_np[aps_np[ap_idx]]
                    # ap_mono[ap_idx] = np.all(vals == vals[0])
                    # Faster:
                    ap_vals = colors_np[aps_np[ap_idx]]
                    ap_mono[ap_idx] = np.all(ap_vals == ap_vals[0])
                improved = True
        
        if current_mono < best_mono:
            best_mono = current_mono
            best_colors = list(colors)
            
        if not improved:
            break
            
    return best_colors, best_mono

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    aps_np = _build_aps_np()
    neighbors, others = _build_neighbors(aps_np)
    
    rng = np.random.default_rng(42)
    
    # Population size for GA
    pop_size = 100
    best_mono = float('inf')
    best_colors = None
    
    # Initialize population
    population = rng.integers(0, 2, size=(pop_size, N)).astype(np.int8)
    
    # Evaluate initial population
    scores = np.array([_evaluate_np(ind, aps_np) for ind in population])
    best_idx = np.argmin(scores)
    best_mono = scores[best_idx]
    best_colors = population[best_idx].tolist()
    
    gen = 0
    while time.time() < deadline:
        # Sort population by score
        sorted_indices = np.argsort(scores)
        # Keep top half
        survivors = [population[i] for i in sorted_indices[:pop_size//2]]
        
        # Generate offspring
        new_population = survivors.copy()
        
        while len(new_population) < pop_size:
            # Select two parents from top half
            p1_idx = rng.integers(0, pop_size//2)
            p2_idx = rng.integers(0, pop_size//2)
            p1 = population[sorted_indices[p1_idx]]
            p2 = population[sorted_indices[p2_idx]]
            
            # Uniform Crossover
            mask = rng.integers(0, 2, size=N).astype(bool)
            child = np.where(mask, p1, p2)
            
            # Mutation
            # Low mutation rate
            mut_mask = rng.random(N) < 0.01
            child[mut_mask] = 1 - child[mut_mask]
            
            new_population.append(child)
        
        population = np.array(new_population, dtype=np.int8)
        
        # Evaluate new population
        scores = np.array([_evaluate_np(ind, aps_np) for ind in population])
        
        # Update global best
        gen_best_idx = np.argmin(scores)
        if scores[gen_best_idx] < best_mono:
            best_mono = scores[gen_best_idx]
            best_colors = population[gen_best_idx].tolist()
            
        # Occasional local search on the global best to polish
        if gen % 5 == 0 and time.time() < deadline - 10:
            polished, polished_mono = _local_search_fast(
                best_colors, aps_np, neighbors, others, max_passes=10
            )
            if polished_mono < best_mono:
                best_mono = polished_mono
                best_colors = polished
                # Inject polished individual into population
                population[0] = np.array(polished, dtype=np.int8)
                scores[0] = polished_mono
        
        gen += 1
    
    return best_colors
# EVOLVE-BLOCK-END
