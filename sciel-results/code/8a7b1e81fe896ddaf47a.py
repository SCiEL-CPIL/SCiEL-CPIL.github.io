"""Optimized construction for minimizing monochromatic 4-term APs in [1,256]."""
import time
import random
import numpy as np

N = 256
DEADLINE = 135.0  # seconds, safety margin before 150s kill

def _build_ap_data():
    """Build all 4-term APs and per-position membership lists."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, store list of (ap_index, position_in_ap)
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_aps[a].append((idx, 0))
        pos_aps[b].append((idx, 1))
        pos_aps[c].append((idx, 2))
        pos_aps[d].append((idx, 3))
    
    return aps, pos_aps

def _count_mono(aps, colors):
    """Count monochromatic 4-APs."""
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count

def _local_search(aps, pos_aps, colors, max_passes=200):
    """Greedy local search with incremental delta scoring."""
    colors = list(colors)
    n = len(colors)
    num_aps = len(aps)
    
    # Track mono status for each AP
    mono = np.zeros(num_aps, dtype=bool)
    for idx, (a, b, c, d) in enumerate(aps):
        mono[idx] = (colors[a] == colors[b] == colors[c] == colors[d])
    
    total_mono = mono.sum()
    
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            # Compute delta if we flip position i
            delta = 0
            for ap_idx, pos_in_ap in pos_aps[i]:
                a, b, c, d = aps[ap_idx]
                # Check if flipping i changes mono status
                if mono[ap_idx]:
                    # Currently mono: all 4 same color
                    # After flip, i has different color -> not mono
                    delta -= 1
                else:
                    # Currently not mono
                    # After flip, check if all 4 would be same
                    # The other 3 positions must all have the flipped color of i
                    new_color = 1 - colors[i]
                    other_positions = [a, b, c, d]
                    other_positions.pop(pos_in_ap)
                    if all(colors[p] == new_color for p in other_positions):
                        delta += 1
            
            if delta < 0:
                colors[i] = 1 - colors[i]
                # Update mono status for affected APs
                for ap_idx, pos_in_ap in pos_aps[i]:
                    a, b, c, d = aps[ap_idx]
                    mono[ap_idx] = (colors[a] == colors[b] == colors[c] == colors[d])
                total_mono += delta
                improved = True
        
        if not improved:
            break
    
    return colors, total_mono

def _simulated_annealing(aps, pos_aps, colors, start_temp=10.0, end_temp=0.1, 
                          n_iter=50000, seed=42):
    """Simulated annealing for global optimization."""
    rng = random.Random(seed)
    colors = list(colors)
    n = len(colors)
    num_aps = len(aps)
    
    # Track mono status
    mono = np.zeros(num_aps, dtype=bool)
    for idx, (a, b, c, d) in enumerate(aps):
        mono[idx] = (colors[a] == colors[b] == colors[c] == colors[d])
    total_mono = mono.sum()
    
    best_colors = list(colors)
    best_mono = total_mono
    
    for it in range(n_iter):
        temp = start_temp * (end_temp / start_temp) ** (it / n_iter)
        
        # Pick random position to flip
        i = rng.randint(0, n - 1)
        new_color = 1 - colors[i]
        
        # Compute delta
        delta = 0
        for ap_idx, pos_in_ap in pos_aps[i]:
            a, b, c, d = aps[ap_idx]
            if mono[ap_idx]:
                delta -= 1
            else:
                other_positions = [a, b, c, d]
                other_positions.pop(pos_in_ap)
                if all(colors[p] == new_color for p in other_positions):
                    delta += 1
        
        if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            colors[i] = new_color
            for ap_idx, pos_in_ap in pos_aps[i]:
                a, b, c, d = aps[ap_idx]
                mono[ap_idx] = (colors[a] == colors[b] == colors[c] == colors[d])
            total_mono += delta
            
            if total_mono < best_mono:
                best_mono = total_mono
                best_colors = list(colors)
    
    return best_colors, best_mono

def construct():
    start_time = time.time()
    aps, pos_aps = _build_ap_data()
    
    # Initial random colorings with different seeds
    best_colors = None
    best_mono = float('inf')
    
    # Try several random initializations
    for seed in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        rng = random.Random(seed)
        colors = [rng.randint(0, 1) for _ in range(N)]
        
        # Local search
        colors_ls, mono_ls = _local_search(aps, pos_aps, colors, max_passes=50)
        
        if mono_ls < best_mono:
            best_mono = mono_ls
            best_colors = colors_ls
    
    # Simulated annealing from best solution
    if time.time() - start_time < DEADLINE - 20:
        colors_sa, mono_sa = _simulated_annealing(
            aps, pos_aps, best_colors, 
            start_temp=5.0, end_temp=0.05,
            n_iter=30000, seed=123
        )
        if mono_sa < best_mono:
            best_mono = mono_sa
            best_colors = colors_sa
    
    # Final local search polish
    if time.time() - start_time < DEADLINE - 5:
        colors_final, mono_final = _local_search(aps, pos_aps, best_colors, max_passes=100)
        if mono_final < best_mono:
            best_mono = mono_final
            best_colors = colors_final
    
    return best_colors
