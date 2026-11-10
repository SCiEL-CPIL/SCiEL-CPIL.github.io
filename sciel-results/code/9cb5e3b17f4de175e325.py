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
    """Build neighbor lists for fast local search."""
    num_aps = aps_np.shape[0]
    neighbors = [[] for _ in range(N)]
    
    for idx in range(num_aps):
        a, b, c, d = aps_np[idx]
        neighbors[a].append(idx)
        neighbors[b].append(idx)
        neighbors[c].append(idx)
        neighbors[d].append(idx)
        
    return neighbors

def _evaluate_np(colors_np, aps_np):
    """Vectorized evaluation of monochromatic APs."""
    vals = colors_np[aps_np]
    all_same = np.all(vals == vals[:, :1], axis=1)
    return np.sum(all_same)

def _compute_mono_count(colors_np, aps_np):
    """Compute monochromatic count and the boolean mask."""
    vals = colors_np[aps_np]
    ap_mono = np.all(vals == vals[:, :1], axis=1)
    return np.sum(ap_mono), ap_mono

def _delta_for_flip(colors_np, ap_mono, aps_np, neighbors, i):
    """Compute delta in monochromatic count if we flip vertex i.
    Returns (delta, list_of_aps_to_update, new_mono_states)
    """
    new_color = 1 - colors_np[i]
    old_color = colors_np[i]
    
    delta = 0
    affected_aps = neighbors[i]
    new_states = []
    
    for ap_idx in affected_aps:
        if ap_mono[ap_idx]:
            # Was mono, flipping breaks it
            delta -= 1
            new_states.append(False)
        else:
            # Was not mono. Check if flipping makes it mono.
            # All other 3 elements must be new_color.
            ap = aps_np[ap_idx]
            others = [ap[j] for j in range(4) if ap[j] != i]
            if (colors_np[others[0]] == new_color and 
                colors_np[others[1]] == new_color and 
                colors_np[others[2]] == new_color):
                delta += 1
                new_states.append(True)
            else:
                new_states.append(False)
    
    return delta, affected_aps, new_states

def _simulated_annealing(colors, aps_np, neighbors, time_limit, rng=None):
    """Simulated Annealing to minimize monochromatic APs."""
    if rng is None:
        rng = np.random.default_rng()
    
    colors = list(colors)
    colors_np = np.array(colors, dtype=np.int8)
    
    current_mono, ap_mono = _compute_mono_count(colors_np, aps_np)
    
    best_mono = current_mono
    best_colors = list(colors)
    
    T = 1.0
    cooling = 0.9995
    min_T = 1e-7
    
    start_time = time.time()
    
    while time.time() < start_time + time_limit and T > min_T:
        # Pick random index
        i = rng.integers(0, N)
        
        delta, affected_aps, new_states = _delta_for_flip(colors_np, ap_mono, aps_np, neighbors, i)
        
        # Metropolis acceptance
        if delta <= 0:
            accept = True
        else:
            prob = np.exp(-delta / T)
            accept = rng.random() < prob
        
        if accept:
            new_color = 1 - colors_np[i]
            colors[i] = new_color
            colors_np[i] = new_color
            current_mono += delta
            
            for ap_idx, new_state in zip(affected_aps, new_states):
                ap_mono[ap_idx] = new_state
            
            if current_mono < best_mono:
                best_mono = current_mono
                best_colors = list(colors)
        
        T *= cooling
        
    return best_colors, best_mono

def _tabu_search(colors, aps_np, neighbors, time_limit, tabu_len=7, rng=None):
    """Tabu Search to minimize monochromatic APs."""
    if rng is None:
        rng = np.random.default_rng()
    
    colors = list(colors)
    colors_np = np.array(colors, dtype=np.int8)
    
    current_mono, ap_mono = _compute_mono_count(colors_np, aps_np)
    
    best_mono = current_mono
    best_colors = list(colors)
    
    tabu = [0] * N  # iteration at which vertex becomes non-tabu
    current_iter = 0
    
    start_time = time.time()
    
    while time.time() < start_time + time_limit:
        best_delta = float('inf')
        best_i = -1
        best_new_states = None
        best_affected = None
        
        # Evaluate all possible moves
        for i in range(N):
            if current_iter < tabu[i]:
                continue  # skip tabu moves
            
            delta, affected_aps, new_states = _delta_for_flip(colors_np, ap_mono, aps_np, neighbors, i)
            
            # Aspiration criterion: accept tabu move if it improves best
            if current_iter >= tabu[i] or delta < best_mono - current_mono:
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_states = new_states
                    best_affected = affected_aps
        
        if best_i == -1:
            # No valid moves, pick random
            best_i = rng.integers(0, N)
            best_delta, best_affected, best_new_states = _delta_for_flip(colors_np, ap_mono, aps_np, neighbors, best_i)
        
        # Apply best move
        new_color = 1 - colors_np[best_i]
        colors[best_i] = new_color
        colors_np[best_i] = new_color
        current_mono += best_delta
        tabu[best_i] = current_iter + tabu_len
        
        for ap_idx, new_state in zip(best_affected, best_new_states):
            ap_mono[ap_idx] = new_state
        
        if current_mono < best_mono:
            best_mono = current_mono
            best_colors = list(colors)
        
        current_iter += 1
        
    return best_colors, best_mono

def _generate_lfsr_sequence(state, taps):
    """Generate a 256-bit sequence from an LFSR."""
    seq = []
    current_state = state[:]
    for _ in range(N):
        seq.append(current_state[0])
        fb = 0
        for t in range(8):
            if taps & (1 << t):
                fb ^= current_state[t]
        current_state = current_state[1:] + [fb]
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    aps_np = _build_aps_np()
    neighbors = _build_neighbors(aps_np)
    
    rng = np.random.default_rng(42)
    
    best_mono = float('inf')
    best_colors = None
    
    # Phase 1: Explore LFSR parameter space
    num_lfsr_samples = 5000
    lfsr_results = []
    
    for _ in range(num_lfsr_samples):
        if time.time() > deadline - 40:
            break
            
        state = rng.integers(0, 2, size=8).tolist()
        taps = rng.integers(1, 256)
        
        seq = _generate_lfsr_sequence(state, taps)
        seq_np = np.array(seq, dtype=np.int8)
        
        mono = _evaluate_np(seq_np, aps_np)
        
        if mono < best_mono:
            best_mono = mono
            best_colors = seq
        
        lfsr_results.append((mono, seq))
        
        seq_complement = [1 - x for x in seq]
        seq_comp_np = np.array(seq_complement, dtype=np.int8)
        mono_comp = _evaluate_np(seq_comp_np, aps_np)
        
        if mono_comp < best_mono:
            best_mono = mono_comp
            best_colors = seq_complement
            
        lfsr_results.append((mono_comp, seq_complement))
    
    # Sort by score
    lfsr_results.sort(key=lambda x: x[0])
    
    # Phase 2: Simulated Annealing + Tabu Search on top LFSR results
    top_k = min(15, len(lfsr_results))
    
    for i in range(top_k):
        if time.time() > deadline - 15:
            break
            
        mono, colors = lfsr_results[i]
        
        # Run SA
        polished, polished_mono = _simulated_annealing(
            colors, aps_np, neighbors, time_limit=8, rng=rng
        )
        
        if polished_mono < best_mono:
            best_mono = polished_mono
            best_colors = polished
        
        # Run Tabu Search from SA result
        if time.time() < deadline - 5:
            tabu_polished, tabu_mono = _tabu_search(
                polished, aps_np, neighbors, time_limit=5, tabu_len=7, rng=rng
            )
            
            if tabu_mono < best_mono:
                best_mono = tabu_mono
                best_colors = tabu_polished
    
    # Phase 3: Random starts with SA + Tabu
    if time.time() < deadline - 25:
        num_random_starts = 15
        for _ in range(num_random_starts):
            if time.time() > deadline - 5:
                break
                
            random_colors = rng.integers(0, 2, size=N).tolist()
            
            # Run SA
            polished, polished_mono = _simulated_annealing(
                random_colors, aps_np, neighbors, time_limit=5, rng=rng
            )
            
            if polished_mono < best_mono:
                best_mono = polished_mono
                best_colors = polished
            
            # Run Tabu Search
            if time.time() < deadline - 3:
                tabu_polished, tabu_mono = _tabu_search(
                    polished, aps_np, neighbors, time_limit=4, tabu_len=5, rng=rng
                )
                
                if tabu_mono < best_mono:
                    best_mono = tabu_mono
                    best_colors = tabu_polished
    
    # Final polish with Tabu on best solution
    if time.time() < deadline - 2:
        final_polished, final_mono = _tabu_search(
            best_colors, aps_np, neighbors, time_limit=3, tabu_len=5, rng=rng
        )
        
        if final_mono < best_mono:
            best_mono = final_mono
            best_colors = final_polished
    
    return best_colors
