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
        neighbors[a].append((idx, 0))
        others[idx, 0] = [b, c, d]
        neighbors[b].append((idx, 1))
        others[idx, 1] = [a, c, d]
        neighbors[c].append((idx, 2))
        others[idx, 2] = [a, b, d]
        neighbors[d].append((idx, 3))
        others[idx, 3] = [a, b, c]
        
    return neighbors, others

def _evaluate_np(colors_np, aps_np):
    """Vectorized evaluation of monochromatic APs."""
    vals = colors_np[aps_np]
    all_same = np.all(vals == vals[:, :1], axis=1)
    return np.sum(all_same)

def _local_search_fast(colors, aps_np, neighbors, others, max_passes=20):
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
                    other_indices = others[ap_idx, pos]
                    if np.all(colors_np[other_indices] == new_color):
                        delta += 1
                        
            if delta < 0:
                # Accept flip
                colors[i] = new_color
                colors_np[i] = new_color
                current_mono += delta
                
                # Update ap_mono for affected APs
                for ap_idx, pos in neighbors[i]:
                    ap_vals = colors_np[aps_np[ap_idx]]
                    ap_mono[ap_idx] = np.all(ap_vals == ap_vals[0])
                improved = True
        
        if current_mono < best_mono:
            best_mono = current_mono
            best_colors = list(colors)
            
        if not improved:
            break
            
    return best_colors, best_mono

def _generate_lfsr_sequence(state, taps):
    """Generate a 256-bit sequence from an LFSR."""
    # state is a list of 8 bits (0 or 1)
    # taps is a bitmask of which bits to XOR
    seq = []
    current_state = state[:]
    for _ in range(N):
        seq.append(current_state[0])
        # Compute feedback bit
        fb = 0
        for t in range(8):
            if taps & (1 << t):
                fb ^= current_state[t]
        # Shift left and insert feedback
        current_state = current_state[1:] + [fb]
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    aps_np = _build_aps_np()
    neighbors, others = _build_neighbors(aps_np)
    
    rng = np.random.default_rng(42)
    
    # Precompute all valid LFSR tap configurations for 8-bit register
    # Primitive polynomials for degree 8: 0x1D, 0x1B, 0x19, 0x13, 0x11, 0x0D, 0x0B, 0x05, 0x03, 0x01
    # But we can also use non-primitive ones, just need to be careful about period
    # For simplicity, let's just use all 256 possible tap masks, but filter out degenerate ones
    # Actually, let's just sample random tap masks and initial states
    
    best_mono = float('inf')
    best_colors = None
    
    # Phase 1: Explore LFSR parameter space
    # We have 2^8 initial states * 2^8 tap masks = 65536 combinations
    # But many will be degenerate (all 0s or all 1s)
    # Let's sample a subset
    
    num_lfsr_samples = 5000
    lfsr_results = []
    
    for _ in range(num_lfsr_samples):
        if time.time() > deadline - 20:
            break
            
        # Random initial state
        state = rng.integers(0, 2, size=8).tolist()
        # Random tap mask (at least one bit set)
        taps = rng.integers(1, 256)
        
        seq = _generate_lfsr_sequence(state, taps)
        # Convert to numpy array
        seq_np = np.array(seq, dtype=np.int8)
        
        # Evaluate
        mono = _evaluate_np(seq_np, aps_np)
        
        if mono < best_mono:
            best_mono = mono
            best_colors = seq
        
        lfsr_results.append((mono, seq))
        
        # Also try the complement
        seq_complement = [1 - x for x in seq]
        seq_comp_np = np.array(seq_complement, dtype=np.int8)
        mono_comp = _evaluate_np(seq_comp_np, aps_np)
        
        if mono_comp < best_mono:
            best_mono = mono_comp
            best_colors = seq_complement
            
        lfsr_results.append((mono_comp, seq_complement))
    
    # Sort by score
    lfsr_results.sort(key=lambda x: x[0])
    
    # Phase 2: Local search on top LFSR results
    top_k = min(20, len(lfsr_results))
    
    for i in range(top_k):
        if time.time() > deadline - 10:
            break
            
        mono, colors = lfsr_results[i]
        
        # Run local search
        polished, polished_mono = _local_search_fast(
            colors, aps_np, neighbors, others, max_passes=15
        )
        
        if polished_mono < best_mono:
            best_mono = polished_mono
            best_colors = polished
    
    # Phase 3: If we have time, do a broader random search with local search
    if time.time() < deadline - 30:
        num_random_starts = 50
        for _ in range(num_random_starts):
            if time.time() > deadline - 10:
                break
                
            # Random initial coloring
            random_colors = rng.integers(0, 2, size=N).tolist()
            
            # Run local search
            polished, polished_mono = _local_search_fast(
                random_colors, aps_np, neighbors, others, max_passes=10
            )
            
            if polished_mono < best_mono:
                best_mono = polished_mono
                best_colors = polished
    
    return best_colors
