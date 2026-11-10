import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_idx(i, j):
    """Map pair (i,j) with i<j to edge index 0..119 in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    """Return array of shape (num_cycles, 4) with edge indices for each 4-cycle."""
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    cycles.append((_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(a, d)))
                    # Cycle 2: a-b-d-c-a
                    cycles.append((_edge_idx(a, b), _edge_idx(b, d), _edge_idx(c, d), _edge_idx(a, c)))
                    # Cycle 3: a-c-b-d-a
                    cycles.append((_edge_idx(a, c), _edge_idx(b, c), _edge_idx(b, d), _edge_idx(a, d)))
    return np.array(cycles, dtype=np.int32)

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

# Precompute for fast vectorized violation checking
# For each cycle, store the 4 edge indices as columns
CYCLE_EDGES = CYCLES  # shape (NUM_CYCLES, 4)

def _count_violations_vectorized(colors):
    """Vectorized count of 4-cycles with 4 distinct edge colors."""
    # colors shape: (NUM_EDGES,)
    # Get colors for all edges in all cycles: shape (NUM_CYCLES, 4)
    cycle_colors = colors[CYCLE_EDGES]  # shape (NUM_CYCLES, 4)
    # A cycle is a violation if all 4 colors are distinct
    # Sort the 4 colors and check if all are different
    sorted_colors = np.sort(cycle_colors, axis=1)
    # Check if all 4 are distinct: no adjacent pair equal
    distinct = (sorted_colors[:, 1] != sorted_colors[:, 0]) & \
               (sorted_colors[:, 2] != sorted_colors[:, 1]) & \
               (sorted_colors[:, 3] != sorted_colors[:, 2])
    return int(np.sum(distinct))

def _is_valid_fast(colors):
    """Check if coloring is valid (no 4-cycle has 4 distinct colors)."""
    return _count_violations_vectorized(colors) == 0

def _algebraic_init(seed_val):
    """Generate an algebraic/rank-1 coloring using modular arithmetic."""
    rng = np.random.RandomState(seed_val)
    
    best_colors = None
    best_distinct = 0
    
    # Structure 1: F_2^k dot product (integer sum)
    # c(i,j) = sum(v_i[m] * v_j[m]) for m in 0..k-1
    # Range: 0..k
    for k in [4, 5, 6, 8, 10]:
        V = rng.randint(0, 2, size=(N, k))
        # Compute all dot products
        # V is (16, k). We need V_i dot V_j.
        # Matrix product V @ V.T gives (16, 16) with dot products.
        M = V @ V.T
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                colors[idx] = M[i, j]
                idx += 1
        
        if _is_valid_fast(colors):
            distinct = len(set(colors.tolist()))
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors.copy()
                
    # Structure 2: F_p multiplicative
    # c(i,j) = (v_i * v_j) % p
    for p in [5, 7, 11, 13, 17, 19]:
        V = rng.randint(1, p, size=N) # Avoid 0 to get more distinct products? Or allow 0.
        # Let's allow 0..p-1
        V = rng.randint(0, p, size=N)
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                colors[idx] = (V[i] * V[j]) % p
                idx += 1
        
        if _is_valid_fast(colors):
            distinct = len(set(colors.tolist()))
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors.copy()
                
    # Structure 3: Original mixed modular structures (kept from before but optimized)
    structures = [
        lambda i, j: (i % 4) * 4 + (j % 4),
        lambda i, j: (i % 5) * 5 + (j % 5),
        lambda i, j: (i % 3) * 3 + (j % 3),
        lambda i, j: (i % 2) * 8 + (j % 8),
        lambda i, j: (i % 8) * 8 + (j % 2),
        lambda i, j: (i * j) % 17,
        lambda i, j: (i * j) % 16,
        lambda i, j: (i * j) % 15,
        lambda i, j: (i ^ j) % 16,
        lambda i, j: (i ^ j) % 12,
        lambda i, j: ((i % 4) * (j % 4) + (i // 4) * (j // 4)) % 16,
        lambda i, j: ((i % 4) + (j % 4) * 4 + (i // 4) * 2 + (j // 4) * 8) % 16,
    ]
    
    for func in structures:
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c = func(i, j)
                if c > MAX_COLOR:
                    c = c % (MAX_COLOR + 1)
                colors[idx] = c
                idx += 1
        
        if _is_valid_fast(colors):
            distinct = len(set(colors.tolist()))
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors.copy()
    
    if best_colors is not None:
        return best_colors
    
    # Fallback: random valid coloring
    return _random_valid_init(seed_val)

def _random_valid_init(seed_val):
    """Generate a random valid coloring by greedy assignment."""
    rng = np.random.RandomState(seed_val)
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # Greedy: assign random colors, check validity
    for attempt in range(100):
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        for e in range(NUM_EDGES):
            # Try random colors
            for trial in range(50):
                c = rng.randint(0, MAX_COLOR + 1)
                colors[e] = c
                if _is_valid_fast(colors):
                    break
            # If no valid color found, keep 0
            colors[e] = 0
        
        if _is_valid_fast(colors):
            return colors
    
    return np.zeros(NUM_EDGES, dtype=np.int32)

def _fast_local_search(colors, deadline, seed_val=42):
    """Vectorized local search to maximize distinct colors while maintaining validity."""
    rng = np.random.RandomState(seed_val)
    current_colors = colors.copy()
    current_distinct = len(set(current_colors.tolist()))
    best_colors = current_colors.copy()
    best_distinct = current_distinct
    
    # Precompute edge-to-cycles mapping for fast violation checking
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci in range(NUM_CYCLES):
        for e in CYCLES[ci]:
            edge_to_cycles[e].append(ci)
    
    # Color counts
    color_counts = np.zeros(MAX_COLOR + 1, dtype=np.int32)
    for c in current_colors:
        color_counts[c] += 1
        
    # List of unused colors for fast sampling
    unused_colors = [i for i in range(MAX_COLOR + 1) if color_counts[i] == 0]
    
    T = 1.0
    cooling = 0.9999
    min_T = 0.001
    steps_since_improve = 0
    max_steps_no_improve = 5000
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # If stuck, cool down faster or restart T?
        if steps_since_improve > max_steps_no_improve:
            T = 0.1 # Lower temperature to settle
            max_steps_no_improve = 5000
            steps_since_improve = 0

        # Pick random edge
        e_idx = rng.randint(0, NUM_EDGES)
        old_color = int(current_colors[e_idx])
        
        # Pick new color
        # 70% chance to try an unused color if any exist
        if rng.random() < 0.7 and len(unused_colors) > 0:
            new_color = unused_colors[rng.randint(0, len(unused_colors))]
        else:
            new_color = int(rng.randint(0, MAX_COLOR + 1))
            
        if new_color == old_color:
            continue
        
        # Check if change creates violations
        affected_cycles = edge_to_cycles[e_idx]
        
        if affected_cycles:
            safe = True
            for ci in affected_cycles:
                cyc = CYCLES[ci]
                c0 = int(current_colors[cyc[0]])
                c1 = int(current_colors[cyc[1]])
                c2 = int(current_colors[cyc[2]])
                c3 = int(current_colors[cyc[3]])
                
                nc0, nc1, nc2, nc3 = c0, c1, c2, c3
                if cyc[0] == e_idx: nc0 = new_color
                elif cyc[1] == e_idx: nc1 = new_color
                elif cyc[2] == e_idx: nc2 = new_color
                elif cyc[3] == e_idx: nc3 = new_color
                
                # Check if new configuration has 4 distinct colors
                # We can use a set check which is fast for 4 elements
                if len({nc0, nc1, nc2, nc3}) == 4:
                    safe = False
                    break
            
            if not safe:
                continue
        
        # Apply change
        current_colors[e_idx] = new_color
        color_counts[old_color] -= 1
        color_counts[new_color] += 1
        
        # Update unused_colors list
        if color_counts[old_color] == 0:
            if old_color not in unused_colors:
                unused_colors.append(old_color)
        if color_counts[new_color] == 1:
            if new_color in unused_colors:
                unused_colors.remove(new_color)
                
        # Update distinct count
        new_distinct_count = current_distinct
        if color_counts[old_color] == 0:
            new_distinct_count -= 1
        if color_counts[new_color] == 1:
            new_distinct_count += 1
            
        # Accept/reject
        if new_distinct_count > current_distinct:
            current_distinct = new_distinct_count
            steps_since_improve = 0
            if current_distinct > best_distinct:
                best_distinct = current_distinct
                best_colors = current_colors.copy()
        else:
            # Accept with probability exp(delta/T)
            delta = new_distinct_count - current_distinct
            if delta >= 0 or rng.random() < np.exp(delta / T):
                current_distinct = new_distinct_count
                steps_since_improve += 1
            else:
                # Revert
                current_colors[e_idx] = old_color
                color_counts[old_color] += 1
                color_counts[new_color] -= 1
                
                if color_counts[old_color] == 1:
                    if old_color in unused_colors:
                        unused_colors.remove(old_color)
                if color_counts[new_color] == 0:
                    if new_color not in unused_colors:
                        unused_colors.append(new_color)
                        
        T *= cooling
        
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_colors = None
    best_distinct = 0
    
    # Multi-start with different algebraic initializations
    num_starts = 10
    time_per_start = (deadline - start_time) / num_starts
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        
        seed_val = 1000 + start_idx * 7919
        
        # Get algebraic initialization
        init_colors = _algebraic_init(seed_val)
        init_distinct = len(set(init_colors.tolist()))
        
        if init_distinct > best_distinct:
            best_distinct = init_distinct
            best_colors = init_colors.copy()
            
        # Run local search from this start
        remaining_time = deadline - 2.0 - time.time()
        if remaining_time < 1.0:
            continue
            
        search_colors, search_distinct = _fast_local_search(
            init_colors, 
            time.time() + min(remaining_time, time_per_start),
            seed_val + 1
        )
        
        if search_distinct > best_distinct:
            best_distinct = search_distinct
            best_colors = search_colors.copy()
            
        # Check if we've used too much time
        if time.time() > deadline - 10.0:
            break
    
    # Final verification
    if best_colors is None:
        return [0] * NUM_EDGES
        
    if _count_violations_vectorized(best_colors) != 0:
        return [0] * NUM_EDGES
        
    return best_colors.tolist()
