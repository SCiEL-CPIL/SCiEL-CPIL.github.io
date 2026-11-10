import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - (i * (i + 1)) // 2 + (j - i - 1)

def _build_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    cycles.append((_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)))
                    cycles.append((_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)))
                    cycles.append((_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)))
    return cycles

def _count_colors(colors):
    return len(set(colors))

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    cycles = _build_cycles()
    num_cycles = len(cycles)
    
    # Precompute: for each edge, which cycles contain it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e1, e2, e3, e4) in enumerate(cycles):
        edge_to_cycles[e1].append(ci)
        edge_to_cycles[e2].append(ci)
        edge_to_cycles[e3].append(ci)
        edge_to_cycles[e4].append(ci)
    
    # Precompute cycle edge sets for fast lookup
    cycle_edges = [set(c) for c in cycles]
    
    def _is_valid_cycle(colors, ci):
        e1, e2, e3, e4 = cycles[ci]
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        return len({c1, c2, c3, c4}) < 4
    
    def _count_violations_fast(colors):
        """Count violations using precomputed structures."""
        count = 0
        for ci in range(num_cycles):
            e1, e2, e3, e4 = cycles[ci]
            c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
            if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                count += 1
        return count
    
    def _delta_violations(colors, edge_idx, old_color, new_color):
        """Calculate change in violation count when changing edge to new_color."""
        delta = 0
        for ci in edge_to_cycles[edge_idx]:
            e1, e2, e3, e4 = cycles[ci]
            # Get colors with the change applied
            c1 = new_color if e1 == edge_idx else colors[e1]
            c2 = new_color if e2 == edge_idx else colors[e2]
            c3 = new_color if e3 == edge_idx else colors[e3]
            c4 = new_color if e4 == edge_idx else colors[e4]
            
            old_c1 = old_color if e1 == edge_idx else colors[e1]
            old_c2 = old_color if e2 == edge_idx else colors[e2]
            old_c3 = old_color if e3 == edge_idx else colors[e3]
            old_c4 = old_color if e4 == edge_idx else colors[e4]
            
            old_violated = (old_c1 != old_c2 and old_c1 != old_c3 and old_c1 != old_c4 and 
                           old_c2 != old_c3 and old_c2 != old_c4 and old_c3 != old_c4)
            new_violated = (c1 != c2 and c1 != c3 and c1 != c4 and 
                           c2 != c3 and c2 != c4 and c3 != c4)
            
            if old_violated and not new_violated:
                delta -= 1
            elif not old_violated and new_violated:
                delta += 1
        return delta
    
    def _structured_baseline():
        """Create a valid structured coloring using vertex partitioning."""
        colors = [0] * NUM_EDGES
        # Use a 4x4 grid structure
        for i in range(N):
            for j in range(i + 1, N):
                # Color based on which "region" the vertices are in
                # This ensures that in any 4-cycle, edges share colors
                ri, rj = i // 4, j // 4
                ci, cj = i % 4, j % 4
                if ri == rj:
                    # Same row: color by column difference
                    colors[_edge_idx(i, j)] = abs(ci - cj)
                elif ci == cj:
                    # Same column: color by row difference  
                    colors[_edge_idx(i, j)] = 4 + abs(ri - rj)
                else:
                    # Different row and column: use a fixed pattern
                    # This ensures cycles have repeated colors
                    colors[_edge_idx(i, j)] = 8 + (ri * 4 + ri) % 4
        return colors
    
    best_colors = None
    best_score = 0
    
    # Start with structured baseline
    colors = _structured_baseline()
    violations = _count_violations_fast(colors)
    score = _count_colors(colors)
    
    if violations == 0:
        best_colors = colors[:]
        best_score = score
    
    random.seed(123)
    
    # Local search with simulated annealing
    T_start = 2.0
    T_end = 0.001
    max_steps = 200000
    step = 0
    
    current_violations = violations
    current_score = score
    
    while step < max_steps and time.time() < deadline - 5.0:
        step += 1
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Choose new color strategy
        if random.random() < 0.4:
            # Try a new color (not currently used)
            used = set(colors)
            available = [c for c in range(120) if c not in used]
            if available:
                new_color = random.choice(available)
            else:
                new_color = random.randint(0, 119)
        else:
            # Try a random color
            new_color = random.randint(0, 119)
        
        if new_color == old_color:
            continue
        
        # Calculate delta in violations
        delta_v = _delta_violations(colors, e, old_color, new_color)
        
        # Calculate score change
        old_score = current_score
        colors[e] = new_color
        new_score = _count_colors(colors)
        score_change = new_score - old_score
        
        # Acceptance criterion
        if delta_v < 0 or (delta_v == 0 and score_change > 0):
            # Always accept improvements
            current_violations += delta_v
            current_score = new_score
            if current_violations == 0 and current_score > best_score:
                best_score = current_score
                best_colors = colors[:]
        elif delta_v == 0 and score_change == 0:
            # Neutral move: accept with probability
            if random.random() < 0.1:
                current_score = new_score
        else:
            # Worse move: accept with probability exp(-delta/T)
            # Penalize violations heavily
            penalty = delta_v * 100 - score_change
            if penalty < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-penalty / max(T, 0.001))
            if random.random() < accept_prob:
                current_violations += delta_v
                current_score = new_score
            else:
                colors[e] = old_color
                continue
        
        # Track best valid solution
        if current_violations == 0 and current_score > best_score:
            best_score = current_score
            best_colors = colors[:]
    
    # Final check
    if best_colors is None:
        # Fallback to structured baseline if it's valid
        baseline = _structured_baseline()
        if _count_violations_fast(baseline) == 0:
            best_colors = baseline
            best_score = _count_colors(baseline)
        else:
            # All zeros as last resort
            best_colors = [0] * NUM_EDGES
            best_score = 1
    
    return best_colors

def construct():
    return _construct_impl()
