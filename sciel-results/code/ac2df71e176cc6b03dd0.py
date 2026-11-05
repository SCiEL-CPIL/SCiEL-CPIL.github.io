import time
import numpy as np
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def edge_vertices(idx):
    i = 0
    while i * N - i * (i + 1) // 2 + (N - i - 1) <= idx:
        i += 1
    i -= 1
    offset = idx - (i * N - i * (i + 1) // 2)
    j = i + 1 + offset
    return i, j

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Precompute all 4-cycles
    # For each 4-set {a,b,c,d}, there are 3 distinct 4-cycles.
    # Cycle 1: (a,b,c,d) -> edges (a,b), (b,c), (c,d), (d,a)
    # Cycle 2: (a,b,d,c) -> edges (a,b), (b,d), (d,c), (c,a)
    # Cycle 3: (a,c,b,d) -> edges (a,c), (c,b), (b,d), (d,a)
    
    all_cycles = []
    vertices = list(range(N))
    
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1
        e1 = np.array([edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)])
        # Cycle 2
        e2 = np.array([edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a)])
        # Cycle 3
        e3 = np.array([edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a)])
        all_cycles.extend([e1, e2, e3])
        
    # Convert to a single 2D array for fast vectorized access
    # Shape: (num_cycles, 4)
    cycles_arr = np.array(all_cycles, dtype=np.int32)
    num_cycles = cycles_arr.shape[0]
    
    # 2. Initial Coloring
    # Start with a valid low-color coloring to ensure validity.
    # (i+j) % 5 is often valid for this constraint class, but let's be safe.
    # A 1-factorization based coloring or just random restarts.
    # Let's use a simple valid base: all 0s, then build up.
    # Or use (i+j)%K for small K.
    
    def get_initial_colors():
        # Random valid-ish start
        # Using (i+j) % 3 is very likely valid and gives 3 colors.
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        for idx in range(NUM_EDGES):
            i, j = edge_vertices(idx)
            colors[idx] = (i + j) % 3
        return colors

    # 3. Vectorized Constraint Checker
    # Input: colors (array of shape (120,))
    # Output: boolean array of shape (num_cycles,) where True means VALID (not 4 distinct)
    def check_constraints_vectorized(colors):
        # Get colors for all edges in all cycles at once
        # cycle_colors shape: (num_cycles, 4)
        cycle_colors = colors[cycles_arr]
        
        # A cycle is INVALID if it has 4 distinct colors.
        # We can check distinctness by sorting or by pairwise comparison.
        # Pairwise comparison is faster for small n.
        # c0, c1, c2, c3
        c0 = cycle_colors[:, 0]
        c1 = cycle_colors[:, 1]
        c2 = cycle_colors[:, 2]
        c3 = cycle_colors[:, 3]
        
        # Invalid if:
        # c0 != c1 AND c0 != c2 AND c0 != c3 AND c1 != c2 AND c1 != c3 AND c2 != c3
        # Actually, just checking if any pair is equal is sufficient for VALID.
        # If ANY pair is equal, it's NOT 4 distinct.
        # So INVALID = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        
        is_valid = ~((c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3))
        return is_valid

    # 4. Fast Single-Edge Move Validator
    # We need to check if changing edge `e` to `new_color` creates a 4-distinct cycle.
    # We only need to check cycles containing `e`.
    # Precompute which cycles contain which edge.
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci in range(num_cycles):
        for e in cycles_arr[ci]:
            edge_to_cycles[e].append(ci)
            
    # Convert to arrays for speed
    edge_cycle_indices = [np.array(c, dtype=np.int32) if c else np.array([], dtype=np.int32) for c in edge_to_cycles]
    
    def is_move_valid(e, new_color, colors):
        old_color = colors[e]
        if new_color == old_color:
            return True
            
        cycle_indices = edge_cycle_indices[e]
        if len(cycle_indices) == 0:
            return True
            
        # Get the other 3 edges in these cycles
        # cycles_arr[cycle_indices] gives shape (num_related_cycles, 4)
        related_cycles = cycles_arr[cycle_indices]
        
        # For each related cycle, we need to check if the other 3 edges + new_color are all distinct.
        # Let the other edges be e1, e2, e3.
        # If any of e1, e2, e3 has the same color as new_color, it's valid (not 4 distinct).
        # If any of e1, e2, e3 has the same color as each other, it's valid.
        # It is INVALID only if:
        # 1. None of e1, e2, e3 have color new_color.
        # 2. e1, e2, e3 are all distinct from each other.
        
        # Extract colors of the other edges
        # We know position of e in the cycle? No, cycles_arr is just indices.
        # We need to mask out e.
        
        # Get colors of all 4 edges in related cycles
        rel_colors = colors[related_cycles] # shape (M, 4)
        
        # Identify which column is 'e'
        # related_cycles contains 'e' in some column.
        # We can find the mask where related_cycles == e
        e_mask = (related_cycles == e) # shape (M, 4), bool
        
        # The other edges are where e_mask is False.
        # Let's extract them.
        # It's easier to just check the condition directly.
        
        # For each cycle m:
        # colors are rel_colors[m, 0..3]
        # One of them is old_color (at the position of e). We replace it with new_color.
        # So the set of colors becomes {new_color, other1, other2, other3}.
        
        # We can check if {new_color, other1, other2, other3} has 4 distinct elements.
        
        # Vectorized approach:
        # Create a copy of rel_colors
        test_colors = rel_colors.copy()
        test_colors[e_mask] = new_color
        
        c0 = test_colors[:, 0]
        c1 = test_colors[:, 1]
        c2 = test_colors[:, 2]
        c3 = test_colors[:, 3]
        
        # Check for 4 distinct
        is_4_distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        
        # If any cycle is 4 distinct, the move is invalid
        if np.any(is_4_distinct):
            return False
        return True

    # 5. Simulated Annealing
    best_colors = get_initial_colors()
    best_score = len(np.unique(best_colors))
    
    rng = np.random.default_rng(42)
    
    # Precompute a list of random edges for sampling
    # We will run multiple independent SA chains in parallel conceptually, 
    # but Python is GIL bound. However, vectorized numpy ops are fast.
    # Let's run a single very fast SA loop with many iterations.
    
    T = 10.0
    cooling = 0.9999
    
    current_colors = best_colors.copy()
    
    # Optimization: Precompute the "other edges" for each edge's cycles to avoid copying large arrays in is_move_valid?
    # The current is_move_valid creates a copy of rel_colors. 
    # rel_colors size is roughly (number of cycles containing e) x 4.
    # Max cycles containing an edge: 
    # Edge (i,j). Cycles use i,j and 2 other vertices k,l.
    # There are 14 other vertices. Pairs of 2 from 14 is C(14,2) = 91.
    # For each pair {k,l}, there are 2 cycles. So 182 cycles.
    # 182 * 4 = 728 integers. Copying this is fast.
    
    steps = 0
    while time.time() < deadline:
        # Batch processing idea: 
        # Instead of 1 move per step, we can't easily batch independent moves because they interfere.
        # So we stick to sequential moves but make the check as fast as possible.
        
        e = rng.integers(0, NUM_EDGES)
        old_color = current_colors[e]
        
        # Choose new color
        # Bias towards unused colors or nearby colors?
        # Random in 0..119 is good for exploration.
        new_color = rng.integers(0, MAX_COLORS)
        
        if new_color == old_color:
            continue
            
        if is_move_valid(e, new_color, current_colors):
            current_colors[e] = new_color
            new_score = len(np.unique(current_colors))
            
            # Acceptance
            if new_score > best_score:
                best_score = new_score
                best_colors = current_colors.copy()
            else:
                # Simulated Annealing acceptance
                # Delta is usually 0 or negative (score change)
                # We want to maximize score.
                # If score decreases, accept with prob exp(-delta/T) where delta = new_score - old_score
                # But score is integer. 
                # Let's use a simpler heuristic: accept if score >= old_score OR random < T
                # Actually, standard SA:
                # Energy = -Score. Minimize Energy.
                # dE = -new_score - (-old_score) = old_score - new_score
                # If dE > 0 (score decreased), accept with prob exp(-dE/T)
                
                old_score = len(np.unique(current_colors)) # Wait, current_colors already updated
                # We need the score BEFORE the move to calculate delta?
                # No, we can just compare.
                # Let's re-fetch old_score properly.
                # We didn't store old_score in the loop.
                # Let's assume we track it.
                
                # Refinement: Track current_score
                pass
                
        steps += 1
        
    # The loop above has a flaw: it doesn't track current_score efficiently.
    # Let's rewrite the loop more carefully.
    
    # Restart with a cleaner SA implementation
    
    best_colors = get_initial_colors()
    best_score = len(np.unique(best_colors))
    
    current_colors = best_colors.copy()
    current_score = best_score
    
    T = 5.0
    cooling = 0.99995
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        e = rng.integers(0, NUM_EDGES)
        old_color = current_colors[e]
        new_color = rng.integers(0, MAX_COLORS)
        
        if new_color == old_color:
            continue
            
        if is_move_valid(e, new_color, current_colors):
            # Tentatively apply
            current_colors[e] = new_color
            new_score = len(np.unique(current_colors))
            
            delta = new_score - current_score
            
            if delta >= 0 or rng.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_colors = current_colors.copy()
            else:
                # Revert
                current_colors[e] = old_color
                
        T *= cooling
        
    # Final check
    if not np.all(check_constraints_vectorized(best_colors)):
        # Fallback to a known valid simple coloring
        fallback = np.zeros(NUM_EDGES, dtype=np.int32)
        for idx in range(NUM_EDGES):
            i, j = edge_vertices(idx)
            fallback[idx] = (i + j) % 3
        return fallback.tolist()
        
    return best_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
    print(colors)
