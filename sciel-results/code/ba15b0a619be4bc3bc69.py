import random
import math
import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    # Map edge (i, j) to index
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Precompute edges
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            
    # Map index to (u, v)
    idx_to_uv = [None] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx_to_uv[edge_idx(i, j)] = (i, j)

    # Precompute affected cycles for each edge
    # For each edge e, list of (cycle_edge_indices, positions_in_cycle)
    # Actually, we just need to check validity of specific cycles.
    # A cycle is defined by 4 vertices.
    # We can precompute for each edge, the list of 4-vertex sets containing it.
    # And for each 4-vertex set, the 3 cycles (as lists of edge indices).
    
    # Let's just iterate over pairs of other vertices for the specific edge during check.
    # It's fast enough: 182 checks per move.
    
    def get_cycles_for_edge(e_idx):
        u, v = idx_to_uv[e_idx]
        others = [x for x in range(N) if x != u and x != v]
        cycles = []
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                x = others[i]
                y = others[j]
                # Cycle 1: u-v-x-y-u
                c1 = [e_idx, edge_idx(v, x), edge_idx(x, y), edge_idx(y, u)]
                cycles.append(c1)
                # Cycle 2: u-v-y-x-u
                c2 = [e_idx, edge_idx(v, y), edge_idx(y, x), edge_idx(x, u)]
                cycles.append(c2)
                # Cycle 3: u-x-v-y-u (does not contain e_idx)
                # We don't need to check this one for validity change of e_idx,
                # because e_idx is not in it.
                # Wait, the task says "all three cycles on every four-vertex subset are checked".
                # If I change e_idx, only cycles containing e_idx can change validity.
                # Cycle 3 does not contain e_idx, so its validity is unchanged.
                # So we only need to check C1 and C2.
        return cycles

    # Precompute cycles for each edge to save time
    edge_cycles = [get_cycles_for_edge(i) for i in range(NUM_EDGES)]

    # Initialize colors
    colors = [0] * NUM_EDGES
    used_colors = {0}
    
    # Count invalid cycles
    # We need a global count? Or just check if any invalid exists?
    # For SA, we need the count to calculate delta.
    # Let's maintain a set of invalid cycles?
    # Or just a count.
    # Since we only change 1 edge, we can re-evaluate the ~182 cycles.
    
    def is_cycle_valid(cycle_edges):
        cols = [colors[e] for e in cycle_edges]
        return len(set(cols)) < 4

    # Initial invalid count
    # We can just assume 0 initially since all colors are 0.
    invalid_count = 0
    
    # To be safe, let's verify initial state is valid (it is).
    
    start_time = time.time()
    timeout = 55 # Leave 5s buffer
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.995
    best_score = 1 # D - M*V. Initially D=1, V=0.
    best_colors = colors[:]
    
    M = 1000 # Penalty for invalid cycles
    
    # Helper to get current score
    def get_score():
        return len(used_colors) - M * invalid_count

    current_score = get_score()
    
    while time.time() - start_time < timeout:
        # Pick random edge
        e_idx = random.randint(0, NUM_EDGES - 1)
        # Pick random color
        new_color = random.randint(0, 119)
        
        if colors[e_idx] == new_color:
            continue
            
        old_color = colors[e_idx]
        
        # Calculate delta
        # 1. Update used_colors logic
        old_used_count = len(used_colors)
        
        # Temporarily apply
        colors[e_idx] = new_color
        
        # Update used_colors
        if new_color not in used_colors:
            used_colors.add(new_color)
        # Note: if old_color was unique, it might be removed?
        # We need to know if old_color is still present elsewhere.
        # This is expensive to check every time.
        # Alternative: Just track the set of colors used in the array.
        # But updating a set by removing an element requires knowing if it's still there.
        # Let's maintain a count of occurrences for each color?
        # Or just re-calculate len(set(colors))? 120 elements. Fast.
        # But we are in a tight loop.
        
        # Let's use a frequency array for colors 0..119
        # freq[c] = count of edges with color c.
        
        # Let's refactor to use freq array.
        # This requires restarting the logic or modifying the existing code significantly.
        # Let's try to make the current structure work with freq.
        
        # Actually, let's just use `len(set(colors))` for simplicity in the delta calculation?
        # No, `set(colors)` is O(120). 182 cycle checks is O(182*4).
        # Total O(200) per move. 100k moves in 1s? Maybe.
        
        # Let's stick to a simpler heuristic if SA is too complex to implement correctly in the edit.
        # But SA is the "new approach".
        
        # Let's implement the freq array properly.
        
        # We'll break out of this loop and rewrite the function body properly in the REPLACE block.
        break
        
    return best_colors
