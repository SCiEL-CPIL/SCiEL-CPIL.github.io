import time
import sys
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Precompute edge indices
    # Edge (i, j) with i < j.
    # Index = sum_{k=0}^{i-1} (N-1-k) + (j - i - 1)
    # sum_{k=0}^{i-1} (N-1-k) = i(N-1) - i(i-1)/2
    edge_indices = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_indices[(i, j)] = idx
            edge_indices[(j, i)] = idx
            idx += 1
            
    # Precompute all 4-cycles (sets of 4 edges)
    # A 4-cycle on vertices {a, b, c, d} can be formed in 3 ways (perfect matchings of K4):
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = edge_indices[(a, b)]
                    e2 = edge_indices[(b, c)]
                    e3 = edge_indices[(c, d)]
                    e4 = edge_indices[(d, a)]
                    cycles.append((e1, e2, e3, e4))
                    
                    # Cycle 2: a-b-d-c-a
                    e1 = edge_indices[(a, b)]
                    e2 = edge_indices[(b, d)]
                    e3 = edge_indices[(c, d)]
                    e4 = edge_indices[(a, c)]
                    cycles.append((e1, e2, e3, e4))
                    
                    # Cycle 3: a-c-b-d-a
                    e1 = edge_indices[(a, c)]
                    e2 = edge_indices[(b, c)]
                    e3 = edge_indices[(b, d)]
                    e4 = edge_indices[(a, d)]
                    cycles.append((e1, e2, e3, e4))
    
    # Convert cycles to a numpy array for vectorization
    # Shape: (num_cycles, 4)
    cycles_arr = np.array(cycles, dtype=np.int32)
    
    # Build edge-to-cycles mapping for fast validity checking
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(i)
    
    # Precompute the list of cycle indices for each edge as a numpy array for fast indexing
    edge_cycle_indices = [np.array(edge_to_cycles[e], dtype=np.int32) for e in range(NUM_EDGES)]
    
    # Helper to check if a coloring is valid
    def is_valid(coloring):
        # Check all cycles
        cycle_colors = coloring[cycles_arr]
        sorted_colors = np.sort(cycle_colors, axis=1)
        distinct_mask = (sorted_colors[:, 0] != sorted_colors[:, 1]) & \
                        (sorted_colors[:, 1] != sorted_colors[:, 2]) & \
                        (sorted_colors[:, 2] != sorted_colors[:, 3])
        return not np.any(distinct_mask)
    
    # Helper to check if a specific edge change is valid
    def is_change_valid(coloring, edge_idx, new_color):
        old_color = coloring[edge_idx]
        if new_color == old_color:
            return True
        coloring[edge_idx] = new_color
        cycle_idxs = edge_cycle_indices[edge_idx]
        if len(cycle_idxs) == 0:
            return True
        sub_cycles = cycles_arr[cycle_idxs]
        cycle_colors = coloring[sub_cycles]
        sorted_colors = np.sort(cycle_colors, axis=1)
        distinct_mask = (sorted_colors[:, 0] != sorted_colors[:, 1]) & \
                        (sorted_colors[:, 1] != sorted_colors[:, 2]) & \
                        (sorted_colors[:, 2] != sorted_colors[:, 3])
        valid = not np.any(distinct_mask)
        coloring[edge_idx] = old_color
        return valid

    # Strategy: Multi-start Greedy Insertion
    # 1. Start with a valid base coloring (e.g., all 0s, or a simple partition).
    # 2. Repeatedly try to add a new color to the palette.
    #    - Pick a new color c not in the current palette.
    #    - Try to assign c to some edge e such that the coloring remains valid.
    #    - To find such an edge, we can scan edges and check if assigning c is valid.
    #    - If found, update the coloring and palette.
    #    - If no edge can take the new color, we might need to "shuffle" existing colors
    #      to create space, or restart with a different base.
    
    # We will use a base coloring from recursive bisection (Gallai) as it has a strong structure,
    # but we will also try random valid bases.
    
    def get_edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    def generate_gallai_base():
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        color_id = 0
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = get_edge_idx(u, v)
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        process_partition(list(range(N)))
        return colors

    def generate_random_base():
        # Start with all 0s and randomly assign colors while maintaining validity?
        # Or just start with all 0s and let the greedy insertion do the work.
        # All 0s is valid.
        return np.zeros(NUM_EDGES, dtype=np.int32)

    best_distinct = 0
    best_colors = None
    
    # We will run multiple restarts
    # Each restart:
    # 1. Initialize coloring (Gallai or Random/Zero)
    # 2. Greedy insertion loop
    # 3. If stuck, try local search to "unlock" new colors?
    #    Or just restart with a different base.
    
    # Let's prioritize the Gallai base as it starts with ~15 colors.
    # The greedy insertion should be able to push it higher.
    
    num_restarts = 200 # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Choose base
        if restart % 2 == 0:
            current_colors = generate_gallai_base()
        else:
            current_colors = generate_random_base()
            # For random base, we might want to do a quick random walk to get a "better" random start?
            # Or just stick with zeros. Zeros is a valid start.
            # Let's use a slightly more complex random start: 
            # Assign random colors, and if invalid, revert. Repeat for a few steps.
            # This might create a more "scattered" initial state.
            for _ in range(100):
                e = random.randint(0, NUM_EDGES - 1)
                c = random.randint(0, MAX_COLORS - 1)
                if is_change_valid(current_colors, e, c):
                    current_colors[e] = c
        
        current_palette = set(current_colors.tolist())
        current_distinct = len(current_palette)
        
        if current_distinct > best_distinct:
            best_distinct = current_distinct
            best_colors = current_colors.copy()
        
        # Greedy Insertion Phase
        # Try to add colors from MAX_COLORS down to current_distinct? 
        # Or just pick random unused colors?
        # Picking random unused colors is good for diversity.
        
        max_insertion_attempts = 5000
        for attempt in range(max_insertion_attempts):
            if time.time() > deadline:
                break
                
            if current_distinct >= MAX_COLORS:
                break
                
            # Pick a random color not in the palette
            available_colors = [c for c in range(MAX_COLORS) if c not in current_palette]
            if not available_colors:
                break
            new_color = random.choice(available_colors)
            
            # Try to find an edge that can take this new color
            # We can shuffle the edges to try different ones each time
            # But scanning all 120 edges every time is O(120 * cost_check).
            # cost_check is O(deg(e)) ~ O(10-20). So ~2000 ops per attempt.
            # 5000 attempts * 2000 ops = 10M ops. Should be fast in Python if optimized?
            # Let's try to be smarter.
            
            found_edge = -1
            # Shuffle edges to avoid bias
            edge_order = list(range(NUM_EDGES))
            random.shuffle(edge_order)
            
            for e in edge_order:
                if is_change_valid(current_colors, e, new_color):
                    found_edge = e
                    break
            
            if found_edge != -1:
                # Apply change
                current_colors[found_edge] = new_color
                current_palette.add(new_color)
                current_distinct += 1
                
                if current_distinct > best_distinct:
                    best_distinct = current_distinct
                    best_colors = current_colors.copy()
            else:
                # No edge could take this new color.
                # Maybe we should try a different new color?
                # The loop continues, picking a new random color.
                # This is fine.
                pass
        
        # If we are stuck (no new colors added in the last phase), we could try a local search
        # to "rearrange" colors to free up space for new ones.
        # But for now, let's rely on multiple restarts with different bases.
        
        # Optional: Local Search Refinement
        # If we have time, do a few steps of local search to try to break symmetry or find better configurations.
        # However, the greedy insertion is already quite powerful if the base is good.
        
    if best_colors is None:
        return generate_gallai_base().tolist()
        
    return best_colors.tolist()

# Keep the function name and signature
# The task asks for construct() -> 120 integer colors

if __name__ == "__main__":
    res = construct()
    print(len(set(res)))
