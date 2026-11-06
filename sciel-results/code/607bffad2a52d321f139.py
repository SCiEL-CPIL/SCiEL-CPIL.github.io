import time
import itertools
import random
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 75  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Build edge index mapping
    edge_index = {}
    edge_list = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            edge_list.append((i, j))
            idx += 1
    
    # Precompute for each edge, the list of 4-sets (as tuples of 4 sorted vertices) that contain it
    # and for each such 4-set, the three cycles (each cycle is a tuple of 4 edge indices)
    # This allows fast re-validation when an edge color changes
    
    # First, enumerate all 4-sets
    all_4sets = list(itertools.combinations(range(N), 4))
    
    # For each 4-set, precompute the three cycles as tuples of edge indices
    # Cycle 1: a-b-c-d-a -> edges (a,b),(b,c),(c,d),(d,a)
    # Cycle 2: a-b-d-c-a -> edges (a,b),(b,d),(d,c),(c,a)
    # Cycle 3: a-c-b-d-a -> edges (a,c),(c,b),(b,d),(d,a)
    def get_cycles(a, b, c, d):
        # Ensure a<b<c<d
        e1 = (edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)])
        e2 = (edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)])
        e3 = (edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)])
        return (e1, e2, e3)
    
    # For each edge, store the list of (4set_index, cycle_index) pairs that involve it
    edge_to_4sets = [[] for _ in range(NUM_EDGES)]
    cycles_by_4set = []
    for s_idx, (a, b, c, d) in enumerate(all_4sets):
        cycles = get_cycles(a, b, c, d)
        cycles_by_4set.append(cycles)
        for c_idx, cycle in enumerate(cycles):
            for e_idx in cycle:
                edge_to_4sets[e_idx].append((s_idx, c_idx))
    
    def is_valid_cycle(colors, cycle):
        """Check if a cycle (tuple of 4 edge indices) has < 4 distinct colors"""
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        # Check if all four are distinct
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
        return True
    
    def check_edge_validity(colors, e_idx):
        """Check all 4-sets/cycles involving edge e_idx"""
        for (s_idx, c_idx) in edge_to_4sets[e_idx]:
            cycle = cycles_by_4set[s_idx][c_idx]
            if not is_valid_cycle(colors, cycle):
                return False
        return True
    
    def full_validation(colors):
        """Full validation of all cycles"""
        for s_idx in range(len(all_4sets)):
            for c_idx in range(3):
                cycle = cycles_by_4set[s_idx][c_idx]
                if not is_valid_cycle(colors, cycle):
                    return False
        return True
    
    def count_colors(colors):
        """Count the number of distinct colors used"""
        return len(set(colors))
    
    # Initialize with the known valid coloring: color(i,j) = i for i<j
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(edge_list):
        colors[idx] = i
    
    # Verify initial coloring is valid
    if not full_validation(colors):
        return [0] * NUM_EDGES
    
    best_colors = colors[:]
    best_count = count_colors(colors)
    
    # Apply known improvement: recolor (13,15) and (14,15) to new colors
    # This achieves 17 colors (15 base + 2 new)
    idx_13_15 = edge_index[(13, 15)]
    idx_14_15 = edge_index[(14, 15)]
    
    colors[idx_13_15] = 15
    colors[idx_14_15] = 16
    
    if full_validation(colors):
        best_colors = colors[:]
        best_count = count_colors(colors)
    else:
        # Fallback to original
        for idx, (i, j) in enumerate(edge_list):
            colors[idx] = i
        best_colors = colors[:]
        best_count = count_colors(colors)
    
    # Now use local search to try to add more colors
    # Strategy: For each edge, try to assign a new color (not currently used)
    # that keeps the coloring valid. Use constraint-aware ordering.
    
    # Count how many 4-sets each edge is involved in (for ordering)
    edge_constraint_count = [len(edge_to_4sets[e]) for e in range(NUM_EDGES)]
    
    # Sort edges by constraint count (most constrained first)
    edge_order = sorted(range(NUM_EDGES), key=lambda e: edge_constraint_count[e], reverse=True)
    
    # Local search: try to assign new colors to edges
    def try_assign_new_color(colors, e_idx, new_color):
        """Try to assign new_color to edge e_idx and check validity"""
        old_color = colors[e_idx]
        colors[e_idx] = new_color
        valid = True
        for (s_idx, c_idx) in edge_to_4sets[e_idx]:
            cycle = cycles_by_4set[s_idx][c_idx]
            if not is_valid_cycle(colors, cycle):
                valid = False
                break
        if not valid:
            colors[e_idx] = old_color
            return False
        return True
    
    # Iteratively try to add new colors
    used_colors = set(best_colors)
    next_color = max(used_colors) + 1 if used_colors else 0
    
    # Multiple passes of local search
    for pass_num in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        progress = False
        for e_idx in edge_order:
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to assign a new color to this edge
            # First, check if the edge already has a unique color
            current_color = best_colors[e_idx]
            
            # Try assigning a brand new color
            if try_assign_new_color(best_colors, e_idx, next_color):
                used_colors.add(next_color)
                next_color += 1
                best_count = count_colors(best_colors)
                progress = True
                # Don't break - continue trying other edges
            else:
                # Try reusing an existing color that might allow more flexibility
                # Actually, we want to MAXIMIZE colors, so only assign new colors
                pass
        
        if not progress:
            # No progress in this pass, try a different strategy
            # Random restart: shuffle edge order
            random.shuffle(edge_order)
            
            # Also try: for each edge, try all possible new colors
            # to see if any work
            for e_idx in edge_order:
                if time.time() - start_time > DEADLINE:
                    break
                
                # Try a range of new colors
                for nc in range(next_color, next_color + 10):
                    if try_assign_new_color(best_colors, e_idx, nc):
                        used_colors.add(nc)
                        next_color = nc + 1
                        best_count = count_colors(best_colors)
                        progress = True
                        break
        
        if not progress:
            # Try a more aggressive approach: random perturbation + local search
            if time.time() - start_time > DEADLINE:
                break
            
            # Make a random valid perturbation
            trial_colors = best_colors[:]
            for _ in range(100):
                if time.time() - start_time > DEADLINE:
                    break
                e_idx = random.randint(0, NUM_EDGES - 1)
                # Try a random new color
                nc = random.randint(0, 119)
                if try_assign_new_color(trial_colors, e_idx, nc):
                    # Check if this increases color count
                    trial_count = count_colors(trial_colors)
                    if trial_count > best_count:
                        best_colors = trial_colors[:]
                        best_count = trial_count
                        used_colors = set(best_colors)
                        next_color = max(used_colors) + 1 if used_colors else 0
                        progress = True
                        break
    
    # Final validation
    if full_validation(best_colors):
        return best_colors
    else:
        # Fallback to the simple valid coloring
        fallback = [0] * NUM_EDGES
        for idx, (i, j) in enumerate(edge_list):
            fallback[idx] = i
        return fallback

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
