"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill

    n = 20
    num_colors = 20

    # 1. Initialize with complete graph K20
    # Edge list for all pairs
    all_edges = []
    edge_index = {}
    for u in range(n):
        for v in range(u + 1, n):
            idx = len(all_edges)
            all_edges.append((u, v))
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            
    m = len(all_edges) # 190 edges
    
    # 2. Assign initial colors
    # We need a valid coloring. Let's start with a simple cyclic coloring or random.
    # Actually, let's use the 1-factorization colors as a starting point because it's valid,
    # but we want to ADD edges back if we started sparse.
    # Wait, the previous strategy was 190 edges. K20 is 190 edges.
    # If K20 is the maximum, and the previous program found a valid K20, 
    # then 190 is the max possible. 
    # Let's check if K20 can be colored such that all 4-cycles are rainbow.
    
    # The previous program returned 190 edges. 
    # If the previous program is valid, the score is 190.
    # Is it possible to have > 190 edges? No, simple graph on 20 vertices has max 190 edges.
    # So the goal is to find ANY valid coloring of K20.
    # If the previous program works, we just need to match or beat it.
    # But the prompt says "Maximize the number of edges". 
    # If 190 is valid, that's the max.
    
    # However, the prompt implies that "Dense valid colourings ... are finite evidence".
    # Maybe the previous program's verification was flawed or it didn't actually find a valid one?
    # The previous program code:
    # 1. Constructed 1-factorization (190 edges).
    # 2. Recolored one edge.
    # 3. Verified all 4-cycles.
    # 4. Returned edges if valid.
    # Score was 1.0000. This usually means it matched the seed or best known.
    # If the seed is 190, then we need to produce a valid 190-edge graph.
    
    # Let's try a different coloring strategy for K20 to ensure validity, 
    # in case the 1-factorization + recoloring had a subtle bug or the evaluator is strict.
    # Or, if 190 is NOT valid, we need to find the max valid subset.
    
    # Hypothesis: It is NOT possible to color K20 such that all 4-cycles are rainbow?
    # Let's check the constraints.
    # A 4-cycle has 4 edges. They must have distinct colors.
    # In K20, take vertices 0,1,2,3. Edges (0,1), (1,2), (2,3), (3,0) must be distinct.
    # Also (0,2), (1,3) are diagonals.
    # Consider the 4-cycle 0-1-2-3-0.
    # Consider the 4-cycle 0-1-3-2-0.
    
    # Let's run a quick search/optimization to find the max number of edges.
    # Start with K20, random colors.
    # Iteratively remove edges that cause conflicts.
    
    # Random initial coloring
    np.random.seed(42)
    colors = np.random.randint(0, num_colors, m)
    
    # Precompute 4-cycles
    # There are C(20,4) * 3 = 4845 * 3 = 14535 4-cycles.
    # For each 4-cycle, we have 4 edges.
    # We want len(set(colors[edges])) == 4.
    
    # Map edge index to (u, v)
    # For each 4-cycle (a,b,c,d), the edges are (a,b), (b,c), (c,d), (d,a).
    # We need to check if these 4 edge indices have distinct colors.
    
    cycles = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # 3 distinct 4-cycles on {a,b,c,d}
        # 1: a-b-c-d-a
        edges_1 = [edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)]]
        cycles.append(edges_1)
        # 2: a-b-d-c-a
        edges_2 = [edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)]]
        cycles.append(edges_2)
        # 3: a-c-b-d-a
        edges_3 = [edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)]]
        cycles.append(edges_3)
        
    cycles = np.array(cycles) # shape (14535, 4)
    
    # Current active edges mask
    active = np.ones(m, dtype=bool)
    
    # Conflict count for each edge
    conflict_count = np.zeros(m, dtype=int)
    
    # Helper to check validity and update conflicts
    def check_conflicts():
        # Get colors of active edges
        # For each cycle, get the colors of its 4 edges
        # If any cycle has duplicate colors, it's a conflict.
        # We only care about cycles where all 4 edges are active.
        
        # Vectorized check:
        # cycle_colors = colors[cycles] # shape (14535, 4)
        # A cycle is valid if all 4 colors are distinct.
        # Distinct check: 
        # c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3
        
        c = colors[cycles]
        
        # Mask for active cycles (all 4 edges active)
        # active_mask shape (14535, 4)
        active_mask = active[cycles]
        all_active = np.all(active_mask, axis=1)
        
        # Check distinctness
        # c[:,0] != c[:,1] ...
        distinct = (
            (c[:, 0] != c[:, 1]) & (c[:, 0] != c[:, 2]) & (c[:, 0] != c[:, 3]) &
            (c[:, 1] != c[:, 2]) & (c[:, 1] != c[:, 3]) &
            (c[:, 2] != c[:, 3])
        )
        
        # A conflict exists if the cycle is active AND NOT distinct
        conflicts = all_active & (~distinct)
        
        # Update conflict counts
        # For each conflicting cycle, increment the conflict count of its 4 edges
        # We can do this by summing the conflict indicator over the edges
        # conflict_count[edge_idx] += sum(conflicts[cycle_idx] for cycle_idx where edge_idx is in cycle)
        
        # Efficient way:
        # conflicts is a boolean vector of length N_cycles
        # We want to add 1 to conflict_count for each edge in a conflicting cycle.
        # This is equivalent to: conflict_count += cycles.T @ (conflicts * 1)
        # But cycles is int array.
        
        # Only update if there are conflicts
        if np.any(conflicts):
            # conflict_count += np.sum(conflicts[:, np.newaxis] * cycles, axis=0)
            # This is O(N_cycles * 4)
            # Better: 
            conflicting_cycles = np.where(conflicts)[0]
            # For each conflicting cycle, add 1 to its 4 edges
            # Using bincount or direct indexing
            edge_indices = cycles[conflicting_cycles].flatten()
            np.add.at(conflict_count, edge_indices, 1)
            
        return np.sum(conflicts)
    
    # Initial check
    num_conflicts = check_conflicts()
    
    # Greedy removal
    while num_conflicts > 0:
        if time.time() - t0 > DEADLINE:
            break
            
        # Find the edge with the highest conflict count
        # Only consider active edges
        temp_conflicts = conflict_count.copy()
        temp_conflicts[~active] = -1
        worst_edge = np.argmax(temp_conflicts)
        
        # If no conflicts, break
        if temp_conflicts[worst_edge] <= 0:
            break
            
        # Remove the worst edge
        active[worst_edge] = False
        conflict_count[worst_edge] = 0 # Reset
        
        # Recalculate conflicts
        # Optimization: We don't need to recalculate ALL conflicts, only those involving the removed edge.
        # But for simplicity and speed with NumPy, full recalc is fast enough for 14k cycles.
        num_conflicts = check_conflicts()
        
        # If we removed too many, we might want to try recoloring instead?
        # The task asks to maximize edges. Removing is a safe fallback.
        # But maybe we can recolor?
        # For now, stick to removal to ensure validity.
        
    # Construct the result
    result_edges = []
    for i, (u, v) in enumerate(all_edges):
        if active[i]:
            result_edges.append([u, v, int(colors[i])])
            
    # Final verification
    col = {}
    for u, v, c in result_edges:
        col[(min(u, v), max(u, v))] = c
        
    valid = True
    # Only check 4-cycles that are fully present in the result graph
    # We can reuse the cycles list
    for cyc in cycles:
        e_idx = cyc
        e1 = e_idx[0]
        e2 = e_idx[1]
        e3 = e_idx[2]
        e4 = e_idx[3]
        
        if not (active[e1] and active[e2] and active[e3] and active[e4]):
            continue
            
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        
        if len({c1, c2, c3, c4}) != 4:
            valid = False
            break
            
    if not valid:
        # Fallback: return a known valid sparse graph if greedy failed
        # The 1-factorization from the seed is a good fallback
        edges = []
        for r in range(19):
            colour = r
            edges.append([19, r, colour])
            for i in range(1, 10):
                u = (r + i) % 19
                v = (r - i) % 19
                edges.append([u, v, colour])
        # Recolor one edge
        for i, e in enumerate(edges):
            if e[0] == 19 and e[1] == 0:
                edges[i][2] = 19
                break
        return edges

    return result_edges
# EVOLVE-BLOCK-END
