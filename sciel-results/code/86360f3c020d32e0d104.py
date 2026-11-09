"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    n = 20
    num_colors = 20
    
    # All possible edges in complete graph
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    total_edges = len(all_edges)
    
    # Build adjacency set for fast lookup
    edge_set_full = set(all_edges)
    
    # Precompute all 4-cycles and their edge membership
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d in order: a-b-c-d-a
    # The edges are (a,b), (b,c), (c,d), (d,a)
    # We need to enumerate all unique 4-cycles
    
    # For each pair of non-adjacent-in-cycle vertices, we can find cycles
    # More systematically: for each set of 4 vertices {a,b,c,d}, there are 3 distinct 4-cycles
    # But we only care about cycles where all 4 edges exist in the graph
    
    # Precompute: for each edge, which 4-cycles (as frozensets of 4 edges) contain it
    # This is expensive to do fully, so let's use a different approach
    
    # Strategy: Use the bipartite-inspired matrix approach from inspiration 2
    # But generalize it: partition vertices into two sets of 10
    # Only consider edges between the two sets (bipartite subgraph)
    # In a bipartite graph, every 4-cycle is of the form u1-v1-u2-v2-u1
    # where u1,u2 are in set U and v1,v2 are in set V
    
    u_nodes = list(range(10))
    v_nodes = list(range(10, 20))
    
    # M[i, j] is the color of edge (u_nodes[i], v_nodes[j]), or -1 if edge not present
    # We'll work with a 10x10 matrix
    
    # Precompute all 4-cycles in the bipartite graph
    # A 4-cycle: (i1, j1, i2, j2) where i1 != i2, j1 != j2
    # Edges: (i1,j1), (i2,j1), (i2,j2), (i1,j2)
    cycle_list = []
    for i1 in range(10):
        for i2 in range(i1 + 1, 10):
            for j1 in range(10):
                for j2 in range(j1 + 1, 10):
                    cycle_list.append(((i1, j1), (i2, j1), (i2, j2), (i1, j2)))
    
    num_cycles = len(cycle_list)
    
    # For each cycle, store the 4 (row, col) positions
    cycle_edges_arr = np.array([(c[0][0], c[0][1], c[1][0], c[1][1], c[2][0], c[2][1], c[3][0], c[3][1]) for c in cycle_list])
    
    # For each cell (i,j), store which cycles contain it
    cell_to_cycles = defaultdict(list)
    for idx, cyc in enumerate(cycle_list):
        for (r, c) in cyc:
            cell_to_cycles[(r, c)].append(idx)
    
    # Convert to numpy arrays for faster access
    cell_to_cycles_arr = {}
    for (r, c), idxs in cell_to_cycles.items():
        cell_to_cycles_arr[(r, c)] = np.array(idxs)
    
    def count_bad_cycles(M):
        """Count number of non-rainbow 4-cycles. M is 10x10 int array, -1 means no edge."""
        # Only consider cycles where all 4 edges are present
        # Get colors for all cycles
        r1, c1, r2, c2, r3, c3, r4, c4 = cycle_edges_arr.T
        cols = np.column_stack((M[r1, c1], M[r2, c2], M[r3, c3], M[r4, c4]))
        
        # A cycle is valid only if all 4 edges are present (color != -1)
        # and all 4 colors are distinct
        all_present = (cols != -1).all(axis=1)
        
        # Check distinctness: for each row, check if all 4 values are different
        # c0 != c1, c0 != c2, c0 != c3, c1 != c2, c1 != c3, c2 != c3
        c0, c1, c2, c3 = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3]
        distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        
        # Bad cycles are those that are present but not distinct
        bad_mask = all_present & ~distinct
        return np.sum(bad_mask), bad_mask
    
    def get_edge_count(M):
        """Count number of edges (non -1 entries) in M."""
        return np.sum(M != -1)
    
    # Greedy construction with local search
    best_M = None
    best_edge_count = 0
    best_bad_count = float('inf')
    
    random.seed(42)
    
    # Phase 1: Try multiple random initializations with greedy addition
    num_attempts = 30
    for attempt in range(num_attempts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize M with all -1 (no edges)
        M = np.full((10, 10), -1, dtype=np.int32)
        
        # Randomly order all cells
        cells = [(i, j) for i in range(10) for j in range(10)]
        random.shuffle(cells)
        
        current_bad, _ = count_bad_cycles(M)
        
        for (i, j) in cells:
            if time.time() > deadline - 10.0:
                break
            
            # Try to add edge (i,j) with some color
            # Try colors in random order
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            added = False
            for color in colors:
                M[i, j] = color
                new_bad, _ = count_bad_cycles(M)
                if new_bad == 0:
                    added = True
                    break
                else:
                    M[i, j] = -1
            
            if not added:
                # Can't add this edge with any color without creating a bad cycle
                pass
        
        edge_count = get_edge_count(M)
        bad_count, _ = count_bad_cycles(M)
        
        if bad_count == 0 and edge_count > best_edge_count:
            best_edge_count = edge_count
            best_M = M.copy()
            best_bad_count = 0
    
    # Phase 2: Local search to improve the best solution found
    if best_M is not None:
        M = best_M.copy()
        current_bad, _ = count_bad_cycles(M)
        current_edges = get_edge_count(M)
        
        # Try to add more edges
        missing_cells = [(i, j) for i in range(10) for j in range(10) if M[i, j] == -1]
        random.shuffle(missing_cells)
        
        for (i, j) in missing_cells:
            if time.time() > deadline - 5.0:
                break
            
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            for color in colors:
                M[i, j] = color
                new_bad, _ = count_bad_cycles(M)
                if new_bad == 0:
                    current_edges += 1
                    break
                else:
                    M[i, j] = -1
        
        if current_edges > best_edge_count:
            best_edge_count = current_edges
            best_M = M.copy()
    
    # Phase 3: If we still have few edges, try a different approach
    # Use the local search from inspiration 2: start with full matrix and fix bad cycles
    if best_edge_count < 50:
        for attempt in range(20):
            if time.time() > deadline - 5.0:
                break
            
            # Start with a random full coloring
            M = np.random.randint(0, num_colors, size=(10, 10), dtype=np.int32)
            
            current_bad, current_bad_mask = count_bad_cycles(M)
            local_best = M.copy()
            local_best_edges = get_edge_count(M)
            local_best_bad = current_bad
            
            # Local search: pick a bad cycle, change one of its edges
            for step in range(5000):
                if time.time() > deadline - 5.0:
                    break
                
                if current_bad == 0:
                    break
                
                # Pick a random bad cycle
                bad_indices = np.where(current_bad_mask)[0]
                if len(bad_indices) == 0:
                    break
                target_idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
                
                # Get the 4 edges in this cycle
                r1, c1, r2, c2, r3, c3, r4, c4 = cycle_edges_arr[target_idx]
                edges_in_cycle = [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
                
                # Pick a random edge from the cycle to change
                r, c = edges_in_cycle[random.randint(0, 3)]
                old_color = M[r, c]
                
                improved = False
                # Try a few random new colors
                for _ in range(15):
                    new_color = random.randint(0, num_colors - 1)
                    if new_color == old_color:
                        continue
                    M[r, c] = new_color
                    new_bad, new_bad_mask = count_bad_cycles(M)
                    
                    if new_bad < current_bad:
                        current_bad = new_bad
                        current_bad_mask = new_bad_mask
                        if current_bad < local_best_bad:
                            local_best_bad = current_bad
                            local_best = M.copy()
                            local_best_edges = get_edge_count(M)
                        improved = True
                        break
                    else:
                        # Revert
                        M[r, c] = old_color
                
                if not improved:
                    # If stuck, do a random move
                    rr = random.randint(0, 9)
                    cc = random.randint(0, 9)
                    M[rr, cc] = random.randint(0, num_colors - 1)
                    current_bad, current_bad_mask = count_bad_cycles(M)
                    if current_bad < local_best_bad:
                        local_best_bad = current_bad
                        local_best = M.copy()
                        local_best_edges = get_edge_count(M)
            
            # Check if local best is valid
            final_bad, _ = count_bad_cycles(local_best)
            if final_bad == 0:
                edges = get_edge_count(local_best)
                if edges > best_edge_count:
                    best_edge_count = edges
                    best_M = local_best.copy()
    
    # If we still have a poor solution, try one more approach:
    # Start with a sparse valid graph and greedily add edges
    if best_M is None or best_edge_count < 30:
        M = np.full((10, 10), -1, dtype=np.int32)
        
        # Add a random spanning structure first
        # Connect all u-nodes to a few v-nodes
        for i in range(10):
            for j in range(10):
                if time.time() > deadline - 2.0:
                    break
                colors = list(range(num_colors))
                random.shuffle(colors)
                for color in colors:
                    M[i, j] = color
                    new_bad, _ = count_bad_cycles(M)
                    if new_bad == 0:
                        break
                    else:
                        M[i, j] = -1
        
        edge_count = get_edge_count(M)
        bad_count, _ = count_bad_cycles(M)
        if bad_count == 0 and edge_count > best_edge_count:
            best_edge_count = edge_count
            best_M = M.copy()
    
    # Final: if we have no valid solution, return a star (fallback)
    if best_M is None:
        # Return a star: vertex 0 connected to all others
        result = []
        for v in range(1, n):
            result.append([0, v, v % num_colors])
        return result
    
    # Convert M to edge list
    # Also try to add intra-set edges if they don't create 4-cycles
    # Intra-set edges (within U or within V) don't participate in bipartite 4-cycles
    # But they can create 4-cycles with other intra-set edges
    # For simplicity, let's first return the bipartite solution
    
    result = []
    for i in range(10):
        for j in range(10):
            if best_M[i, j] != -1:
                u = u_nodes[i]
                v = v_nodes[j]
                c = int(best_M[i, j])
                result.append([u, v, c])
    
    # Try to add intra-set edges
    # Edges within U: (u_nodes[i], u_nodes[k]) for i != k
    # Edges within V: (v_nodes[j], v_nodes[l]) for j != l
    # These edges don't form 4-cycles with bipartite edges alone
    # But two intra-U edges and two bipartite edges can form a 4-cycle
    # e.g., u1-u2 (intra), u2-v1 (bip), v1-v2 (intra V? no, v1,v2 in V)
    # Actually: u1-u2 (intra U), u2-v1 (bip), v1-v2 (intra V), v2-u1 (bip)
    # This is a 4-cycle! So we need to be careful.
    
    # For now, let's just return the bipartite solution
    # We could try to add intra-set edges with careful checking, but that's complex
    
    return result

# EVOLVE-BLOCK-END
