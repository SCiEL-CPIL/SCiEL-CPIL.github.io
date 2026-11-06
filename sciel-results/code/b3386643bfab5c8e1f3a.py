import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator kills at 150s)

    n = 20
    n_colors = 20

    # 1. Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            idx = len(all_edges)
            all_edges.append((u, v))
            edge_to_idx[(u, v)] = idx
    num_edges = len(all_edges)

    # 2. Precompute all 4-cycles
    # A 4-cycle is a set of 4 distinct vertices {a,b,c,d} and a specific cycle order.
    # In a complete graph, there are 3 distinct 4-cycles for each set of 4 vertices.
    # Cycle 1: a-b-c-d-a  (edges: ab, bc, cd, da)
    # Cycle 2: a-b-d-c-a  (edges: ab, bd, dc, ca)
    # Cycle 3: a-c-b-d-a  (edges: ac, cb, bd, da)
    
    all_4cycles = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d
        e1 = edge_to_idx[(a, b)]
        e2 = edge_to_idx[(b, c)]
        e3 = edge_to_idx[(c, d)]
        e4 = edge_to_idx[(d, a)]
        all_4cycles.append((e1, e2, e3, e4))
        
        # Cycle 2: a-b-d-c
        e1 = edge_to_idx[(a, b)]
        e2 = edge_to_idx[(b, d)]
        e3 = edge_to_idx[(d, c)]
        e4 = edge_to_idx[(c, a)]
        all_4cycles.append((e1, e2, e3, e4))
        
        # Cycle 3: a-c-b-d
        e1 = edge_to_idx[(a, c)]
        e2 = edge_to_idx[(c, b)]
        e3 = edge_to_idx[(b, d)]
        e4 = edge_to_idx[(d, a)]
        all_4cycles.append((e1, e2, e3, e4))

    num_4cycles = len(all_4cycles)

    # 3. Precompute edge to cycles mapping
    edge_cycles = [[] for _ in range(num_edges)]
    for ci, cyc in enumerate(all_4cycles):
        for ei in cyc:
            edge_cycles[ei].append(ci)

    # State
    present = [False] * num_edges
    colors = [0] * num_edges
    viol = [False] * num_4cycles
    total_viol = 0
    edge_count = 0

    def is_cycle_rainbow(cyc):
        e1, e2, e3, e4 = cyc
        if not (present[e1] and present[e2] and present[e3] and present[e4]):
            return True
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        return len({c1, c2, c3, c4}) == 4

    # Initialize with empty graph
    best_edge_count = 0
    best_present = [False] * num_edges
    best_colors = [0] * num_edges

    # We will perform multiple restarts of a greedy construction algorithm
    # The greedy algorithm will try to add edges in a randomized order,
    # choosing a random valid color for each edge.
    
    while time.time() < deadline:
        # Reset state
        for i in range(num_edges):
            present[i] = False
            colors[i] = 0
        for i in range(num_4cycles):
            viol[i] = False
        total_viol = 0
        edge_count = 0

        # Create a list of edges and shuffle
        edges_list = list(range(num_edges))
        random.shuffle(edges_list)

        # Greedy construction
        for ei in edges_list:
            if time.time() > deadline - 1.0:
                break
                
            # Find a valid color for this edge
            valid_color = -1
            for c in range(n_colors):
                colors[ei] = c # Temporarily set to check
                all_ok = True
                for ci in edge_cycles[ei]:
                    if not is_cycle_rainbow(all_4cycles[ci]):
                        all_ok = False
                        break
                if all_ok:
                    valid_color = c
                    break
                else:
                    colors[ei] = 0 # Reset
                    
            if valid_color != -1:
                present[ei] = True
                colors[ei] = valid_color
                edge_count += 1
                
                # Update violations for cycles containing this edge
                # Actually, since we checked validity, no new violations should be created
                # if the previous state was valid. But let's be safe and update.
                for ci in edge_cycles[ei]:
                    if not is_cycle_rainbow(all_4cycles[ci]):
                        viol[ci] = True
                        total_viol += 1
                    else:
                        viol[ci] = False
                        # If it was True, we should decrement total_viol?
                        # Wait, if it was False before, it stays False.
                        # If it was True before, it becomes False, so total_viol -= 1.
                        # But we start from a valid state (empty graph), so all viol are False.
                        # Adding a valid edge cannot create a violation.
                        # So total_viol should remain 0.
                        pass
            else:
                # No valid color, skip this edge
                pass

        # Local Search / Repair phase
        # Try to add skipped edges?
        # Try to recolor to allow more edges?
        
        # Simple local search: try to recolor random edges to see if it unlocks new edges?
        # Or try to add edges that were skipped?
        
        # Let's do a simple pass to try to add edges again, maybe with different random choices?
        # No, the greedy pass already tried all edges.
        
        # Let's try a "swap" move: remove an edge, add another?
        # Or just recolor an existing edge to see if it allows a skipped edge to be added.
        
        # A better local search:
        # Pick a random edge that is NOT present.
        # Try to recolor some of its incident cycle edges to make it possible to add it.
        # This is complex.
        
        # Alternative: Simulated Annealing on top of the greedy result.
        # But we want to be fast.
        
        # Let's just record the best found so far.
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_present = present.copy()
            best_colors = colors.copy()

    # Build result
    result = []
    for ei in range(num_edges):
        if best_present[ei]:
            u, v = all_edges[ei]
            c = best_colors[ei]
            result.append([u, v, c])
            
    return result
