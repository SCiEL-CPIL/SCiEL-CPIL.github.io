import time
import random
import sys

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin

    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Precompute all 4-cycles: for each 4-vertex subset {a,b,c,d},
    # the three cycles are (a,b,c,d), (a,b,d,c), (a,c,b,d)
    # Each cycle is (e1, e2, e3, e4) where edges are (a,b),(b,c),(c,d),(d,a)
    # We need: for every 4-cycle, not all 4 edge colors are distinct.
    
    # Build list of 4-cycles. For vertices a<b<c<d:
    # Cycle 1: (a,b,c,d) -> edges (a,b),(b,c),(c,d),(d,a)
    # Cycle 2: (a,b,d,c) -> edges (a,b),(b,d),(d,c),(c,a)
    # Cycle 3: (a,c,b,d) -> edges (a,c),(c,b),(b,d),(d,a)
    
    four_cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    four_cycles.append(((a,b), (b,c), (c,d), (d,a)))
                    # Cycle 2: a-b-d-c-a
                    four_cycles.append(((a,b), (b,d), (d,c), (c,a)))
                    # Cycle 3: a-c-b-d-a
                    four_cycles.append(((a,c), (c,b), (b,d), (d,a)))
    
    num_cycles = len(four_cycles)
    
    # For each edge, precompute which cycles contain it
    edge_to_cycles = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_cycles[(i,j)] = []
    
    for ci, cycle in enumerate(four_cycles):
        for e in cycle:
            e_norm = (min(e[0], e[1]), max(e[0], e[1]))
            edge_to_cycles[e_norm].append(ci)
    
    # Edge ordering: use a fixed order (lexicographic)
    all_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            all_edges.append((i, j))
    
    num_edges = len(all_edges)
    
    # For backtracking: assign edges one by one
    # State: colors array, used_colors set
    # Constraint: for any 4-cycle where 3 edges are assigned, the 4th must not create 4 distinct colors
    # More generally: for any 4-cycle, if all 4 edges are assigned, they must not have 4 distinct colors
    
    # Efficient constraint checking during assignment:
    # When assigning edge e to color c, check all cycles containing e:
    #   If the other 3 edges in the cycle are already assigned, verify not all 4 are distinct
    #   If some are unassigned, we can't check yet (defer)
    
    # For speed, track for each cycle how many edges are assigned and what colors
    # cycle_state[ci] = (num_assigned, color_mask or list)
    
    # Actually, let's use a simpler approach:
    # When we assign edge e to color c, for each cycle containing e:
    #   Check if all other 3 edges are assigned
    #   If yes, verify the 4 colors are not all distinct
    
    # Let's also track: for each cycle, the set of assigned colors (to detect when 3 distinct colors are already present)
    # If a cycle already has 3 distinct colors among assigned edges, the 4th edge (if unassigned) must match one of them
    
    # Domain reduction: for each unassigned edge, its domain is reduced by cycles where 3 edges are assigned with 3 distinct colors
    
    # Let's implement with:
    # - colors: list of -1 (unassigned) or color value
    # - For each cycle, track assigned colors
    
    # For efficiency with 120 edges and ~3264 cycles, we need fast checks
    
    # Precompute: for each edge index, the list of (cycle_index, position_in_cycle)
    edge_idx_to_cycles = {}
    for ei, (i, j) in enumerate(all_edges):
        edge_idx_to_cycles[ei] = edge_to_cycles[(i,j)]
    
    # For each cycle, store the 4 edge indices (normalized)
    cycle_edges = []
    for cycle in four_cycles:
        edges_in_cycle = []
        for e in cycle:
            e_norm = (min(e[0], e[1]), max(e[0], e[1]))
            # Find edge index
            ei = get_idx(e_norm[0], e_norm[1])
            edges_in_cycle.append(ei)
        cycle_edges.append(edges_in_cycle)
    
    # Now implement backtracking with MRV
    # We'll use a recursive approach but with iteration limit
    
    # Actually, for 120 edges, full backtracking might be too slow.
    # Let's use a hybrid: greedy assignment with constraint propagation, then local search.
    
    # Better approach: sequential assignment with constraint checking
    # Assign edges in a smart order, try colors in "new color first" order
    
    # Let's use a different strategy: 
    # 1. Start with all edges uncolored
    # 2. Repeatedly pick the edge with most constraints (most cycles where other edges are colored)
    # 3. Try to assign a new color first, then existing colors
    # 4. If stuck, backtrack
    
    # For speed, let's do a simpler greedy + repair approach:
    
    # Strategy: Assign edges in order, for each edge try new colors first (to maximize diversity)
    # If no new color works, try existing colors
    
    colors = [-1] * num_edges
    used_colors = set()
    
    # For fast constraint checking, maintain:
    # cycle_assigned[ci] = list of (edge_idx, color) for assigned edges in cycle ci
    # But this is expensive. Instead, just check on the fly.
    
    # For each cycle, track how many edges are assigned and the set of colors
    cycle_num_assigned = [0] * num_cycles
    cycle_colors = [set() for _ in range(num_cycles)]
    
    # For each edge, track which cycles it's in
    # When assigning edge ei to color c:
    #   For each cycle ci containing ei:
    #     cycle_num_assigned[ci] += 1
    #     cycle_colors[ci].add(c)
    #     If cycle_num_assigned[ci] == 4:
    #       if len(cycle_colors[ci]) == 4: INVALID
    
    # Also need to check: if cycle has 3 assigned edges with 3 distinct colors,
    # the 4th edge must use one of those 3 colors (domain restriction)
    
    # Let's implement a sequential greedy with backtracking
    
    # Edge ordering: process edges in an order that maximizes constraint propagation
    # Use a dynamic ordering: always pick the edge that is in the most "critical" cycles
    # (cycles with 3 edges already assigned)
    
    # For simplicity, let's use a fixed order but with smart color selection
    
    # Fixed edge order: lexicographic
    # Color order: prefer new colors (higher numbers not yet used), then existing
    
    # Actually, let's try a different structural approach:
    # Use the fact that if we partition vertices into groups and use group-based coloring,
    # we can avoid rainbow cycles more easily.
    
    # But the directive says to use constraint satisfaction with backtracking.
    
    # Let me implement a proper backtracking with MRV:
    
    # State:
    # - colors: array of length num_edges, -1 if unassigned
    # - For each cycle: set of assigned colors, count of assigned edges
    
    # MRV: pick unassigned edge with smallest domain
    # Domain of edge e: colors c such that for every cycle containing e,
    #   if the other 3 edges are assigned and have 3 distinct colors, c must be in that set
    
    # This is complex. Let's simplify:
    # Just do sequential assignment with immediate constraint checking.
    # If we get stuck, backtrack.
    
    # For 120 edges, even with backtracking, we need to be careful about time.
    
    # Let's try: sequential assignment, no backtracking initially, but with smart color choice.
    # Then use local search to improve.
    
    # Actually, the key insight from the knowledge is: "Backtracking with MRV and 'new color first' 
    # heuristic explores the solution space more effectively than greedy"
    
    # Let me implement a limited backtracking:
    
    # We'll assign edges one by one. For each edge, try colors in order:
    # 1. New colors (not yet used) - to maximize diversity
    # 2. Existing colors
    
    # If no color works for an edge, backtrack to the previous edge.
    
    # To keep it fast, limit backtracking depth.
    
    # Let's implement:
    
    max_backtrack_depth = 50  # Limit backtracking to keep time bounded
    
    def try_assign(ei, depth, backtrack_count):
        """Try to assign edge ei. Return True if successful (and all remaining edges can be assigned)."""
        if time.time() > deadline - 5:
            return False
        
        if ei == num_edges:
            return True
        
        # Get domain: try new colors first, then existing
        new_colors = [c for c in range(120) if c not in used_colors]
        existing_colors = list(used_colors)
        
        # Shuffle for diversity
        random.shuffle(new_colors)
        random.shuffle(existing_colors)
        
        domain = new_colors + existing_colors
        
        for c in domain:
            if time.time() > deadline - 5:
                return False
            
            # Check if assigning colors[ei] = c is valid
            valid = True
            affected_cycles = []
            
            for ci in edge_idx_to_cycles[ei]:
                # Check this cycle
                edges_in_cycle = cycle_edges[ci]
                cycle_col_set = set()
                for e2 in edges_in_cycle:
                    if e2 == ei:
                        cycle_col_set.add(c)
                    else:
                        col2 = colors[e2]
                        if col2 != -1:
                            cycle_col_set.add(col2)
                        else:
                            # Unassigned, can't check fully yet
                            pass
                
                # If all 4 edges are now assigned (including this one), check
                all_assigned = True
                for e2 in edges_in_cycle:
                    if e2 != ei and colors[e2] == -1:
                        all_assigned = False
                        break
                
                if all_assigned and len(cycle_col_set) == 4:
                    valid = False
                    break
            
            if valid:
                # Assign
                colors[ei] = c
                used_colors.add(c)
                
                # Recurse
                if try_assign(ei + 1, depth + 1, backtrack_count):
                    return True
                
                # Unassign
                colors[ei] = -1
                if len(used_colors) > 0:
                    # Check if c is still used by another edge
                    still_used = False
                    for e2 in range(num_edges):
                        if e2 != ei and colors[e2] == c:
                            still_used = True
                            break
                    if not still_used:
                        used_colors.discard(c)
                
                # Backtrack limit
                if depth > max_backtrack_depth:
                    return False
        
        return False
    
    # Initialize random seed
    random.seed(42)
    
    # Try the backtracking approach
    success = try_assign(0, 0, 0)
    
    if not success:
        # Fallback: simple valid coloring
        colors = [0] * num_edges
        for i in range(N):
            for j in range(i + 1, N):
                colors[get_idx(i, j)] = i % 7
        used_colors = set(colors)
    
    # Local search to add more colors
    if time.time() < deadline - 10:
        ec = [[-1] * N for _ in range(N)]
        for ei, (i, j) in enumerate(all_edges):
            c = colors[ei]
            ec[i][j] = c
            ec[j][i] = c
        
        attempts = 0
        max_attempts = 200000
        
        while time.time() < deadline - 3 and attempts < max_attempts:
            attempts += 1
            
            # Pick a random edge
            edge_idx = random.randint(0, num_edges - 1)
            u, v = all_edges[edge_idx]
            old_color = colors[edge_idx]
            
            # Try to change to a new color
            unused = [c for c in range(120) if c not in used_colors]
            if not unused:
                break
            
            new_color = random.choice(unused)
            
            # Check if changing this edge to new_color keeps validity
            # Check all 4-cycles containing this edge
            valid = True
            for x in range(N):
                if x == u or x == v:
                    continue
                cx = ec[u][x]
                for y in range(N):
                    if y == u or y == v or y == x:
                        continue
                    cy = ec[x][y]
                    cz = ec[y][v]
                    if len({new_color, cx, cy, cz}) == 4:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                colors[edge_idx] = new_color
                ec[u][v] = new_color
                ec[v][u] = new_color
                used_colors.add(new_color)
    
    # Convert to required format
    result = [0] * (N * (N - 1) // 2)
    for ei, (i, j) in enumerate(all_edges):
        result[get_idx(i, j)] = colors[ei]
    
    # Final verification
    ec_final = [[-1] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            c = result[get_idx(i, j)]
            ec_final[i][j] = c
            ec_final[j][i] = c
    
    valid_final = True
    for i in range(N):
        if not valid_final:
            break
        for j in range(i + 1, N):
            if not valid_final:
                break
            c1 = ec_final[i][j]
            for x in range(N):
                if x == i or x == j:
                    continue
                c2 = ec_final[i][x]
                for y in range(N):
                    if y == i or y == j or y == x:
                        continue
                    c3 = ec_final[x][y]
                    c4 = ec_final[y][j]
                    if len({c1, c2, c3, c4}) == 4:
                        valid_final = False
                        break
                if not valid_final:
                    break
            if not valid_final:
                break
        if not valid_final:
            break
    
    if not valid_final:
        # Fallback to simple valid coloring
        result = [0] * (N * (N - 1) // 2)
        for i in range(N):
            for j in range(i + 1, N):
                result[get_idx(i, j)] = i % 7
    
    return result
# EVOLVE-BLOCK-END
