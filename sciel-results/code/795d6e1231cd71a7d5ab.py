"""Combined approach: structural precomputation + delta-based local search."""
import numpy as np
import time
import random
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles in K_n
    # For 4 sorted vertices (a<b<c<d), three distinct 4-cycles:
    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    all_4cycles = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        all_4cycles.append((a, b, c, d))  # a-b-c-d-a
        all_4cycles.append((a, b, d, c))  # a-b-d-c-a
        all_4cycles.append((a, c, b, d))  # a-c-b-d-a
    
    # For each cycle, precompute the 4 edges as (min, max) tuples
    cycle_edges = []
    for cyc in all_4cycles:
        u, v, w, x = cyc
        edges = [(min(u,v), max(u,v)), (min(v,w), max(v,w)),
                 (min(w,x), max(w,x)), (min(x,u), max(x,u))]
        cycle_edges.append(edges)
    
    num_cycles = len(all_4cycles)
    
    # Build edge_to_cycles: maps each edge to list of cycle indices containing it
    edge_to_cycles = {}
    for i, edges in enumerate(cycle_edges):
        for e in edges:
            if e not in edge_to_cycles:
                edge_to_cycles[e] = []
            edge_to_cycles[e].append(i)
    
    # Precompute for each cycle the 4 edge positions
    # cycle_edge_idx[i] = list of 4 (u, v) pairs for cycle i
    # Already in cycle_edges
    
    # State
    adj = np.zeros((n, n), dtype=np.uint8)
    colors = np.zeros((n, n), dtype=np.int32)
    
    # Initialize: start with complete graph, random colors
    for u in range(n):
        for v in range(u + 1, n):
            adj[u, v] = 1
            adj[v, u] = 1
            c = random.randint(0, num_colors - 1)
            colors[u, v] = c
            colors[v, u] = c
    
    current_edges = n * (n - 1) // 2
    
    def count_bad_cycles(adj, colors):
        """Count 4-cycles where colors are not all distinct (only count existing cycles)."""
        bad = 0
        for i in range(num_cycles):
            edges = cycle_edges[i]
            # Check if all 4 edges exist
            e0, e1, e2, e3 = edges
            if not (adj[e0[0], e0[1]] and adj[e1[0], e1[1]] and 
                    adj[e2[0], e2[1]] and adj[e3[0], e3[1]]):
                continue
            c1 = colors[e0[0], e0[1]]
            c2 = colors[e1[0], e1[1]]
            c3 = colors[e2[0], e2[1]]
            c4 = colors[e3[0], e3[1]]
            if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                bad += 1
        return bad
    
    def get_bad_cycles_list(adj, colors):
        """Return list of bad 4-cycle indices."""
        bad = []
        for i in range(num_cycles):
            edges = cycle_edges[i]
            e0, e1, e2, e3 = edges
            if not (adj[e0[0], e0[1]] and adj[e1[0], e1[1]] and 
                    adj[e2[0], e2[1]] and adj[e3[0], e3[1]]):
                continue
            c1 = colors[e0[0], e0[1]]
            c2 = colors[e1[0], e1[1]]
            c3 = colors[e2[0], e2[1]]
            c4 = colors[e3[0], e3[1]]
            if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                bad.append(i)
        return bad
    
    current_bad = count_bad_cycles(adj, colors)
    
    best_bad = current_bad
    best_edges = current_edges
    best_adj = adj.copy()
    best_colors = colors.copy()
    
    # Phase 1: Greedy edge removal to eliminate all bad cycles
    while current_bad > 0 and time.time() < deadline - 10:
        bad_cycles = get_bad_cycles_list(adj, colors)
        if not bad_cycles:
            current_bad = 0
            break
        
        # Count how many bad cycles each edge participates in
        edge_count = Counter()
        for idx in bad_cycles:
            edges = cycle_edges[idx]
            for e in edges:
                edge_count[e] += 1
        
        if not edge_count:
            break
            
        # Remove the most frequent edge in bad cycles
        max_edge = max(edge_count, key=edge_count.get)
        u, v = max_edge
        adj[u, v] = 0
        adj[v, u] = 0
        colors[u, v] = 0
        colors[v, u] = 0
        current_edges -= 1
        current_bad = count_bad_cycles(adj, colors)
        
        if current_bad < best_bad or (current_bad == best_bad and current_edges > best_edges):
            best_bad = current_bad
            best_edges = current_edges
            best_adj = adj.copy()
            best_colors = colors.copy()
    
    # Phase 2: Local search with delta evaluation for recoloring
    # Key insight: when recoloring edge e from c_old to c_new, we only need to
    # check cycles containing e. For each such cycle, we can compute the delta in bad count.
    
    def delta_recolor(u, v, new_c):
        """Compute change in bad cycle count if we recolor edge (u,v) to new_c.
        Returns (delta_bad, feasible) where feasible means the move doesn't create
        new bad cycles that weren't already bad."""
        e = (min(u, v), max(u, v))
        old_c = colors[u, v]
        if new_c == old_c:
            return 0, True
        
        delta = 0
        for idx in edge_to_cycles.get(e, []):
            edges = cycle_edges[idx]
            # Check if all 4 edges in this cycle exist
            e0, e1, e2, e3 = edges
            if not (adj[e0[0], e0[1]] and adj[e1[0], e1[1]] and 
                    adj[e2[0], e2[1]] and adj[e3[0], e3[1]]):
                continue
            
            # Get current colors of the 4 edges
            c1 = colors[e0[0], e0[1]]
            c2 = colors[e1[0], e1[1]]
            c3 = colors[e2[0], e2[1]]
            c4 = colors[e3[0], e3[1]]
            
            # Determine which position in the cycle is edge e
            # The cycle edges are (u,v), (v,w), (w,x), (x,u) in order
            # e is one of these 4 edges
            pos = -1
            for p in range(4):
                ep = edges[p]
                if ep[0] == min(u,v) and ep[1] == max(u,v):
                    pos = p
                    break
            
            if pos == -1:
                continue
            
            # Current rainbow status
            cur_colors = [c1, c2, c3, c4]
            cur_rainbow = len(set(cur_colors)) == 4
            
            # New colors
            new_colors = list(cur_colors)
            new_colors[pos] = new_c
            new_rainbow = len(set(new_colors)) == 4
            
            if not cur_rainbow and new_rainbow:
                delta -= 1  # fixed a bad cycle
            elif cur_rainbow and not new_rainbow:
                delta += 1  # created a new bad cycle
            # else: no change
        
        return delta, True
    
    # Phase 2: Iterative improvement via recoloring
    max_iter = 5000
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        if current_bad == 0:
            break
        
        bad_cycles = get_bad_cycles_list(adj, colors)
        if not bad_cycles:
            current_bad = 0
            break
        
        # Pick a random bad cycle
        idx = random.choice(bad_cycles)
        edges = cycle_edges[idx]
        
        best_delta = 0
        best_move = None
        
        # Try recoloring each edge in the cycle
        for e in edges:
            u, v = e
            old_c = colors[u, v]
            for new_c in range(num_colors):
                if new_c == old_c:
                    continue
                delta, _ = delta_recolor(u, v, new_c)
                if delta < best_delta:
                    best_delta = delta
                    best_move = (u, v, new_c)
        
        if best_move is not None and best_delta < 0:
            u, v, new_c = best_move
            colors[u, v] = new_c
            colors[v, u] = new_c
            current_bad += best_delta
            if current_bad < best_bad or (current_bad == best_bad and current_edges > best_edges):
                best_bad = current_bad
                best_edges = current_edges
                best_adj = adj.copy()
                best_colors = colors.copy()
        else:
            # No improving recoloring found, try removing an edge
            e = random.choice(edges)
            u, v = e
            adj[u, v] = 0
            adj[v, u] = 0
            colors[u, v] = 0
            colors[v, u] = 0
            current_edges -= 1
            current_bad = count_bad_cycles(adj, colors)
            if current_bad < best_bad or (current_bad == best_bad and current_edges > best_edges):
                best_bad = current_bad
                best_edges = current_edges
                best_adj = adj.copy()
                best_colors = colors.copy()
    
    # Phase 3: Try to add back removed edges
    # For each missing edge, check if it can be added with some valid color
    if time.time() < deadline - 2:
        all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_possible_edges)
        
        for u, v in all_possible_edges:
            if time.time() > deadline - 2:
                break
            if adj[u, v] == 1:
                continue
            
            # Try to find a valid color for this edge
            e = (u, v)
            found_color = None
            for new_c in range(num_colors):
                # Check all cycles containing this edge
                valid = True
                for idx in edge_to_cycles.get(e, []):
                    edges = cycle_edges[idx]
                    e0, e1, e2, e3 = edges
                    # Check if all other 3 edges exist
                    other_exist = True
                    for ep in edges:
                        if ep[0] == min(u,v) and ep[1] == max(u,v):
                            continue
                        if not adj[ep[0], ep[1]]:
                            other_exist = False
                            break
                    if not other_exist:
                        continue
                    
                    # Get colors of other 3 edges
                    other_colors = []
                    for ep in edges:
                        if ep[0] == min(u,v) and ep[1] == max(u,v):
                            continue
                        other_colors.append(colors[ep[0], ep[1]])
                    
                    # Check if other 3 are distinct
                    if len(set(other_colors)) < 3:
                        valid = False
                        break
                    
                    # Check if new_c is distinct from other colors
                    if new_c in other_colors:
                        valid = False
                        break
                
                if valid:
                    found_color = new_c
                    break
            
            if found_color is not None:
                adj[u, v] = 1
                adj[v, u] = 1
                colors[u, v] = found_color
                colors[v, u] = found_color
                current_edges += 1
                current_bad = count_bad_cycles(adj, colors)
                
                if current_bad == 0:
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj.copy()
                        best_colors = colors.copy()
                else:
                    # Revert
                    adj[u, v] = 0
                    adj[v, u] = 0
                    colors[u, v] = 0
                    colors[v, u] = 0
                    current_edges -= 1
    
    # Use best state found
    if best_bad > 0:
        # Fallback: use the best state we have, even if not perfectly valid
        # But the evaluator requires validity, so we need to ensure validity
        adj = best_adj.copy()
        colors = best_colors.copy()
        current_bad = count_bad_cycles(adj, colors)
        
        if current_bad > 0:
            # Remove edges until valid
            while current_bad > 0 and time.time() < deadline - 1:
                bad_cycles = get_bad_cycles_list(adj, colors)
                if not bad_cycles:
                    break
                edge_count = Counter()
                for idx in bad_cycles:
                    edges = cycle_edges[idx]
                    for e in edges:
                        edge_count[e] += 1
                if edge_count:
                    max_edge = max(edge_count, key=edge_count.get)
                    u, v = max_edge
                    adj[u, v] = 0
                    adj[v, u] = 0
                    colors[u, v] = 0
                    colors[v, u] = 0
                    current_edges -= 1
                    current_bad = count_bad_cycles(adj, colors)
                else:
                    break
    
    # Construct output
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 1:
                c = int(colors[u, v])
                result.append([u, v, c])
    
    return result
