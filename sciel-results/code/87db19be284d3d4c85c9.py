"""Path decomposition of graphs using ILP (Minimum Cost Flow) formulation."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    best_results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: each edge as its own path
            edges = _instance(*spec)
            best_results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        result = _decompose_ilp(edges, deadline)
        if result is not None:
            best_results.append(result)
        else:
            # Fallback: each edge as its own path
            best_results.append([list(e) for e in edges])
    
    return best_results


def _instance(n, p, seed):
    """Generate the instance graph exactly as in the verifier."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_ilp(edges, deadline):
    """
    Decompose edges into minimum number of edge-disjoint simple paths using ILP.
    """
    if not edges:
        return []
        
    n = max(max(u, v) for u, v in edges) + 1 if edges else 0
    if n == 0:
        return []
        
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Nodes in ILP:
    # 0 to n-1: v_out
    # n to 2n-1: v_in
    # 2n: S
    # 2n+1: T
    # Total nodes: 2n + 2
    
    num_nodes = 2 * n + 2
    S = 2 * n
    T = 2 * n + 1
    
    # Arcs:
    # 1. Internal: v_in -> v_out (index n+v -> v)
    # 2. Edge arcs: u_out -> v_in (u -> n+v) for (u,v) in E
    # 3. Source arcs: S -> v_out (S -> v)
    # 4. Sink arcs: v_in -> T (n+v -> T)
    
    # We will create a variable for each arc.
    # Let's list all arcs.
    arcs = []
    arc_to_edge = {} # arc_index -> (u, v) original edge or None
    
    # 1. Internal arcs
    for v in range(n):
        arcs.append((n + v, v)) # v_in -> v_out
        
    # 2. Edge arcs
    # We need to map original edges to these arcs.
    # For each edge (u, v), we have two potential arcs: u_out -> v_in and v_out -> u_in
    # We will add both as variables.
    edge_arc_indices = {}
    for u, v in edges:
        arc1 = (u, n + v) # u_out -> v_in
        arc2 = (v, n + u) # v_out -> u_in
        idx1 = len(arcs)
        idx2 = len(arcs) + 1
        arcs.append(arc1)
        arcs.append(arc2)
        edge_arc_indices[(u, v)] = (idx1, idx2)
        
    # 3. Source arcs
    for v in range(n):
        arcs.append((S, v)) # S -> v_out
        
    # 4. Sink arcs
    for v in range(n):
        arcs.append((n + v, T)) # v_in -> T
        
    num_arcs = len(arcs)
    
    # Variables: x_i for each arc i. 0 <= x_i <= 1.
    # Objective: Minimize sum of source arcs.
    # Source arcs are the last n arcs? No.
    # Let's identify indices.
    # Internal: 0 to n-1
    # Edge arcs: n to n + 2*len(edges) - 1
    # Source arcs: n + 2*len(edges) to n + 2*len(edges) + n - 1
    # Sink arcs: n + 2*len(edges) + n to n + 2*len(edges) + 2n - 1
    
    start_src_idx = n + 2 * len(edges)
    start_sink_idx = start_src_idx + n
    
    c = np.zeros(num_arcs)
    for i in range(start_src_idx, start_src_idx + n):
        c[i] = 1.0
        
    # Constraints:
    # 1. Flow conservation at v_in (node n+v):
    #    Inflow from edge arcs (u_out -> v_in) = Outflow to internal arc (v_in -> v_out)
    #    Sum(x_{u_out, v_in}) - x_{v_in, v_out} = 0
    
    # 2. Flow conservation at v_out (node v):
    #    Inflow from internal arc (v_in -> v_out) + Inflow from Source (S -> v_out) = Outflow to edge arcs (v_out -> w_in) + Outflow to Sink (v_out -> T)?
    #    Wait, my arcs are:
    #    Internal: v_in -> v_out
    #    Edge: u_out -> v_in
    #    Source: S -> v_out
    #    Sink: v_in -> T
    
    # Let's re-verify the flow direction.
    # Path: S -> u_out -> u_in -> v_in -> v_out -> w_in ... -> T
    # Arcs used:
    # S -> u_out (Source)
    # u_out -> u_in (Internal? No, Internal is v_in -> v_out)
    # My Internal arc is v_in -> v_out.
    # So flow goes: ... -> u_out -> u_in (via edge arc from prev) -> u_out (via internal)?
    # No, if Internal is v_in -> v_out, then:
    # u_out receives from S or from w_in (via edge w-u)?
    # Edge arc is w_out -> u_in.
    # So u_in receives from w_out.
    # u_in sends to u_out (Internal).
    # u_out sends to v_in (Edge arc u-v) or T? No, Sink is v_in -> T.
    # So u_out sends to v_in.
    
    # Let's check conservation at v_in (node n+v):
    # In: From edge arcs u_out -> v_in (where (u,v) in E).
    # Out: To internal arc v_in -> v_out.
    # Constraint: Sum(x_{u_out, v_in}) - x_{v_in, v v_out} = 0.
    
    # Conservation at v_out (node v):
    # In: From internal arc v_in -> v_out. From Source S -> v_out.
    # Out: To edge arcs v_out -> w_in (where (v,w) in E). To Sink? No, Sink is v_in -> T.
    # Wait, if Sink is v_in -> T, then the path ends at v_in.
    # Path: ... -> u_out -> v_in -> T.
    # This skips v_out.
    # Is that valid?
    # If we end at v_in, we used edge u-v. We didn't use v_out.
    # This is fine. The path is simple.
    
    # So Conservation at v_out:
    # In: x_{v_in, v_out} + x_{S, v_out}
    # Out: Sum(x_{v_out, w_in})
    # Constraint: x_{v_in, v_out} + x_{S, v_out} - Sum(x_{v_out, w_in}) = 0.
    
    # Conservation at S:
    # Out: Sum(x_{S, v_out})
    # In: 0
    # No constraint needed if we just minimize sum.
    
    # Conservation at T:
    # In: Sum(x_{v_in, T})
    # Out: 0
    
    # Mandatory Edge Constraint:
    # For each edge (u, v), x_{u_out, v_in} + x_{v_out, u_in} = 1.
    
    A_rows = []
    b_lower = []
    b_upper = []
    
    # 1. Flow conservation at v_in (n+v)
    for v in range(n):
        row = np.zeros(num_arcs)
        # Outflow: internal arc v_in -> v_out (index v)
        row[v] = -1.0
        
        # Inflow: edge arcs u_out -> v_in
        for u in adj[v]:
            # Find index of arc (u, n+v)
            # We stored edge_arc_indices[(u,v)] or (v,u)
            if (u, v) in edge_arc_indices:
                idx1, _ = edge_arc_indices[(u, v)]
            else:
                _, idx1 = edge_arc_indices[(v, u)]
            row[idx1] = 1.0
            
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(0.0)
        
    # 2. Flow conservation at v_out (v)
    for v in range(n):
        row = np.zeros(num_arcs)
        # Inflow: internal arc v_in -> v_out (index v)
        row[v] = 1.0
        # Inflow: source arc S -> v_out (index start_src_idx + v)
        row[start_src_idx + v] = 1.0
        
        # Outflow: edge arcs v_out -> w_in
        for w in adj[v]:
            if (v, w) in edge_arc_indices:
                _, idx2 = edge_arc_indices[(v, w)]
            else:
                idx2, _ = edge_arc_indices[(w, v)]
            row[idx2] = -1.0
            
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(0.0)
        
    # 3. Mandatory edge constraints
    for (u, v), (idx1, idx2) in edge_arc_indices.items():
        row = np.zeros(num_arcs)
        row[idx1] = 1.0
        row[idx2] = 1.0
        A_rows.append(row)
        b_lower.append(1.0)
        b_upper.append(1.0)
        
    A = np.array(A_rows)
    b_lower = np.array(b_lower)
    b_upper = np.array(b_upper)
    
    constraints = LinearConstraint(A, b_lower, b_upper)
    integrality = np.ones(num_arcs)
    bounds = Bounds(0, 1)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    except Exception:
        return None
        
    if not res.success:
        return None
        
    x = res.x
    
    # Reconstruct paths
    # Build adjacency of the selected directed arcs
    # selected_arcs: list of (from, to)
    selected_arcs = []
    for i, (u, v) in enumerate(arcs):
        if x[i] > 0.5:
            selected_arcs.append((u, v))
            
    # Map node IDs back to original vertices
    # 0..n-1: v_out
    # n..2n-1: v_in
    # 2n: S
    # 2n+1: T
    
    # Build successor map
    # From v_out, where do we go?
    # From v_in, where do we go?
    
    succ_out = {}
    succ_in = {}
    
    for u, v in selected_arcs:
        if u < n: # u is v_out
            succ_out[u] = v
        elif u >= n and u < 2*n: # u is v_in
            succ_in[u] = v
        elif u == S:
            pass # Source
        # We don't need to track T explicitly for path building, but we can check termination
        
    # Identify paths and cycles
    # A path starts at S -> v_out.
    # A cycle is a loop not involving S or T.
    
    paths = []
    
    # Find starts
    for v in range(n):
        if succ_out.get(S, None) is not None: # S is not in succ_out keys? 
            # S -> v_out is arc (S, v). 
            # In selected_arcs, we have (S, v).
            pass
            
    # Let's just trace from S.
    # For each v, if (S, v) is selected, start a path.
    for v in range(n):
        # Check if arc (S, v) is selected
        # Find index of (S, v)
        # It is start_src_idx + v
        if x[start_src_idx + v] > 0.5:
            current_node = v # v_out
            path_vertices = [v]
            
            while True:
                # Move from current_node (which is v_out)
                # Go to v_in via internal arc
                # Check if internal arc (n+v, v) is selected?
                # Wait, my internal arc is (n+v, v). i.e., v_in -> v_out.
                # So flow is v_in -> v_out.
                # So from v_out, we came from v_in.
                # Where do we go from v_out?
                # To w_in via edge arc (v, n+w).
                
                next_in = succ_out.get(current_node)
                if next_in is None:
                    break # Should not happen if flow conservation holds and it's a path
                # next_in is w_in (index n+w)
                w = next_in - n
                path_vertices.append(w)
                
                # Now at w_in. Where do we go?
                # To w_out via internal arc (n+w, w).
                # Check if internal arc is selected.
                # Index of internal arc for w is w.
                if x[w] < 0.5:
                    # If internal arc is not selected, it must go to T?
                    # Sink arc is (n+w, T). Index start_sink_idx + w.
                    if x[start_sink_idx + w] > 0.5:
                        break # Path ends at w
                    else:
                        # Error?
                        break
                # If internal arc selected, go to w_out
                current_node = w
                
            paths.append(path_vertices)
            
    # Find cycles
    # Cycles are components not connected to S.
    # We can find them by looking for nodes not visited in paths.
    visited = set()
    for p in paths:
        visited.update(p)
        
    # Unvisited nodes might be in cycles.
    # We need to trace cycles in the graph of selected arcs.
    # Build a generic successor map for all nodes involved in selected arcs.
    # Nodes: 0..2n-1
    # Succ map: node -> next_node
    
    succ = {}
    for u, v in selected_arcs:
        if u < 2*n: # Not S or T
            succ[u] = v
            
    # Find cycles
    # We only care about cycles that consist of original vertices.
    # A cycle in the split graph corresponds to a cycle in the original graph.
    # Example: v_out -> w_in -> w_out -> x_in -> ... -> v_out
    
    unvisited_nodes = [v for v in range(2*n) if v not in visited]
    # Note: visited contains original vertices. We need to check both v_out and v_in.
    # If v is in a path, v_out and v_in are used.
    # If v is in a cycle, v_out and v_in are used.
    
    # Let's just find all cycles in the directed graph defined by succ.
    # Since it's a functional graph (each node has at most 1 successor), cycles are easy to find.
    
    seen = set()
    cycles = []
    
    for start_node in range(2*n):
        if start_node in seen or start_node not in succ:
            continue
            
        path_nodes = []
        curr = start_node
        while curr in succ and curr not in seen:
            path_nodes.append(curr)
            seen.add(curr)
            curr = succ[curr]
            
        # If curr is in path_nodes, we found a cycle
        if curr in path_nodes:
            cycle_start_idx = path_nodes.index(curr)
            cycle_nodes = path_nodes[cycle_start_idx:]
            
            # Convert cycle nodes to original vertices
            # Cycle nodes are mixed v_out and v_in.
            # Extract unique original vertices.
            cycle_verts = set()
            for node in cycle_nodes:
                if node < n:
                    cycle_verts.add(node)
                else:
                    cycle_verts.add(node - n)
            
            # Break cycle into paths
            # A cycle of k edges becomes 2 paths.
            # We need to generate the actual path vertex lists.
            # Trace the cycle in original graph.
            
            # Get the sequence of original vertices in the cycle.
            # The cycle in split graph: v_out -> w_in -> w_out -> ...
            # This corresponds to edges (v,w), (w,x)...
            
            # Let's reconstruct the path of vertices.
            # Pick a starting vertex from cycle_verts.
            if cycle_verts:
                # We need the order.
                # Let's trace the cycle in the split graph and extract vertices.
                seq = []
                # Start from the first node in cycle_nodes
                curr = cycle_nodes[0]
                for _ in range(len(cycle_nodes)):
                    if curr < n:
                        seq.append(curr)
                    else:
                        seq.append(curr - n)
                    curr = succ[curr]
                    
                # seq contains vertices with duplicates at the end?
                # cycle_nodes is a cycle, so succ[last] = first.
                # The sequence of original vertices in the cycle is unique.
                
                # To break the cycle, pick an edge to remove.
                # Let's remove the edge between seq[-2] and seq[-1] (which is the same as seq[0]).
                # Actually, seq is the list of vertices in the cycle.
                # Path 1: seq[0] ... seq[k-1]
                # Path 2: seq[k-1] ... ?
                # Wait, if we remove edge (u,v), we have path u...v and path v...u?
                # No, we have one path covering k-1 edges and one path covering 1 edge?
                # Or two paths?
                # If we have cycle 1-2-3-1.
                # Remove (1,2).
                # Path 1: 2-3-1 (edges 2-3, 3-1)
                # Path 2: 1 (edge 1-2? No, 1-2 is removed).
                # Wait, we must cover all edges.
                # If we remove (1,2) from the cycle, we have path 2-3-1.
                # The edge (1,2) is still there? No, the cycle used (1,2).
                # If we break the cycle, we just stop the path.
                # The "cycle" was a set of edges.
                # If we decompose a cycle into paths, we can just take the cycle as a path? No, simple path.
                # We can take path 1-2-3 and path 3-1?
                # Path 1-2-3 uses edges (1,2), (2,3).
                # Path 3-1 uses edge (3,1).
                # This covers all edges. 2 paths.
                
                # So, generate:
                # Path A: seq[0] to seq[-1] (excluding the last edge? No, including all but one)
                # Let's say we cut after seq[-1] (which connects to seq[0]).
                # Path 1: seq[0] ... seq[-1]
                # Path 2: seq[-1] ... seq[0] ? No, that's the same edges.
                # We need to partition the edges.
                # Cut edge (seq[-1], seq[0]).
                # Path 1: seq[0] -> ... -> seq[-1]
                # Path 2: seq[-1] -> seq[0] (single edge)
                
                p1 = seq[:]
                p2 = [seq[-1], seq[0]]
                
                paths.append(p1)
                paths.append(p2)
                
    return paths

# EVOLVE-BLOCK-END
