import time
import random
from random import Random
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

DEADLINE_MARGIN = 10  # seconds before 150s

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def build_adj(edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def find_2_factor(edges):
    """
    Find a 2-factor (collection of edge-disjoint cycles covering all vertices 
    with degree 2 in the subgraph) using Minimum Weight Perfect Matching on a 
    bipartite graph representation.
    
    Each undirected edge (u, v) in the original graph corresponds to two directed 
    arcs (u, v) and (v, u) in the bipartite graph.
    The left nodes are u_out, right nodes are v_in.
    A matching of size N (where N is the number of vertices) corresponds to a 2-factor.
    However, we only have edges that exist in the graph.
    
    Actually, a 2-factor in a general graph G is a spanning subgraph where every 
    vertex has degree exactly 2. This is equivalent to a perfect matching in the 
    "bipartite double cover" of G, but with the constraint that we cannot select 
    both (u, v) and (v, u) if we want simple cycles? No, a 2-factor is a set of 
    cycles. In the bipartite matching formulation for 2-factors:
    Create bipartite graph (U, V, E') where U = {u_out | u in V(G)}, V = {v_in | v in V(G)}.
    For each undirected edge {u, v} in G, add directed edges u_out -> v_in and v_out -> u_in.
    A perfect matching in this bipartite graph corresponds to a set of cycles covering all vertices.
    Note: If the graph is not 2-regular, a perfect matching might not exist.
    We want to maximize the number of vertices covered by cycles? 
    The problem asks for decomposition into fewest pieces.
    Pieces are cycles or single edges.
    Let k be the number of cycles, m be the number of single edges.
    Total pieces = k + m.
    Total edges E = sum(len(cycle)) + m.
    If we find a 2-factor (covering all N vertices), it consists of some cycles. 
    The number of pieces would be (number of cycles in 2-factor) + (E - N) [since 2-factor uses N edges].
    Wait, a 2-factor uses exactly N edges (one per vertex out, one per vertex in).
    So if we find a 2-factor, we cover N edges. The remaining E - N edges must be covered by single edges or other cycles?
    No, the remaining edges are just edges. They can form more cycles or be single edges.
    
    Strategy:
    1. Find a 2-factor (maximal set of vertex-disjoint cycles covering all vertices). 
       This is possible if and only if the graph has a 2-factor.
       For random graphs G(n, p), with n=20..30 and p=0.3..0.4, the average degree is ~6-10. 
       It is very likely that a 2-factor exists.
    2. Remove the edges of the 2-factor.
    3. The remaining graph has N vertices and E-N edges.
    4. Recursively or greedily decompose the remaining graph.
       However, the remaining graph might not have a 2-factor.
       
    Better Strategy:
    We want to minimize pieces.
    A cycle of length L counts as 1 piece and uses L edges.
    A single edge counts as 1 piece and uses 1 edge.
    Ratio of edges per piece: Cycle is L/1, Single is 1/1.
    We want to maximize the number of edges covered by cycles.
    
    Let's try to find the maximum set of edge-disjoint cycles.
    This is the "Maximum Cycle Packing" problem, which is NP-hard.
    However, finding a 2-factor is a good approximation or a first step.
    
    If we find a 2-factor, we have covered N edges with C cycles (where C is the number of cycles in the 2-factor).
    Remaining edges: E - N.
    We can try to find more cycles in the remaining graph.
    
    Let's implement a greedy approach that uses 2-factor extraction repeatedly if possible, 
    otherwise falls back to longest cycle or single edges.
    
    Actually, a simpler structural change: 
    Use the 2-factor to get a large set of cycles. 
    Then, on the remaining edges, just take single edges? 
    No, we can find more cycles.
    
    Let's refine:
    1. Find a 2-factor in the current graph.
    2. Extract those cycles as pieces.
    3. Remove those edges.
    4. Repeat on the remaining graph.
    5. When no 2-factor exists (or matching fails), find longest cycle greedily, then single edges.
    """
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    
    if len(nodes) < 3:
        return []
    
    # Build bipartite graph for matching
    # Left side: u_out, Right side: v_in
    # We want a perfect matching on the set of nodes present in the graph.
    # But a 2-factor must cover ALL vertices in the graph? 
    # Yes, a 2-factor is a spanning subgraph.
    # If the graph is not 2-regular, a perfect matching in the bipartite double cover 
    # corresponds to a 2-factor IF the matching is perfect on the set of vertices.
    
    # Let N = len(nodes).
    # We create a cost matrix of size N x N.
    # Cost is 0 if edge exists, INF otherwise.
    # We want to minimize cost, i.e., find a perfect matching using only existing edges.
    
    node_list = sorted(list(nodes))
    node_to_idx = {u: i for i, u in enumerate(node_list)}
    N = len(node_list)
    
    # Initialize cost matrix with large value (INF)
    # We use a large number as INF for linear_sum_assignment
    INF = 10**9
    cost_matrix = np.full((N, N), INF, dtype=int)
    
    # Fill in edges
    for u, v in edges:
        i = node_to_idx[u]
        j = node_to_idx[v]
        # u_out -> v_in
        cost_matrix[i, j] = 0
        # v_out -> u_in
        cost_matrix[j, i] = 0
    
    # Solve assignment
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except Exception:
        return []
    
    # Check if it's a valid perfect matching (all assigned costs are 0)
    if any(cost_matrix[r, c] == INF for r, c in zip(row_ind, col_ind)):
        return []
    
    # Reconstruct cycles from the matching
    # The matching gives a permutation P where u -> P(u).
    # This permutation decomposes into disjoint cycles.
    visited = [False] * N
    cycles = []
    
    for i in range(N):
        if not visited[i]:
            cycle_nodes = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle_nodes.append(node_list[curr])
                curr = col_ind[row_ind[curr]] # Wait, row_ind and col_ind are aligned.
                # Actually, linear_sum_assignment returns row_ind, col_ind such that 
                # matching is (row_ind[k], col_ind[k]).
                # So for row i, the matched col is col_ind[k] where row_ind[k] == i.
                # It's easier to build a dict.
            
    # Let's rebuild properly
    match_dict = {}
    for r, c in zip(row_ind, col_ind):
        match_dict[r] = c
        
    visited = [False] * N
    cycles = []
    for i in range(N):
        if not visited[i]:
            cycle_nodes = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle_nodes.append(node_list[curr])
                curr = match_dict[curr]
            
            # A cycle of length 1 or 2 in the permutation corresponds to 
            # a self-loop or 2-cycle in the directed sense, which is a single undirected edge.
            # We only want cycles of length >= 3 in the original graph.
            if len(cycle_nodes) >= 3:
                cycles.append(cycle_nodes)
            # If len < 3, it's just an edge (u,v). We can treat it as a single edge piece 
            # or ignore it here and let the remaining edges logic handle it?
            # The 2-factor uses these edges. If we ignore them, they are not "covered" by a cycle piece.
            # But they are part of the 2-factor.
            # A 2-cycle in the permutation (u->v, v->u) means edges (u,v) and (v,u) are used.
            # In an undirected graph, this is just the single edge (u,v).
            # So a 2-cycle in the matching corresponds to a single edge in the graph.
            # We should record these as single edge pieces? Or just remove them and treat as remaining?
            # If we treat them as single edge pieces, they count as 1 piece.
            # If we leave them in "remaining", they will be found as single edges later.
            # It's better to extract them as single edges now to reduce the graph size.
            elif len(cycle_nodes) == 2:
                cycles.append(cycle_nodes) # Mark as single edge later
            elif len(cycle_nodes) == 1:
                # Self loop? Not possible in simple graph unless multigraph. 
                # In simple graph, u->u implies edge (u,u) which doesn't exist.
                # So this shouldn't happen if cost_matrix[i,i] is INF.
                pass

    return cycles

def decompose_graph(edges, max_time=40.0):
    start_time = time.time()
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    while remaining:
        if time.time() - start_time > max_time:
            break
            
        # Try to find a 2-factor
        cycles = find_2_factor(remaining)
        
        if cycles:
            # Filter and add cycles
            added_any = False
            edges_to_remove = set()
            for c in cycles:
                if len(c) >= 3:
                    # It's a proper cycle
                    pieces.append(c)
                    # Add edges to remove
                    for i in range(len(c)):
                        u = c[i]
                        v = c[(i+1) % len(c)]
                        e = (u, v) if u < v else (v, u)
                        edges_to_remove.add(e)
                    added_any = True
                elif len(c) == 2:
                    # It's a single edge
                    u, v = c
                    e = (u, v) if u < v else (v, u)
                    if e in remaining:
                        pieces.append([u, v])
                        edges_to_remove.add(e)
                        added_any = True
            
            if added_any:
                for e in edges_to_remove:
                    remaining.discard(e)
                continue
            else:
                # No valid cycles found, fallback
                pass
        
        # Fallback: Find longest cycle greedily or take single edge
        # To save time, just take single edges if 2-factor failed or found nothing useful
        # Or do a quick greedy cycle search
        adj = build_adj(list(remaining))
        nodes_in_rem = set([u for (u,v) in remaining] + [v for (u,v) in remaining])
        
        found_cycle = None
        # Simple greedy: try to find a cycle starting from a random node
        # This is expensive, so we limit it.
        # For now, just take single edges to ensure progress and correctness.
        # The 2-factor step should have handled most of the structure.
        
        if remaining:
            e = next(iter(remaining))
            pieces.append([e[0], e[1]])
            remaining.discard(e)
            
    # Add any remaining edges as single pieces
    for e in remaining:
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    start = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for spec in specs:
        edges = instance(*spec)
        cur_time = time.time()
        remaining_time = max(1.0, DEADLINE_MARGIN - (cur_time - start))
        
        pieces = decompose_graph(edges, max_time=remaining_time)
        results.append(pieces)
        
        cur_time = time.time()
        if cur_time - start > DEADLINE_MARGIN - 3:
            # Out of time, fill rest with single edges
            for remaining_spec in specs[specs.index(spec) + 1:]:
                e = instance(*remaining_spec)
                results.append([list(e) for e in e])
            break
    
    while len(results) < 3:
        results.append([])
        
    return results[:3]
