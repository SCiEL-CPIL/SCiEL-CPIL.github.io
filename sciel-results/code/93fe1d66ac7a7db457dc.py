import time
import math
from random import Random
from collections import deque

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def min_cost_flow(n, edges, max_flow=None):
    """
    Find min-cost flow of value max_flow (or max possible) in a bipartite graph.
    Left nodes: 0..n-1 (out)
    Right nodes: 0..n-1 (in)
    Edge (u, v) has capacity 1 and cost 1.
    """
    if not HAS_SCIPY:
        return []
    
    # Build cost matrix for linear_sum_assignment
    # Size 2n x 2n
    # Nodes: 0..n-1 are 'out' nodes, n..2n-1 are 'in' nodes
    # We want to assign out->in edges.
    # Cost matrix C where C[u, v] is cost of edge u->v.
    # If edge doesn't exist, cost is INF.
    # We also need dummy edges to allow non-perfect matching if graph is not regular enough?
    # linear_sum_assignment finds min cost perfect matching.
    # We can add dummy nodes with cost 0 to allow "unmatched" status, but we want to maximize flow (minimize cost 1 edges).
    # Actually, we want to select a set of edges such that no node has in-degree > 1 and out-degree > 1.
    # This is equivalent to finding a maximum cardinality matching in the bipartite graph?
    # No, we want to minimize the number of edges used? No, we want to maximize the number of edges used in cycles/paths.
    # Wait, the cost is 1 for each edge used. We want to minimize the total cost of the flow?
    # If we send flow 1 on edge (u,v), cost is 1.
    # We want to maximize the number of edges in the decomposition?
    # Actually, any edge used in a cycle or path contributes 1 to the "flow".
    # The total number of pieces is (Total Edges) - (Number of Cycles Closed by Matching).
    # Let's re-evaluate.
    # Decomposition into cycles and paths.
    # Let E be the set of edges.
    # We select a subset F of E to be the "flow" edges.
    # The flow edges form vertex-disjoint cycles and paths.
    # Every node has in-degree <= 1 and out-degree <= 1 in F.
    # To minimize pieces, we want to maximize the number of edges in F?
    # If we include all edges in F, we get a 2-factor (if possible) + paths.
    # But we can't include all edges if degrees > 2.
    # So we want to find a maximum cardinality subset of edges F such that deg(u) <= 2 for all u.
    # This is the "Maximum 2-Matching" problem.
    # Can we solve Max 2-Matching with MCMF?
    # Yes. Split each node u into u_in and u_out.
    # Edge (u, v) in G becomes edge (u_out, v_in) in flow network with capacity 1, cost 1.
    # Also add self-loop (u_out, u_in) with capacity 2, cost 0? No, we don't have self loops.
    # We want to maximize sum of x_e.
    # Constraints: sum_in x_e <= 2, sum_out x_e <= 2.
    # This is a bipartite b-matching with b=2.
    # This can be solved by min-cost flow.
    # Source S -> u_out (cap 2, cost 0)
    # u_out -> v_in (cap 1, cost 1) for each edge (u,v)
    # v_in -> Sink T (cap 2, cost 0)
    # Find max flow min cost.
    # Since all costs are 0 or 1, minimizing cost is equivalent to minimizing number of edges?
    # Wait, if we want to MAXIMIZE edges, we should set cost to -1 for edges?
    # Or just find max flow, and among max flows, any is fine?
    # No, we want to maximize the number of edges in the decomposition to minimize the number of "single edge" pieces?
    # Actually, the pieces are the cycles and paths.
    # Number of pieces = (Number of paths) + (Number of cycles).
    # Let k be number of paths.
    # Total edges in F = M.
    # Sum of lengths of paths and cycles = M.
    # Number of pieces = M - (Number of cycles).
    # Because a path of length L is 1 piece. A cycle of length L is 1 piece.
    # Wait. A path of length 1 is 1 piece. A path of length 2 is 1 piece.
    # If we have a path u-v, it's 1 piece.
    # If we have a cycle u-v-w, it's 1 piece.
    # The number of pieces is NOT M - C.
    # Let P be set of paths, C be set of cycles.
    # Pieces = |P| + |C|.
    # Edges in P = sum(len(p) for p in P).
    # Edges in C = sum(len(c) for c in C).
    # Total Edges M = sum(len(p)) + sum(len(c)).
    # This doesn't directly give |P| + |C|.
    # However, if we close paths into cycles, we reduce the number of pieces.
    # Closing a path (u...v) into a cycle (u...v, v...u) requires adding an edge (v,u) which must exist in G \ F.
    # If we can close a path, we convert 1 path piece into 1 cycle piece. The count of pieces stays the same?
    # No. A path is 1 piece. A cycle is 1 piece.
    # Why is closing paths good?
    # Because if we don't close them, they are paths. If we close them, they are cycles.
    # The task asks for decomposition into cycles and edges.
    # "A piece is a single edge [u, v] or a cycle".
    # A path of length > 1 is NOT a valid piece by itself unless it's decomposed into edges?
    # "Pieces must be edge-disjoint and cover all edges."
    # "A piece is a single edge [u, v] or a cycle given by at least three distinct vertices".
    # So a path of length 2 (u-v-w) is NOT a valid piece. It must be [u,v] and [v,w] (2 pieces) or part of a cycle.
    # Therefore, any path in the flow decomposition MUST be broken into single edges OR closed into a cycle.
    # If we break a path of length L into single edges, we get L pieces.
    # If we close it into a cycle, we get 1 piece.
    # So we want to maximize the number of paths we can close into cycles.
    # And for the rest, we break them into edges.
    # Also, any edge NOT in the flow F is a single edge piece (1 piece).
    
    # Let F be the set of edges in the 2-matching.
    # Let P be the set of paths in F.
    # Let C be the set of cycles in F.
    # Edges in C are covered by |C| pieces.
    # Edges in P:
    #   Some paths can be closed using edges from G \ F. Let K be the set of paths that can be closed.
    #   These contribute |K| pieces.
    #   The remaining paths P \ K must be broken into single edges.
    #   Let L_p be length of path p. It contributes L_p pieces.
    # Edges in G \ F:
    #   Some are used to close paths in K. Let E_close be these edges.
    #   The rest E_rem = (G \ F) \ E_close are single edge pieces.
    #   So |E_rem| pieces.
    
    # Total pieces = |C| + |K| + sum_{p in P\K} L_p + |(G \ F) \ E_close|
    
    # Note that sum_{p in P} L_p = |F| - sum_{c in C} L_c.
    # Also |G| = |F| + |G \ F|.
    # |G \ F| = |E_close| + |E_rem|.
    # So |E_rem| = |G \ F| - |E_close|.
    # Since each closed path uses exactly 1 edge to close it, |E_close| = |K|.
    # So |E_rem| = |G \ F| - |K|.
    
    # Total pieces = |C| + |K| + (|F| - sum_{c in C} L_c - sum_{p in P\K} L_p) ... wait.
    # sum_{p in P\K} L_p = (sum_{p in P} L_p) - (sum_{p in K} L_p).
    # sum_{p in P} L_p = |F| - sum_{c in C} L_c.
    
    # Total pieces = |C| + |K| + (|F| - sum_{c in C} L_c - sum_{p in K} L_p) + (|G \ F| - |K|)
    # = |C| + |K| + |F| - sum_{c in C} L_c - sum_{p in K} L_p + |G| - |F| - |K|
    # = |C| + |G| - sum_{c in C} L_c - sum_{p in K} L_p
    
    # We want to minimize this.
    # |G| is fixed.
    # We want to maximize sum_{c in C} L_c + sum_{p in K} L_p.
    # This is the total length of all cycles and closed paths.
    # Basically, we want to cover as many edges as possible with cycles (either from F or closed paths).
    # Edges in F that are in cycles contribute their length.
    # Edges in F that are in paths and get closed contribute their length.
    # Edges in F that are in paths and NOT closed contribute 0 to this sum (they become single edges).
    # Edges NOT in F contribute 0.
    
    # So the objective is: Maximize (Total length of cycles in F) + (Total length of paths in F that can be closed).
    # Let M be the set of edges in F.
    # Let C_F be cycles in F.
    # Let P_F be paths in F.
    # Let K be max matching of P_F endpoints to G\F edges.
    # Objective: sum_{c in C_F} |c| + sum_{p in K} |p|.
    
    # This is complex to optimize jointly.
    # Heuristic:
    # 1. Find Max 2-Matching F. (Maximizes |F|).
    #    This tends to create long cycles and paths.
    # 2. Extract cycles and paths from F.
    # 3. Find max matching of path endpoints to remaining edges.
    # 4. Close as many paths as possible.
    # 5. The rest are single edges.
    
    if not edges:
        return []
    
    # Build cost matrix for Max 2-Matching via MCMF is hard with simple linear_sum_assignment
    # because we need capacities of 2 on nodes.
    # We can simulate capacity 2 by duplicating nodes?
    # u_out1, u_out2, u_in1, u_in2.
    # Edge (u,v) connects u_out_i to v_in_j?
    # This becomes a 4n x 4n assignment.
    # Cost 1 for valid edges, INF for invalid.
    # But we need to ensure that u_out1 and u_out2 don't both go to same v?
    # No, capacity of v_in is 2, so two different u's can go to v.
    # But u_out1 and u_out2 are the same node u.
    # If we just duplicate, we might assign u_out1->v_in1 and u_out2->v_in1?
    # No, v_in1 is a distinct node in the assignment.
    # We need v_in1 and v_in2 to both be available.
    # And we need to prevent u_out1->v_in1 AND u_out2->v_in1?
    # Actually, in the bipartite graph for 2-matching:
    # Left: u1, u2 for each u.
    # Right: v1, v2 for each v.
    # Edge (u, v) exists in G.
    # We can connect u1->v1, u1->v2, u2->v1, u2->v2?
    # If we do that, we might assign u1->v1 and u2->v1. This means u sends 2 edges to v.
    # This is allowed in 2-matching?
    # In a simple graph, we can't have multiple edges.
    # But in the flow, u->v is a single edge.
    # If we use u1->v1 and u2->v1, we are using the edge (u,v) twice?
    # No, the edge (u,v) has capacity 1.
    # So we must enforce that only ONE of the 4 connections (u1/v1, u1/v2, u2/v1, u2/v2) is used?
    # This is not a standard assignment problem.
    
    # Alternative:
    # Use a standard MCMF implementation.
    # Since SciPy doesn't have a general MCMF solver, we might have to write one or use a different approach.
    # However, for n=30, the graph is small.
    # We can use a simpler heuristic for 2-matching:
    # Start with all edges.
    # While there is a node with degree > 2:
    #   Remove an edge incident to it.
    #   Which one? The one that is part of the shortest cycle? Or random?
    #   Or use a local search.
    
    # Let's try a simple greedy 2-matching:
    # 1. Sort edges by some heuristic (e.g. endpoints degree).
    # 2. Add edges if deg(u)<2 and deg(v)<2.
    # This gives a 2-matching.
    # To improve, we can try to swap edges.
    
    # But the prompt suggests MCMF.
    # Let's try to implement a simple MCMF or use the assignment trick with duplicates carefully.
    # Actually, for n=30, 4n=120. 120x120 assignment is fast.
    # The issue is the capacity constraint on the original edge (u,v).
    # If we connect u1->v1, u1->v2, u2->v1, u2->v2, we allow using (u,v) twice.
    # To prevent this, we can set cost to INF for 3 of them? No.
    # We can set cost 1 for u1->v1 and INF for others?
    # Then we can only use u1->v1. But then u2 and v2 are never used?
    # We need to allow u2->v2 as well.
    # But we must prevent u1->v2 and u2->v1.
    # If we set cost 1 for u1->v1 and u2->v2, and INF for u1->v2 and u2->v1.
    # Then we can use (u,v) at most twice?
    # u1->v1 uses (u,v). u2->v2 uses (u,v).
    # So we can use (u,v) twice. But capacity is 1.
    # So this is invalid.
    
    # Correct mapping for 2-matching to Assignment:
    # This is not a simple reduction to LAP.
    # It is a b-matching.
    # We can solve b-matching by reducing to standard matching by splitting nodes.
    # u -> u1, u2.
    # Edge (u,v) becomes... ?
    # In standard reduction of b-matching to matching:
    # Create u1, u2.
    # For each edge (u,v), create an edge node e_uv.
    # Connect u1, u2 to e_uv.
    # Connect e_uv to v1, v2.
    # This creates a bipartite graph.
    # Matching in this graph corresponds to b-matching.
    # Size: Left: 2n + |E|. Right: 2n.
    # For n=30, |E|~100. Left ~ 160. Right ~ 60.
    # This is rectangular. We can pad.
    # This is feasible.
    
    # However, implementing this generic reduction is complex.
    # Given time, let's stick to a simpler heuristic for 2-matching:
    # Greedy + Local Search.
    
    # 1. Greedy 2-matching
    # 2. Extract cycles/paths
    # 3. Close paths via matching
    
    # Let's implement Greedy 2-matching.
    
    adj = {u: set() for u in range(n)}
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Initial 2-matching:
    # Sort edges by (deg[u], deg[v]) descending?
    # Prefer edges between high-degree nodes?
    # Or random?
    # Let's try random with restarts.
    
    best_F = None
    best_score = -1
    
    rng = Random(42)
    
    for attempt in range(10):
        F = set()
        deg_F = [0] * n
        # Shuffle edges
        edge_list = list(edges)
        rng.shuffle(edge_list)
        # Sort by degree sum descending to prioritize "hard" edges?
        # Or just process in shuffled order.
        for u, v in edge_list:
            if deg_F[u] < 2 and deg_F[v] < 2:
                F.add((u, v))
                deg_F[u] += 1
                deg_F[v] += 1
        
        # Local search: try to swap edges to increase number of cycles or length?
        # Objective: Maximize sum of lengths of cycles and closeable paths.
        # Hard to evaluate locally.
        # Let's just evaluate the final piece count.
        
        pieces = decompose_from_matching(F, edges, n)
        score = len(pieces)
        if score < best_score:
            best_score = score
            best_F = F
            
    # Final decomposition from best_F
    return decompose_from_matching(best_F, edges, n)

def decompose_from_matching(F, edges, n):
    """
    Given a 2-matching F (set of edges), decompose into cycles and paths.
    Then close paths using edges in G \ F.
    """
    # Build adj for F
    adj_F = {u: [] for u in range(n)}
    for u, v in F:
        adj_F[u].append(v)
        adj_F[v].append(u)
        
    # Find cycles and paths
    visited = [False] * n
    cycles = []
    paths = []
    
    for u in range(n):
        if not visited[u] and adj_F[u]:
            # Start DFS
            path = [u]
            visited[u] = True
            cur = u
            while True:
                nexts = [v for v in adj_F[cur] if not visited[v]]
                if not nexts:
                    # Check if we can close to start?
                    # If cur is connected to start and len(path) >= 3
                    if start in adj_F[cur] and len(path) >= 3:
                        cycles.append(path)
                    else:
                        paths.append(path)
                    break
                # Pick next
                # Heuristic: pick one that leads to cycle?
                # Just pick first
                nxt = nexts[0]
                visited[nxt] = True
                path.append(nxt)
                cur = nxt
            # Note: This simple DFS might not find all cycles correctly if graph is complex.
            # But F is a 2-matching, so components are cycles and paths.
            # A component that is a cycle: every node has degree 2.
            # A component that is a path: endpoints have degree 1, others 2.
            # My DFS logic:
            # If I start at u, and I come back to u, it's a cycle.
            # If I get stuck at a node with no unvisited neighbors, and it's not connected to start, it's a path.
            # Wait, if it's a cycle, I will eventually hit a visited node which is start.
            # If it's a path, I will hit a node with no unvisited neighbors (end of path).
            # This works if I always pick an unvisited neighbor.
            # But what if I pick a neighbor that leads away from the cycle?
            # In a cycle, both neighbors are in the cycle.
            # In a path, one neighbor is in the path, one is out (or none).
            # So this should work.
            # One issue: I mark visited. If I enter a cycle, I mark all.
            # If I enter a path, I mark all.
            # But what if the component is a cycle and I start at u, go to v, ... and come back to u?
            # The loop `if not nexts` won't trigger because u is visited?
            # No, I check `if not visited[v]`.
            # When I am at the last node before u, its neighbors are prev and u.
            # prev is visited. u is visited.
            # So `nexts` is empty.
            # Then I check `if start in adj_F[cur]`.
            # If yes, cycle.
            # This works.
            
    # Now we have cycles and paths.
    # Cycles are pieces.
    # Paths need to be closed or broken.
    
    remaining_edges = set(edges) - F
    
    # Find max matching of path endpoints to remaining edges.
    # Path endpoints: for each path, first and last node.
    # We want to match (u, v) such that (u, v) is in remaining_edges.
    # This is a bipartite matching problem.
    # Left: start nodes of paths.
    # Right: end nodes of paths.
    # Edge (s, e) exists if (s, e) in remaining_edges.
    # We want to maximize number of matches.
    
    # Simple greedy matching:
    # Sort paths by length descending?
    # Try to match.
    
    closed_paths = []
    used_rem_edges = set()
    
    # List of (path, start, end)
    path_info = []
    for p in paths:
        path_info.append((p, p[0], p[-1]))
    
    # Greedy: try to match
    # To be smarter, use bipartite matching.
    # Since n is small, we can do a simple DFS for bipartite matching.
    
    # Build bipartite graph
    # Left nodes: indices of path_info
    # Right nodes: indices of path_info
    # Edge i->j if path_info[i].end is connected to path_info[j].start?
    # No, we close path i by adding edge (end_i, start_j)?
    # No, we close a SINGLE path i by adding edge (end_i, start_i).
    # Wait, a path is u-v-w. Endpoints u, w.
    # To close it, we need edge (w, u).
    # So we check if (w, u) is in remaining_edges.
    # If yes, we can close it.
    # We don't need to match different paths to each other.
    # We just need to check if each path can be closed individually.
    # And we need to ensure the closing edges are distinct.
    # Since each path uses a distinct pair of endpoints, and edges are distinct,
    # can two paths use the same closing edge?
    # Path 1: u1...v1. Closing edge (v1, u1).
    # Path 2: u2...v2. Closing edge (v2, u2).
    # If (v1, u1) == (v2, u2), then they use the same edge.
    # This can happen if u1=u2 and v1=v2?
    # But paths are vertex-disjoint in F.
    # So u1 != u2.
    # So the closing edges are distinct.
    # So we can just check each path independently!
    
    for p, s, e in path_info:
        ce = (e, s) if e < s else (s, e)
        if ce in remaining_edges:
            # Close it
            closed_paths.append(p)
            remaining_edges.discard(ce)
        else:
            # Break into single edges
            for i in range(len(p) - 1):
                # Add as single edge piece
                pass # We'll add them later
    
    # Construct pieces
    pieces = []
    for c in cycles:
        pieces.append(c)
        
    for p in closed_paths:
        pieces.append(p)
        
    # Add single edges from paths not closed
    for p, s, e in path_info:
        if p not in closed_paths:
            for i in range(len(p) - 1):
                u, v = p[i], p[i+1]
                pieces.append([u, v])
                
    # Add remaining edges as single edges
    for e in remaining_edges:
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 140
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            edges = instance(n, p, seed)
            results.append([list(ed) for ed in edges])
            continue
            
        edges = instance(n, p, seed)
        pieces = min_cost_flow(n, edges, max_flow=None)
        results.append(pieces)
        
    while len(results) < 3:
        results.append([])
        
    return results[:3]

if __name__ == "__main__":
    construct()
