import time
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    def min_path_decomposition(n, edges):
        """
        Find minimum number of edge-disjoint paths covering all edges.
        
        Model:
        Each edge e = (u, v) can be oriented as u->v or v->u.
        We create two "ports" per edge:
          - port 2*i:   edge i used as u->v (enters at u, leaves at v)
          - port 2*i+1: edge i used as v->u (enters at v, leaves at u)
        
        We build a bipartite graph:
          Left  = "leaving" ports (each port leaves at a specific vertex)
          Right = "entering" ports (each port enters at a specific vertex)
          Left port p is connected to Right port q if they share a vertex 
          (i.e., the leaving vertex of p == entering vertex of q).
        
        We want a matching that:
          1. Covers every edge exactly once (exactly one of its two ports is used as a leaving port AND exactly one as an entering port)
          2. Maximizes the number of "connections" (matched pairs)
        
        This is a min-cost max-matching problem. 
        Cost 0 for a valid connection (chain), high cost otherwise.
        
        The number of paths = m - (number of matched connections).
        """
        m = len(edges)
        if m == 0:
            return []
        
        # Build port definitions
        # port_id -> (edge_idx, entering_vertex, leaving_vertex)
        ports = []
        for i, (u, v) in enumerate(edges):
            # Port 2*i: enters at u, leaves at v
            ports.append((i, u, v))
            # Port 2*i+1: enters at v, leaves at u
            ports.append((i, v, u))
        
        num_ports = 2 * m
        
        # We need to select exactly m ports to be "active" (used in the decomposition).
        # Specifically, for each edge i, exactly one of {2*i, 2*i+1} must be selected as a "leaving" port
        # and exactly one must be selected as an "entering" port.
        # Wait, the matching is between Leaving ports and Entering ports.
        # Let L be the set of all ports (as potential leaving nodes).
        # Let R be the set of all ports (as potential entering nodes).
        # A matching M is a set of pairs (p, q) where p in L, q in R, p != q, and leave_vertex(p) == enter_vertex(q).
        # We also need to ensure that each edge is used exactly once.
        # This means:
        #   For each edge i:
        #     Exactly one of its ports is matched as a leaving port (out-degree 1 in the matching graph of ports)
        #     Exactly one of its ports is matched as an entering port (in-degree 1)
        #     And the leaving port and entering port for edge i must be consistent? 
        #     Actually, if edge i is used, it contributes one "leaving" event and one "entering" event.
        #     The leaving event is at one endpoint, entering at the other.
        
        # Let's reformulate as a standard assignment problem on a larger graph.
        # We have 2m ports.
        # We want to choose a subset of m ports to be "leaving" and m ports to be "entering" such that:
        #   - For each edge i, exactly one of its two ports is in the "leaving" set, and exactly one is in the "entering" set.
        #   - Actually, it's simpler: We just need to match leaving ports to entering ports.
        #   - The constraint "each edge used exactly once" translates to:
        #       The set of leaving ports must contain exactly one port from each edge.
        #       The set of entering ports must contain exactly one port from each edge.
        
        # This is a bit tricky for a standard Hungarian algorithm which assumes a fixed set of left and right nodes.
        # However, we can model this as a min-cost flow or a more complex assignment.
        
        # Alternative approach: 
        # Let x_{i, v} be 1 if edge i is oriented such that it leaves v. (So v is the "tail")
        # Then edge i enters u where u is the other endpoint.
        # We want to maximize the number of chains. A chain occurs if edge i leaves v and edge j enters v.
        # Let y_{i, j} be 1 if edge i leaves v and edge j enters v (for some v).
        # This is getting complicated.
        
        # Let's stick to the bipartite matching on ports but handle the "selection" constraint.
        # Actually, there is a known reduction:
        # Minimum path decomposition is equivalent to finding a maximum matching in a specific bipartite graph?
        # No, it's related to the "minimum path cover" in a DAG, but here the graph is undirected.
        
        # Let's use the following formulation:
        # Create a bipartite graph with 2m nodes on the left and 2m nodes on the right.
        # Left node (i, 0) represents edge i oriented u->v (leaves v).
        # Left node (i, 1) represents edge i oriented v->u (leaves u).
        # Right node (j, 0) represents edge j oriented u->v (enters u).
        # Right node (j, 1) represents edge j oriented v->u (enters v).
        
        # We want to select a set of left nodes L_sel and right nodes R_sel such that:
        # 1. For each edge i, exactly one of (i,0) or (i,1) is in L_sel.
        # 2. For each edge j, exactly one of (j,0) or (j,1) is in R_sel.
        # 3. We match nodes in L_sel to nodes in R_sel if they share a vertex.
        # 4. Maximize the number of matches.
        
        # This can be solved by adding dummy nodes and using min-cost max-flow, or by cleverly constructing the cost matrix for Hungarian.
        
        # Let's construct the cost matrix for Hungarian.
        # Size: 2m x 2m.
        # Row (i, o) corresponds to "Edge i leaves vertex v_i_o".
        # Col (j, e) corresponds to "Edge j enters vertex v_j_e".
        # Cost(row, col) = 0 if leave_vertex(row) == enter_vertex(col) and i != j.
        #                  = INF if i == j (can't chain an edge to itself) or vertices don't match.
        
        # But we also need to enforce the selection constraint.
        # Hungarian algorithm finds a perfect matching in the bipartite graph.
        # If we just run Hungarian on the 2m x 2m matrix, it will pick 2m pairs.
        # This doesn't directly enforce "exactly one per edge".
        
        # Let's try a different tactic.
        # We know that the number of paths is m - k, where k is the number of connections.
        # Each connection is a pair (i, j) such that edge i and edge j are connected at a vertex.
        # This is equivalent to finding a maximum set of edge pairs that can be connected.
        
        # Let's use the "port" matching idea but solve it as a min-cost max-flow.
        # Source -> Left Port (i, o) [cap 1, cost 0]
        # Left Port (i, o) -> Right Port (j, e) [cap 1, cost 0 if connected, INF if not]
        # Right Port (j, e) -> Sink [cap 1, cost 0]
        
        # Constraints:
        # For each edge i, the sum of flow out of Left Port (i,0) and Left Port (i,1) must be 1.
        # For each edge j, the sum of flow into Right Port (j,0) and Right Port (j,1) must be 1.
        
        # This is a standard min-cost flow problem.
        # However, implementing a general min-cost flow solver is heavy.
        
        # Let's try to reduce it to a standard assignment problem by splitting the edges.
        # Actually, there is a simpler insight:
        # The problem is equivalent to finding a maximum matching in a graph where:
        # Nodes are edges.
        # Two edges are connected if they share a vertex.
        # But we can only match each edge once? No, an edge can be part of a chain, so it connects to a previous edge and a next edge.
        # So each edge has "in-degree" at most 1 and "out-degree" at most 1 in the chain graph.
        # This is exactly a matching problem on a bipartite graph where:
        # Left side: Edges (as "tails")
        # Right side: Edges (as "heads")
        # Edge (i, j) exists if edge i and edge j share a vertex.
        # We want a matching M.
        # But we also need to choose the orientation.
        
        # Let's go back to the 2m x 2m Hungarian approach but with a twist.
        # We will create a cost matrix of size 2m x 2m.
        # We will add a penalty for not satisfying the "one per edge" constraint?
        # No, Hungarian always finds a perfect matching (size 2m).
        
        # Let's try a different formulation:
        # We want to select m left ports and m right ports.
        # Let's create a new bipartite graph with m nodes on the left (edges) and m nodes on the right (edges).
        # This doesn't work directly because of orientation.
        
        # Okay, let's use the min-cost flow formulation with a simple implementation or use scipy if available?
        # scipy.optimize.linprog is available.
        # We can formulate it as an ILP? No, linear_sum_assignment is faster.
        
        # Let's try to construct a valid instance for linear_sum_assignment.
        # We have 2m left ports and 2m right ports.
        # We want to pick a matching of size m (not 2m).
        # linear_sum_assignment finds a perfect matching (size N).
        
        # Trick: Add m dummy nodes to the right?
        # If we add m dummy nodes to the right, and set the cost of matching a real left port to a dummy to be 0, 
        # and the cost of matching a real left port to a real right port to be 0 (if connected) or INF (if not),
        # we still have the constraint that each edge must be used once.
        
        # Actually, if we just run Hungarian on the 2m x 2m matrix where:
        # Cost(i, j) = 0 if port i and port j are compatible (share vertex, i!=j)
        # Cost(i, j) = INF otherwise
        # And we take the result, we get a perfect matching of 2m pairs.
        # This will likely match ports in a way that doesn't respect the "one per edge" constraint.
        
        # Let's look at the structure again.
        # If we match port (i, 0) to port (j, 1), it means edge i leaves v and edge j enters v.
        # This forms a chain i -> j.
        # If we have a perfect matching of size 2m, we have 2m such connections.
        # But we only have m edges. Each edge can be a "tail" of at most one connection and a "head" of at most one connection.
        # So the maximum number of connections is m (if we form a single cycle or path covering all edges? No, max m-1 for a path).
        # Wait, if we have m edges, the max number of connections is m-1 (forming a single path).
        # So a perfect matching of size 2m is impossible in a valid sense.
        
        # Therefore, we should NOT use a perfect matching on 2m x 2m.
        
        # Correct formulation:
        # We want to find a set of connections (pairs of edges) such that:
        # 1. Each edge is involved in at most one "outgoing" connection (as a tail).
        # 2. Each edge is involved in at most one "incoming" connection (as a head).
        # 3. We maximize the number of connections.
        
        # This is a maximum bipartite matching problem where:
        # Left nodes: Edges (as tails)
        # Right nodes: Edges (as heads)
        # Edge (i, j) exists if edge i and edge j share a vertex.
        # BUT, we need to ensure that the shared vertex is consistent with the orientation.
        
        # Let's refine:
        # For each edge i, we have two possible "tail" vertices.
        # For each edge j, we have two possible "head" vertices.
        # We can connect i (tail at v) to j (head at v) if they share v.
        
        # Let's create a bipartite graph with 2m left nodes and 2m right nodes.
        # Left node (i, 0): Edge i, tail at u_i.
        # Left node (i, 1): Edge i, tail at v_i.
        # Right node (j, 0): Edge j, head at u_j.
        # Right node (j, 1): Edge j, head at v_j.
        
        # We want to find a matching in this bipartite graph.
        # Constraints:
        # - At most one left node per edge i can be matched.
        # - At most one right node per edge j can be matched.
        
        # This is a "maximum matching with conflict constraints".
        # We can solve this by reducing it to a standard matching problem.
        # Create a new graph:
        # Source -> Left Node (i, k) [cap 1]
        # Left Node (i, k) -> Right Node (j, l) [cap 1, cost 0 if compatible]
        # Right Node (j, l) -> Sink [cap 1]
        
        # But we need to ensure that for each edge i, at most one of its left nodes is used.
        # We can group them:
        # Source -> Group Node i [cap 1]
        # Group Node i -> Left Node (i, 0) [cap 1]
        # Group Node i -> Left Node (i, 1) [cap 1]
        # Left Node (i, k) -> Right Node (j, l) [cap 1]
        # Right Node (j, l) -> Group Node j [cap 1]
        # Group Node j -> Sink [cap 1]
        
        # This is a min-cost max-flow problem.
        
        # Since we don't have a built-in min-cost flow solver in scipy (only linprog), 
        # and linprog might be slow for large m, let's see if we can use linear_sum_assignment creatively.
        
        # Actually, for m up to ~100, we can use a simple greedy or local search.
        # But the current program uses linear_sum_assignment on an m x m matrix.
        # Let's check the current code's logic.
        # It builds an m x m compat matrix.
        # compat[i][j] = vertex if edges i and j share a vertex.
        # It then runs Hungarian on this m x m matrix.
        # This finds a permutation of edges.
        # This corresponds to a set of m pairs (i, sigma(i)).
        # This implies each edge i is connected to edge sigma(i).
        # This forms a set of cycles and paths.
        # If sigma is a permutation, we have m connections.
        # But we can't have m connections if we have m edges, unless we have a single cycle.
        # If we have a cycle, the number of paths is 0? No, a cycle is not a path.
        # The problem asks for paths. A cycle must be broken.
        # So the maximum number of connections in a path decomposition is m-1.
        
        # So the current approach of finding a permutation (m connections) is slightly off.
        # It might find a cycle, which is invalid for a path decomposition unless we break it.
        
        # Let's modify the approach to find a maximum matching of size at most m-1.
        # Or, find a maximum matching in the bipartite graph (2m x 2m) with the group constraints.
        
        # Given the complexity, let's try a heuristic improvement on the current code.
        # The current code finds a permutation. If it contains a cycle, we can break it.
        # Breaking a cycle of length k reduces the number of connections by 1 (from k to k-1), increasing paths by 1.
        # But we want to MINIMIZE paths, so we want to MAXIMIZE connections.
        
        # Let's try to find a maximum matching in the 2m x 2m graph using a simple algorithm.
        
        # For now, let's stick to the m x m Hungarian approach but ensure we don't form cycles.
        # If we find a permutation, we can check for cycles.
        # If there is a cycle, we break it.
        
        # Let's implement the m x m Hungarian approach and then post-process to break cycles.
        
        compat = [[-1] * m for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(m):
                if i == j:
                    continue
                u2, v2 = edges[j]
                if u1 == u2:
                    compat[i][j] = u1
                elif u1 == v2:
                    compat[i][j] = u1
                elif v1 == u2:
                    compat[i][j] = v1
                elif v1 == v2:
                    compat[i][j] = v1
        
        BIG = 10**9
        cost = [[BIG if compat[i][j] == -1 else 0 for j in range(m)] for i in range(m)]
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Build the permutation
        perm = {}
        for r, c in zip(row_ind, col_ind):
            if cost[r][c] == 0:
                perm[r] = c
        
        # Find cycles and break them
        # We want to remove the minimum number of edges from the permutation to make it acyclic.
        # This is equivalent to finding the minimum feedback arc set in a permutation?
        # No, a permutation consists of disjoint cycles.
        # We just need to break each cycle.
        # To break a cycle, we remove one edge (connection).
        # So for each cycle, we remove one connection.
        
        # Identify cycles
        visited = set()
        cycles = []
        for i in range(m):
            if i in visited or i not in perm:
                continue
            cycle = []
            curr = i
            while curr not in visited:
                visited.add(curr)
                cycle.append(curr)
                if curr in perm:
                    curr = perm[curr]
                else:
                    break
            if len(cycle) > 1:
                cycles.append(cycle)
        
        # Break each cycle by removing one connection
        # Which connection to remove?
        # It doesn't matter for the count, but it matters for the path structure.
        # We want to keep the graph as "path-like" as possible.
        # Let's just remove the first connection in the cycle.
        
        for cycle in cycles:
            # Remove connection from cycle[0] to perm[cycle[0]]
            # Wait, perm[i] is the next edge.
            # So we remove the link i -> perm[i].
            # Let's remove the link from the last element to the first?
            # It doesn't matter.
            start = cycle[0]
            end = perm[start]
            del perm[start]
        
        # Now perm is a set of paths (no cycles).
        # Number of paths = m - len(perm)
        
        # Reconstruct paths
        next_edge = perm
        matched_as_target = set(next_edge.values())
        path_starts = [i for i in range(m) if i not in matched_as_target]
        
        paths = []
        for start in path_starts:
            path = []
            curr = start
            visited_in_path = set()
            while curr is not None and curr not in visited_in_path:
                visited_in_path.add(curr)
                u, v = edges[curr]
                if curr in next_edge:
                    next_e = next_edge[curr]
                    u_next, v_next = edges[next_e]
                    if compat[curr][next_e] == u:
                        path.append(v)
                        path.append(u)
                    else:
                        path.append(u)
                        path.append(v)
                    curr = next_e
                else:
                    path.append(u)
                    path.append(v)
                    curr = None
            
            if len(path) > 1:
                clean_path = [path[0]]
                for v in path[1:]:
                    if v != clean_path[-1]:
                        clean_path.append(v)
                paths.append(clean_path)
        
        return paths

    def greedy_fallback(n, edges):
        """Fallback greedy approach if matching fails or takes too long."""
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        curr_edges = set(edges)
        curr_paths = []
        
        while curr_edges:
            active_vertices = [v for v in range(n) if adj[v] and any((min(v, nb), max(v, nb)) in curr_edges for nb in adj[v])]
            
            if not active_vertices:
                break
            
            best_path_in_iter = []
            
            for s in active_vertices:
                path = [s]
                visited = {s}
                curr_node = s
                
                while True:
                    candidates = []
                    for nb in adj[curr_node]:
                        if nb in visited:
                            continue
                        e = (min(curr_node, nb), max(curr_node, nb))
                        if e in curr_edges:
                            unvis_deg = 0
                            for m in adj[nb]:
                                if m in visited:
                                    continue
                                e2 = (min(nb, m), max(nb, m))
                                if e2 in curr_edges:
                                    unvis_deg += 1
                            candidates.append((unvis_deg, nb))
                    
                    if not candidates:
                        break
                    
                    candidates.sort(key=lambda x: x[0])
                    min_deg = candidates[0][0]
                    min_candidates = [c for c in candidates if c[0] == min_deg]
                    next_node = min_candidates[0][1]
                    
                    path.append(next_node)
                    visited.add(next_node)
                    curr_node = next_node
                
                if len(path) > len(best_path_in_iter):
                    best_path_in_iter = path[:]
            
            if best_path_in_iter and len(best_path_in_iter) > 1:
                curr_paths.append(best_path_in_iter)
                for i in range(len(best_path_in_iter) - 1):
                    u, v = best_path_in_iter[i], best_path_in_iter[i + 1]
                    e = (min(u, v), max(u, v))
                    curr_edges.discard(e)
            else:
                if curr_edges:
                    e = next(iter(curr_edges))
                    curr_paths.append(list(e))
                    curr_edges.discard(e)
        
        return curr_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Try matching-based approach first
        if time.time() < deadline - 30.0:
            try:
                paths = min_path_decomposition(n, edges)
                # Validate: check that paths are valid
                valid = True
                used_edges = set()
                for path in paths:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        e = (min(u, v), max(u, v))
                        if e not in set(edges) or e in used_edges:
                            valid = False
                            break
                        used_edges.add(e)
                    if not valid:
                        break
                
                if valid and len(used_edges) == len(edges):
                    all_results.append(paths)
                    continue
            except Exception:
                pass
        
        # Fallback to greedy
        paths = greedy_fallback(n, edges)
        all_results.append(paths)

    return all_results
