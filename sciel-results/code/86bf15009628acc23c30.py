import time
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_2factor(adj, n):
        """
        Find a maximum 2-factor (vertex-disjoint cycles) using bipartite matching.
        This extracts a large set of edge-disjoint cycles simultaneously.
        """
        edge_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_list.append((u, v))
        
        m = len(edge_list)
        if m == 0:
            return []
        
        # We want to select a subset of edges such that every vertex has degree 2 in the subset.
        # This is a b-matching problem where b(v) = 2.
        # Reduction to standard bipartite matching:
        # Left side: Edges (0..m-1)
        # Right side: Vertex slots. Each vertex v has 2 slots: 2v, 2v+1.
        # An edge (u, v) can be assigned to any of u's 2 slots or v's 2 slots.
        # If edge e is assigned to slot s of vertex v, it means e is "used" by v.
        # For a valid 2-factor, every selected edge must be assigned to exactly one of its endpoints?
        # No, that's not quite right. An edge connects u and v. It contributes 1 to deg(u) and 1 to deg(v).
        # So edge (u,v) consumes 1 slot of u and 1 slot of v.
        # Standard reduction for b-matching in general graph is complex.
        # However, for a 2-factor, we can use the following trick:
        # Split each vertex v into v_in and v_out.
        # Direct each edge (u, v) as u_out -> v_in AND v_out -> u_in.
        # Find a cycle cover in this directed graph.
        # A cycle cover in the directed graph corresponds to a 2-factor in the undirected graph.
        
        # Build cost matrix for Hungarian Algorithm (Min Cost Max Flow / Assignment)
        # Nodes: 2n (n_in, n_out)
        # Left: n_out (0..n-1)
        # Right: n_in (0..n-1)
        # Cost[i][j] is 0 if edge (i, j) exists in original graph, INF otherwise.
        # We want to assign every out-node to an in-node.
        # This ensures every vertex has exactly one outgoing edge and one incoming edge in the directed sense,
        # which means degree 2 in the undirected sense (unless it's a 2-cycle, which is fine).
        
        INF = 10**9
        cost = [[INF] * n for _ in range(n)]
        
        for u in range(n):
            for v in adj[u]:
                if u != v:
                    cost[u][v] = 0
        
        try:
            row_ind, col_ind = linear_sum_assignment(cost)
        except:
            return []
        
        # Check if a perfect matching was found
        valid_match = 0
        for i, j in zip(row_ind, col_ind):
            if cost[i][j] < INF:
                valid_match += 1
        
        if valid_match < n:
            return []
        
        # Extract cycles from the permutation defined by the matching
        # perm[i] = j means i_out -> j_in, i.e., edge (i, j) is in the 2-factor
        perm = {}
        for i, j in zip(row_ind, col_ind):
            if cost[i][j] < INF:
                perm[i] = j
        
        visited = set()
        cycles = []
        
        for start in range(n):
            if start in visited:
                continue
            
            path = [start]
            visited.add(start)
            curr = start
            while True:
                nxt = perm.get(curr)
                if nxt is None:
                    break
                if nxt == start:
                    if len(path) >= 3:
                        cycles.append(path)
                    break
                if nxt in visited:
                    break
                path.append(nxt)
                visited.add(nxt)
                curr = nxt
        
        # Verify cycles are valid in the original graph
        valid_cycles = []
        for c in cycles:
            ok = True
            for i in range(len(c)):
                u = c[i]
                v = c[(i+1) % len(c)]
                if v not in adj[u]:
                    ok = False
                    break
            if ok:
                valid_cycles.append(c)
        
        return valid_cycles

    def find_cycle_greedy(adj, start, rng, max_len=50):
        """Find a cycle starting and ending at start using DFS with heuristics."""
        if len(adj[start]) < 2:
            return None
        
        best_cycle = None
        
        for attempt in range(10):
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = [n for n in adj[current] if n not in visited]
                
                if len(path) >= 3 and start in adj[current]:
                    cycle = path + [start]
                    if best_cycle is None or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    break
                
                if not neighbors:
                    break
                
                # Heuristic: choose neighbor with fewest options (Warnsdorff)
                neighbors.sort(key=lambda x: len(adj[x] - visited))
                
                # Randomize slightly to escape local minima
                if rng.random() < 0.2 and len(neighbors) > 1:
                    next_node = neighbors[1]
                else:
                    next_node = neighbors[0]
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
                
                if len(path) > max_len:
                    break
        
        return best_cycle

    def decompose(edges, n, deadline):
        """
        Decompose edges into cycles and single edges.
        Strategy:
        1. Repeatedly find a 2-factor (set of cycles) and remove them.
        2. If no 2-factor, find individual cycles greedily.
        3. Local search / restarts to improve.
        """
        rng = random.Random(42)
        
        def score(pieces):
            return len(pieces)
        
        best_pieces = None
        best_score = float('inf')
        
        # Time per restart
        max_restarts = 50
        restart_time = (deadline - time.time()) / max_restarts if time.time() < deadline else 0
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            current_edges = set(edges)
            adj = defaultdict(set)
            for u, v in current_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            pieces = []
            
            # Phase 1: 2-Factor Extraction
            # We can extract multiple 2-factors.
            # Each 2-factor covers 2*|V'| edges where |V'| is number of vertices in cycles.
            # This is very efficient.
            
            while True:
                if time.time() > deadline - 5:
                    break
                
                # Try to find a 2-factor
                # To make it stochastic, we can randomly remove some edges before finding 2-factor?
                # Or just find the max 2-factor.
                # The max 2-factor is deterministic given the graph.
                # To add variety, we can shuffle the order of vertices in the cost matrix?
                # linear_sum_assignment is deterministic.
                # Let's try finding a 2-factor. If it exists, take it.
                
                cycles = find_2factor(adj, n)
                
                if cycles:
                    # Remove all these cycles
                    for c in cycles:
                        pieces.append(c)
                        for i in range(len(c)):
                            u = c[i]
                            v = c[(i+1) % len(c)]
                            e = (min(u, v), max(u, v))
                            current_edges.discard(e)
                            if v in adj[u]:
                                adj[u].discard(v)
                            if u in adj[v]:
                                adj[v].discard(u)
                else:
                    # No 2-factor found (or graph is too sparse)
                    # Fall back to finding single cycles
                    found_cycle = False
                    # Try a few random starts
                    candidates = [u for u in adj if len(adj[u]) >= 2]
                    rng.shuffle(candidates)
                    
                    for start in candidates[:5]:
                        if time.time() > deadline - 5:
                            break
                        c = find_cycle_greedy(adj, start, rng)
                        if c:
                            pieces.append(c)
                            for i in range(len(c) - 1):
                                u, v = c[i], c[i+1]
                                e = (min(u, v), max(u, v))
                                current_edges.discard(e)
                                if v in adj[u]:
                                    adj[u].discard(v)
                                if u in adj[v]:
                                    adj[v].discard(u)
                            found_cycle = True
                            break
                    
                    if not found_cycle:
                        break
            
            # Remaining edges are single pieces
            for e in current_edges:
                pieces.append([e[0], e[1]])
            
            s = score(pieces)
            if s < best_score:
                best_score = s
                best_pieces = pieces
            
            # Early stop if we reached a very low score?
            # Theoretical minimum is roughly E/ (avg cycle length).
            # For these graphs, E is around 100-150.
            # If we get below 30 pieces, that's good.
            if best_score < 25:
                break

        if best_pieces is None:
            # Fallback: all edges
            return [[u, v] for u, v in edges]
        
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > DEADLINE:
            # Emergency: return trivial decomposition (all edges)
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
            
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Allocate time per graph
        time_left = DEADLINE - time.time()
        # Simple allocation: 1/3 of remaining time for each, but ensure minimum time
        # Actually, just pass the global deadline and let the internal loop handle it.
        # But we should be careful not to spend all time on the first graph.
        # Let's give each graph 1/3 of the total budget roughly.
        # Total budget ~ 145s. 3 graphs -> ~45s each.
        graph_deadline = start_time + (DEADLINE - start_time) / 3.0 * (len(results) + 1)
        
        # Actually, simpler: just use the global deadline but the loop inside decompose
        # will stop if time is up. The risk is the first graph takes 100s.
        # Let's enforce a per-graph limit.
        per_graph_limit = (DEADLINE - time.time()) / (3 - len(results))
        if per_graph_limit < 10:
             per_graph_limit = 10
             
        graph_deadline = time.time() + per_graph_limit
        
        pieces = decompose(edges, n, graph_deadline)
        results.append(pieces)
        
    return results
