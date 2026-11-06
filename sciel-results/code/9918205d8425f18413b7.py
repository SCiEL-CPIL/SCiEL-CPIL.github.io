import time
import sys
from collections import defaultdict, deque
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_eulerian(edges, seed_offset=0):
        """
        Decompose edges into paths using an Eulerian-path-inspired approach.
        Key insight: if we add dummy edges to make all vertices even-degree,
        we can find an Eulerian circuit, then break it at the dummy edges.
        The number of paths = (number of odd-degree vertices) / 2.
        
        Actually, a simpler approach: 
        - Count odd-degree vertices. Let k = number of odd vertices.
        - The minimum number of paths is at least k/2.
        - We can achieve exactly k/2 paths by pairing up odd vertices and finding
          edge-disjoint paths between each pair, then combining with the Eulerian
          structure of the remaining even subgraph.
        
        Algorithm:
        1. Find all odd-degree vertices.
        2. Pair them up (using a matching or greedy).
        3. For each pair, find a path between them (using the remaining edges).
        4. The remaining edges (after removing these paths) should form an even-degree
           subgraph, which can be decomposed into Eulerian circuits.
        5. Break each Eulerian circuit at appropriate points to form paths.
        
        Actually, a cleaner approach based on the theory:
        - The minimum number of edge-disjoint paths covering all edges equals
          max(1, number_of_odd_vertices / 2) for connected graphs? No, that's not quite right.
        
        Let me think more carefully. The standard result is:
        A connected graph can be decomposed into k edge-disjoint paths where
        k = max(1, |odd|/2) if and only if ... actually this is always achievable.
        
        Wait, let me reconsider. For any connected graph, the minimum number of
        edge-disjoint paths needed to cover all edges is exactly ceil(number_of_odd_vertices / 2).
        
        Algorithm to achieve this:
        1. Let odd = list of vertices with odd degree.
        2. Pair up odd vertices: (odd[0], odd[1]), (odd[2], odd[3]), etc.
        3. For each pair (u, v), find a simple path from u to v in the current graph.
        4. Remove the edges of this path from the graph.
        5. After removing all such paths, the remaining graph has all even degrees.
        6. Decompose the remaining even-degree graph into Eulerian circuits.
        7. Each Eulerian circuit becomes a cycle, but we can break it at any point
           to form a path. However, we need to be careful: the cycles from step 6
           are edge-disjoint from the paths in step 3, and from each other.
        
        But wait - breaking a cycle into a path means we're using all edges of the cycle
        in one path. So each Eulerian circuit contributes exactly 1 path.
        
        Total paths = (number of odd pairs) + (number of Eulerian circuits in remaining graph)
        
        Hmm, but that might not be minimal. Let me think again...
        
        Actually, the correct approach is:
        - The minimum number of paths is exactly |odd|/2 when |odd| > 0, and 1 when |odd| = 0 (for connected graphs with edges).
        
        Wait, that's not right either. Let me look at this more carefully.
        
        For a connected graph G, the minimum number of edge-disjoint paths that cover all edges
        is equal to max(1, |odd(G)|/2) where odd(G) is the set of vertices with odd degree.
        
        No wait, that's for a single path decomposition. Let me reconsider.
        
        Actually, I think the answer is: the minimum number of edge-disjoint paths covering
        all edges of a connected graph is exactly |odd|/2 if |odd| >= 2, and 1 if |odd| = 0.
        
        Let me verify with examples:
        - A tree with n edges: all leaves have odd degree (if we consider degree in the tree).
          Actually, in a tree, the number of odd-degree vertices is even. For a path graph
          with 2 vertices and 1 edge: both vertices have degree 1 (odd), so |odd| = 2, 
          minimum paths = 1. Correct.
        - A star with 3 leaves: center has degree 3 (odd), leaves have degree 1 (odd).
          |odd| = 4, minimum paths = 2. We can cover the 3 edges with 2 paths: 
          leaf1-center-leaf2 and leaf3-center. Wait, that uses edge center-leaf3 twice? No.
          Path 1: leaf1-center-leaf2 (uses edges leaf1-center, center-leaf2)
          Path 2: leaf3-center (uses edge leaf3-center)
          Yes, 2 paths. Correct.
        
        So the algorithm is:
        1. Find all odd-degree vertices.
        2. Pair them up arbitrarily.
        3. For each pair, find a path between them and remove those edges.
        4. The remaining graph has all even degrees, so it can be decomposed into 
           Eulerian circuits. But actually, after removing the paths in step 3, 
           we don't need additional paths because the paths in step 3 already cover
           all edges? No, that's not right.
        
        I think I'm overcomplicating this. Let me use a different approach:
        
        Hierholzer's algorithm variant for path decomposition:
        
        Actually, the simplest correct approach:
        1. While there are edges remaining:
           a. Find a vertex with odd degree in the remaining graph (if any).
              If all degrees are even, pick any vertex with positive degree.
           b. Follow edges greedily (using a stack-based approach like Hierholzer) 
              to form a path.
           c. Remove the edges used.
        
        But this doesn't guarantee minimality.
        
        Let me try a different structural approach: use the fact that we can add 
        dummy edges to make all degrees even, find an Eulerian circuit, then break 
        at the dummy edges.
        
        Algorithm:
        1. Let odd = list of odd-degree vertices.
        2. Pair up odd vertices: pairs = [(odd[0],odd[1]), (odd[2],odd[3]), ...]
        3. Add dummy edges for each pair (these are not real edges).
        4. Now all vertices have even degree. Find an Eulerian circuit in the 
           augmented graph.
        5. The Eulerian circuit will use the dummy edges. Break the circuit at 
           each dummy edge to get paths in the original graph.
        6. The number of paths = number of dummy edges = |odd|/2.
        
        But we need to find the actual paths in the original graph. The Eulerian 
        circuit in the augmented graph uses both real and dummy edges. When we 
        break at dummy edges, each segment between two consecutive dummy edges 
        (or between the start/end and a dummy edge) is a path in the original graph.
        
        Let me implement this:
        
        1. Build the graph with real edges.
        2. Identify odd-degree vertices and pair them.
        3. Create an augmented adjacency that includes dummy edges.
        4. Run Hierholzer's algorithm on the augmented graph to get an Eulerian circuit.
        5. Traverse the circuit, breaking at dummy edges to extract paths.
        
        This should give exactly |odd|/2 paths (or 1 if |odd| = 0 and there are edges).
        """
        if not edges:
            return []
        
        # Build adjacency for real edges
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add(tuple(sorted((u, v))))
        
        # Count degrees
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        # Find odd-degree vertices
        odd_vertices = [v for v in degree if degree[v] % 2 == 1]
        
        if not odd_vertices:
            # All even degrees: find Eulerian circuit and break into one path
            # Actually, if all degrees are even and the graph is connected, 
            # we can find an Eulerian circuit. But we need a path, not a circuit.
            # We can just take the Eulerian circuit and remove the last edge 
            # to make it a path? No, that would leave an edge uncovered.
            # 
            # Actually, for an Eulerian circuit, we can break it at any point 
            # to form a path that covers all edges. Wait, no - an Eulerian circuit 
            # is a closed walk. To make it a path, we'd need to remove one edge, 
            # but then we haven't covered all edges.
            #
            # Hmm, I think for the all-even case, the minimum number of paths is 1
            # only if the graph is a single cycle. Otherwise, it might be more.
            #
            # Wait, no. If all degrees are even and the graph is connected, 
            # it has an Eulerian circuit. An Eulerian circuit uses all edges 
            # exactly once and returns to the start. To convert this to a path,
            # we can just... actually, we can't directly convert a circuit to a 
            # path without removing an edge.
            #
            # I think I was wrong. Let me reconsider.
            #
            # For a connected graph with all even degrees, the minimum number of 
            # edge-disjoint paths covering all edges is 1 if and only if the graph 
            # is a single cycle. Otherwise, it could be more.
            #
            # Actually, I think the correct statement is:
            # - If the graph has exactly 2 odd-degree vertices, min paths = 1.
            # - If the graph has 0 odd-degree vertices, min paths = 1 (by taking 
            #   an Eulerian circuit and "opening" it, but this requires the circuit 
            #   to be simple, which it's not in general).
            #
            # I'm getting confused. Let me just use a practical approach:
            # Use a greedy algorithm that tries to find long paths, but with 
            # the Eulerian insight to guide the pairing.
            
            # For now, fall back to greedy
            pass
        
        # For simplicity and correctness, let's use a different approach:
        # Use the pairing method properly.
        
        # Pair up odd vertices
        pairs = []
        for i in range(0, len(odd_vertices), 2):
            pairs.append((odd_vertices[i], odd_vertices[i + 1]))
        
        # We'll use a modified Hierholzer that accounts for the pairing.
        # Actually, let's just do a clean greedy with good heuristics.
        
        return None  # Signal to use fallback

    def decompose_greedy_improved(edges, seed_offset=0):
        """
        Improved greedy: at each step, find the longest possible simple path
        using a DFS with backtracking limited by time budget.
        """
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths = []
        rng = random.Random(12345 + seed_offset)
        
        while remaining_edges:
            if time.time() > deadline - 5:
                # Emergency: add all remaining edges as single-edge paths
                for e in remaining_edges:
                    paths.append([e[0], e[1]])
                break
            
            active_nodes = set()
            for u, v in remaining_edges:
                active_nodes.add(u)
                active_nodes.add(v)
            
            if not active_nodes:
                break
            
            best_path = []
            best_edges_used = set()
            
            # Try multiple starting points
            # Sort by degree in remaining graph, descending
            deg_in_remaining = defaultdict(int)
            for u, v in remaining_edges:
                deg_in_remaining[u] += 1
                deg_in_remaining[v] += 1
            
            nodes_sorted = sorted(active_nodes, key=lambda x: (-deg_in_remaining[x], x))
            
            # Limit the number of starts we try to save time
            max_starts = min(len(nodes_sorted), 15)
            starts_to_try = nodes_sorted[:max_starts]
            
            for start_node in starts_to_try:
                # DFS to find a long simple path
                # Use iterative DFS with backtracking
                stack = [(start_node, [start_node], {start_node}, start_node)]
                local_best = [start_node]
                
                while stack:
                    current, path, visited, last = stack.pop()
                    
                    if len(path) > len(local_best):
                        local_best = list(path)
                    
                    # Find candidates
                    candidates = []
                    for nb in adj[current]:
                        edge = tuple(sorted((current, nb)))
                        if edge in remaining_edges and nb not in visited:
                            # Heuristic: prefer neighbors with lower degree in remaining graph
                            # to avoid getting stuck in dense areas
                            candidates.append((deg_in_remaining[nb], nb))
                    
                    if not candidates:
                        continue
                    
                    # Sort by degree ascending (prefer going to less connected nodes)
                    candidates.sort(key=lambda x: x[0])
                    
                    # Try the best candidate first
                    for _, nb in candidates[:3]:  # Limit branching
                        new_path = path + [nb]
                        new_visited = visited | {nb}
                        stack.append((nb, new_path, new_visited, nb))
                
                if len(local_best) > len(best_path):
                    best_path = local_best
                    # Compute edges used
                    best_edges_used = set()
                    for i in range(len(best_path) - 1):
                        u, v = best_path[i], best_path[i + 1]
                        best_edges_used.add(tuple(sorted((u, v))))
            
            if not best_path or len(best_path) < 2:
                # Fallback: pick any edge
                e = next(iter(remaining_edges))
                best_path = [e[0], e[1]]
                best_edges_used = {e}
            
            # Extract the path
            paths.append(best_path)
            for e in best_edges_used:
                remaining_edges.discard(e)
        
        return paths

    def decompose_with_local_search(edges, seed_offset=0, max_time=20):
        """
        Start with a greedy decomposition, then use local search to improve.
        Moves: 
        1. Merge two paths if they can be connected by reassigning edges.
        2. Split a path and recombine with another.
        """
        if not edges:
            return []
        
        # Get initial solution from greedy
        initial = decompose_greedy_improved(edges, seed_offset)
        if not initial:
            return []
        
        paths = [list(p) for p in initial]
        rng = random.Random(9999 + seed_offset)
        end_time = start_time + max_time
        
        # Build edge-to-path mapping
        def build_edge_map(paths):
            edge_map = {}
            for pi, path in enumerate(paths):
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i + 1])))
                    edge_map[e] = pi
            return edge_map
        
        def path_edges(path):
            es = set()
            for i in range(len(path) - 1):
                es.add(tuple(sorted((path[i], path[i + 1]))))
            return es
        
        def try_merge(paths, i, j):
            """Try to merge paths i and j by finding a connection."""
            path_i = paths[i]
            path_j = paths[j]
            
            # Try all combinations of endpoints
            # Path i can be traversed in two directions
            # Path j can be traversed in two directions
            
            for rev_i in [False, True]:
                for rev_j in [False, True]:
                    pi = list(reversed(path_i)) if rev_i else path_i
                    pj = list(reversed(path_j)) if rev_j else path_j
                    
                    # Try to connect end of pi to start of pj
                    # Check if there's an edge from pi[-1] to pj[0]
                    e = tuple(sorted((pi[-1], pj[0])))
                    if e in path_edges(pi) or e in path_edges(pj):
                        continue  # Edge already used
                    
                    # Check if the edge exists in the original graph
                    adj_set = set()
                    for p in paths:
                        adj_set.update(path_edges(p))
                    if e not in adj_set:
                        # Need to check if this edge is in the original graph
                        # Actually, we need to check if this edge exists in the graph
                        # and is not already used
                        pass
                    
                    # This is getting complex. Let's use a simpler merge heuristic.
                    # Check if pi[-1] and pj[0] are connected by an unused edge.
                    # We need to know which edges are unused.
                    
            return None  # Simplified for now
        
        # For time efficiency, just return the greedy result
        return initial

    def decompose_final(edges, seed_offset=0, time_budget=30):
        """
        Final decomposition using improved greedy with multiple restarts.
        """
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        num_restarts = 5
        time_per_restart = time_budget / num_restarts
        
        for r in range(num_restarts):
            if time.time() > start_time + time_budget:
                break
            
            paths = decompose_greedy_improved(edges, seed_offset + r * 100)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
        
        return best_paths if best_paths else [[u, v] for u, v in edges]

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    # Allocate time budget
    time_per_graph = 40  # 3 graphs * 40s = 120s, leaving 25s margin
    
    for idx, spec in enumerate(specs):
        if time.time() > deadline - 5:
            break
        
        edges = instance(*spec)
        
        # Use the final decomposition with time budget
        remaining_time = min(time_per_graph, deadline - 10 - time.time())
        if remaining_time < 5:
            # Emergency fallback
            results.append([[u, v] for u, v in edges])
            continue
        
        paths = decompose_final(edges, seed_offset=idx * 1000, time_budget=remaining_time)
        results.append(paths)
    
    # Fill remaining with trivial edge paths if we ran out of time
    while len(results) < 3:
        spec = specs[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
