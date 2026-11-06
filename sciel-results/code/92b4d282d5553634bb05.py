import time
import random
import heapq
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

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

    def solve_min_path_decomposition(n, edges):
        """
        Solve minimum path decomposition using min-cost flow.
        
        We model this as a min-cost flow problem:
        - Each edge (u,v) in the graph becomes an arc in the flow network.
        - We need to decompose edges into minimum number of paths.
        - This is equivalent to finding a minimum-cost circulation/flow where
          each edge is used exactly once, and the number of paths equals the
          number of "sources" (vertices with net outflow > inflow by 1).
        
        The standard reduction:
        - Create a flow network where each original edge (u,v) is represented
          as two directed arcs: u->v and v->u, each with capacity 1 and cost 0.
        - For each vertex v, we need to balance the flow. A path starts at a
          vertex with outflow-inflow=1 and ends at a vertex with inflow-outflow=1.
        - The minimum number of paths = (number of vertices with odd degree) / 2
          if we're looking for a path cover, but we want to minimize the number
          of paths, which is at least max(1, odd_count/2).
        
        Actually, the minimum path decomposition problem can be solved by:
        1. Find an Eulerian augmentation: add edges to make all degrees even,
           creating an Eulerian multigraph.
        2. The number of added edges = odd_count/2.
        3. Find an Eulerian circuit in the augmented graph.
        4. Remove the added edges; the remaining segments are the paths.
        
        But this gives odd_count/2 paths, which may not be minimal if we can
        have fewer paths by having paths that start and end at the same vertex
        (cycles). Actually, a path decomposition into k paths means we need
        exactly k vertices with net outflow 1 and k vertices with net inflow 1
        (or fewer if some paths are cycles).
        
        The minimum number of paths in a path decomposition of a connected graph
        is max(1, ceil(odd_count/2)) where odd_count is the number of vertices
        with odd degree. This is because each path has exactly two endpoints
        (or zero if it's a cycle), and each endpoint must be a vertex of odd degree
        in the original graph.
        
        So the lower bound is max(1, odd_count // 2), and this bound is achievable.
        
        To achieve it, we can:
        1. Pair up odd-degree vertices.
        2. Add edges between each pair (creating a perfect matching on odd vertices).
        3. The resulting multigraph is Eulerian.
        4. Find an Eulerian circuit.
        5. Remove the added edges; the remaining connected components are the paths.
        
        But we need to be careful: the added edges must not create multi-edges that
        complicate things, and we need to find a pairing that minimizes the number
        of paths, which is always odd_count/2 (or 1 if odd_count=0).
        
        Let's implement this approach.
        """
        # Compute degrees
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
        odd_count = len(odd_vertices)
        
        if odd_count == 0:
            # Eulerian graph: one Eulerian circuit
            # Find Eulerian circuit
            adj = defaultdict(list)
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Hierholzer's algorithm
            circuit = []
            stack = [0]
            visited_edges = set()
            
            def edge_key(u, v):
                return (min(u, v), max(u, v))
            
            while stack:
                v = stack[-1]
                if adj[v]:
                    w = adj[v].pop()
                    ek = edge_key(v, w)
                    if ek not in visited_edges:
                        visited_edges.add(ek)
                        stack.append(w)
                else:
                    circuit.append(stack.pop())
            
            circuit.reverse()
            return [circuit]
        
        # Pair up odd vertices greedily (or use a matching)
        # For simplicity, pair them in order
        pairs = []
        for i in range(0, odd_count, 2):
            pairs.append((odd_vertices[i], odd_vertices[i+1]))
        
        # Add the pairing edges to create an Eulerian multigraph
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        added_edges = set()
        for u, v in pairs:
            adj[u].append(v)
            adj[v].append(u)
            ek = (min(u, v), max(u, v))
            added_edges.add(ek)
        
        # Find Eulerian circuit using Hierholzer's
        def edge_key(u, v):
            return (min(u, v), max(u, v))
        
        # We need to track which edges are used, but now we have multi-edges
        # Use a counter for each edge
        from collections import Counter
        edge_count = Counter()
        for u, v in edges:
            ek = edge_key(u, v)
            edge_count[ek] += 1
        for u, v in pairs:
            ek = edge_key(u, v)
            edge_count[ek] += 1
        
        adj2 = defaultdict(list)
        for u, v in edges:
            adj2[u].append(v)
            adj2[v].append(u)
        for u, v in pairs:
            adj2[u].append(v)
            adj2[v].append(u)
        
        circuit = []
        stack = [0]
        
        while stack:
            v = stack[-1]
            if adj2[v]:
                w = adj2[v].pop()
                ek = edge_key(v, w)
                if edge_count[ek] > 0:
                    edge_count[ek] -= 1
                    stack.append(w)
                else:
                    # Put it back
                    adj2[v].append(w)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        # Remove the added edges to get the paths
        paths = []
        current_path = []
        for i in range(len(circuit)):
            v = circuit[i]
            if i > 0:
                prev = circuit[i-1]
                ek = edge_key(prev, v)
                if ek in added_edges:
                    # This is an added edge, so split the path here
                    if current_path:
                        paths.append(current_path)
                        current_path = []
            current_path.append(v)
        
        if current_path:
            paths.append(current_path)
        
        # Remove empty paths
        paths = [p for p in paths if len(p) > 1]
        
        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for spec_idx, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        if time.time() > deadline - 5.0:
            # Not enough time, just return edge list as fallback
            all_results.append([list(e) for e in edges])
            continue
        
        paths = solve_min_path_decomposition(n, edges)
        all_results.append(paths)

    return all_results
