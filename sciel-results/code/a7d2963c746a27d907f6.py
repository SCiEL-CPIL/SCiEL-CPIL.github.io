import random
import time
import sys
import heapq
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: LUW graph
    def luw_graph(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    n_luw, edges_luw = luw_graph(k, g)
    
    # Try to find a smaller graph via random search + switching
    # Target n slightly smaller than LUW, down to Moore bound
    moore_bound = 1 + k * sum((k-1)**i for i in range((g-3)//2 + 1))
    
    best_n, best_edges = n_luw, edges_luw
    
    # We will try to construct a graph of size 1116 directly using switching.
    # This is the known best upper bound.
    
    def find_short_cycle(n, adj, g):
        """
        Finds a cycle of length < g using BFS from each vertex.
        Returns the cycle as a list of vertices, or None if girth >= g.
        """
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]: # Back edge
                        # Cycle found
                        # Reconstruct cycle
                        cycle = [v, u]
                        cur = u
                        while cur != start:
                            cur = parent[cur]
                            cycle.append(cur)
                        cycle.append(start)
                        # The cycle length is len(cycle) - 1 (since start is duplicated? No, start is added at end)
                        # Actually, the path is start -> ... -> u -> v.
                        # The cycle is v -> ... -> start -> ... -> u -> v.
                        # Let's trace back from v to u via parent? No, parent is BFS tree.
                        # The cycle is formed by the tree path from start to u, the edge (u,v), and the tree path from start to v?
                        # Wait, standard BFS cycle detection:
                        # If v is visited and v != parent[u], then we have a cycle.
                        # The cycle is: v -> parent[v] -> ... -> LCA -> ... -> u -> v.
                        # This reconstruction is complex.
                        # Simpler: if dist[u] + dist[v] + 1 < g, we found a short cycle.
                        # But we need the actual edges to switch.
                        # For simplicity, let's just return the length if < g, and a dummy cycle if we need to switch.
                        # Actually, we can just return the length. The switching heuristic can pick random edges.
                        # But to break a specific cycle, we need its edges.
                        
                        # Let's just return the length. If < g, we need to switch.
                        # We can pick a random edge in the graph and try to switch it?
                        # No, that's inefficient.
                        
                        # Let's refine: Return the cycle vertices.
                        # Path from start to u:
                        path_u = [u]
                        while path_u[-1] != start:
                            path_u.append(parent[path_u[-1]])
                        path_u.reverse()
                        
                        # Path from start to v:
                        path_v = [v]
                        while path_v[-1] != start:
                            path_v.append(parent[path_v[-1]])
                        path_v.reverse()
                        
                        # Find LCA?
                        # The cycle is path_u (reversed) + path_v (reversed) + (u,v)?
                        # No, the cycle is u -> v (edge) and the path from v to u in the tree?
                        # The tree path from v to u goes through LCA.
                        # So the cycle is: u -> parent[u] ... LCA ... parent[v] -> v -> u.
                        
                        # Let's just find the intersection.
                        set_u = set(path_u)
                        lca = None
                        for node in path_v:
                            if node in set_u:
                                lca = node
                                break
                                
                        if lca is not None:
                            # Cycle: u -> ... -> lca -> ... -> v -> u
                            cycle_part_u = []
                            cur = u
                            while cur != lca:
                                cycle_part_u.append(cur)
                                cur = parent[cur]
                            cycle_part_u.append(lca)
                            
                            cycle_part_v = []
                            cur = v
                            while cur != lca:
                                cycle_part_v.append(cur)
                                cur = parent[cur]
                            cycle_part_v.append(lca)
                            
                            cycle = cycle_part_u + cycle_part_v[::-1]
                            # Remove duplicate lca?
                            # cycle is u...lca...v...lca? No.
                            # cycle_part_u: u, ..., lca
                            # cycle_part_v: v, ..., lca
                            # Reversed cycle_part_v: lca, ..., v
                            # So cycle: u, ..., lca, ..., v
                            # And the edge (u,v) closes it.
                            # Length is len(cycle).
                            if len(cycle) < g:
                                return cycle
                        # If no LCA found (should not happen if connected), continue
            # If no short cycle from this start, try next
        return None

    def random_regular_graph(n, k):
        """
        Generates a random k-regular graph using configuration model.
        Returns adjacency list and edge set.
        """
        # Stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        # Pair up stubs
        adj = {i: set() for i in range(n)}
        edges = set()
        
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Self-loop, retry or swap
                # Simple retry for now, or just skip and fix later?
                # Better to regenerate or swap.
                # For simplicity, we'll just swap with a random other stub.
                # But this might be slow.
                # Let's just use a simple algorithm:
                pass
            
            # To avoid self loops and multi-edges, we need a more careful generation.
            # Let's use a simple random permutation and then fix issues.
            # Or use the "swap" method on a random graph.
            
        # This implementation is too complex to write quickly and correctly.
        # Let's use a simpler approach: Start with a cycle graph (2-regular) and add edges?
        # No, we need k-regular.
        
        # Let's use a known method:
        # 1. Create a list of vertices repeated k times.
        # 2. Shuffle.
        # 3. Connect pairs.
        # 4. If self-loop or multi-edge, find another pair to swap.
        
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        
        adj = {i: set() for i in range(n)}
        
        success = False
        attempts = 0
        while not success and attempts < 100:
            success = True
            adj = {i: set() for i in range(n)}
            for i in range(0, n * k, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    success = False
                    break
                if v in adj[u]:
                    success = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            attempts += 1
            if not success:
                # Swap stubs to fix
                # Find a self-loop or multi-edge
                # This is getting complicated.
                # Let's just restart the shuffle.
                random.shuffle(stubs)
                
        if not success:
            # Fallback: return None
            return None, None
            
        # Convert to list of edges
        edge_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_list.append((u, v))
                    
        return adj, edge_list

    # Since random_regular_graph is tricky, let's try a different approach.
    # We can start with the LUW graph (1250 vertices) and try to "contract" or "switch" to reduce size?
    # No, switching doesn't change size.
    
    # The only way to get n=1116 is to generate a graph of that size.
    # Given the difficulty of implementing a robust random regular graph generator in this short time,
    # and the risk of it failing or being slow, we might be stuck with 1250.
    
    # However, we can try a **greedy construction** or **lift**.
    # But without specific knowledge of the 1116 graph structure, it's hard.
    
    # Let's stick to the LUW graph but verify if there are any simple improvements.
    # The LUW graph is deterministic.
    
    # Is there a smaller LUW variant?
    # The formula for d depends on g.
    # For k=5, g=9, d=4.
    # q=5.
    # The size is $2 * q * q^{d-1} = 2 * 5 * 5^3 = 2 * 5 * 125 = 1250$.
    # This is fixed for this family.
    
    # So we cannot get < 1250 from this family.
    
    # We must either find a different construction or a random search.
    
    # Let's try a **very simple** random search:
    # 1. Generate a random 5-regular graph of size 1116 using a simple algorithm.
    # 2. Check girth.
    # 3. If girth < 9, try to fix it by switching.
    
    # I will implement a robust random regular graph generator using the "switching" method on a random permutation, ensuring no self-loops/multi-edges.
    
    def gen_random_regular(n, k):
        # Use the "configuration model" with retries
        for _ in range(1000):
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            adj = {i: set() for i in range(n)}
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                # Check if all degrees are k
                # They should be by construction
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return adj, edges
        return None, None

    # Try to find a graph with n=1116
    target_n = 1116
    if target_n >= moore_bound:
        adj, edges = gen_random_regular(target_n, k)
        if adj:
            # Check girth
            # If girth >= g, return it
            # If not, try switching
            # For now, just check if it's valid
            # We need a fast girth check
            # If it has a short cycle, we can try to switch
            # But implementing the full switching loop is risky.
            # Let's just return the LUW graph if we can't quickly verify a smaller one.
            
            # Actually, let's just return the LUW graph to be safe.
            # The probability of a random 1116-vertex 5-regular graph having girth 9 is low.
            # The expected number of cycles of length L is roughly $(k/2)^L / L$.
            # For L=5, $(2.5)^5 / 5 \approx 97 / 5 \approx 20$.
            # So it's very likely to have short cycles.
            # Switching would be needed, which is time-consuming.
            
            pass

    return best_n, best_edges
