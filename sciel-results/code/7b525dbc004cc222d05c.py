import time
import sys
from collections import deque
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before the 150s kill
    
    # If k != 3 or g is very small, fall back to a simple construction or error
    # The task specifies k=3, g=16, but we should be robust.
    if k != 3:
        # Fallback for other k: return a trivial graph or raise
        # For this specific task, we assume k=3.
        # If k is not 3, we can try a simple cycle if k=2, or complete graph logic?
        # Let's just return a dummy valid graph for k=3, g=16 context.
        # Actually, the contract says "evaluator passes this task's degree k and girth g".
        # It implies k=3, g=16.
        pass

    # --- Helper: Fast Girth Check using Non-Backtracking Walks ---
    # We need to verify that no cycle of length < g exists.
    # A cycle of length L < g exists iff there is a non-backtracking walk of length L-1 
    # that returns to the start? 
    # Actually, it's easier to check: 
    # For each vertex v, perform a BFS-like expansion but tracking the previous node 
    # to avoid backtracking. If we return to v at depth d < g, we have a cycle.
    # However, simple BFS from v finds the shortest cycle *containing v*? No, 
    # BFS from v finds the shortest distance to any node, but cycles can be found 
    # by looking for edges (u, w) where dist[u] + dist[w] + 1 < g?
    # The most robust and fast method for small g (like 16) and moderate n (like 1000) 
    # is to check for non-backtracking walks of length g-1 that close the loop?
    # No, that's too expensive (branching factor 2, depth 15 -> 32768 paths per node).
    
    # Better method: 
    # The girth is the minimum length of a cycle.
    # We can check if the ball of radius r = (g-1)//2 around every vertex is a tree.
    # If the ball of radius r is a tree, then there is no cycle of length <= 2r+1.
    # If g is even, g=16, r=7. 2*7+1 = 15. 
    # If the ball of radius 7 is a tree, girth >= 16.
    # If the ball of radius 7 is NOT a tree, there is a cycle of length <= 15.
    
    def check_girth(adj, n, g):
        r = (g - 1) // 2
        # For each start node, BFS up to depth r.
        # If we encounter a node that is already visited (and not the parent), 
        # and the sum of depths + 1 <= 2*r (which is g-1 if g even, g-2 if g odd), 
        # we have a short cycle.
        # Actually, if we find ANY cross-edge within the BFS tree of depth r, 
        # the cycle length is dist[u] + dist[v] + 1.
        # We need to ensure dist[u] + dist[v] + 1 >= g for all such edges?
        # No, if dist[u] + dist[v] + 1 < g, we fail.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # v is already visited
                        # If v is parent of u, it's a tree edge, skip.
                        if parent[u] == v:
                            continue
                        # If u is parent of v, it's a tree edge (but v would have been visited earlier? 
                        # No, if v is parent of u, we skip. If u is parent of v, then dist[v] = dist[u]-1.
                        # In that case, we saw v earlier. This is a back-edge to parent, skip.)
                        if parent[v] == u:
                            continue
                        
                        # This is a cross-edge or back-edge to a non-parent ancestor.
                        # The cycle length is dist[u] + dist[v] + 1.
                        # Wait, this formula is only valid if the LCA of u and v is the start node?
                        # No. The cycle formed by the path start->u, edge (u,v), path v->start 
                        # is not necessarily simple if LCA != start.
                        # However, if the ball of radius r is a tree, there are NO cross edges 
                        # within the ball. 
                        # If we find a cross edge (u,v) where dist[u] <= r and dist[v] <= r,
                        # then the cycle length is at most dist[u] + dist[v] + 1 <= 2r + 1.
                        # If g is even, 2r+1 = g-1 < g. Fail.
                        # If g is odd, 2r+1 = g-2 < g. Fail.
                        # So, if we find ANY cross edge within depth r, girth < g.
                        return False
        return True

    # --- Initial Graph Construction ---
    # Generate a random 3-regular graph on n vertices.
    # We want n to be small. Let's start with n = 512 (lower bound) or slightly higher.
    # Moore bound for (3,16) is 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 * 127 = 254? 
    # Wait, Moore bound M(k,g) for even g=2r is 2 * sum_{i=0}^{r-1} (k-1)^i.
    # r=8. Sum = 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255.
    # M = 2 * 255 = 510.
    # So n must be >= 510.
    # Let's try n = 512.
    
    # We will try several n values starting from the lower bound.
    # Since random 3-regular graphs on small n likely have small girth, 
    # we need to "anneal" them to increase girth.
    
    def generate_random_3_regular(n):
        # n must be even.
        if n % 2 != 0:
            n += 1
        # Configuration model: create 3n stubs, pair them randomly.
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        random.shuffle(stubs)
        
        edges = set()
        adj = [[] for _ in range(n)]
        
        # Pair stubs
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Self-loop, retry? Configuration model might produce self-loops/multiedges.
                # For a simple graph, we need to reject or fix.
                # Given n is small, probability of self-loop is low but non-zero.
                # Let's just restart if we get a self-loop or multiedge.
                return generate_random_3_regular(n)
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                # Multiedge
                return generate_random_3_regular(n)
            edges.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
            
        # Check degree
        for i in range(n):
            if len(adj[i]) != 3:
                return generate_random_3_regular(n)
                
        return adj, n

    def get_shortest_cycle_len(adj, n):
        # Returns the girth, or a large number if none found within limit
        # We can use the same logic as check_girth but return the min length found.
        # For efficiency, we just need to know if girth < g.
        # But for annealing, knowing the exact girth helps.
        # Let's just use check_girth for now, it's binary.
        pass

    # --- Simulated Annealing ---
    
    # We want to minimize n. 
    # Strategy:
    # 1. Pick a target n (start from 512).
    # 2. Generate random graph.
    # 3. If girth < g, perform local search to increase girth.
    #    Move: Swap endpoints of two edges.
    #    Edge (u, v) and (x, y) -> (u, y) and (x, v).
    #    This preserves 3-regularity.
    #    We accept moves that reduce the number of short cycles (or increase girth).
    # 4. If we succeed, try smaller n? No, n is fixed in the graph structure.
    #    We can try to find a graph with n=512, then 514, etc.
    #    Actually, we want the SMALLEST n. So we should try n=512 first.
    #    If we can't find a valid graph at n=512 within time, try 514, etc.
    
    best_n = None
    best_edges = None
    
    # Time budget for each n
    # Total time 145s.
    # Let's try n=512 for 30s, n=514 for 30s, etc.
    
    current_n = 512
    while time.time() < deadline - 5.0:
        if current_n % 2 != 0:
            current_n += 1
            
        # Try to find a valid graph of size current_n
        # Use a time slice for this n
        slice_deadline = time.time() + 20.0 # 20s per n attempt
        
        found = False
        while time.time() < slice_deadline and not found:
            adj, n = generate_random_3_regular(current_n)
            
            # Check initial girth
            if check_girth(adj, n, g):
                found = True
                best_n = n
                # Convert adj to edges
                best_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            best_edges.append((u, v))
                break
            
            # Local Search (Annealing)
            # We want to eliminate short cycles.
            # Metric: Number of cycles of length < g? 
            # Simpler: Just try to increase girth.
            # Accept a move if it doesn't decrease the girth (or increases it).
            # Since check_girth is expensive, we can use a faster proxy?
            # Or just run check_girth occasionally.
            
            # Let's use a greedy hill-climbing with random restarts.
            # Actually, let's just do random swaps and check girth every K swaps.
            
            swaps_per_check = 50
            for _ in range(swaps_per_check):
                # Pick two random edges
                # We need a list of edges for easy access
                # Rebuilding edge list every time is expensive.
                # Let's maintain an edge list.
                
                # This is getting complex. Let's simplify.
                # Just do random swaps on the adjacency list?
                # Hard to pick random edges from adj list without a list.
                
                # Let's maintain an edge list.
                if not hasattr(adj, '_edge_list'):
                    adj._edge_list = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                adj._edge_list.append((u, v))
                
                # Pick two random edges
                e1_idx = random.randint(0, len(adj._edge_list) - 1)
                e2_idx = random.randint(0, len(adj._edge_list) - 1)
                if e1_idx == e2_idx:
                    continue
                
                u1, v1 = adj._edge_list[e1_idx]
                u2, v2 = adj._edge_list[e2_idx]
                
                # Check if swap creates self-loop or multiedge
                # New edges: (u1, v2) and (u2, v1)
                # Or (u1, u2) and (v1, v2)? 
                # Standard 2-opt swap: (u1, v1) and (u2, v2) -> (u1, v2) and (u2, v1).
                
                if u1 == v2 or u2 == v1:
                    continue # Self-loop
                
                # Check multiedge
                # We need to check if (u1, v2) or (u2, v1) already exists.
                # Using a set of edges for O(1) lookup.
                if not hasattr(adj, '_edge_set'):
                    adj._edge_set = set(adj._edge_list)
                
                e_new1 = (min(u1, v2), max(u1, v2))
                e_new2 = (min(u2, v1), max(u2, v1))
                
                if e_new1 in adj._edge_set or e_new2 in adj._edge_set:
                    continue
                
                # Perform swap
                # Remove old edges
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj._edge_set.remove((min(u1, v1), max(u1, v1)))
                adj._edge_set.remove((min(u2, v2), max(u2, v2)))
                
                # Add new edges
                adj[u1].append(v2)
                adj[v2].append(u1)
                adj[u2].append(v1)
                adj[v1].append(u2)
                
                adj._edge_set.add(e_new1)
                adj._edge_set.add(e_new2)
                
                adj._edge_list[e1_idx] = e_new1
                adj._edge_list[e2_idx] = e_new2
                
            # Check girth after swaps
            if check_girth(adj, n, g):
                found = True
                best_n = n
                best_edges = list(adj._edge_set)
                break
        
        if found:
            # We found a valid graph of size current_n.
            # Since we are iterating n upwards, this is the smallest n found so far.
            # We can stop? 
            # Yes, because we want the minimum n.
            # However, we might have failed to find a valid graph at a smaller n 
            # due to time limits, but found one at current_n.
            # Is it possible a smaller n exists? Yes.
            # But we tried them. 
            # So we can return this.
            break
        
        current_n += 2 # Next even number

    if best_n is None:
        # Fallback: Return a large graph that is definitely valid?
        # Or use the LUW construction from the previous program as a fallback.
        # Let's implement a quick LUW fallback to ensure we return *something* valid.
        
        def construct_luw(k, g):
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

        best_n, best_edges = construct_luw(k, g)

    return best_n, best_edges
