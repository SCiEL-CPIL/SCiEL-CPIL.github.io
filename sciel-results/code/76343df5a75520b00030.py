import time
import random

def construct(k, g):
    if k != 3:
        return 0, []

    start_time = time.time()
    time_limit = 110.0

    def generate_random_3_regular(n):
        if n < 2 or n % 2 != 0:
            return None
        stubs = list(range(n)) * 3
        random.shuffle(stubs)
        edges = []
        seen = set()
        for i in range(0, 3*n, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            key = (u, v) if u < v else (v, u)
            if key in seen:
                return None
            edges.append(key)
            seen.add(key)
        return edges

    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def find_shortest_cycle(n, adj, start):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        min_cycle = float('inf')
        cycle_edges = []
        for u in queue:
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        # Reconstruct cycle
                        # This is expensive, so we just return the length for now
                        # Or we can return the edges
        return min_cycle

    # We will try to find a graph.
    # Start with N=512.
    # If we fail, increase N.
    
    current_n = 512
    while current_n <= 1200:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Try a few random instances for this N
        found = False
        for _ in range(20):
            if time.time() - start_time > time_limit - 5:
                break
            edges = generate_random_3_regular(current_n)
            if edges is None:
                continue
            
            # Check girth
            if check_girth(current_n, edges, g):
                return current_n, edges
            
            # If not, try to improve via local search
            # (Simplified: just restart for now, as full local search is complex)
            # But random restarts won't work.
            # I will add a simple edge swap heuristic.
            
            # Build adj for swaps
            adj = [[] for _ in range(current_n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            improved = False
            for swap_iter in range(100):
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Find a short cycle
                # We'll just pick a random edge and try to swap it
                # This is a blind search.
                
                # Pick random edge index
                idx = random.randint(0, len(edges)-1)
                u1, v1 = edges[idx]
                
                # Find a partner edge (u2, v2) such that u1 != u2, v1 != u2, etc.
                # And not adjacent.
                partner_idx = random.randint(0, len(edges)-1)
                if partner_idx == idx: continue
                u2, v2 = edges[partner_idx]
                
                # Check validity of swap (u1, u2) and (v1, v2)
                # Must not be loops
                if u1 == u2 or v1 == v2: continue
                # Must not be multi-edges
                # Check if (u1, u2) exists in adj
                if u2 in adj[u1] or v2 in adj[v1]: continue
                # Check if (u1, v2) exists? No, we are swapping to (u1, u2) and (v1, v2).
                # Wait, original edges: (u1, v1) and (u2, v2).
                # New edges: (u1, u2) and (v1, v2).
                # We need to ensure (u1, u2) is not an existing edge (other than the ones being removed).
                # And (v1, v2) is not an existing edge.
                
                # This check is O(degree) which is small.
                
                # Perform swap
                edges[idx] = (u1, u2)
                edges[partner_idx] = (v1, v2)
                
                # Update adj
                # Remove old, add new.
                # This is messy to maintain.
                # I will rebuild adj after swap if we want to check girth.
                # But checking girth is O(N^2). 100 swaps * O(N^2) is too slow.
                
                # So I will only check girth after 100 swaps.
                # This is a blind hill climb.
            
            # After 100 swaps, check girth
            if check_girth(current_n, edges, g):
                return current_n, edges

        current_n += 16

    return 0, []
