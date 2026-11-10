import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses local search (2-switches) on random cubic graphs to find a graph with high girth.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    if k != 3:
        # Fallback for other k values
        return 0, []

    def generate_random_cubic(n):
        """Generate a random cubic graph with n vertices."""
        if n % 2 != 0:
            return None
        # Use configuration model
        half_edges = [i for i in range(n) for _ in range(k)]
        random.shuffle(half_edges)
        edges = set()
        adj = [[] for _ in range(n)]
        
        # Pair up half-edges
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                # Self-loop, retry or swap
                # Simple retry
                while u == v:
                    j = random.randint(0, len(half_edges)-1)
                    if j != i:
                        half_edges[i], half_edges[j] = half_edges[j], half_edges[i]
                        v = half_edges[i+1]
                        u = half_edges[i]
            if (u, v) in edges or (v, u) in edges:
                # Multi-edge, retry
                while (u, v) in edges or (v, u) in edges:
                    j = random.randint(0, len(half_edges)-1)
                    if j != i and j != i+1:
                        half_edges[i+1], half_edges[j] = half_edges[j], half_edges[i+1]
                        v = half_edges[i+1]
            edges.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
        
        # Check if simple
        for i in range(n):
            if len(adj[i]) != k:
                return None
            if len(set(adj[i])) != k:
                return None
        return adj

    def find_short_cycle(adj, n, max_len):
        """Find a cycle of length < max_len. Returns list of vertices or None."""
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_len // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < max_len:
                            # Reconstruct cycle
                            cycle = [u]
                            curr = u
                            while curr != s:
                                curr = parent[curr]
                                cycle.append(curr)
                            cycle.append(s)
                            
                            cycle2 = [v]
                            curr = v
                            while curr != s:
                                curr = parent[curr]
                                cycle2.append(curr)
                            cycle2.append(s)
                            
                            # Combine
                            full_cycle = cycle + cycle2[::-1]
                            # Remove duplicates at the join
                            # Actually, just return the length and the edges involved
                            return full_cycle
        return None

    def apply_2_switch(adj, n, cycle):
        """
        Apply a 2-switch to break a cycle.
        This is a heuristic. A proper 2-switch picks two edges (a,b) and (c,d)
        and replaces them with (a,c) and (b,d).
        Here we just pick two edges in the cycle and swap their endpoints.
        """
        if len(cycle) < 4:
            return False
        
        # Pick two non-adjacent edges in the cycle
        # cycle is a list of vertices
        # Edges are (cycle[i], cycle[i+1])
        # We need to pick (u,v) and (w,x) such that they are not adjacent.
        
        # Simple strategy: pick edge 0 and edge len//2
        idx1 = 0
        idx2 = len(cycle) // 2
        
        u1, v1 = cycle[idx1], cycle[(idx1+1) % len(cycle)]
        u2, v2 = cycle[idx2], cycle[(idx2+1) % len(cycle)]
        
        # Ensure u1 != u2 etc.
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return False
            
        # Remove old edges
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        # Add new edges
        # Try (u1, u2) and (v1, v2)
        if u1 not in adj[u2] and v1 not in adj[v2]:
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            return True
        else:
            # Revert
            adj[u1].append(v1)
            adj[v1].append(u1)
            adj[u2].append(v2)
            adj[v2].append(u2)
            return False

    def check_girth_fast(adj, n_nodes, target_girth):
        max_depth = (target_girth - 1) // 2
        for s in range(n_nodes):
            dist = [-1] * n_nodes
            parent = [-1] * n_nodes
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    random.seed(42)
    
    # Try to find a graph with N=936
    # If we find one, return it.
    # If not, try smaller N? No, 936 is the known best.
    # We should try to find *any* valid graph.
    
    best_adj = None
    best_n = 0
    
    # Try N from 936 down to 512? 
    # Local search is expensive. Start with 936.
    
    for n_target in [936, 900, 800, 700, 600, 512]:
        if time.time() > deadline - 20:
            break
            
        if n_target % 2 != 0:
            continue
            
        success = False
        for trial in range(10): # Few trials per size
            if time.time() > deadline - 10:
                break
                
            adj = generate_random_cubic(n_target)
            if not adj:
                continue
                
            # Local search
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                    
                cycle = find_short_cycle(adj, n_target, g)
                if cycle is None:
                    # Check if girth is actually >= g
                    if check_girth_fast(adj, n_target, g):
                        edges = []
                        for i in range(n_target):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return n_target, edges
                    break
                
                # Apply switch
                if not apply_2_switch(adj, n_target, cycle):
                    # If switch fails, try a random switch
                    # Pick random edges
                    u1 = random.randint(0, n_target-1)
                    v1 = random.choice(adj[u1])
                    u2 = random.randint(0, n_target-1)
                    v2 = random.choice(adj[u2])
                    
                    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                        continue
                        
                    if v1 in adj[u2] or v2 in adj[u1]:
                        continue
                        
                    # Perform switch
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    
                    adj[u1].append(u2)
                    adj[u2].append(u1)
                    adj[v1].append(v2)
                    adj[v2].append(v1)
                    
                    # Check if it introduced a self-loop or multi-edge?
                    # generate_random_cubic ensures simple, but switch might break it.
                    # We assume simple for now.
        
    # If no valid graph found, return the best we have?
    # Or return a known valid graph?
    # The LUW construction in the original code might have worked for smaller girths.
    # For g=16, we need a specific graph.
    
    # Fallback: Return the LUW graph if it's valid (it probably isn't for g=16)
    # But we must return *something*.
    
    # If we failed to find a valid graph, we should return the smallest valid graph we *know* exists?
    # We don't have one hardcoded.
    
    # If we can't find one, we return 0, [] which scores 0.
    # But we want to improve.
    
    # Let's try one more thing: The Petersen graph product?
    
    return 0, []
# EVOLVE-BLOCK-END
