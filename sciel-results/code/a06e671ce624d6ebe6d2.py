import time
import sys
from collections import deque

N = 40
DEADLINE_SECONDS = 55

def construct():
    t0 = time.time()
    
    # 1. Initialize with the Fan
    # Path 0-1-...-39
    # Spokes 0-i for i=2..39
    edges = set()
    for i in range(N - 1):
        edges.add((i, i + 1))
    for i in range(2, N):
        edges.add((0, i))
        
    # List of edges for iteration
    edge_list = list(edges)
    
    # Helper to check if edge exists
    def get_edge_set(edge_list):
        s = set()
        for u, v in edge_list:
            s.add((min(u, v), max(u, v)))
        return s
        
    # Current edge set
    current_edges = get_edge_set(edge_list)
    
    # Helper to build adjacency
    def build_adjacency(edges):
        adj = {i: [] for i in range(N)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Helper to check 2-connectivity
    def is_2_connected(edges):
        if len(edges) < N:
            return False
        adj = build_adjacency(edges)
        # Check for articulation points
        # Simple DFS
        visited = [False] * N
        disc = [0] * N
        low = [0] * N
        parent = [-1] * N
        time_counter = [0]
        
        def dfs(u):
            visited[u] = True
            disc[u] = low[u] = time_counter[0]
            time_counter[0] += 1
            children = 0
            for v in adj[u]:
                if not visited[v]:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if parent[u] != -1 and low[v] >= disc[u]:
                        return True
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
            if parent[u] == -1 and children > 1:
                return True
            return False
            
        for i in range(N):
            if not visited[i]:
                if dfs(i):
                    return False
        return True

    # Helper to check if a cycle of length k exists
    # Returns True if exists
    def has_cycle_of_length(k, edges):
        if k < 3 or k > N:
            return False
        if k == N:
            # Hamiltonian cycle check
            # Use DFS with pruning
            adj = build_adjacency(edges)
            # Simple backtracking
            path = [0]
            visited = [False] * N
            visited[0] = True
            
            def backtrack(current, length):
                if length == k:
                    # Check if current connects to start
                    return 0 in adj[current]
                
                for neighbor in adj[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        path.append(neighbor)
                        if backtrack(neighbor, length + 1):
                            return True
                        path.pop()
                        visited[neighbor] = False
                return False
                
            return backtrack(0, 1)

        # For k < N, use DFS
        adj = build_adjacency(edges)
        # Try starting from each node?
        # To save time, we can just try from 0?
        # No, cycle might not include 0.
        # But in our sparse graphs, likely it does.
        # Let's try from all nodes if N is small?
        # N=40 is too big for all nodes x all paths.
        
        # Optimization: 
        # If k is small, we can just search.
        # If k is large, we can check if the graph is "pancyclic-like".
        
        # Let's use a simple DFS from each node, but limit depth.
        # This might be slow.
        
        # Alternative: 
        # We know the graph is derived from a fan.
        # Maybe we can just check if the girth is <= k and the max cycle is >= k?
        # No.
        
        # Let's use a BFS for shortest path? No.
        
        # For the purpose of this search, we will assume that if the graph is 2-connected 
        # and has a triangle, and we are removing edges from a known pancyclic graph,
        # we can use a heuristic: 
        # Check if a cycle of length k exists by running a limited DFS.
        
        for start in range(N):
            visited = [False] * N
            visited[start] = True
            path = [start]
            
            def dfs(current, depth):
                if depth == k:
                    return start in adj[current]
                for neighbor in adj[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        path.append(neighbor)
                        if dfs(neighbor, depth + 1):
                            return True
                        path.pop()
                        visited[neighbor] = False
                return False
                
            if dfs(start, 1):
                return True
        return False

    def is_pancyclic(edges):
        if not is_2_connected(edges):
            return False
        for k in range(3, N + 1):
            if not has_cycle_of_length(k, edges):
                return False
        return True

    # Greedy Removal
    # Order: Try to remove spokes first? Or path edges?
    # Spokes are "redundant" for connectivity but crucial for specific cycles.
    # Path edges are crucial for the main structure.
    
    # Let's sort edges: Spokes first (higher index first?), then Path.
    # Actually, let's just iterate.
    
    for e in edge_list:
        if time.time() - t0 > DEADLINE_SECONDS - 5:
            break
            
        # Try to remove e
        new_edges = current_edges - {e}
        if len(new_edges) < N: # Must have at least N edges? No, N+1 min.
            continue
            
        # Check if still pancyclic
        # This is the bottleneck.
        # We can skip full check if we just removed an edge that wasn't in any cycle of a specific length?
        # Hard to know.
        
        # Let's run the check.
        if is_pancyclic(new_edges):
            current_edges = new_edges

    # Final result
    edges_list = list(current_edges)
    
    # Generate cycles for the final graph
    # We need to provide explicit cycles.
    # We can find them using the same DFS logic, or just return the fan cycles if they still exist?
    # If we removed edges, the fan cycles might be broken.
    # We must find valid cycles.
    
    cycles = []
    for k in range(3, N + 1):
        # Find a cycle of length k
        found = None
        # Reuse logic
        adj = build_adjacency(current_edges)
        for start in range(N):
            visited = [False] * N
            visited[start] = True
            path = [start]
            
            def dfs(current, depth):
                if depth == k:
                    if start in adj[current]:
                        return path + [start]
                    return None
                for neighbor in adj[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        path.append(neighbor)
                        res = dfs(neighbor, depth + 1)
                        if res:
                            return res
                        path.pop()
                        visited[neighbor] = False
                return None
                
            res = dfs(start, 1)
            if res:
                found = res
                break
        if found:
            cycles.append(found)
        else:
            # Fallback: If we failed to find a cycle, the graph is not pancyclic.
            # This should not happen if is_pancyclic was correct.
            # But if it does, we return the original fan?
            # No, we must return a valid graph.
            # If the greedy search broke it, we should have kept the old edges.
            # So this case implies a bug in is_pancyclic or cycle finding.
            # We'll raise an error or return a known valid graph.
            pass

    # To be safe, if we didn't find all cycles, we revert to the Fan
    if len(cycles) != N - 2:
        # Revert to Fan
        edges = set()
        for i in range(N - 1):
            edges.add((i, i + 1))
        for i in range(2, N):
            edges.add((0, i))
        edges_list = list(edges)
        cycles = []
        for k in range(3, N + 1):
            cycles.append(list(range(k)))

    return {'edges': edges_list, 'cycles': cycles}
