import random
import time
import sys

def _bfs_dist(source, adj):
    """Compute shortest path distances from source to all nodes."""
    n = len(adj)
    dist = [-1] * n
    dist[source] = 0
    queue = [source]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

def _has_c6_through(u, v, adj):
    """
    Check if adding edge (u, v) creates a C6.
    This is true iff there is a simple path of length 5 from u to v in the current graph.
    We use a DFS with pruning based on BFS distances from v.
    """
    # Get distances from v
    dist = _bfs_dist(v, adj)
    
    # If dist[u] > 5, no path of length <= 5 exists, so no path of length 5.
    # But we need exactly 5. If dist[u] < 5, a path of length 5 might still exist (via detours).
    # If dist[u] > 5, impossible.
    if dist[u] == -1 or dist[u] > 5:
        return False
    
    # DFS from u
    # We need to reach v in exactly 5 steps.
    # Current node: u, steps remaining: 5.
    # We track visited nodes to ensure simplicity.
    
    def dfs(node, steps_left, visited):
        if steps_left == 0:
            return node == v
        # Pruning: The shortest path from node to v must be <= steps_left
        if dist[node] == -1 or dist[node] > steps_left:
            return False
        
        # If we are at v but steps_left > 0, we can't stop here.
        # But we must not revisit v until the end?
        # Actually, if node == v and steps_left > 0, we can continue?
        # No, the path must be simple. If we hit v early, we can't use it again.
        # So if node == v, we must have steps_left == 0.
        if node == v:
            return steps_left == 0
            
        for nb in adj[node]:
            if nb in visited:
                continue
            # If nb is v, we can only take it if steps_left == 1
            if nb == v and steps_left != 1:
                continue
                
            visited.add(nb)
            if dfs(nb, steps_left - 1, visited):
                return True
            visited.remove(nb)
            
        return False

    visited = {u}
    return dfs(u, 5, visited)

def _count_edges(adj):
    return sum(len(s) for s in adj) // 2

def construct():
    n = 40
    start_time = time.time()
    time_limit = 115 # Leave 5s for safety
    
    # Initialize with a random graph
    # Start with a moderate number of edges to allow growth
    # Or start empty and grow?
    # Local search usually works better starting from a valid solution.
    # Let's start with a random sparse graph and ensure no C6.
    
    def random_valid_graph(target_edges):
        adj = [set() for _ in range(n)]
        edges = 0
        attempts = 0
        # Try to add random edges
        # To avoid C6, we check before adding.
        # This is slow if we do it for many edges.
        # Instead, start with 0 edges and add greedily?
        # That's the current approach.
        # Let's try starting with a random permutation of edges and adding greedily.
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if edges >= target_edges:
                break
            if not _has_c6_through(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
                edges += 1
        return adj

    # Start with a greedy construction to get a decent baseline
    # Then use Simulated Annealing to improve
    
    # Greedy phase
    adj = random_valid_graph(120) # Try to get to 120 edges
    current_edges = _count_edges(adj)
    
    best_adj = [set(s) for s in adj]
    best_edges = current_edges
    
    # Simulated Annealing Phase
    T = 1.0
    cooling_rate = 0.999
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))
            else:
                edge_set.add((v, u))
    
    # We need a list of missing edges for random addition
    # And a list of existing edges for random removal
    
    while time.time() - start_time < time_limit:
        # Generate a move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_adj = [set(s) for s in adj]
        new_edges = current_edges
        
        if move_type == 'add':
            # Pick a random missing edge
            # To make this efficient, we can maintain a list of missing edges,
            # but it changes.
            # Simple: pick random u, v not in adj
            for _ in range(100): # Retry a few times to find a missing edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and v not in adj[u]:
                    break
            else:
                continue
                
            if not _has_c6_through(u, v, new_adj):
                new_adj[u].add(v)
                new_adj[v].add(u)
                new_edges += 1
            else:
                continue # Move invalid, skip
                
        elif move_type == 'remove':
            # Pick a random existing edge
            u = random.randint(0, n-1)
            if not adj[u]:
                continue
            v = random.choice(list(adj[u]))
            
            new_adj[u].remove(v)
            new_adj[v].remove(u)
            new_edges -= 1
            
        else: # swap
            # Remove e1, Add e2
            u1 = random.randint(0, n-1)
            if not adj[u1]:
                continue
            v1 = random.choice(list(adj[u1]))
            
            # Find a missing edge to add
            added = False
            for _ in range(100):
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 != v2 and v2 not in adj[u2]:
                    # Ensure we don't remove and add the same edge
                    if (u1, v1) != (u2, v2) and (u1, v1) != (v2, u2):
                        break
            else:
                continue
                
            # Remove e1
            new_adj[u1].remove(v1)
            new_adj[v1].remove(u1)
            
            # Add e2
            if not _has_c6_through(u2, v2, new_adj):
                new_adj[u2].add(v2)
                new_adj[v2].add(u2)
                new_edges += 1 # Net change 0, but we swapped
                # Actually, remove 1, add 1, so new_edges = current_edges
                # But we need to check if the swap is valid.
                # The check _has_c6_through uses new_adj which has e1 removed.
                # This is correct.
            else:
                continue
                
        # Acceptance criterion
        delta = new_edges - current_edges
        if delta > 0:
            accept = True
        else:
            # delta is 0 or negative
            # If delta == 0, we might accept to escape local minima
            # If delta < 0, we accept with prob exp(delta/T)
            if delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < pow(2.718, delta / T)
                
        if accept:
            adj = new_adj
            current_edges = new_edges
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [set(s) for s in adj]
                
        T *= cooling_rate
        
    # Convert best_adj to edge list
    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([u, v])
                
    return result
