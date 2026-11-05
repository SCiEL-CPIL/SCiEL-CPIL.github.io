# EVOLVE-BLOCK-START
import sys
from collections import defaultdict, deque
import heapq
from random import Random

def construct():
    # Generate the three instance graphs
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    results = []
    for spec in specs:
        edges = instance(*spec)
        n = spec[0]
        
        best_result = None
        best_count = float('inf')
        
        # Run multiple times with different random seeds to find a better decomposition
        for run_seed in range(20):
            result = decompose_graph(n, edges, run_seed)
            count = len(result)
            if count < best_count:
                best_count = count
                best_result = result
                
        results.append(best_result)
    
    return results

def decompose_graph(n, edges, seed=0):
    """
    Decompose the undirected graph into minimum number of cycles and edges.
    Uses a randomized greedy approach: repeatedly find a cycle, remove it.
    Improvement: Select the longest cycle among multiple random attempts.
    """
    if not edges:
        return []
    
    rng = Random(seed)
    
    # Build adjacency structure
    edge_set = set()
    adj = defaultdict(list) # List for shuffling
    for u, v in edges:
        e = (min(u,v), max(u,v))
        edge_set.add(e)
        adj[u].append(v)
        adj[v].append(u)
    
    # Shuffle adjacency lists to randomize cycle selection
    for u in adj:
        rng.shuffle(adj[u])
    
    pieces = []
    remaining_edges = set(edge_set)
    
    max_iterations = 1000
    iteration = 0
    while remaining_edges and iteration < max_iterations:
        iteration += 1
        
        # Build adjacency from remaining edges for the current step
        rem_adj = defaultdict(list)
        for u, v in remaining_edges:
            rem_adj[u].append(v)
            rem_adj[v].append(u)
            
        # Find the best (longest) cycle among several random attempts
        best_cycle = None
        best_len = 0
        
        # Try multiple random shuffles to find a longer cycle
        for _ in range(10):
            # Create a copy of rem_adj to shuffle
            temp_adj = defaultdict(list)
            for u in rem_adj:
                temp_adj[u] = rem_adj[u][:]
                rng.shuffle(temp_adj[u])
            
            # Find a cycle using the shuffled adjacency
            cycle = find_cycle(temp_adj, n)
            if cycle:
                # Verify validity (should be valid by construction of find_cycle on temp_adj)
                # But we need to ensure edges exist in remaining_edges
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    e = (min(u,v), max(u,v))
                    if e not in remaining_edges:
                        valid = False
                        break
                
                if valid and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
        
        if best_cycle is None:
            break
        
        # Remove cycle edges
        for i in range(len(best_cycle)):
            u = best_cycle[i]
            v = best_cycle[(i+1) % len(best_cycle)]
            e = (min(u,v), max(u,v))
            remaining_edges.discard(e)
        
        # Add cycle as a piece
        pieces.append(best_cycle)
    
    # Remaining edges become individual pieces
    for u, v in remaining_edges:
        pieces.append([u, v])
    
    return pieces

def find_cycle(adj, n):
    """Find a simple cycle in the undirected graph using DFS. Returns list of vertices in cycle or None."""
    visited = [False] * n
    parent = [-1] * n
    
    def dfs(start):
        stack = [(start, iter(adj[start]))]
        visited[start] = True
        while stack:
            node, it = stack[-1]
            advanced = False
            for neighbor in it:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    stack.append((neighbor, iter(adj[neighbor])))
                    advanced = True
                    break
                elif neighbor != parent[node]:
                    # Found a back edge
                    # Reconstruct cycle
                    cycle = [node]
                    curr = node
                    while curr != neighbor:
                        curr = parent[curr]
                        cycle.append(curr)
                    if len(cycle) >= 3:
                        return cycle
            if not advanced:
                stack.pop()
        return None
    
    for start in range(n):
        if not visited[start] and adj[start]:
            cycle = dfs(start)
            if cycle:
                return cycle
    return None
# EVOLVE-BLOCK-END
