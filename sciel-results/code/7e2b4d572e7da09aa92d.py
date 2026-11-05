"""Decompositions of three fixed graphs into few cycles and edges (Erdős-Gallai)."""
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
        # We need to ensure we only traverse existing edges
        rem_adj = defaultdict(list)
        for u, v in remaining_edges:
            rem_adj[u].append(v)
            rem_adj[v].append(u)
            
        # Shuffle again to maintain randomness in subsequent steps
        for u in rem_adj:
            rng.shuffle(rem_adj[u])
        
        # Find a cycle
        cycle = find_cycle(rem_adj, n)
        if cycle is None:
            break
        
        # Verify the cycle is valid
        cycle_edges = []
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u,v), max(u,v))
            if e not in remaining_edges:
                valid = False
                break
            cycle_edges.append(e)
        
        if not valid or len(cycle) < 3:
            break
        
        # Remove cycle edges
        for e in cycle_edges:
            remaining_edges.discard(e)
        
        # Add cycle as a piece
        pieces.append(cycle)
    
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
