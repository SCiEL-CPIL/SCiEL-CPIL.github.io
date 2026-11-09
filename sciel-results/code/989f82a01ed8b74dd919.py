import time
import random
import numpy as np
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 40
    # adjacency as set of frozensets for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    
    def has_path_len5(u, v, depth_limit=5):
        """Check if there's a simple path of length 5 from u to v (4 intermediate vertices)."""
        # BFS/DFS from u, looking for v at exactly distance 5
        # State: (current, visited_frozenset)
        # This could be expensive, so let's be smart
        
        # Use iterative deepening or just BFS with visited tracking
        # Since n=40 and we only go depth 5, this is manageable
        from collections import defaultdict
        
        # BFS layer by layer
        # Layer 0: {u}
        # Layer 1: neighbors of u (excluding v to avoid short paths)
        # ...
        # Layer 5: should contain v
        
        # Track: layer -> set of (node, frozenset_of_visited)
        # This can blow up, so let's use a different approach
        
        # Actually, for depth 5 with 40 vertices, let's just do DFS with pruning
        def dfs(current, target, remaining_depth, visited):
            if remaining_depth == 0:
                return current == target
            for nb in adj[current]:
                if nb == target and remaining_depth == 1:
                    return True
                if nb not in visited and nb != target:  # don't go to target early
                    visited.add(nb)
                    if dfs(nb, target, remaining_depth - 1, visited):
                        visited.discard(nb)
                        return True
                    visited.discard(nb)
            return False
        
        visited = {u}
        return dfs(u, v, 5, visited)
    
    def can_add_edge(u, v):
        """Check if adding edge (u,v) creates a C6."""
        if u == v or (u, v) in edge_set or (v, u) in edge_set:
            return False
        # Check if there's a path of length 5 from u to v
        return not has_path_len5(u, v)
    
    # Greedy randomized approach
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if time.time() > deadline:
            break
        if can_add_edge(u, v):
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
    
    # Try multiple restarts with different random orders
    best_edges = len(edge_set)
    best_adj = [s.copy() for s in adj]
    best_edge_set = set(edge_set)
    
    while time.time() < deadline - 10:
        # New restart
        adj = [set() for _ in range(n)]
        edge_set = set()
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline - 10:
                break
            if can_add_edge(u, v):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
        
        if len(edge_set) > best_edges:
            best_edges = len(edge_set)
            best_adj = [s.copy() for s in adj]
            best_edge_set = set(edge_set)
    
    # Return edges
    return [[u, v] for u, v in best_edge_set if u < v]
