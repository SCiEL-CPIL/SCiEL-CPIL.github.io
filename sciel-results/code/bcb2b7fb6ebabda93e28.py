# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 110  # Safety margin
    n = 20
    
    # Pre-compute all edges
    all_edges = list(combinations(range(n), 2))
    m = len(all_edges)
    
    # Map edge to index
    edge_idx = {e: i for i, e in enumerate(all_edges)}
    
    def is_valid_addition(adj, cn, u, v):
        """
        Check if adding edge (u,v) to current state (adj, cn) is valid.
        Returns True if valid.
        """
        # 1. u and v must have < 3 common neighbors
        # cn[u][v] is the count of common neighbors.
        if cn[u][v] >= 3:
            return False
        
        # 2. Check impact on other pairs
        # For any w != u, v:
        # If v is neighbor of w, then v becomes common neighbor of u and w.
        # So cn[u][w] increases by 1. We need new cn[u][w] < 3, so old cn[u][w] <= 1.
        # If u is neighbor of w, then u becomes common neighbor of v and w.
        # So cn[v][w] increases by 1. We need new cn[v][w] < 3, so old cn[v][w] <= 1.
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                if cn[u][w] >= 2:
                    return False
            if adj[u][w]:
                if cn[v][w] >= 2:
                    return False
            
        return True

    def add_edge(adj, cn, u, v):
        adj[u][v] = 1
        adj[v][u] = 1
        
        # Update cn
        # cn[u][v] unchanged (definition: common neighbors are distinct from u,v)
        # For w != u,v:
        # if adj[v][w] was 1, cn[u][w] += 1
        # if adj[u][w] was 1, cn[v][w] += 1
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                cn[u][w] += 1
                cn[w][u] += 1
            if adj[u][w]:
                cn[v][w] += 1
                cn[w][v] += 1

    def remove_edge(adj, cn, u, v):
        adj[u][v] = 0
        adj[v][u] = 0
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                cn[u][w] -= 1
                cn[w][u] -= 1
            if adj[u][w]:
                cn[v][w] -= 1
                cn[w][v] -= 1

    # Phase 1: Greedy Initialization
    best_count = 0
    best_edges = []
    
    def greedy_init():
        nonlocal best_count, best_edges
        adj = np.zeros((n, n), dtype=int)
        cn = np.zeros((n, n), dtype=int)
        count = 0
        edges_used = []
        
        # Try multiple random orderings
        for trial in range(20):
            random.seed(trial * 1000 + 42)
            adj = np.zeros((n, n), dtype=int)
            cn = np.zeros((n, n), dtype=int)
            count = 0
            edges_used = []
            shuffled_edges = all_edges.copy()
            random.shuffle(shuffled_edges)
            
            for u, v in shuffled_edges:
                if is_valid_addition(adj, cn, u, v):
                    add_edge(adj, cn, u, v)
                    count += 1
                    edges_used.append((u, v))
            
            if count > best_count:
                best_count = count
                best_edges = edges_used.copy()
                
    greedy_init()
    
    # Phase 2: Branch and Bound
    # We will use a recursive search.
    # To speed up, we maintain the list of "available" edges.
    # However, the set of available edges depends on the current state.
    
    adj = np.zeros((n, n), dtype=int)
    cn = np.zeros((n, n), dtype=int)
    
    # Load best greedy solution into adj/cn
    for u, v in best_edges:
        add_edge(adj, cn, u, v)
        
    current_count = best_count
    
    # Helper to get list of edges that can be added
    def get_addable_edges():
        addable = []
        for i, (u, v) in enumerate(all_edges):
            if adj[u][v] == 0:
                if is_valid_addition(adj, cn, u, v):
                    addable.append(i)
        return addable

    # Branch and Bound function
    def bnb(depth, current_count):
        nonlocal best_count, best_edges, adj, cn
        
        if time.time() - start_time > DEADLINE:
            return
        
        # Pruning: Even if we add all remaining addable edges, can we beat best?
        # We need a tighter bound.
        # Simple bound: current_count + (m - current_count) is not helpful.
        # We calculate addable edges.
        addable = get_addable_edges()
        
        if current_count + len(addable) <= best_count:
            return
            
        if not addable:
            if current_count > best_count:
                best_count = current_count
                # Reconstruct edges from adj
                best_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
            return

        # Heuristic: Pick an edge to branch on.
        # We can pick the first one, or one that is "most constrained".
        # For simplicity and speed, we just iterate through addable edges.
        # To avoid symmetry/redundancy, we can fix an order.
        
        # Optimization: If we just want to find a better solution, we can try adding edges.
        # But B&B explores the space.
        
        # To speed up, let's try a "greedy within B&B" approach or just deep dive.
        # Given N=20, the search tree is huge.
        # We need aggressive pruning.
        
        # Let's try adding edges in a specific order.
        # We'll iterate through addable edges.
        # To reduce branching factor, we can use a "conflict" based ordering?
        
        # For now, simple iteration.
        for idx in addable:
            u, v = all_edges[idx]
            
            # Add edge
            add_edge(adj, cn, u, v)
            bnb(depth + 1, current_count + 1)
            
            # Remove edge
            remove_edge(adj, cn, u, v)
            
            if time.time() - start_time > DEADLINE:
                return

    # Run B&B
    # To make it effective, we should start from a good state.
    # We already have best_edges.
    # We will run B&B starting from the empty graph? No, that's too big.
    # We should run B&B starting from the greedy solution?
    # B&B usually works best from scratch with good bounds, or from a solution with "unfixed" edges.
    
    # Alternative: Local Search / Hill Climbing with Random Restarts is often faster for N=20.
    # The directive says "Branch-and-Bound".
    # Let's try B&B from scratch but with strong pruning.
    
    adj = np.zeros((n, n), dtype=int)
    cn = np.zeros((n, n), dtype=int)
    
    # To help B&B, we can use the greedy solution to order edges or provide a bound.
    # Let's run B&B.
    
    # We need to be careful with recursion depth (190 is fine) but time.
    # We will use a time-limited B&B.
    
    # To improve B&B performance, we can use a "branch on a vertex" strategy?
    # Or simply rely on the `addable` pruning.
    
    # Let's try running B&B.
    bnb(0, 0)
    
    # If B&B didn't finish or didn't improve, we might want to do a local polish on the best found.
    
    # Convert best_edges to list of lists
    edges = [[u, v] for u, v in best_edges]
    
    return edges

# EVOLVE-BLOCK-END
