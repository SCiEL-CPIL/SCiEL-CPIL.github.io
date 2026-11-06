import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    n = 40
    
    # Initial graph: 8 disjoint K5s
    # Vertices 0-4, 5-9, ..., 35-39
    adj = np.zeros((n, n), dtype=int)
    
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                adj[u, v] = 1
                adj[v, u] = 1
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
    
    def compute_power_5(a):
        # Compute a^5
        a2 = a @ a
        a3 = a2 @ a
        a4 = a3 @ a
        a5 = a4 @ a
        return a5
    
    a5 = compute_power_5(adj)
    
    # Greedy addition
    # We want to add edges (u, v) where adj[u,v]==0 and a5[u,v]==0
    # To maximize edges, we should probably add edges that are "most central" or just iterate.
    # Let's try a few random restarts or different orders.
    
    best_edges = edges
    best_count = len(edges)
    
    # Try multiple random permutations of candidate edges to avoid local minima
    for restart in range(50): # Adjust based on time
        if time.time() > deadline:
            break
        
        # Restore initial state
        adj = np.zeros((n, n), dtype=int)
        for k in range(8):
            base = k * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    u = base + i
                    v = base + j
                    adj[u, v] = 1
                    adj[v, u] = 1
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        
        a5 = compute_power_5(adj)
        
        # Candidate edges
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 0 and a5[u, v] == 0:
                    candidates.append((u, v))
        
        # Shuffle candidates to try different orders
        random.shuffle(candidates)
        
        added = 0
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            # Check if still valid (adj might have changed)
            if adj[u, v] == 0:
                # Re-check a5 condition? 
                # If we add edges, a5 changes. An edge that was safe might become unsafe.
                # So we MUST recompute a5 after every addition.
                # This is expensive.
                
                # Optimization: 
                # If we recompute a5 every time, the loop is slow.
                # Let's estimate cost.
                # Max edges ~ 100.
                # 100 recomputations of 40x40 matrix power 5.
                # 40^3 * 5 * 100 = 3.2 * 10^6 operations. Very fast.
                
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append([u, v])
                added += 1
                a5 = compute_power_5(adj)
                
                # Remove (u,v) from candidates? No, we are iterating a static list.
                # We just check if it's already added or if it became unsafe.
                # Actually, if we add an edge, other candidates might become unsafe.
                # But we are just adding (u,v) if it was safe *before* this addition?
                # No, the condition for adding (u,v) is that it doesn't create a C6 in the *current* graph.
                # The current graph is the one with previous additions.
                # So we need to check if a5[u,v] == 0 in the *current* a5.
                
                # My loop structure above is flawed because I pre-computed candidates.
                # I should re-evaluate candidates or check condition dynamically.
                
                # Let's restructure:
                # While there are safe edges, add one.
        
        # Let's restart the loop logic inside the restart block properly.
        
        # Re-do the restart logic correctly
        adj = np.zeros((n, n), dtype=int)
        for k in range(8):
            base = k * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    u = base + i
                    v = base + j
                    adj[u, v] = 1
                    adj[v, u] = 1
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        
        a5 = compute_power_5(adj)
        
        progress = True
        while progress:
            if time.time() > deadline:
                break
            
            progress = False
            # Find all safe edges
            safe_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v] == 0 and a5[u, v] == 0:
                        safe_edges.append((u, v))
            
            if not safe_edges:
                break
            
            # Pick one random safe edge to add
            # Or pick the one with max degree sum?
            # Random helps escape local minima.
            u, v = random.choice(safe_edges)
            
            adj[u, v] = 1
            adj[v, u] = 1
            edges.append([u, v])
            progress = True
            
            # Recompute a5
            a5 = compute_power_5(adj)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    return best_edges
