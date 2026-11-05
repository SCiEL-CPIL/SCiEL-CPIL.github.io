import numpy as np
import time
import random

def construct():
    n = 40
    deadline = time.time() + 105.0
    
    def random_graph_with_k5():
        """Create initial graph with 8 disjoint K5s plus some random edges."""
        A = np.zeros((n, n), dtype=np.int64)
        adj = [set() for _ in range(n)]
        
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    A[u, v] = A[v, u] = 1
                    adj[u].add(v)
                    adj[v].add(u)
        
        return A, adj
    
    def compute_P5_fast(adj):
        """
        Compute P5[u][v] = 1 if there's a simple path of length 5 from u to v.
        Uses iterative BFS with bitmask tracking for efficiency.
        """
        P5 = [[0]*n for _ in range(n)]
        
        for start in range(n):
            # BFS layers: layer[d][v] = set of masks of paths of length d from start to v
            # But that's too expensive. Instead, use the fact that we just need existence.
            # Use DFS with memoization on (node, depth, visited_mask)
            
            # For each start, do DFS to find all nodes reachable in exactly 5 steps
            visited_mask = 1 << start
            found = [False]*n
            
            # Iterative DFS with stack: (node, depth, mask)
            stack = [(start, 0, visited_mask)]
            while stack:
                node, depth, mask = stack.pop()
                if depth == 5:
                    continue
                for nb in adj[node]:
                    if mask & (1 << nb):
                        continue
                    new_mask = mask | (1 << nb)
                    if depth + 1 == 5:
                        found[nb] = True
                    else:
                        stack.append((nb, depth + 1, new_mask))
            
            for v in range(n):
                if found[v]:
                    P5[start][v] = 1
        
        return P5
    
    def get_valid_non_edges(A, P5):
        """Get non-edges (u,v) where P5[u][v] == 0."""
        valid = []
        for u in range(n):
            for v in range(u+1, n):
                if A[u,v] == 0 and P5[u][v] == 0:
                    valid.append((u, v))
        return valid
    
    def get_all_edges(A):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if A[u,v] == 1:
                    edges.append((u, v))
        return edges
    
    def count_blocked_edges(A, P5):
        """Count how many non-edges are blocked by P5."""
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if A[u,v] == 0 and P5[u][v] == 1:
                    count += 1
        return count
    
    def weighted_valid_edge_selection(valid_edges, A):
        """Select an edge with probability weighted by something useful."""
        if not valid_edges:
            return None
        # Weight by degree sum - prefer connecting high-degree vertices
        weights = []
        for u, v in valid_edges:
            w = len(A[u]) + len(A[v]) + 1
            weights.append(w)
        total = sum(weights)
        r = random.random() * total
        cumsum = 0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                return valid_edges[i]
        return valid_edges[-1]
    
    best_m = -1
    best_A = None
    best_adj = None
    
    # Multiple restarts with different random seeds
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        A, adj = random_graph_with_k5()
        m = 80  # 8 * C(5,2) = 8 * 10 = 80
        
        # Try to add edges greedily
        P5 = compute_P5_fast(adj)
        
        # Greedy phase: add valid edges until stuck
        while True:
            valid = get_valid_non_edges(A, P5)
            if not valid:
                break
            if time.time() > deadline - 5:
                break
            
            # Use weighted selection
            u, v = weighted_valid_edge_selection(valid, A)
            A[u,v] = A[v,u] = 1
            adj[u].add(v)
            adj[v].add(u)
            m += 1
            
            # Recompute P5
            P5 = compute_P5_fast(adj)
        
        # Local search phase
        steps = 0
        max_steps = 500
        while steps < max_steps and time.time() < deadline - 5:
            valid = get_valid_non_edges(A, P5)
            if not valid:
                # Kick: remove a random edge
                edges = get_all_edges(A)
                if not edges:
                    break
                u, v = random.choice(edges)
                A[u,v] = A[v,u] = 0
                adj[u].discard(v)
                adj[v].discard(u)
                m -= 1
                P5 = compute_P5_fast(adj)
                steps += 1
                continue
            
            # Try adding an edge
            u, v = weighted_valid_edge_selection(valid, A)
            A[u,v] = A[v,u] = 1
            adj[u].add(v)
            adj[v].add(u)
            m += 1
            P5 = compute_P5_fast(adj)
            
            # Try to add more edges after this addition
            for _ in range(3):
                valid2 = get_valid_non_edges(A, P5)
                if not valid2:
                    break
                u2, v2 = weighted_valid_edge_selection(valid2, A)
                A[u2,v2] = A[v2,u2] = 1
                adj[u2].add(v2)
                adj[v2].add(u2)
                m += 1
                P5 = compute_P5_fast(adj)
            
            steps += 1
        
        if m > best_m:
            best_m = m
            best_A = A.copy()
            best_adj = [s.copy() for s in adj]
    
    # Convert best graph to edge list
    result = []
    if best_A is not None:
        for u in range(n):
            for v in range(u+1, n):
                if best_A[u,v] == 1:
                    result.append([u, v])
    
    return result
