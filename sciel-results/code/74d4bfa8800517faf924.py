import time
import random

def construct():
    deadline = time.time() + 115.0
    
    # Base graph: Polarity graph of PG(2,5)
    graph = _build_polarity_graph(5)
    n = graph['n']
    edges = graph['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Try to extend
    best_n = n
    best_edges = list(edges)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        # Randomize the order of vertices for neighbor selection
        current_adj = [set(s) for s in adj]
        current_edges = list(edges)
        current_n = n
        
        for attempt in range(12):
            if time.time() > deadline - 5:
                break
                
            new_v = current_n
            # Find a valid set of neighbors for new_v
            # Neighbors must be pairwise "incompatible" in terms of common neighbors
            # i.e., for u, w in S, N(u) & N(w) is empty.
            
            # Shuffle vertices to try different combinations
            candidates = list(range(current_n))
            random.shuffle(candidates)
            
            S = []
            for v in candidates:
                # Check if v can be added to S
                # v must have no common neighbor with any u in S
                compatible = True
                for u in S:
                    if current_adj[v] & current_adj[u]:
                        compatible = False
                        break
                if compatible:
                    S.append(v)
                    
            # If S is empty, new_v is isolated.
            # Isolated vertex increases alpha by 1.
            # We need to check if alpha + 1 <= 10.
            # We don't have exact alpha, but we can estimate.
            # For now, prefer non-empty S.
            
            if not S:
                # Try to add isolated vertex?
                # Only if we are confident alpha is small.
                # Skip for now to avoid increasing alpha too fast.
                continue
                
            # Add new vertex with neighbors S
            # Check if this maintains C4-free (guaranteed by S construction)
            # Check if alpha <= 10.
            
            # Heuristic check for alpha:
            # Greedy independent set size
            remaining = set(range(current_n + 1))
            new_adj = [set(s) for s in current_adj] + [set(S)]
            for v in S:
                new_adj[v].add(new_v)
                
            # Simple greedy alpha check
            alpha_est = 0
            temp_remaining = set(range(current_n + 1))
            while temp_remaining:
                # Pick min degree
                min_deg = float('inf')
                best_v = None
                for v in temp_remaining:
                    deg = len(new_adj[v] & temp_remaining)
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                alpha_est += 1
                temp_remaining.discard(best_v)
                temp_remaining -= new_adj[best_v]
                if alpha_est > 10:
                    break
                    
            if alpha_est > 10:
                # Try a smaller S?
                # Or just skip this attempt
                continue
                
            # Accept
            for v in S:
                current_edges.append([new_v, v])
                current_adj[v].add(new_v)
            current_adj.append(set(S))
            current_n += 1
            
        if current_n > best_n:
            best_n = current_n
            best_edges = list(current_edges)
            
    return {'n': best_n, 'edges': best_edges}

def _build_polarity_graph(q):
    # Same as before
    n = q * q + q + 1
    points = []
    for x in range(q):
        for y in range(q):
            points.append([x, y, 1])
    for x in range(q):
        points.append([x, 1, 0])
        
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            pi = points[i]
            pj = points[j]
            val = (pj[1] * pi[0] + pj[0] * pi[1] + pj[2] * pi[2]) % q
            if val == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
