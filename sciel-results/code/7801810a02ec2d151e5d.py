import numpy as np
import random
import math

def construct():
    n = 31
    m = 91
    
    # Initialize random graph
    A = np.zeros((n, n), dtype=int)
    edges = set()
    
    # Generate m distinct edges
    while len(edges) < m:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            if u > v:
                u, v = v, u
            if (u, v) not in edges:
                edges.add((u, v))
                A[u, v] = 1
                A[v, u] = 1
                
    # Calculate M (common neighbors)
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            # M[i, j] = sum(A[i, k] * A[j, k])
            # Using dot product of rows
            M[i, j] = np.dot(A[i], A[j])
            M[j, i] = M[i, j]
            
    # Calculate S = sum_{i < j} comb(M[i, j], 2)
    S = 0
    for i in range(n):
        for j in range(i + 1, n):
            S += M[i, j] * (M[i, j] - 1) // 2
            
    # C4 count is S // 2
    # We minimize S
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling = 0.999
    
    # Precompute list of edges and non-edges for sampling?
    # Or just sample randomly.
    
    for step in range(100000): # Enough iterations
        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or A[u, v] == 0:
            continue
            
        # Pick random non-edge
        x = random.randint(0, n - 1)
        y = random.randint(0, n - 1)
        if x == y or A[x, y] == 1:
            continue
            
        # Check if disjoint
        if len({u, v, x, y}) < 4:
            continue
            
        # Calculate Delta S
        delta_S = 0
        
        # Remove (u, v)
        # Update M[u, w] and M[v, w]
        for w in range(n):
            if w == u or w == v:
                continue
            # M[u, w] decreases by A[v, w]
            if A[v, w]:
                old_mu = M[u, w]
                new_mu = old_mu - 1
                delta_S += (new_mu * (new_mu - 1) // 2) - (old_mu * (old_mu - 1) // 2)
                # M[w, u] is same as M[u, w], but we sum i<j.
                # If u < w, M[u, w] is in S. If w < u, M[w, u] is in S.
                # We are updating the value. The contribution to S is based on the pair {u, w}.
                # So we just need to calculate the change in binomial coefficient.
                
            # M[v, w] decreases by A[u, w]
            if A[u, w]:
                old_mv = M[v, w]
                new_mv = old_mv - 1
                delta_S += (new_mv * (new_mv - 1) // 2) - (old_mv * (old_mv - 1) // 2)
                
        # Add (x, y)
        for w in range(n):
            if w == x or w == y:
                continue
            # M[x, w] increases by A[y, w]
            if A[y, w]:
                old_mx = M[x, w]
                new_mx = old_mx + 1
                delta_S += (new_mx * (new_mx - 1) // 2) - (old_mx * (old_mx - 1) // 2)
                
            # M[y, w] increases by A[x, w]
            if A[x, w]:
                old_my = M[y, w]
                new_my = old_my + 1
                delta_S += (new_my * (new_my - 1) // 2) - (old_my * (old_my - 1) // 2)
                
        # Accept/Reject
        if delta_S < 0 or random.random() < math.exp(-delta_S / T):
            # Update A
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            # Update M
            for w in range(n):
                if w == u or w == v:
                    continue
                if A[v, w]:
                    M[u, w] -= 1
                    M[w, u] -= 1
                if A[u, w]:
                    M[v, w] -= 1
                    M[w, v] -= 1
                    
            for w in range(n):
                if w == x or w == y:
                    continue
                if A[y, w]:
                    M[x, w] += 1
                    M[w, x] += 1
                if A[x, w]:
                    M[y, w] += 1
                    M[w, y] += 1
                    
            S += delta_S
            
        T *= cooling
        if T < T_min:
            break
            
    # Construct edge list
    edges_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges_list.append([i, j])
                
    return edges_list
