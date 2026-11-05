import numpy as np
import time
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
import networkx as nx # Wait, is networkx allowed? 
# "Python standard library, NumPy and SciPy are available."
# NetworkX is NOT in the standard library. I must not use it.
# I need to implement Betweenness or use SciPy.
# SciPy has `scipy.sparse.csgraph` but does it have betweenness?
# No, `scipy.sparse.csgraph` has shortest paths, etc.
# Implementing Brandes' algorithm for Betweenness Centrality in Python for N=200 is feasible.
# Or, use a simpler proxy.

# Let's stick to NumPy/SciPy only.
# I can compute Betweenness Centrality using a simple BFS-based algorithm (Brandes) for unweighted graphs.
# N=200, M ~ 10000 (for p=0.1). 
# Brandes is O(N*M). 200 * 10000 = 2,000,000 operations. Very fast.

def brandes_betweenness(adjacency):
    n = adjacency.shape[0]
    betweenness = np.zeros(n)
    # Convert to list of neighbors
    neighbors = [np.where(adjacency[i] == 1)[0] for i in range(n)]
    
    for s in range(n):
        # BFS from s
        # Stack for order of finishing
        stack = []
        # Predecessors
        pred = [list() for _ in range(n)]
        # Number of shortest paths
        sigma = np.zeros(n)
        sigma[s] = 1
        # Distance
        dist = np.full(n, -1)
        dist[s] = 0
        
        # Queue
        queue = [s]
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            stack.append(v)
            for w in neighbors[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        
        # Delta
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]
                
    # Symmetrize? Brandes usually accumulates for both directions if we run from all sources.
    # Actually, the standard implementation sums delta for all w != s.
    # This gives the betweenness for the source s.
    # Summing over all s gives the total betweenness.
    return betweenness

def priority(adjacency):
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    
    # Compute Betweenness Centrality
    # For N=200, this is fast.
    betweenness = brandes_betweenness(adjacency)
    
    # Normalize betweenness to be comparable to degree?
    # Degree range: 0 to N-1.
    # Betweenness range: 0 to O(N^3).
    # We should scale them.
    # Max degree ~ N. Max betweenness ~ N^3.
    # Let's normalize betweenness by max betweenness.
    max_b = np.max(betweenness) if np.max(betweenness) > 0 else 1
    norm_b = betweenness / max_b
    
    # Score: We want to pick high betweenness, low degree.
    # Score = norm_b - degrees / (n-1)
    # Or just: Score = norm_b - degrees
    # Let's try: Score = norm_b * n - degrees
    # This makes the scales similar.
    
    scores = norm_b * n - degrees
    
    return scores.astype(float)
