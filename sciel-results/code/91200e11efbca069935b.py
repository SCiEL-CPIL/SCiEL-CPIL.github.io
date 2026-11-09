"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    n = 16
    target_edges = 65
    
    # We want to find a graph on n vertices with exactly target_edges edges
    # that minimizes the maximum triangle degree sum.
    
    # Strategy: Simulated Annealing
    # State: adjacency matrix (symmetric, 0/1)
    # Objective: minimize max triangle degree sum.
    # To make it differentiable/smooth for SA, we can minimize sum of squares of triangle degree sums,
    # but the task asks to minimize the MAX. 
    # However, minimizing the max is hard. 
    # Let's try a hybrid: minimize (max_sum)^2 + lambda * sum_of_squares to encourage uniformity.
    # Or just directly minimize the max sum with SA.
    
    # Initial graph: Start with a random graph with target_edges edges, 
    # or start with the known good structure (K_{8,8} + 1 edge) and perturb.
    
    # Let's start with the seed structure as a good baseline, then improve via SA.
    # Seed: K_{8,8} + edge (0,1)
    # Degrees in seed:
    # Vertices 0,1: deg 9 (8 from other part + 1 internal)
    # Vertices 2..7: deg 8
    # Vertices 8..15: deg 8
    # Triangles: only those involving edge (0,1) and a vertex in 8..15.
    # Triangle (0,1,k) for k in 8..15.
    # deg(0)+deg(1)+deg(k) = 9+9+8 = 26.
    # So seed score is 26.
    
    # Can we do better?
    # Asymptotic bound is ~23.4.
    # If we could get max triangle sum to 24 or 23, that would be great.
    
    # Let's implement SA.
    
    def compute_triangle_sums(adj):
        """
        adj: numpy array n x n, 0/1 symmetric.
        Returns list of (u, v, w, sum) for all triangles.
        """
        triangles = []
        # Find all triangles
        # For each edge (u, v), find common neighbors w
        # To avoid duplicates, only consider u < v < w
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == 1:
                    # common neighbors of u and v
                    # w must be > v to ensure u < v < w
                    common = np.where((adj[u, :v+1] == 0) & (adj[v, :v+1] == 0))[0] # This is wrong, need w > v
                    # Let's just iterate w > v
                    for w in range(v+1, n):
                        if adj[u, w] == 1 and adj[v, w] == 1:
                            triangles.append((u, v, w))
        
        sums = []
        for (u, v, w) in triangles:
            s = adj[u].sum() + adj[v].sum() + adj[w].sum()
            sums.append(s)
        return triangles, sums

    def get_score(adj):
        """
        Returns the maximum triangle degree sum.
        If no triangles, return 0 (but we must have triangles).
        """
        triangles, sums = compute_triangle_sums(adj)
        if not sums:
            return 0
        return max(sums)

    # Initial state
    # Use the seed graph
    adj = np.zeros((n, n), dtype=int)
    # K_{8,8}
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    # Edge (0,1)
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    current_edges = int(adj.sum() // 2)
    assert current_edges == 65, f"Initial edges {current_edges} != 65"
    
    current_score = get_score(adj)
    
    # SA Parameters
    T = 5000 # Temperature steps
    t_start = 10.0
    t_end = 0.1
    cooling = (t_end / t_start) ** (1.0 / T)
    
    best_adj = adj.copy()
    best_score = current_score
    
    for step in range(T):
        t = t_start * (cooling ** step)
        
        # Move: Flip a random edge
        # Choose two distinct vertices u, v
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Check if flipping (u,v) keeps edge count = 65?
        # If we flip an edge, count changes by +/- 1.
        # We need to keep count exactly 65.
        # So we must perform a "swap": remove one edge and add another.
        # Or, flip two edges: remove (u1,v1) and add (u2,v2) or vice versa.
        
        # Let's do a swap move:
        # Pick two existing edges (a,b) and (c,d) and swap to (a,c) and (b,d) or similar?
        # Simpler: Pick one existing edge to remove, and one non-existing edge to add.
        
        # Find an existing edge to remove
        # Find a non-existing edge to add
        
        # To be efficient, let's just pick random u,v and if it's an edge, remove it and add another random non-edge.
        # If it's not an edge, add it and remove another random edge.
        
        # Let's pick a random pair (u,v)
        # If adj[u,v] == 1, we want to remove it. We need to add another edge (x,y) where adj[x,y]==0.
        # If adj[u,v] == 0, we want to add it. We need to remove another edge (x,y) where adj[x,y]==1.
        
        # Let's just pick two random pairs (u,v) and (x,y) and swap their states if valid.
        # Actually, simpler:
        # 1. Pick random edge (a,b) to remove.
        # 2. Pick random non-edge (c,d) to add.
        
        # To pick random edge:
        edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
        if not edge_list:
            continue
        a, b = random.choice(edge_list)
        
        non_edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 0]
        if not non_edge_list:
            continue
        c, d = random.choice(non_edge_list)
        
        # Ensure we don't create a self-loop or duplicate (already ensured by i<j)
        # Also ensure (c,d) != (a,b) (ensured by one being edge, one non-edge)
        
        # Apply move
        adj[a, b] = 0
        adj[b, a] = 0
        adj[c, d] = 1
        adj[d, c] = 1
        
        new_score = get_score(adj)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
        else:
            # Accept with probability exp(-delta / t)
            # Since scores are integers, delta is integer.
            # t is temperature.
            if delta == 0:
                current_score = new_score
            else:
                prob = np.exp(-delta / t)
                if random.random() < prob:
                    current_score = new_score
        
        # Undo move if not accepted? No, SA usually keeps the state.
        # But if we rejected, we should revert to keep the state consistent with current_score?
        # Standard SA: if rejected, revert the move.
        if delta >= 0:
            prob = np.exp(-delta / t) if delta > 0 else 1.0
            if random.random() >= prob:
                # Revert
                adj[a, b] = 1
                adj[b, a] = 1
                adj[c, d] = 0
                adj[d, c] = 0
                # current_score remains unchanged
        
        # Optimization: If best_score is very low, we might stop early?
        # But let's run full T.
        
        # Re-optimization: The list generation for edge_list and non_edge_list is O(n^2) every step.
        # n=16, n^2=256. T=5000. 5000*256 = 1.28M operations. This is fast.
        # compute_triangle_sums is also O(n^3) worst case, but n=16 is tiny.
        
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v] == 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
