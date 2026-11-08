import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # 1. Degrees
    degrees = adj.sum(axis=1)
    
    # 2. Sum of neighbor degrees (A @ degrees)
    # This measures how "central" the neighbors are.
    # High value means this vertex is connected to hubs.
    # Picking it blocks hubs, which is generally good (hubs block many others).
    neighbor_degree_sum = adj @ degrees
    
    # 3. Triangle counts
    # T(v) is the number of triangles containing v.
    # A^3 diagonal gives 2 * T(v).
    # We want high triangle density in the neighborhood.
    # High T(v) means neighbors are connected, so blocking them is "efficient" (less waste).
    A2 = adj @ adj
    A3 = A2 @ adj
    triangle_counts = np.diag(A3) / 2.0
    
    # 4. Construct Score
    # We want to maximize the score.
    # Components:
    # + neighbor_degree_sum: Reward connecting to hubs.
    # + triangle_density: Reward dense neighborhoods (efficient blocking).
    # - degrees: Penalty for having many neighbors (direct blocking cost).
    
    # Normalize or weight?
    # Degrees are O(N). Neighbor degree sum is O(N^2) in dense graphs.
    # Triangle counts are O(N^3) worst case.
    
    # Let's use a weighted sum.
    # If we just add them, the magnitude of neighbor_degree_sum might dominate.
    # In a star graph:
    # Leaf: nds=N-1, t=0, d=1. Score ~ N.
    # Center: nds=N-1, t=0, d=N-1. Score ~ N - N = 0.
    # This works without explicit scaling if the relative magnitudes align.
    
    # However, in a random graph, nds is roughly d * avg_deg.
    # So nds is proportional to d.
    # If nds = k * d, then score = k*d + t/d - d = (k-1)d + t/d.
    # If k > 1 (graph is assortative or hubs connected to hubs?), we prefer high degree?
    # In ER graphs, degrees are Poisson. nds is not strongly correlated with d in a way that k>1 always.
    # But generally, high degree nodes are connected to other high degree nodes (rich club).
    # So nds is high for hubs.
    # We want to pick the "periphery" of the hubs, not the hubs themselves.
    # The periphery nodes have low d, but high nds (connected to hub).
    # The hub has high d, high nds.
    # The penalty -d should tip the balance in favor of the periphery node.
    
    # Let's refine the penalty.
    # Maybe -d^2?
    # Or just -d is enough.
    
    # Let's add a small term for triangle density.
    # t / max(d, 1)
    
    safe_degrees = np.maximum(degrees, 1.0)
    triangle_density = triangle_counts / safe_degrees
    
    # Score = neighbor_degree_sum + triangle_density - degrees
    scores = neighbor_degree_sum + triangle_density - degrees
    
    # Check time
    if time.time() > deadline:
         # Fallback to simple degree if something went wrong or took too long
         return -degrees.astype(float)
         
    return scores.astype(float)
