import numpy as np

def priority(adjacency):
       N = len(adjacency)
       deg = adjacency.sum(axis=1).astype(float)
       
       # Compute core numbers
       core = np.zeros(N, dtype=float)
       remaining = np.ones(N, dtype=bool)
       current_deg = deg.copy().astype(int)
       
       for _ in range(N):
           if not remaining.any():
               break
           min_deg = np.min(current_deg[remaining])
           candidates = np.where(remaining & (current_deg == min_deg))[0]
           core[candidates] = min_deg
           remaining[candidates] = False
           
           # Update neighbors
           # Vectorized update:
           # neighbors of all candidates
           # We can do:
           # adj_sub = adjacency[candidates] # shape (num_cands, N)
           # This is a boolean matrix.
           # We need to decrement current_deg for columns where any candidate is connected.
           # But we must only decrement for neighbors that are still remaining.
           # And we decrement by 1 for each edge removed.
           # Since we remove all candidates at once, each neighbor loses (number of candidates it is connected to) degrees.
           # connections = adjacency[candidates].sum(axis=0) # shape (N,)
           # But we only care about remaining neighbors.
           # Also, we must not decrement the candidates themselves (they are removed).
           # And we must not decrement non-remaining neighbors (they are already processed/removed).
           
           # Let's just loop over candidates for simplicity, N is small.
           for v in candidates:
               # Get neighbors that are still remaining
               neigh = np.where(adjacency[v] & remaining)[0]
               if len(neigh) > 0:
                   current_deg[neigh] -= 1

       # Score
       # We want to maximize the score.
       # Score = (sum of neighbor degrees + sum of neighbor cores) / (deg + 1)
       neighbor_deg_sum = adjacency @ deg
       neighbor_core_sum = adjacency @ core
       
       numerator = neighbor_deg_sum + neighbor_core_sum
       denominator = deg + 1.0
       
       score = numerator / denominator
       
       return score
