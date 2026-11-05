import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    
    # Precompute some features
    # 1. Degrees
    # 2. Neighbor Degree Sum
    neigh_deg_sum = adjacency @ degrees
    avg_neigh_deg = neigh_deg_sum / np.where(degrees > 0, degrees, 1)
    
    # 3. Clustering Coeff (approximate or exact)
    # Exact clustering is O(n^3) or O(n^2 * deg). For n=200, O(n^3) = 8M ops, ok.
    # C_v = 2 * E(N(v)) / (deg(v) * (deg(v)-1))
    # E(N(v)) is number of edges between neighbors.
    # Can compute via adjacency matrix squared?
    # (A^2)[i,j] is number of length 2 paths.
    # For i != j, (A^2)[i,j] is number of common neighbors.
    # Edges in N(v) = sum_{u in N(v)} deg_N(v, u) / 2?
    # Actually, number of edges in induced subgraph on N(v) is:
    # sum_{u in N(v)} sum_{w in N(v), w > u} A[u,w]
    # Or simply: trace of A[N(v), N(v)]?
    # Let's use a simpler proxy: Common Neighbors count.
    # A2 = adjacency @ adjacency
    # common_neighbors = A2[degrees > 0, :].sum(axis=1)? No.
    # For vertex v, number of edges among neighbors is:
    # sum_{u in N(v)} deg(u, N(v)) / 2
    # deg(u, N(v)) is number of neighbors of u that are also in N(v).
    # This is row sum of A[N(v), u].
    # Too complex to vectorize simply without loops.
    # Let's skip exact clustering or use a fast approximation.
    # Approximation: (A^2)[v, v] is deg(v).
    # (A^2)[u, w] for u,w in N(v) is common neighbors of u,w.
    # Let's just use Degree and Neigh Deg.
    
    # 4. Fiedler Vector
    # L = D - A
    # D = diag(degrees)
    # L = np.diag(degrees) - adjacency
    # Eigenvectors
    # This might be the strongest structural signal.
    
    candidates = []
    
    # C1: Min Degree
    candidates.append(-degrees.astype(float))
    
    # C2: Min Degree + Avg Neigh Deg
    candidates.append(-degrees.astype(float) + avg_neigh_deg.astype(float))
    
    # C3: Min Degree + Sum Neigh Deg
    candidates.append(-degrees.astype(float) + neigh_deg_sum.astype(float))
    
    # C4: Min Degree^2
    candidates.append(-degrees.astype(float)**2)
    
    # C5: Fiedler Vector
    try:
        # Ensure symmetric
        A = (adjacency + adjacency.T) / 2.0
        D = np.diag(degrees)
        L = D - A
        # Compute eigenvalues/vectors
        # For n=200, this is fast.
        vals, vecs = np.linalg.eigh(L)
        # Fiedler vector is index 1 (0 is trivial, all 1s)
        fiedler = vecs[:, 1]
        # We want to pick "peripheral" or "central"?
        # Usually, ordering by Fiedler vector gives a good spatial ordering.
        # For IS, maybe we want to pick high magnitude first?
        # Let's try both signs.
        candidates.append(fiedler)
        candidates.append(-fiedler)
    except:
        pass
        
    # C6: Random (Deterministic)
    rng = np.random.RandomState(42)
    candidates.append(rng.rand(n))
    
    # C7: Min Degree + Random
    candidates.append(-degrees.astype(float) + 0.1 * rng.rand(n))
    
    # Simulate Greedy IS for each candidate
    best_score = -1
    best_scores = None
    
    # Helper to simulate
    def simulate(scores):
        order = np.argsort(-scores)
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not blocked[v] and not selected[v]:
                selected[v] = True
                count += 1
                # Block neighbors
                neighbors = np.where(adjacency[v])[0]
                blocked[neighbors] = True
        return count
        
    # Time guard
    start_time = time.time()
    timeout = 100 # Leave margin
    
    for i, s in enumerate(candidates):
        if time.time() - start_time > timeout:
            break
        # Ensure scores are distinct to avoid tie-breaking ambiguity in argsort
        # Add tiny noise?
        # argsort is stable?
        # We can just use the scores as is.
        size = simulate(s)
        if size > best_score:
            best_score = size
            best_scores = s
            
    if best_scores is None:
        # Fallback to min degree
        best_scores = -degrees.astype(float)
        
    return best_scores
