import numpy as np
import time

def priority(adjacency):
    """
    Calculates priority scores for vertices to maximize the size of the independent set
    selected by a greedy decoder (descending score).
    
    Strategy: Hybrid Spectral-Local Heuristic with Meta-Heuristic Selection.
    1. Compute graph Laplacian L = D - A.
    2. Find the Fiedler vector (eigenvector of the second-smallest eigenvalue).
    3. Partition the graph into V+ and V- based on the Fiedler vector signs.
    4. Generate candidate scoring functions that combine:
       - Local degree (lower is better for MIS).
       - Neighbor degree (higher is better, as picking a node removes high-degree neighbors).
       - Spectral position (distance from the cut, side size, and "territory" score).
    5. Simulate the greedy MIS decoder for each candidate.
    6. Return the scores from the candidate that yields the largest independent set.
    
    This approach leverages the global structure (spectral partition) to identify
    regions where independent sets can be larger, while using local degree information
    to refine the ordering within those regions.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    A = np.array(adjacency, dtype=bool)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    degrees = A.sum(axis=1).astype(int)
    
    # Helper function to simulate greedy MIS given scores
    def simulate_greedy(scores, A, n):
        # Sort vertices by descending score
        order = np.argsort(-scores, kind='mergesort')
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                # Check if any neighbor is selected
                # A[v] is a boolean mask of neighbors
                if not selected[A[v]].any():
                    selected[v] = True
                    count += 1
        return count

    # Precompute common features
    # 1. Degree
    deg = degrees.astype(float)
    
    # 2. Sum of neighbor degrees (A @ deg)
    neighbor_deg_sum = A.astype(float) @ deg
    
    # 3. Average neighbor degree (safe division)
    avg_neighbor_deg = neighbor_deg_sum / (deg + 1e-9)
    
    # 4. Spectral features
    # Compute Laplacian
    D = np.diag(deg)
    L = D - A.astype(float)
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Sort eigenvalues (eigh returns them in ascending order usually, but let's be sure)
        # Index 0 is 0 (connected component indicator), Index 1 is Fiedler value
        # If disconnected, there might be multiple 0s. We want the smallest positive one.
        # For simplicity, assuming connected or using index 1 as standard Fiedler.
        fiedler_vec = eigvecs[:, 1]
        
        # Normalize Fiedler vector to have mean 0 and std 1 for stable scaling
        f_mean = np.mean(fiedler_vec)
        f_std = np.std(fiedler_vec)
        if f_std > 1e-9:
            fiedler_norm = (fiedler_vec - f_mean) / f_std
        else:
            fiedler_norm = np.zeros(n)
            
        # Partition into V+ and V-
        positive_mask = fiedler_vec > 0
        negative_mask = ~positive_mask
        
        size_pos = np.sum(positive_mask)
        size_neg = np.sum(negative_mask)
        
        # Calculate cross-cut edges
        # Edges between V+ and V-
        # Sum of A[i, j] where i in V+, j in V-
        # We can compute this as: sum of degrees in V+ - sum of internal edges in V+ - sum of internal edges in V-?
        # Easier: mask the matrix
        cross_edges = np.sum(A[positive_mask][:, negative_mask])
        
        # Territory Score:
        # For a vertex, how "safe" is its side?
        # If we are in the smaller side, we might have fewer internal conflicts?
        # Or if the cut is small, the sides are more independent?
        # Let's define a "territory" score based on the size of the side and the cross-cut ratio.
        # A vertex in the larger side might have more "room" but also more internal edges.
        # Let's try: Score_side = size_of_side - cross_edges (normalized?)
        
        # Let's create a per-vertex territory score
        # Vertices in V+ get a score based on V+ properties, V- based on V- properties.
        territory_score = np.zeros(n)
        
        # Metric: (Size of side) - (Cross edges incident to side? or total cross edges?)
        # Let's use: Size of side - 0.5 * Cross Edges (since each cross edge touches both sides)
        # This rewards sides that are large but have few connections to the other side.
        val_pos = size_pos - 0.5 * cross_edges
        val_neg = size_neg - 0.5 * cross_edges
        
        territory_score[positive_mask] = val_pos
        territory_score[negative_mask] = val_neg
        
        # Normalize territory score to [0, 1] range roughly for combination
        t_min = np.min(territory_score)
        t_max = np.max(territory_score)
        if t_max > t_min:
            territory_norm = (territory_score - t_min) / (t_max - t_min)
        else:
            territory_norm = np.zeros(n)
            
    except Exception:
        fiedler_vec = np.zeros(n)
        fiedler_norm = np.zeros(n)
        territory_norm = np.zeros(n)
        positive_mask = np.zeros(n, dtype=bool)
        negative_mask = np.ones(n, dtype=bool)
        size_pos = 0
        size_neg = n
        cross_edges = 0
        territory_score = np.zeros(n)

    # Normalize degree features
    max_deg = np.max(deg) if np.max(deg) > 0 else 1
    deg_norm = deg / max_deg
    
    max_avg_deg = np.max(avg_neighbor_deg) if np.max(avg_neighbor_deg) > 0 else 1
    avg_deg_norm = avg_neighbor_deg / max_avg_deg

    # Generate Candidates
    candidates = []
    
    # 1. Pure Degree (LDF)
    candidates.append(("deg", -deg_norm))
    
    # 2. Degree + Neighbor Degree (Inspiration 1)
    candidates.append(("deg_nei", -deg_norm + avg_deg_norm))
    
    # 3. Pure Spectral (Fiedler magnitude)
    candidates.append(("spec_mag", -np.abs(fiedler_norm)))
    
    # 4. Spectral (Fiedler signed)
    candidates.append(("spec_signed", fiedler_norm))
    
    # 5. Spectral (Fiedler squared)
    candidates.append(("spec_sq", -fiedler_norm**2))
    
    # 6. Territory Score
    candidates.append(("territory", territory_norm))
    
    # 7. Combined: Degree + Territory
    for alpha in [0.5, 1.0, 2.0]:
        candidates.append((f"deg_ter_{alpha}", -deg_norm + alpha * territory_norm))
        
    # 8. Combined: Degree + Neighbor + Territory
    for alpha in [0.5, 1.0, 2.0]:
        candidates.append((f"deg_nei_ter_{alpha}", -deg_norm + avg_deg_norm + alpha * territory_norm))
        
    # 9. Combined: Degree + Fiedler
    for alpha in [0.5, 1.0, 2.0]:
        candidates.append((f"deg_spec_{alpha}", -deg_norm - alpha * np.abs(fiedler_norm)))
        
    # 10. Combined: Degree + Fiedler + Territory
    for alpha in [0.5, 1.0]:
        candidates.append((f"deg_spec_ter_{alpha}", -deg_norm - 0.5 * np.abs(fiedler_norm) + alpha * territory_norm))

    # Evaluate candidates
    best_size = -1
    best_scores = candidates[0][1]  # Default to first candidate
    
    for name, scores in candidates:
        if time.time() > deadline:
            break
            
        size = simulate_greedy(scores, A, n)
        if size > best_size:
            best_size = size
            best_scores = scores
            
    return best_scores
