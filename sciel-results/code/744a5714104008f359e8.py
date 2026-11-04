import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin
    
    A = np.asarray(adjacency, dtype=bool)
    N = A.shape[0]
    
    # 1. Compute Features
    # Degree
    D = A.sum(axis=1).astype(float)
    
    # Sum of Neighbor Degrees
    SND = (A @ D)
    
    # Number of triangles containing each vertex
    Tri = np.diag(A @ A)
    
    # 2. Normalize Features to [0, 1]
    def normalize(x):
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())
    
    D_norm = normalize(D)
    SND_norm = normalize(SND)
    Tri_norm = normalize(Tri)
    
    # 3. Define Weight Grid
    w1_vals = [0.0, 0.5, 1.0, 2.0]
    w2_vals = [0.0, 0.5, 1.0, 2.0]
    w3_vals = [0.0, 0.5, 1.0]
    
    best_size = -1
    best_scores = None
    
    # Fallback: Seed strategy (Degree only)
    fallback_scores = -D_norm
    
    for w1 in w1_vals:
        if time.time() > DEADLINE:
            break
        for w2 in w2_vals:
            if time.time() > DEADLINE:
                break
            for w3 in w3_vals:
                if time.time() > DEADLINE:
                    break
                
                # Compute scores
                scores = -(w1 * D_norm + w2 * SND_norm + w3 * Tri_norm)
                
                # Greedy Selection
                indices = np.argsort(-scores)
                
                is_picked = np.zeros(N, dtype=bool)
                count = 0
                
                for v in indices:
                    if not is_picked[v]:
                        count += 1
                        is_picked[v] = True
                        is_picked[A[v]] = True
                
                if count > best_size:
                    best_size = count
                    best_scores = scores
    
    if best_scores is None:
        return fallback_scores
        
    return best_scores
