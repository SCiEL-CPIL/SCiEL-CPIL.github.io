import numpy as np

N = 16

def construct():
    points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(points)
    coords = np.array(points)
    
    # Precompute squared distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    
    # Precompute isosceles triples
    # is_iso[i, j, k] is True if i, j, k form an isosceles triangle
    d1 = dist2[:, :, np.newaxis]
    d2 = dist2[:, np.newaxis, :]
    d3 = dist2[np.newaxis, :, :]
    
    is_iso = (d1 == d2) | (d1 == d3) | (d2 == d3)
    
    # Exclude degenerate cases (i==j, i==k, j==k)
    # We can use np.fill_diagonal on slices, but easier to loop
    for i in range(n_points):
        is_iso[i, i, :] = False
        is_iso[i, :, i] = False
        is_iso[:, i, i] = False
        
    S = []
    U = np.ones(n_points, dtype=bool)
    blocked = np.zeros(n_points, dtype=bool)
    
    while np.any(U):
        candidates = U & ~blocked
        
        if not np.any(candidates):
            break
            
        best_p = -1
        best_score = -1
        best_new_blocked = None
        
        cand_indices = np.where(candidates)[0]
        
        for p in cand_indices:
            if not S:
                new_blocked = blocked
            else:
                # Get rows for indices in S, column p
                # Shape: (len(S), 256)
                sub = is_iso[S, p, :]
                union_mask = np.any(sub, axis=0)
                new_blocked = blocked | union_mask
            
            cover = new_blocked & U
            score = np.sum(cover)
            
            if score > best_score:
                best_score = score
                best_p = p
                best_new_blocked = new_blocked
        
        S.append(best_p)
        blocked = best_new_blocked
        U = U & ~blocked
        
    return [points[i] for i in S]
