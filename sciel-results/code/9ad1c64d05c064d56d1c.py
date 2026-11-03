<<<<<<< SEARCH
def _select_best_subset(pts, N):
    """
    From a larger set of points, select N points that maximize the minimum distance.
    Use greedy selection.
    """
    if len(pts) <= N:
        return pts
    
    selected = []
    available = set(range(len(pts)))
    
    for _ in range(N):
        if not available:
            break
        
        # Find the point that maximizes the minimum distance to already selected points
        best_idx = None
        best_min_dist = -1
        
        for idx in available:
            if not selected:
                min_dist = np.inf
            else:
                min_dist = np.min([np.linalg.norm(pts[idx] - pts[s]) for s in selected])
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        
        selected.append(best_idx)
        available.discard(best_idx)
    
    return pts[selected]
=======
def _select_best_subset(pts, N):
    """
    From a larger set of points, select N points that maximize the minimum distance.
    Use greedy selection.
    """
    if len(pts) <= N:
        return pts
    
    selected = []
    available = set(range(len(pts)))
    
    for _ in range(N):
        if not available:
            break
        
        # Find the point that maximizes the minimum distance to already selected points
        best_idx = None
        best_min_dist = -1
        
        for idx in available:
            if not selected:
                min_dist = np.inf
            else:
                min_dist = np.min([np.linalg.norm(pts[idx] - pts[s]) for s in selected])
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        
        selected.append(best_idx)
        available.discard(best_idx)
    
    return pts[selected]
>>>>>>> REPLACE

