import numpy as np
import time
import random

# Pre-compute triple indices for N=30
N = 30
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)
idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def fast_score(points):
    """
    points: numpy array of shape (N, 2)
    Returns: max multiplicity of a positive doubled area
    """
    if points.shape[0] < 3:
        return 0
        
    # Extract coordinates for all triples
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    
    # det = (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
    det = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
          (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
          
    valid_dets = np.abs(det)
    valid_dets = valid_dets[valid_dets > 0]
    
    if len(valid_dets) == 0:
        return 0
        
    # Count frequencies
    # Max area is bounded. 
    # Use bincount for speed
    counts = np.bincount(valid_dets)
    return counts.max()

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Generate Initial Candidates
    
    candidates = []
    
    # Rectangular Grids
    for W in [1, 2, 3, 5, 6, 10, 15, 30]:
        H = 30 // W
        if W * H != 30: continue
        pts = []
        for i in range(W):
            for j in range(H):
                # Center the grid
                x = i - W // 2
                y = j - H // 2
                pts.append((x, y))
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            candidates.append(np.array(pts, dtype=int))
            
    # Sheared Grids (Unimodular transforms don't change area multiset, 
    # but might help with distinctness or bounds? No, areas are invariant. 
    # Skip sheared grids as they are equivalent in score to rectangular ones.)
    
    # "Clustered" starts
    for _ in range(3):
        pts = []
        base_x = random.randint(-20, 20)
        base_y = random.randint(-20, 20)
        while len(pts) < N:
            p = (base_x + random.randint(-5, 5), base_y + random.randint(-5, 5))
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))

    # Random starts
    for _ in range(3):
        pts = []
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))

    # Evaluate and sort
    scored_candidates = []
    for cand in candidates:
        s = fast_score(cand)
        scored_candidates.append((s, cand))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_global_score = -1
    best_global_pts = None
    
    # Run local search on top candidates
    for initial_score, initial_pts in scored_candidates[:5]:
        current_pts = initial_pts.copy()
        current_score = initial_score
        
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_pts = current_pts.copy()
            
        T = 1.0
        T_min = 0.01
        alpha = 0.995
        no_improve_count = 0
        max_no_improve = 1000 # Increased patience
        
        while time.time() < deadline:
            if current_score > best_global_score:
                best_global_score = current_score
                best_global_pts = current_pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            # Move Selection
            r = random.random()
            new_pts = current_pts.copy()
            
            if r < 0.6:
                # 1-Opt Local
                idx = np.random.randint(N)
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0: continue
                new_p = current_pts[idx] + np.array([dx, dy])
                if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30: continue
                if any(np.array_equal(p, new_p) for p in current_pts if p is not current_pts[idx]):
                     # Faster check:
                     if np.any(np.all(new_pts == new_p, axis=1)): 
                         # Wait, new_pts[idx] is old. We need to check others.
                         # Let's just do a simple loop or set check for distinctness
                         pass
                # Distinctness check
                is_distinct = True
                for i in range(N):
                    if i != idx and np.array_equal(current_pts[i], new_p):
                        is_distinct = False
                        break
                if not is_distinct: continue
                new_pts[idx] = new_p
                
            elif r < 0.8:
                # 1-Opt Global
                idx = np.random.randint(N)
                new_p = np.array([random.randint(-30, 30), random.randint(-30, 30)])
                if any(np.array_equal(p, new_p) for p in current_pts if p is not current_pts[idx]):
                    # Check distinctness
                    is_distinct = True
                    for i in range(N):
                        if i != idx and np.array_equal(current_pts[i], new_p):
                            is_distinct = False
                            break
                    if not is_distinct: continue
                new_pts[idx] = new_p
                
            else:
                # 2-Opt Swap
                i1, i2 = np.random.choice(N, 2, replace=False)
                # Swap points i1 and i2? No, that's just reordering, score is invariant to permutation.
                # Move two points?
                # Let's try moving two points locally.
                dx1 = random.randint(-3, 3)
                dy1 = random.randint(-3, 3)
                dx2 = random.randint(-3, 3)
                dy2 = random.randint(-3, 3)
                
                new_p1 = current_pts[i1] + np.array([dx1, dy1])
                new_p2 = current_pts[i2] + np.array([dx2, dy2])
                
                # Check bounds
                if new_p1[0] < -30 or new_p1[0] > 30 or new_p1[1] < -30 or new_p1[1] > 30: continue
                if new_p2[0] < -30 or new_p2[0] > 30 or new_p2[1] < -30 or new_p2[1] > 30: continue
                
                # Check distinctness
                # Temporarily set them to None to check against others?
                # Simpler: check if new_p1 == new_p2 or if they collide with others.
                if np.array_equal(new_p1, new_p2): continue
                
                # Check collision with other points
                valid = True
                for i in range(N):
                    if i == i1 or i == i2: continue
                    if np.array_equal(current_pts[i], new_p1) or np.array_equal(current_pts[i], new_p2):
                        valid = False
                        break
                if not valid: continue
                
                new_pts[i1] = new_p1
                new_pts[i2] = new_p2
            
            new_score = fast_score(new_pts)
            
            # Acceptance
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
            else:
                delta = current_score - new_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_pts = new_pts
                        current_score = new_score
                elif random.random() < 0.5:
                    current_pts = new_pts
            
            T = max(T_min, T * alpha)

    return [(int(p[0]), int(p[1])) for p in best_global_pts]
