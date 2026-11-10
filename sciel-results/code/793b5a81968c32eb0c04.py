import numpy as np
import time
import random

# Precompute upper-triangular indices for N=30
_N = 30
_i, _j, _k = np.where(np.triu(np.ones((_N, _N, _N), dtype=bool), 1).reshape(_N, _N, _N))
# Actually, we need i < j < k, so use a different approach
_triu_indices = []
for a in range(_N):
    for b in range(a + 1, _N):
        for c in range(b + 1, _N):
            _triu_indices.append((a, b, c))
_triu_indices = np.array(_triu_indices, dtype=np.int32)
_TRUI_A = _triu_indices[:, 0]
_TRUI_B = _triu_indices[:, 1]
_TRUI_C = _triu_indices[:, 2]
_NUM_TRIPLES = len(_triu_indices)

def fast_score(points):
    """
    points: numpy array of shape (N, 2)
    Returns: max multiplicity of a positive doubled area
    """
    n = points.shape[0]
    if n < 3:
        return 0
    
    # Use precomputed indices
    pa = points[_TRUI_A]
    pb = points[_TRUI_B]
    pc = points[_TRUI_C]
    
    det = (pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) - \
          (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])
    
    valid_dets = np.abs(det)
    valid_dets = valid_dets[valid_dets > 0]
    
    if len(valid_dets) == 0:
        return 0
        
    counts = np.bincount(valid_dets)
    return counts.max()

def fast_score_delta(points, idx, old_pt, new_pt):
    """
    Fast score update: compute only the dets involving point idx.
    Returns new max multiplicity.
    """
    n = points.shape[0]
    
    # Compute dets for all pairs (i, j) with i < j, where at least one is idx
    # Case 1: idx is the first point (idx, j, k) with idx < j < k
    # Case 2: idx is the middle point (i, idx, k) with i < idx < k
    # Case 3: idx is the last point (i, j, idx) with i < j < idx
    
    new_dets = []
    
    # Case 1: (idx, j, k) for j > idx, k > j
    for j in range(idx + 1, n):
        for k in range(j + 1, n):
            p1 = new_pt
            p2 = points[j]
            p3 = points[k]
            det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            det = abs(det)
            if det > 0:
                new_dets.append(det)
    
    # Case 2: (i, idx, k) for i < idx, k > idx
    for i in range(idx):
        for k in range(idx + 1, n):
            p1 = points[i]
            p2 = new_pt
            p3 = points[k]
            det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            det = abs(det)
            if det > 0:
                new_dets.append(det)
    
    # Case 3: (i, j, idx) for i < j < idx
    for i in range(idx):
        for j in range(i + 1, idx):
            p1 = points[i]
            p2 = points[j]
            p3 = new_pt
            det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            det = abs(det)
            if det > 0:
                new_dets.append(det)
    
    if not new_dets:
        return 0
        
    counts = np.bincount(new_dets)
    return counts.max()

def construct():
    N = 30
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit
    
    # 1. Generate Initial Candidates - focus on high-potential grid seeds
    candidates = []
    
    # Rectangular Grids W x H such that W*H = 30
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
    
    # Add perturbed grid starts - jitter grid points slightly
    for W in [3, 5, 6]:
        H = 30 // W
        if W * H != 30: continue
        for trial in range(2):
            pts = []
            for i in range(W):
                for j in range(H):
                    x = i - W // 2 + random.randint(-1, 1)
                    y = j - H // 2 + random.randint(-1, 1)
                    pts.append((x, y))
            # Ensure distinctness
            unique_pts = []
            for p in pts:
                if p not in unique_pts and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                    unique_pts.append(p)
            if len(unique_pts) == N:
                candidates.append(np.array(unique_pts, dtype=int))
            
    # Add a "clustered" start: points close to each other
    for _ in range(3):
        pts = []
        base_x = random.randint(-20, 20)
        base_y = random.randint(-20, 20)
        while len(pts) < N:
            # Random offset in a small box
            p = (base_x + random.randint(-5, 5), base_y + random.randint(-5, 5))
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))

    # Add random starts
    for _ in range(3):
        pts = []
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))

    # Evaluate candidates and pick the best few for local search
    scored_candidates = []
    for cand in candidates:
        s = fast_score(cand)
        scored_candidates.append((s, cand))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_global_score = -1
    best_global_pts = None
    
    # Run local search on the top 5 initial candidates
    for initial_score, initial_pts in scored_candidates[:5]:
        current_pts = initial_pts.copy()
        current_score = initial_score
        
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_pts = current_pts.copy()
            
        # Simulated Annealing Parameters
        T = 1.0
        T_min = 0.01
        alpha = 0.995 # Cooling rate
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline:
            if current_score > best_global_score:
                best_global_score = current_score
                best_global_pts = current_pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            # Decide move type: 50% single point move, 30% swap, 20% global
            move_type = random.random()
            
            if move_type < 0.5:
                # Single point local move
                idx = np.random.randint(N)
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_p = current_pts[idx] + np.array([dx, dy])
                
                # Check bounds
                if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
                    continue
                    
                # Check distinctness
                is_distinct = True
                for i in range(N):
                    if i != idx and np.array_equal(current_pts[i], new_p):
                        is_distinct = False
                        break
                        
                if not is_distinct:
                    continue
                    
                new_pts = current_pts.copy()
                new_pts[idx] = new_p
                
            elif move_type < 0.8:
                # Swap two points
                idx1, idx2 = np.random.choice(N, 2, replace=False)
                new_pts = current_pts.copy()
                new_pts[idx1], new_pts[idx2] = current_pts[idx2].copy(), current_pts[idx1].copy()
            else:
                # Global move: random point in box
                idx = np.random.randint(N)
                new_p = np.array([random.randint(-30, 30), random.randint(-30, 30)])
                
                # Check bounds
                if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
                    continue
                    
                # Check distinctness
                is_distinct = True
                for i in range(N):
                    if i != idx and np.array_equal(current_pts[i], new_p):
                        is_distinct = False
                        break
                        
                if not is_distinct:
                    continue
                    
                new_pts = current_pts.copy()
                new_pts[idx] = new_p
            
            # Calculate new score
            new_score = fast_score(new_pts)
            
            # Accept if better
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
            else:
                # Accept if worse with probability exp(-delta / T)
                delta = current_score - new_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_pts = new_pts
                        current_score = new_score
                # If equal, accept (to explore)
                elif random.random() < 0.5:
                    current_pts = new_pts
                    # current_score stays same
            
            # Cool down
            T = max(T_min, T * alpha)

    # Final Polish Phase: Targeted local search with tighter moves
    # We use the best found configuration and perform a refined local search
    if best_global_pts is not None:
        current_pts = best_global_pts.copy()
        current_score = best_global_score
        
        # Smaller step sizes and more iterations if time permits
        T = 0.1
        T_min = 0.001
        alpha = 0.999
        no_improve_count = 0
        max_no_improve = 1000 # Allow more stagnation to try to jump out of local opt
        
        while time.time() < deadline:
            if current_score > best_global_score:
                best_global_score = current_score
                best_global_pts = current_pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            # Decide move type: 60% single point, 40% swap
            move_type = random.random()
            
            if move_type < 0.6:
                # Single point local move for polishing
                idx = np.random.randint(N)
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                if dx == 0 and dy == 0:
                    continue
                new_p = current_pts[idx] + np.array([dx, dy])
                    
                # Check bounds
                if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
                    continue
                    
                # Check distinctness
                is_distinct = True
                for i in range(N):
                    if i != idx and np.array_equal(current_pts[i], new_p):
                        is_distinct = False
                        break
                        
                if not is_distinct:
                    continue
                    
                new_pts = current_pts.copy()
                new_pts[idx] = new_p
            else:
                # Swap two points
                idx1, idx2 = np.random.choice(N, 2, replace=False)
                new_pts = current_pts.copy()
                new_pts[idx1], new_pts[idx2] = current_pts[idx2].copy(), current_pts[idx1].copy()
            
            # Calculate new score
            new_score = fast_score(new_pts)
            
            # Accept if better
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
            else:
                # Accept if worse with probability exp(-delta / T)
                delta = current_score - new_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_pts = new_pts
                        current_score = new_score
                # If equal, accept (to explore)
                elif random.random() < 0.5:
                    current_pts = new_pts
                    # current_score stays same
            
            # Cool down
            T = max(T_min, T * alpha)

    return [(int(p[0]), int(p[1])) for p in best_global_pts]
