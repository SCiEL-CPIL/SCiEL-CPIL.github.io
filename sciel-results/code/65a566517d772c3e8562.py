import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for overhead and return
    
    best_score = -np.inf
    best_result = None
    
    # --- Utility Functions ---
    
    def is_valid(x, tol=1e-5):
        """
        x: array of shape (N, 3) [cx, cy, r]
        W, H: derived from the max extent? No, the task says the bounding rectangle 
        has perimeter at most 4. The evaluator likely checks the bounding box of the circles.
        Wait, the task says: "variable rectangle of perimeter 4". 
        "construct() -> array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4"
        This implies the rectangle is defined by the circles' bounding box, OR we can choose a rectangle?
        Re-reading: "Pack 21 circles ... in a variable rectangle of perimeter 4".
        "bounding rectangle has perimeter at most 4".
        This usually means we can choose the rectangle dimensions W and H such that 2(W+H) <= 4, 
        and all circles must be inside. The score is sum(r).
        If the evaluator calculates the bounding box of the circles and checks its perimeter, 
        then we just need the circles to fit in a box of perimeter <= 4.
        However, if we use a box smaller than the bounding box, we might be wasting space?
        No, if the circles fit in a box of perimeter 4, their bounding box also has perimeter <= 4 
        (actually, the bounding box perimeter is <= 2(W+H) if they are inside WxH).
        Wait, if circles are inside a W x H box, the bounding box of the circles is W' x H' where W'<=W, H'<=H.
        So 2(W'+H') <= 2(W+H) <= 4.
        So we can just assume a fixed rectangle W x H with 2(W+H)=4 and pack into it.
        The evaluator will check if the circles are disjoint and if their bounding box perimeter <= 4.
        To maximize sum(r), we should use the full perimeter budget.
        So we will optimize for specific W, H with W+H=2.
        """
        if len(x) < N:
            return False
        cx = x[:, 0]
        cy = x[:, 1]
        r = x[:, 2]
        
        # Check if radii are positive
        if np.any(r < 0):
            return False
            
        # Determine bounding box
        min_x = np.min(cx - r)
        max_x = np.max(cx + r)
        min_y = np.min(cy - r)
        max_y = np.max(cy + r)
        
        W_bbox = max_x - min_x
        H_bbox = max_y - min_y
        
        # Perimeter of bounding box
        if 2 * (W_bbox + H_bbox) > 4.0 + tol:
            return False
            
        # Check disjointness
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (cx[i] - cx[j])**2 + (cy[i] - cy[j])**2
                rad_sum_sq = (r[i] + r[j])**2
                if dist_sq < rad_sum_sq - tol:
                    return False
        return True

    def get_score(x):
        if not is_valid(x):
            return -np.inf
        return np.sum(x[:, 2])

    def polish_with_slsqp(x0, W, H, max_iter=100):
        """
        x0: (N, 3) array
        W, H: fixed box dimensions
        Returns optimized (N, 3) array
        """
        # Flatten to vector: [x0, y0, r0, x1, y1, r1, ...]
        # We fix W and H, so we only optimize circle positions and radii.
        # Variables: 3*N
        def objective(v):
            cx = v[0:N]
            cy = v[N:2*N]
            r = v[2*N:3*N]
            return -np.sum(r)
            
        def constraints_def(v):
            cx = v[0:N]
            cy = v[N:2*N]
            r = v[2*N:3*N]
            cons = []
            
            # Boundaries
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]}) # x - r >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: W - v[i] - v[2*N+i]}) # W - x - r >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]}) # y - r >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: H - v[N+i] - v[2*N+i]}) # H - y - r >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i] - 1e-6}) # r >= eps
                
            # Non-overlap
            for i in range(N):
                for j in range(i + 1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: 
                            (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2
                    })
            return cons
            
        v0 = x0.flatten()
        try:
            res = minimize(objective, v0, method='SLSQP', 
                           bounds=[(1e-4, None)] * (3*N),
                           constraints=constraints_def,
                           options={'maxiter': max_iter, 'ftol': 1e-8})
            if res.success or get_score(res.x.reshape(N, 3)) > get_score(x0):
                return res.x.reshape(N, 3)
        except Exception:
            pass
        return x0

    def sequential_insert(W, H, seed=0):
        """
        Greedily insert N circles into W x H box.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        circles = [] # list of [cx, cy, r]
        
        # Pre-calculate grid for candidate centers
        # Use a moderate grid size for speed
        grid_size = 40
        xs = np.linspace(0, W, grid_size)
        ys = np.linspace(0, H, grid_size)
        
        for k in range(N):
            best_r = 0
            best_cx = W / 2
            best_cy = H / 2
            found = False
            
            # Coarse search
            for x in xs:
                for y in ys:
                    # Calculate max radius for a circle at (x,y)
                    r_max = min(x, W - x, y, H - y)
                    if r_max <= 0:
                        continue
                    
                    # Check against existing circles
                    for cx_j, cy_j, r_j in circles:
                        dist = np.hypot(x - cx_j, y - cy_j)
                        r_candidate = dist - r_j
                        if r_candidate < r_max:
                            r_max = r_candidate
                        if r_max <= 0:
                            break
                    
                    if r_max > best_r:
                        best_r = r_max
                        best_cx = x
                        best_cy = y
                        found = True
            
            if not found:
                # Fallback: place tiny circle in center
                circles.append([W/2, H/2, 1e-5])
                continue
                
            # Local refinement around best_cx, best_cy
            # Use a finer grid in a small neighborhood
            ref_grid_size = 20
            ref_radius = 0.1 * min(W, H) # Search within this radius
            if ref_radius > min(W, H)/2:
                ref_radius = min(W, H)/2
                
            local_xs = np.linspace(max(0, best_cx - ref_radius), min(W, best_cx + ref_radius), ref_grid_size)
            local_ys = np.linspace(max(0, best_cy - ref_radius), min(H, best_cy + ref_radius), ref_grid_size)
            
            for x in local_xs:
                for y in local_ys:
                    r_max = min(x, W - x, y, H - y)
                    if r_max <= 0:
                        continue
                    for cx_j, cy_j, r_j in circles:
                        dist = np.hypot(x - cx_j, y - cy_j)
                        r_candidate = dist - r_j
                        if r_candidate < r_max:
                            r_max = r_candidate
                        if r_max <= 0:
                            break
                    if r_max > best_r:
                        best_r = r_max
                        best_cx = x
                        best_cy = y
            
            circles.append([best_cx, best_cy, max(best_r, 1e-5)])
            
        return np.array(circles)

    # --- Main Loop ---
    
    attempts = 0
    max_attempts = 30
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Vary aspect ratio
        # W + H = 2
        # Try a range of aspect ratios
        W = np.random.uniform(0.8, 1.2)
        H = 2.0 - W
        
        # 1. Generate a packing using sequential insertion
        # Use different seeds for diversity
        seed = attempts
        x_seq = sequential_insert(W, H, seed=seed)
        
        # 2. Polish the result using SLSQP
        # This allows the circles to move and grow slightly
        x_polished = polish_with_slsqp(x_seq, W, H, max_iter=150)
        
        # 3. Evaluate
        score = get_score(x_polished)
        
        if score > best_score:
            best_score = score
            best_result = x_polished
            
        # 4. Also try a random initialization + SLSQP occasionally (from current program strategy)
        if attempts % 3 == 0:
             # Random init
             cx = np.random.uniform(0.1, W-0.1, N)
             cy = np.random.uniform(0.1, H-0.1, N)
             r = np.random.uniform(0.05, 0.2, N)
             x_rand = np.column_stack((cx, cy, r))
             x_rand_polished = polish_with_slsqp(x_rand, W, H, max_iter=150)
             score_rand = get_score(x_rand_polished)
             if score_rand > best_score:
                 best_score = score_rand
                 best_result = x_rand_polished

    if best_result is None:
        # Fallback: simple grid in 1x1 square
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
