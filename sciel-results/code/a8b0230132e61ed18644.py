import numpy as np
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def time_left():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def hex_support(angle, rot):
    """
    Support function of a unit regular hexagon (side 1, circumradius 1)
    rotated by 'rot' in direction 'angle'.
    h(alpha) = max_k cos(alpha - rot - k*pi/3)
    """
    # The vertices are at angles rot + k*pi/3
    # The support function is the max projection of vertices on the direction 'angle'
    # proj_k = cos(angle - (rot + k*pi/3))
    # Due to symmetry, we can map the angle difference to [0, pi/3)
    # But let's just compute the 6 values to be safe and simple, N is small.
    k = np.arange(6)
    angles = rot + k * np.pi / 3
    projs = np.cos(angle - angles)
    return np.max(projs)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_A = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Precompute outer hexagon normals (apothem directions)
    # Outer hexagon centered at origin, rotation 0.
    # Normals are at pi/6 + k*pi/3
    outer_normals = np.pi/6 + np.arange(6) * np.pi / 3
    outer_normals_vec = np.column_stack((np.cos(outer_normals), np.sin(outer_normals)))
    
    def check_validity(centers, rots, A):
        """
        Check if the configuration is valid (all hexagons inside and non-overlapping).
        Returns True if valid, False otherwise.
        """
        # 1. Containment Constraints
        for i in range(N):
            ci = centers[i]
            thi = rots[i]
            for k in range(6):
                nk_vec = outer_normals_vec[k]
                proj_center = np.dot(ci, nk_vec)
                sup = hex_support(outer_normals[k], thi)
                val = proj_center + sup
                if val > A + 1e-6:
                    return False
        
        # 2. Non-overlap Constraints
        for i in range(N):
            for j in range(i + 1, N):
                ci = centers[i]
                cj = centers[j]
                thi = rots[i]
                thj = rots[j]
                
                d_vec = cj - ci
                min_margin = float('inf')
                
                for k in range(6):
                    # Axis from i
                    ang_i = thi + np.pi/6 + k * np.pi/3
                    d_i = np.array([np.cos(ang_i), np.sin(ang_i)])
                    proj_dist = np.dot(d_vec, d_i)
                    sup_i = hex_support(ang_i, thi)
                    sup_j = hex_support(ang_i + np.pi, thj)
                    margin = proj_dist - (sup_i + sup_j)
                    if margin < min_margin:
                        min_margin = margin
                    
                    # Axis from j
                    ang_j = thj + np.pi/6 + k * np.pi/3
                    d_j = np.array([np.cos(ang_j), np.sin(ang_j)])
                    proj_dist_j = np.dot(d_vec, d_j)
                    sup_i_j = hex_support(ang_j, thi)
                    sup_j_j = hex_support(ang_j + np.pi, thj)
                    margin_j = proj_dist_j - (sup_i_j + sup_j_j)
                    if margin_j < min_margin:
                        min_margin = margin_j
                
                # If min_margin < 0, they overlap
                if min_margin < -1e-6:
                    return False
        
        return True

    def compute_min_A(centers, rots):
        """
        Compute the minimum A required to contain the given configuration.
        """
        min_A = 1.0
        for i in range(N):
            ci = centers[i]
            thi = rots[i]
            for k in range(6):
                nk_vec = outer_normals_vec[k]
                proj_center = np.dot(ci, nk_vec)
                sup = hex_support(outer_normals[k], thi)
                val = proj_center + sup
                if val > min_A:
                    min_A = val
        return min_A

    # Initial guesses
    # We can try a few random restarts
    np.random.seed(42)
    
    num_restarts = 100
    restarts_done = 0
    
    while restarts_done < num_restarts and time.time() < deadline:
        # Random initialization
        # A initial guess: start with a large enough A to allow packing, e.g. 4.0
        A0 = 4.0
        centers0 = np.random.uniform(-1.5, 1.5, (N, 2))
        rots0 = np.random.uniform(0, np.pi/3, N)
        
        # Optimize using Nelder-Mead to minimize A + penalty
        # But first, let's just try to find a valid configuration and then minimize A
        # Actually, let's use a simpler approach: random search with local refinement
        
        # Try to find a valid configuration near the random start
        # by using a local optimizer to minimize the penalty
        def penalty_only(params):
            centers = params[:3*N].reshape(N, 2)
            rots = params[3*N:3*N+N]
            A = params[-1]
            
            if A < 1.0:
                return 1e10
                
            penalty = 0.0
            
            # 1. Containment Constraints
            for i in range(N):
                ci = centers[i]
                thi = rots[i]
                for k in range(6):
                    nk_vec = outer_normals_vec[k]
                    proj_center = np.dot(ci, nk_vec)
                    sup = hex_support(outer_normals[k], thi)
                    val = proj_center + sup
                    if val > A:
                        penalty += (val - A) ** 2
            
            # 2. Non-overlap Constraints
            for i in range(N):
                for j in range(i + 1, N):
                    ci = centers[i]
                    cj = centers[j]
                    thi = rots[i]
                    thj = rots[j]
                    
                    d_vec = cj - ci
                    min_margin = float('inf')
                    
                    for k in range(6):
                        ang_i = thi + np.pi/6 + k * np.pi/3
                        d_i = np.array([np.cos(ang_i), np.sin(ang_i)])
                        proj_dist = np.dot(d_vec, d_i)
                        sup_i = hex_support(ang_i, thi)
                        sup_j = hex_support(ang_i + np.pi, thj)
                        margin = proj_dist - (sup_i + sup_j)
                        if margin < min_margin:
                            min_margin = margin
                        
                        ang_j = thj + np.pi/6 + k * np.pi/3
                        d_j = np.array([np.cos(ang_j), np.sin(ang_j)])
                        proj_dist_j = np.dot(d_vec, d_j)
                        sup_i_j = hex_support(ang_j, thi)
                        sup_j_j = hex_support(ang_j + np.pi, thj)
                        margin_j = proj_dist_j - (sup_i_j + sup_j_j)
                        if margin_j < min_margin:
                            min_margin = margin_j
                    
                    if min_margin < 0:
                        penalty += (-min_margin) ** 2
            
            return penalty
        
        x0 = np.concatenate([centers0.flatten(), rots0, [A0]])
        
        try:
            # Use Nelder-Mead to minimize penalty
            res = so.minimize(penalty_only, x0, method='Nelder-Mead', 
                              options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < 1e-4:
                # Configuration is valid, now compute the actual A
                params = res.x
                centers = params[:3*N].reshape(N, 2)
                rots = params[3*N:3*N+N]
                
                A_val = compute_min_A(centers, rots)
                
                if A_val < best_A:
                    # Verify validity
                    if check_validity(centers, rots, A_val):
                        best_A = A_val
                        best_inner = []
                        for i in range(N):
                            best_inner.append([centers[i][0], centers[i][1], np.degrees(rots[i])])
        except Exception as e:
            pass

        restarts_done += 1
        
        if time.time() > deadline - 5:
            break
            
    if best_inner is None:
        # Fallback to a known valid configuration
        # Grid of hexagons
        inner = []
        for i in range(N):
            x = (i % 4) * 2.2
            y = (i // 4) * 1.9
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    return best_inner, (0.0, 0.0), best_A, 0.0

# EVOLVE-BLOCK-END
