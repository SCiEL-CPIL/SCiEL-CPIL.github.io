    def _min_outer_side(hexagons):
        all_points = []
        for cx, cy, ang, s in hexagons:
            verts = _hex_vertices(cx, cy, s, ang)
            all_points.extend(verts)
        all_points = np.array(all_points)
        
        if len(all_points) < 3:
            return 0.0
            
        try:
            hull = ConvexHull(all_points)
            hull_points = all_points[hull.vertices]
        except Exception:
            return float('inf')
            
        best_S = float('inf')
        
        # Search rotations 0 to 30 degrees
        for theta_deg in np.arange(0, 30.1, 0.5):
            theta = math.radians(theta_deg)
            normals = np.array([
                [math.cos(theta + math.radians(30 + 60*i)), math.sin(theta + math.radians(30 + 60*i))]
                for i in range(6)
            ])
            
            # Project hull points onto normals
            # Projections shape: (6, N_hull)
            projections = normals @ hull_points.T
            
            L = projections.max(axis=1) # Max projection in each normal dir
            R = projections.min(axis=1) # Min projection in each normal dir
            
            # We want to find min S such that there exists C
            # Constraints on C:
            # n_k . C <= R_k + a
            # -n_k . C <= a - L_k
            # where a = S * sqrt(3) / 2
            
            # Binary search on S (or a)
            # Lower bound: max(W_k) / sqrt(3) ? No.
            # Width W_k = L_k - R_k.
            # The hexagon width in direction k is 2*a.
            # But the center can shift.
            # The minimum a is at least max(W_k)/2? No.
            # For a rectangle, min enclosing rect side is max(width).
            # For hexagon, it's similar.
            
            low_a = 0.0
            high_a = 100.0 # Arbitrary large upper bound
            
            # Check feasibility
            def is_feasible(a):
                # A_ub, b_ub for linprog
                # Minimize 0 subject to A_ub x <= b_ub
                A_ub = np.zeros((12, 2))
                b_ub = np.zeros(12)
                
                for i in range(6):
                    # n_i . C <= R_i + a
                    A_ub[2*i, 0] = normals[i, 0]
                    A_ub[2*i, 1] = normals[i, 1]
                    b_ub[2*i] = R[i] + a
                    
                    # -n_i . C <= a - L_i
                    A_ub[2*i+1, 0] = -normals[i, 0]
                    A_ub[2*i+1, 1] = -normals[i, 1]
                    b_ub[2*i+1] = a - L[i]
                    
                try:
                    res = linprog(c=np.zeros(2), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None)], method='highs')
                    return res.success
                except Exception:
                    return False

            # Binary search
            # We can start low_a = max((L-R)/2) / (sqrt(3)/2) ? 
            # Actually, just start low_a = 0.
            # But we can get a better lower bound.
            # The width of the hexagon in direction k is 2*a.
            # The projection of the point set in direction k has width W_k.
            # The center C must be such that the interval [C.n - a, C.n + a] covers [R, L].
            # This requires 2a >= L - R ? No, because C can be anywhere.
            # Wait, if C is free, the condition is that the intersection of the 6 strips is non-empty.
            # The width of the strip is 2a.
            # A necessary condition is that the "width" of the feasible region for C is non-negative.
            
            # Let's just binary search a.
            for _ in range(20): # ~1e-6 precision
                mid_a = (low_a + high_a) / 2
                if is_feasible(mid_a):
                    high_a = mid_a
                else:
                    low_a = mid_a
            
            S_candidate = high_a * 2.0 / math.sqrt(3)
            if S_candidate < best_S:
                best_S = S_candidate
                
        return best_S
    