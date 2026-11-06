import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    # Distance squared on triangular lattice
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # 1. Generate Candidate Points
    candidates = []
    R = 6 # Hex radius
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if hex_dist([a, b]) <= R:
                candidates.append([a, b])
    
    N = len(candidates)
    
    # 2. Build Graph of Unit Distances
    edges = []
    edge_map = {}
    for i in range(N):
        for j in range(i+1, N):
            if dist_sq(candidates[i], candidates[j]) == 1:
                edges.append((i, j))
                edge_map[(i, j)] = len(edges) - 1
                edge_map[(j, i)] = len(edges) - 1
                
    M = len(edges)
    
    # 3. ILP Formulation
    c = np.zeros(N + M)
    c[N:] = -1.0
    
    A_eq = np.zeros((1, N + M))
    b_eq = np.array([32.0])
    A_eq[0, :N] = 1.0
    
    A_ub = np.zeros((3*M, N + M))
    b_ub = np.zeros(3*M)
    
    for k, (i, j) in enumerate(edges):
        row = 3*k
        A_ub[row, N+k] = 1.0
        A_ub[row, i] = -1.0
        b_ub[row] = 0.0
        
        row = 3*k + 1
        A_ub[row, N+k] = 1.0
        A_ub[row, j] = -1.0
        b_ub[row] = 0.0
        
        row = 3*k + 2
        A_ub[row, N+k] = 1.0
        A_ub[row, i] = -1.0
        A_ub[row, j] = -1.0
        b_ub[row] = -1.0
        
    bounds = [(0, 1)] * (N + M)
    integrality = np.ones(N + M)
    
    ilp_success = False
    result = None
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        constraints = [
            LinearConstraint(A_eq, 32.0, 32.0),
            LinearConstraint(A_ub, -np.inf, b_ub)
        ]
        
        # Add time limit to prevent hanging
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb=np.zeros(N+M), ub=np.ones(N+M)),
            options={"time_limit": 60.0}
        )
        
        if res.status == 0 and res.x is not None:
            x_vars = res.x[:N]
            selected_indices = [i for i in range(N) if x_vars[i] > 0.5]
            
            if len(selected_indices) == 32:
                result = [candidates[i][:] for i in selected_indices]
                
                def count_edges(points):
                    count = 0
                    n = len(points)
                    lookup = set((p[0], p[1]) for p in points)
                    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
                    for p in points:
                        a, b = p
                        for da, db in neighbors:
                            if (a+da, b+db) in lookup:
                                count += 1
                    return count // 2
                
                edge_count = count_edges(result)
                valid = all(abs(p[0]) <= 64 and abs(p[1]) <= 64 for p in result)
                
                if valid:
                    ilp_success = True
                    return result
                    
    except Exception as e:
        pass
        
    # 4. Fallback: Deterministic Local Search
    # Start with hex spiral
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
        
    candidates_sorted = sorted(candidates, key=lambda p: (hex_dist(p), angle(p)))
    current_points = [p[:] for p in candidates_sorted[:32]]
    
    def count_edges_fast(points):
        lookup = set((p[0], p[1]) for p in points)
        count = 0
        neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        for p in points:
            a, b = p
            for da, db in neighbors:
                if (a+da, b+db) in lookup:
                    count += 1
        return count // 2
        
    best_score = count_edges_fast(current_points)
    best_points = [p[:] for p in current_points]
    
    # Local Search: Try swapping any point with any candidate not in the set
    # We do this for a fixed number of iterations or until no improvement
    # To make it faster, we only consider candidates within a certain distance
    # But since N is small (91), we can just try all swaps
    
    max_iters = 500
    for _ in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        if best_score >= 76:
            break
            
        improved = False
        # Create a set of current points for quick lookup
        current_set = set((p[0], p[1]) for p in current_points)
        
        # Try moving each point to a better location
        # We iterate through points and try to swap with candidates
        # To avoid O(N^2) checks per iteration, we can just pick a random point and try random swaps
        # But deterministic is better. Let's try to find the best swap in one pass.
        
        best_swap_score = best_score
        best_swap_idx = -1
        best_swap_new_point = None
        
        # Limit search to keep it fast
        # Try a subset of swaps
        for i in range(32):
            if time.time() > deadline - 5.0:
                break
            p_i = current_points[i]
            # Try swapping with candidates not in current set
            # We can limit to candidates close to p_i
            for j in range(N):
                if (candidates[j][0], candidates[j][1]) in current_set:
                    continue
                # Check if swap improves
                # Remove p_i, add candidates[j]
                # This is expensive to do fully for every pair.
                # Instead, let's just do a random restart hill climb with fixed seed
                pass
        
        # Simpler: Random hill climb with fixed seed
        # But we want deterministic.
        # Let's just use the random hill climb but with a fixed seed
        random.seed(42)
        
        neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        
        def try_move(points, idx, new_a, new_b):
            a, b = points[idx]
            if new_a == a and new_b == b:
                return False
            if abs(new_a) > 64 or abs(new_b) > 64:
                return False
            for j, p in enumerate(points):
                if j == idx:
                    continue
                if p[0] == new_a and p[1] == new_b:
                    return False
            return True

        max_iters_hc = 5000
        for _ in range(max_iters_hc):
            if time.time() > deadline - 5.0:
                break
            if best_score >= 76:
                break
                
            idx = random.randint(0, 31)
            a, b = current_points[idx]
            
            da = random.randint(-3, 3)
            db = random.randint(-3, 3)
            new_a = a + da
            new_b = b + db
            
            if not try_move(current_points, idx, new_a, new_b):
                continue
                
            old_pos = (a, b)
            current_points[idx] = [new_a, new_b]
            new_score = count_edges_fast(current_points)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[idx] = [a, b]
                
    return best_points
