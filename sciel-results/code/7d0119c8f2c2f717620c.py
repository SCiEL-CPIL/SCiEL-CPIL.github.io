import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    # Distance squared on triangular lattice
    # Points are a + b*omega, omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # (a1 + b1*omega) - (a2 + b2*omega) = (a1-a2) + (b1-b2)*omega
    # |z|^2 = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)(b1-b2)
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # 1. Generate Candidate Points
    # We expect the optimal 32 points to be within a small radius.
    # A hexagon of radius 5 has 91 points. This is a safe upper bound for the search space.
    candidates = []
    R = 6 # Hex radius
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if hex_dist([a, b]) <= R:
                candidates.append([a, b])
    
    N = len(candidates)
    
    # 2. Build Graph of Unit Distances
    # We need to identify all pairs (i, j) such that dist_sq(candidates[i], candidates[j]) == 1
    edges = []
    edge_map = {} # (i, j) -> index in edges list
    for i in range(N):
        for j in range(i+1, N):
            if dist_sq(candidates[i], candidates[j]) == 1:
                edges.append((i, j))
                edge_map[(i, j)] = len(edges) - 1
                edge_map[(j, i)] = len(edges) - 1
                
    M = len(edges)
    
    # 3. ILP Formulation
    # Variables:
    # x_i for i in 0..N-1 (binary)
    # y_k for k in 0..M-1 (binary), corresponding to edge k
    
    # Objective: Maximize sum(y_k)
    # In scipy.optimize.milp, we minimize. So we minimize -sum(y_k).
    # c vector:
    c = np.zeros(N + M)
    c[N:] = -1.0 # Minimize -sum(y) -> Maximize sum(y)
    
    # Constraints:
    # 1. Sum(x_i) == 32
    # 2. y_k <= x_i
    # 3. y_k <= x_j
    # 4. y_k >= x_i + x_j - 1  =>  -y_k + x_i + x_j >= 1
    
    # We will build the constraint matrix A_ub and b_ub for A_ub @ vars <= b_ub
    # And A_eq and b_eq for A_eq @ vars == b_eq
    
    # Equality constraint: sum(x) = 32
    A_eq = np.zeros((1, N + M))
    b_eq = np.array([32.0])
    A_eq[0, :N] = 1.0
    
    # Inequality constraints
    # For each edge k (i, j):
    # y_k - x_i <= 0
    # y_k - x_j <= 0
    # -y_k + x_i + x_j >= 1  =>  y_k - x_i - x_j <= -1
    
    # Total inequality constraints: 3 * M
    A_ub = np.zeros((3*M, N + M))
    b_ub = np.zeros(3*M)
    
    for k, (i, j) in enumerate(edges):
        # y_k - x_i <= 0
        row = 3*k
        A_ub[row, N+k] = 1.0
        A_ub[row, i] = -1.0
        b_ub[row] = 0.0
        
        # y_k - x_j <= 0
        row = 3*k + 1
        A_ub[row, N+k] = 1.0
        A_ub[row, j] = -1.0
        b_ub[row] = 0.0
        
        # y_k - x_i - x_j <= -1
        row = 3*k + 2
        A_ub[row, N+k] = 1.0
        A_ub[row, i] = -1.0
        A_ub[row, j] = -1.0
        b_ub[row] = -1.0
        
    # Variable bounds
    bounds = [(0, 1)] * (N + M)
    
    # Integrality
    integrality = np.ones(N + M) # All variables are integer (0 or 1)
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        constraints = [
            LinearConstraint(A_eq, 32.0, 32.0),
            LinearConstraint(A_ub, -np.inf, b_ub)
        ]
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb=np.zeros(N+M), ub=np.ones(N+M))
        )
        
        if res.status == 0 and res.x is not None:
            x_vars = res.x[:N]
            selected_indices = [i for i in range(N) if x_vars[i] > 0.5]
            
            if len(selected_indices) == 32:
                result = [candidates[i][:] for i in selected_indices]
                
                # Verify validity and count edges
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
                
                # Check bounds
                valid = all(abs(p[0]) <= 64 and abs(p[1]) <= 64 for p in result)
                
                if valid:
                    return result
                else:
                    # Fallback if bounds violated (should not happen with R=6)
                    pass
                    
    except Exception as e:
        pass
        
    # 4. Fallback: Heuristic Search if ILP fails or times out
    # Generate initial hex spiral
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
        
    candidates_sorted = sorted(candidates, key=lambda p: (hex_dist(p), angle(p)))
    result = [p[:] for p in candidates_sorted[:32]]
    
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
        
    best_score = count_edges_fast(result)
    best_points = [p[:] for p in result]
    
    # Simple Local Search / Hill Climbing
    # We will try to swap points or move points to improve edge count
    # Given the time limit, a quick random restart hill climb is safer than complex SA
    
    current_points = [p[:] for p in result]
    current_score = best_score
    
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    def try_move(points, idx, new_a, new_b):
        a, b = points[idx]
        if new_a == a and new_b == b:
            return False
        if abs(new_a) > 64 or abs(new_b) > 64:
            return False
        # Check distinctness
        for j, p in enumerate(points):
            if j == idx:
                continue
            if p[0] == new_a and p[1] == new_b:
                return False
        # Distance constraint is automatically satisfied for distinct lattice points
        return True

    max_iters = 20000
    for _ in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        if best_score >= 76:
            break
            
        idx = random.randint(0, 31)
        a, b = current_points[idx]
        
        # Try moving to a random nearby spot
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
