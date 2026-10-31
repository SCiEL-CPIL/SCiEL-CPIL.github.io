"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    # Equilateral triangle vertices
    h = np.sqrt(3) / 2
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    
    # Function to check if points are inside the triangle
    # Returns True if all points are inside, False otherwise.
    # Uses barycentric coordinate check via areas.
    def are_points_inside(points):
        # points: (N, 2)
        # Calculate area of triangle ABC
        # A=V[0], B=V[1], C=V[2]
        # Area = 0.5 * sqrt(3)
        total_area = 0.5 * np.sqrt(3)
        
        # For each point, check if it is inside
        # We can vectorize this.
        # P is inside if P = a*A + b*B + c*C with a,b,c >= 0 and a+b+c=1
        # This is equivalent to:
        # 1. y >= 0 (above AB)
        # 2. y <= sqrt(3) * x (below AC? No, AC is from (0,0) to (0.5, h). Slope is h/0.5 = sqrt(3). So y <= sqrt(3)x for x in [0, 0.5]?)
        # Let's use the standard half-plane tests.
        # Edge AB: y >= 0
        # Edge BC: Line from (1,0) to (0.5, h). Slope = (h-0)/(0.5-1) = -sqrt(3). Eq: y - 0 = -sqrt(3)(x - 1) => y = -sqrt(3)x + sqrt(3) => sqrt(3)x + y <= sqrt(3)
        # Edge CA: Line from (0.5, h) to (0,0). Slope = h/0.5 = sqrt(3). Eq: y = sqrt(3)x. Point must be to the right/below? 
        # The triangle is x >= 0, y >= 0, y <= sqrt(3)x (for left part) and y <= -sqrt(3)(x-1) (for right part).
        # Actually, simpler:
        # 1. y >= -eps
        # 2. sqrt(3)*x + y <= sqrt(3) + eps
        # 3. y <= sqrt(3)*x + eps  (Wait, for x > 0.5, this is not the boundary. The boundary is BC.)
        
        # Let's use the cross product sign test for robustness.
        # For point P, it is inside if it is on the same side of each edge as the interior.
        # Edge AB (from A to B): Vector AB = (1,0). Normal (0,1). P should have dot(P-A, (0,1)) >= 0 => Py >= 0.
        # Edge BC (from B to C): Vector BC = (-0.5, h). Normal (h, 0.5). P-B dot (h, 0.5) >= 0?
        #   (Px-1)*h + (Py-0)*0.5 >= 0 => h*Px + 0.5*Py >= h.
        # Edge CA (from C to A): Vector CA = (-0.5, -h). Normal (-h, 0.5). P-C dot (-h, 0.5) >= 0?
        #   (Px-0.5)*(-h) + (Py-h)*0.5 >= 0 => -h*Px + 0.5h + 0.5Py - 0.5h >= 0 => 0.5Py >= h*Px => Py >= sqrt(3)*Px.
        
        # Let's verify the third one.
        # A=(0,0), C=(0.5, h). Line AC: y = sqrt(3)x.
        # Interior point, e.g., centroid (0.5, h/3).
        # h/3 >= sqrt(3)*0.5 ? h/3 = sqrt(3)/6. sqrt(3)*0.5 = sqrt(3)/2.
        # sqrt(3)/6 < sqrt(3)/2. So the inequality direction is wrong.
        # The interior is to the "right" of CA if traversing C->A?
        # Let's just use barycentric coordinates calculated via linear algebra or area ratios.
        
        # Area method is safer.
        # Area(P, B, C) + Area(A, P, C) + Area(A, B, P) == Area(A, B, C)
        # And all sub-areas >= 0.
        
        # Vectorized area calculation
        # P is (N, 2), B, C, A are (2,)
        # Area(P, B, C) = 0.5 * | (B-P) x (C-P) |
        # This is expensive to vectorize fully without loops, but N=11 is small.
        
        for i in range(len(points)):
            p = points[i]
            # Check 1: y >= 0
            if p[1] < -1e-9:
                return False
            # Check 2: Line BC: h*x + 0.5*y <= h
            # h*px + 0.5*py <= h
            if h * p[0] + 0.5 * p[1] > h + 1e-9:
                return False
            # Check 3: Line AC: y <= sqrt(3)*x  =>  sqrt(3)*x - y >= 0
            # Wait, let's re-verify the centroid.
            # Centroid (0.5, h/3).
            # sqrt(3)*0.5 - h/3 = sqrt(3)/2 - sqrt(3)/6 = sqrt(3)/3 > 0.
            # So condition is sqrt(3)*x - y >= 0.
            if np.sqrt(3) * p[0] - p[1] < -1e-9:
                return False
                
        return True

    def min_triangle_area(points):
        # points: (N, 2)
        # Calculate all pairwise triangle areas
        # N=11, so 11*10*9/6 = 165 triangles.
        min_area = float('inf')
        n = len(points)
        # Use numpy for speed
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * | (p2-p1) x (p3-p1) |
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                    if area < min_area:
                        min_area = area
        return min_area

    def objective(x):
        # x: flattened array of 2*N coordinates
        points = x.reshape(-1, 2)
        
        # Check bounds and triangle containment
        # x in [0,1], y in [0, h]
        if np.any(points[:, 0] < 0) or np.any(points[:, 0] > 1) or np.any(points[:, 1] < 0) or np.any(points[:, 1] > h):
            return 1e6 # Large penalty
            
        if not are_points_inside(points):
            return 1e6 # Large penalty
            
        area = min_triangle_area(points)
        return -area

    # Initial bounds for DE
    bounds = [(0, 1), (0, h)] * N
    
    # Run Differential Evolution
    # maxiter and popsize control runtime.
    # N=11, 22 variables.
    # 120s limit.
    de_result = differential_evolution(
        objective, 
        bounds, 
        maxiter=500, 
        popsize=30, 
        seed=42,
        tol=1e-6,
        init='sobol'
    )
    
    best_x = de_result.x
    
    # Refine with SLSQP
    # We need to define constraints for SLSQP if we use it, or just use the penalty method again.
    # SLSQP is fast for local refinement.
    def objective_local(x):
        points = x.reshape(-1, 2)
        if not are_points_inside(points):
            # Penalty for being outside
            penalty = 0
            for i in range(N):
                p = points[i]
                if p[1] < 0: penalty += (p[1])**2
                if h * p[0] + 0.5 * p[1] > h: penalty += (h * p[0] + 0.5 * p[1] - h)**2
                if np.sqrt(3) * p[0] - p[1] < 0: penalty += (np.sqrt(3) * p[0] - p[1])**2
            return 1e6 + penalty
            
        area = min_triangle_area(points)
        return -area

    # Bounds for SLSQP
    slsqp_bounds = [(0, 1), (0, h)] * N
    
    # Local refinement
    local_result = minimize(
        objective_local, 
        best_x, 
        method='SLSQP', 
        bounds=slsqp_bounds,
        options={'maxiter': 200, 'ftol': 1e-12}
    )
    
    final_points = local_result.x.reshape(-1, 2)
    
    # Final check
    if not are_points_inside(final_points):
        # Fallback to DE result if local optimization pushed it out
        final_points = best_x.reshape(-1, 2)

    return final_points
# EVOLVE-BLOCK-END
