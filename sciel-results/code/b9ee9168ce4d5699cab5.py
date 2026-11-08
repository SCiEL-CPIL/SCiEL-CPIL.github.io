import numpy as np
import time
import random

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0
H = np.sqrt(3) / 2.0

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 of the points.
    Optimized for n=11 using vectorized operations where possible.
    Returns normalized area (divided by total triangle area).
    """
    n = len(points)
    min_area = float('inf')
    
    # Precompute all pairwise differences? 
    # For n=11, triple loop is 165 iterations. Vectorizing the inner loop is faster.
    for i in range(n):
        pi = points[i]
        # Vectors from i to j, k where j > i, k > j
        # We can vectorize over k for a fixed j, or just do the loop.
        # Let's try to vectorize the area calculation for all triples involving i.
        # Actually, for n=11, pure python loops are quite slow if not careful.
        # Let's use a semi-vectorized approach:
        # For each i, compute cross products for all pairs (j, k) with j,k > i?
        # No, standard area formula: 0.5 * | (Pj-Pi) x (Pk-Pi) |
        
        # Let's stick to a tight loop but with numpy scalar ops if needed, 
        # or just standard python. 165 iterations is tiny.
        # The bottleneck is usually the function call overhead in SA.
        
        for j in range(i + 1, n):
            v1 = points[j] - pi
            # Vectorize over k > j
            if j + 1 < n:
                v2s = points[j+1:] - pi
                # Cross product z-component: v1[0]*v2[1] - v1[1]*v2[0]
                crosses = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
                areas = 0.5 * np.abs(crosses)
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    
    return min_area / AREA

def _is_inside_triangle(p):
    """Check if a point (x, y) is inside the equilateral triangle."""
    x, y = p
    if y < -1e-9: return False
    if x < -1e-9 or x > 1.0 + 1e-9: return False
    # Line equations:
    # Left edge: y = sqrt(3) x  =>  y - sqrt(3)x <= 0
    # Right edge: y = -sqrt(3) x + sqrt(3) => y + sqrt(3)x - sqrt(3) <= 0
    # Bottom edge: y >= 0
    if y - np.sqrt(3) * x > 1e-9: return False
    if y + np.sqrt(3) * x - np.sqrt(3) > 1e-9: return False
    return True

def _is_inside_triangle_batch(points):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    # y >= 0
    if np.any(y < -1e-9): return False
    # y <= sqrt(3) x
    if np.any(y - np.sqrt(3) * x > 1e-9): return False
    # y <= -sqrt(3) x + sqrt(3)
    if np.any(y + np.sqrt(3) * x - np.sqrt(3) > 1e-9): return False
    return True

def _project_to_triangle(points):
    """
    Project points to the nearest point inside the triangle.
    This ensures validity without rejecting moves.
    """
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # Clip y to >= 0
    y = np.clip(y, 0.0, H)
    
    # For each point, check against the two slanted edges.
    # If outside, project onto the edge.
    # Left edge: x - y/sqrt(3) >= 0  => x >= y/sqrt(3)
    # Right edge: (1-x) - y/sqrt(3) >= 0 => x <= 1 - y/sqrt(3)
    
    # Calculate violations
    left_viol = y / np.sqrt(3) - x
    right_viol = x - (1.0 - y / np.sqrt(3))
    
    # If left_viol > 0, point is to the left of the left edge.
    # Project onto line y = sqrt(3) x.
    # Normal vector n = (-1, 1/sqrt(3)) normalized? 
    # Simpler: The closest point on the line y = sqrt(3)x to (x,y)
    # is given by projection.
    # Actually, for small violations, we can just adjust x and y.
    # But a robust projection is better.
    
    # Let's use a simple iterative correction or direct formula.
    # Direct formula for projection onto line ax+by+c=0 is complex for multiple edges.
    # Given the small domain, a simple rejection or clipping might suffice, 
    # but projection allows smoother SA.
    
    # Let's try a simple "push back" method.
    # If x < y/sqrt(3), set x = y/sqrt(3). This moves it onto the edge.
    # If x > 1 - y/sqrt(3), set x = 1 - y/sqrt(3).
    
    mask_left = left_viol > 0
    mask_right = right_viol > 0
    
    x[mask_left] = y[mask_left] / np.sqrt(3)
    x[mask_right] = 1.0 - y[mask_right] / np.sqrt(3)
    
    # It's possible that adjusting x for left violation creates a right violation or vice versa?
    # Unlikely for small steps, but let's do a second pass just in case.
    left_viol2 = y / np.sqrt(3) - x
    right_viol2 = x - (1.0 - y / np.sqrt(3))
    
    mask_left2 = left_viol2 > 0
    mask_right2 = right_viol2 > 0
    
    x[mask_left2] = y[mask_left2] / np.sqrt(3)
    x[mask_right2] = 1.0 - y[mask_right2] / np.sqrt(3)
    
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # 1. Generate a good initial configuration
    # Greedy construction: place points one by one to maximize the min distance to existing points.
    # This is a standard heuristic for TSP/packing and provides a strong starting point.
    
    def greedy_init():
        pts = []
        # Start with vertices? No, usually interior points are better for area.
        # Let's start with centroid and then add points.
        # Or just random + greedy refinement.
        
        # Initialize with random points
        current_pts = _project_to_triangle(rng.uniform(0, 1, (11, 2)))
        
        # Greedy improvement:
        # For each point, try to move it to a position that maximizes the min area contribution?
        # That's expensive. 
        # Alternative: Random restarts with SA is usually enough if SA is good.
        # Let's use a "Structured Random" init: 
        # Place points on a grid-like structure inside the triangle.
        
        # A known good pattern for n=11 involves a hexagonal packing distortion.
        # Let's try to generate points based on a distorted hex grid.
        
        # Hex grid points in unit triangle:
        # Row 0 (bottom): 4 points?
        # Row 1: 3 points?
        # Row 2: 2 points?
        # Row 3: 1 point?
        # Total 10. Need 11.
        
        # Let's just use a few random inits and let SA do the work.
        # But a better init helps SA escape.
        
        # Let's try placing 11 points randomly, then running a quick local search to "unclump" them.
        
        best_init = None
        best_init_score = 0
        
        for _ in range(20):
            pts = _project_to_triangle(rng.uniform(0, 1, (11, 2)))
            # Quick local search to spread them out
            for _ in range(50):
                improved = False
                for i in range(11):
                    # Try small random moves
                    for _ in range(5):
                        p_new = pts[i] + rng.normal(0, 0.05, 2)
                        p_new = _project_to_triangle(p_new.reshape(1,2))[0]
                        pts[i] = p_new
                # Evaluate
                score = _min_triangle_area(pts)
                if score > best_init_score:
                    best_init_score = score
                    best_init = pts.copy()
            
            if best_init_score > 0.08: # Good enough threshold?
                break
                
        if best_init is None:
            best_init = _project_to_triangle(rng.uniform(0, 1, (11, 2)))
            
        return best_init

    best_points = greedy_init()
    best_score = _min_triangle_area(best_points)
    
    # 2. Simulated Annealing
    # Objective: Maximize min_area.
    # We will perform many steps.
    
    T = 0.01  # Initial temperature
    cooling = 0.9995
    T_min = 1e-6
    
    # We need to estimate how many steps we can do.
    # _min_triangle_area takes ~50us?
    # 150s / 50us = 3,000,000 steps.
    # This is plenty.
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Adaptive step size
    step_size = 0.05
    
    while time.time() < deadline:
        # Generate a random move
        # Perturb a random point
        i = rng.integers(0, 11)
        delta = rng.normal(0, step_size, 2)
        
        new_point = current_points[i] + delta
        # Project to triangle
        new_point = _project_to_triangle(new_point.reshape(1, 2))[0]
        
        if np.allclose(new_point, current_points[i]):
            continue # No change
        
        # Calculate new score
        # Optimization: Only recalculate triangles involving point i?
        # For n=11, full recalculation is fast enough, but let's try to be smart.
        # Actually, _min_triangle_area is already optimized.
        
        temp_points = current_points.copy()
        temp_points[i] = new_point
        new_score = _min_triangle_area(temp_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = temp_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            # Increase step size slightly if improving?
            step_size = min(step_size * 1.01, 0.2)
        else:
            # Metropolis criterion
            delta_e = new_score - current_score
            if T > T_min:
                accept_prob = np.exp(delta_e / T)
                if rng.random() < accept_prob:
                    current_points = temp_points
                    current_score = new_score
            # Decrease step size if stuck?
            step_size = max(step_size * 0.999, 0.01)
        
        T *= cooling
        
        # If T gets too low, reset to a higher T to escape local minima?
        # Or just let it cool. With 3M steps, it should cool down nicely.
        
    return best_points

# Fallback if construct fails or times out unexpectedly
if __name__ == "__main__":
    # Just to ensure it runs
    pass
