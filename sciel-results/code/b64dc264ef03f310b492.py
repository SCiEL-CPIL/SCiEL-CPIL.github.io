import numpy as np
import random

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

def min_area(pts):
    # pts is (N, 2)
    # Calculate areas of all C(N,3) triangles
    # N=11 -> 165 triangles
    # Vectorized approach:
    # Generate indices
    # This might be overkill for N=11, a simple loop is probably fast enough and less memory overhead per call.
    # However, for speed in SA, let's try to vectorize.
    
    # Pre-computed indices for efficiency?
    # We'll just use a nested loop, it's small.
    min_a = float('inf')
    for i in range(N-2):
        pi = pts[i]
        for j in range(i+1, N-1):
            pj = pts[j]
            for k in range(j+1, N):
                pk = pts[k]
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                cross = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
                area = 0.5 * abs(cross)
                if area < min_a:
                    min_a = area
    return min_a

def project_to_triangle(p):
    # Check if p is inside ABC
    # Using barycentric coordinates or cross products
    # Cross product sign method:
    # For CCW triangle A, B, C
    # Edge AB: cross(B-A, p-A) >= 0
    # Edge BC: cross(C-B, p-B) >= 0
    # Edge CA: cross(A-C, p-C) >= 0
    
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
    c1 = cross(A, B, p)
    c2 = cross(B, C, p)
    c3 = cross(C, A, p)
    
    has_neg = (c1 < 0) or (c2 < 0) or (c3 < 0)
    has_pos = (c1 > 0) or (c2 > 0) or (c3 > 0)
    
    if not has_neg or not has_pos:
        return p # Inside or on boundary
        
    # Find closest edge
    # Distance from point to line segment
    def dist_to_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)
        
    d1 = dist_to_segment(p, A, B)
    d2 = dist_to_segment(p, B, C)
    d3 = dist_to_segment(p, C, A)
    
    min_d = min(d1, d2, d3)
    if min_d == d1:
        # Project onto AB
        ab = B - A
        ap = p - A
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))
        return A + t * ab
    elif min_d == d2:
        bc = C - B
        bp = p - B
        t = np.dot(bp, bc) / np.dot(bc, bc)
        t = max(0, min(1, t))
        return B + t * bc
    else:
        ca = A - C
        cp = p - C
        t = np.dot(cp, ca) / np.dot(ca, ca)
        t = max(0, min(1, t))
        return C + t * ca

def construct():
    # Initial guess: 
    # Let's try a few random starts and a structured start.
    best_points = None
    best_score = 0
    
    # Structured start: Triangular grid
    # We need 11 points.
    # Grid of side 4 has 10 points. Add center?
    # Or just use random restarts with SA.
    
    # Let's do 10 restarts of SA.
    # Time limit 120s. 
    # Each SA run should be fast.
    
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        # Initialize points
        # Random inside triangle
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v <= 1:
                    p = u * B + v * C
                    # Check if inside (it is by construction)
                    pts[i] = p
                    break
        
        current_score = min_area(pts)
        
        # Simulated Annealing
        T = 0.01
        T_min = 1e-6
        cooling = 0.999
        steps = 5000
        
        for step in range(steps):
            # Pick random point
            i = rng.integers(0, N)
            # Random move
            # Step size proportional to T?
            move_size = 0.05 * (T / 0.01) # Scale down as T drops? Or fixed?
            # Let's use a fixed step size that decreases slightly
            move_size = 0.02
            delta = rng.normal(0, move_size, 2)
            new_p = pts[i] + delta
            new_p = project_to_triangle(new_p)
            
            old_p = pts[i].copy()
            pts[i] = new_p
            
            new_score = min_area(pts)
            
            if new_score > current_score:
                current_score = new_score
            else:
                # Accept with probability exp((new - old) / T)
                # Note: We are maximizing, so delta E = new - old
                # If new < old, delta E is negative.
                prob = np.exp((new_score - current_score) / T)
                if rng.random() > prob:
                    pts[i] = old_p
                    # Keep current_score as is? 
                    # If we reject, current_score remains the score of the accepted state.
                    # But we calculated new_score for the rejected state.
                    # So current_score is still valid for pts.
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if current_score > best_score:
            best_score = current_score
            best_points = pts.copy()
            
    return best_points
