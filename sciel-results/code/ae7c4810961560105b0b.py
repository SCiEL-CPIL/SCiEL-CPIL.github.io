"""Heilbronn triangles - place 11 points to maximize minimum triangle area."""
import time
import numpy as np
from itertools import combinations

N = 11
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4  # area of the unit equilateral triangle


def bary_to_cart(b):
    """Convert barycentric coordinates (b0, b1, b2) with b0+b1+b2=1 to cartesian."""
    # b is shape (n, 2) where b[:, 0] = b1, b[:, 1] = b2, b0 = 1 - b1 - b2
    b1 = b[:, 0]
    b2 = b[:, 1]
    b0 = 1.0 - b1 - b2
    x = b0 * V0[0] + b1 * V1[0] + b2 * V2[0]
    y = b0 * V0[1] + b1 * V1[1] + b2 * V2[1]
    return np.column_stack([x, y])


def cart_to_bary(p):
    """Convert cartesian to barycentric (b1, b2)."""
    x = p[:, 0]
    y = p[:, 1]
    # Solve: p = b0*V0 + b1*V1 + b2*V2, b0+b1+b2=1
    # x = b1 + 0.5*b2
    # y = (sqrt(3)/2)*b2
    b2 = y / (np.sqrt(3) / 2)
    b1 = x - 0.5 * b2
    return np.column_stack([b1, b2])


def min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                  (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def valid_bary(b):
    """Check if barycentric coordinates are valid (inside or on triangle)."""
    b1 = b[:, 0]
    b2 = b[:, 1]
    b0 = 1.0 - b1 - b2
    eps = 1e-9
    return np.all(b0 >= -eps) and np.all(b1 >= -eps) and np.all(b2 >= -eps)


def project_to_triangle(b):
    """Project barycentric coordinates back into the triangle if they go outside."""
    b1 = b[:, 0].copy()
    b2 = b[:, 1].copy()
    b0 = 1.0 - b1 - b2
    # Clip to keep inside triangle
    for i in range(len(b1)):
        if b0[i] < 0:
            # Move toward the edge opposite to V0
            scale = -b0[i] / (b1[i] + b2[i]) if (b1[i] + b2[i]) > 1e-15 else 0
            b1[i] *= (1 + scale)
            b2[i] *= (1 + scale)
        elif b1[i] < 0:
            scale = -b1[i] / (b0[i] + b2[i]) if (b0[i] + b2[i]) > 1e-15 else 0
            b0[i] *= (1 + scale)
            b2[i] *= (1 + scale)
        elif b2[i] < 0:
            scale = -b2[i] / (b0[i] + b1[i]) if (b0[i] + b1[i]) > 1e-15 else 0
            b0[i] *= (1 + scale)
            b1[i] *= (1 + scale)
    # Normalize to ensure b0+b1+b2=1
    s = (1.0 - b1) - b2  # this is b0
    # Actually just clip and renormalize
    b1 = np.clip(b1, 0, 1)
    b2 = np.clip(b2, 0, 1)
    b0 = 1.0 - b1 - b2
    if b0 < 0:
        # Scale b1, b2 down
        total = b1 + b2
        if total > 1e-15:
            b1 *= -b0 / total
            b2 *= -b0 / total
        else:
            b1 = 0.5
            b2 = 0.5
    return np.column_stack([b1, b2])


def local_search(b, max_iter=5000, base_step=0.05, decay=0.999, min_step=1e-6):
    """Iterative local search in barycentric coordinates."""
    n = len(b)
    step = base_step
    current_b = b.copy()
    pts = bary_to_cart(current_b)
    current_area = min_triangle_area(pts)
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_b = current_b.copy()
                    new_b[i, 0] += dx
                    new_b[i, 1] += dy
                    new_b = project_to_triangle(new_b)
                    new_pts = bary_to_cart(new_b)
                    new_area = min_triangle_area(new_pts)
                    if new_area > current_area + 1e-12:
                        current_b = new_b
                        pts = new_pts
                        current_area = new_area
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            step *= decay
            if step < min_step:
                break
    return current_b, current_area


def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    best_b = None
    best_area = 0.0
    
    # Multiple random restarts
    n_restarts = 50
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization in barycentric coordinates
        b = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            b[i, 0] = u
            b[i, 1] = v
        
        pts = bary_to_cart(b)
        area = min_triangle_area(pts)
        
        if area > best_area:
            best_area = area
            best_b = b.copy()
        
        # Local search
        if time.time() > deadline - 10:
            break
        b, area = local_search(b, max_iter=2000, base_step=0.08, decay=0.998, min_step=1e-5)
        
        if area > best_area:
            best_area = area
            best_b = b.copy()
    
    # Final polish: very fine local search
    if best_b is not None and time.time() < deadline - 2:
        best_b, best_area = local_search(best_b, max_iter=3000, base_step=0.01, decay=0.999, min_step=1e-7)
    
    pts = bary_to_cart(best_b)
    return pts


# Keep the original function name and interface
def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    best_b = None
    best_area = 0.0
    
    # Multiple random restarts
    n_restarts = 80
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization in barycentric coordinates
        b = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            b[i, 0] = u
            b[i, 1] = v
        
        pts = bary_to_cart(b)
        area = min_triangle_area(pts)
        
        if area > best_area:
            best_area = area
            best_b = b.copy()
        
        # Local search
        if time.time() > deadline - 10:
            break
        b, area = local_search(b, max_iter=1500, base_step=0.08, decay=0.998, min_step=1e-5)
        
        if area > best_area:
            best_area = area
            best_b = b.copy()
    
    # Final polish: very fine local search
    if best_b is not None and time.time() < deadline - 2:
        best_b, best_area = local_search(best_b, max_iter=2000, base_step=0.01, decay=0.999, min_step=1e-7)
    
    pts = bary_to_cart(best_b)
    return pts
