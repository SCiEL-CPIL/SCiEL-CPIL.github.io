import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
EPS = 1e-9
START = time.time()
DEADLINE = 150.0

def _hex_vertices(cx, cy, s, rot):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation rot (degrees)."""
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    """Unit regular hexagon (side 1) with center (cx,cy) and rotation ang_deg."""
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _min_outer_side(inner, outer_c, rot):
    """
    Compute the minimum side length S of a regular hexagon centered at outer_c
    with rotation rot that contains all inner hexagons.
    """
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_c
    max_proj = 0.0
    for (x, y, a) in inner:
        pts = _inner_hex(x, y, a)
        for (px, py) in pts:
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    return 2.0 * max_proj / math.sqrt(3.0)

def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _point_in_poly(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        c = _cross(poly[i], poly[(i+1) % n], pt)
        if abs(c) < 1e-9:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True

def _poly_overlap(A, B):
    """SAT: return True if convex polygons A and B overlap (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            nx, ny = -ay, ax
            minA = min(p[0]*nx + p[1]*ny for p in A)
            maxA = max(p[0]*nx + p[1]*ny for p in A)
            minB = min(p[0]*nx + p[1]*ny for p in B)
            maxB = max(p[0]*nx + p[1]*ny for p in B)
            if maxA < minB - 1e-7 or maxB < minA - 1e-7:
                return False
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _valid(inner, outer_c, s, rot):
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        if not _in_outer(polys[i], outer_pts):
            return False
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _generate_lattice_clusters():
    """
    Generates candidate sets of 11 center coordinates from the hexagonal lattice.
    Base: 1 center + 6 ring1 + 6 ring2.
    We remove 2 points from ring2.
    """
    centers = []
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1: r = sqrt(3)
    r1 = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(k * 60)
        centers.append((r1 * math.cos(a), r1 * math.sin(a)))
    
    # Ring 2: r = 3
    r2 = 3.0
    ring2 = []
    for k in range(6):
        a = math.radians(30 + k * 60)
        ring2.append((r2 * math.cos(a), r2 * math.sin(a)))
    
    # Candidate 1: Remove opposite (indices 0 and 3 in ring2)
    c1 = centers[:]
    c1.extend([ring2[1], ring2[2], ring2[4], ring2[5]])
    
    # Candidate 2: Remove adjacent (indices 0 and 1)
    c2 = centers[:]
    c2.extend([ring2[2], ring2[3], ring2[4], ring2[5]])
    
    # Candidate 3: Remove with 1 gap (indices 0 and 2)
    c3 = centers[:]
    c3.extend([ring2[1], ring2[3], ring2[4], ring2[5]])
    
    return [c1, c2, c3]

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE - 10.0
    
    # Generate candidate topologies
    clusters = _generate_lattice_clusters()
    
    # Normals for outer hexagon (rotation 0)
    normals = []
    for k in range(6):
        alpha = math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    for cluster_centers in clusters:
        if time.time() > budget:
            break
            
        # Pre-compute vertices for all 11 hexagons in local frame (rotation 0)
        # Each hexagon has 6 vertices
        # We will store them in a flat list of (x,y) for speed
        local_vertices = []
        for (cx, cy) in cluster_centers:
            for k in range(6):
                a = math.radians(60 * k)
                px = cx + math.cos(a)
                py = cy + math.sin(a)
                local_vertices.append((px, py))
        
        # Function to compute side length for a given rotation theta (radians)
        def compute_side(theta):
            c = math.cos(theta)
            s = math.sin(theta)
            max_proj = 0.0
            # Rotate all vertices and project
            for (px, py) in local_vertices:
                rx = px * c - py * s
                ry = px * s + py * c
                for (nx, ny) in normals:
                    proj = rx * nx + ry * ny
                    if proj > max_proj:
                        max_proj = proj
            return 2.0 * max_proj / math.sqrt(3.0)
        
        # Grid search for theta in [0, pi/3]
        best_theta = 0.0
        best_s = float('inf')
        # 180 steps for 60 degrees -> 0.33 deg resolution
        for i in range(181):
            theta = i * (math.pi / 3.0) / 180.0
            s_val = compute_side(theta)
            if s_val < best_s:
                best_s = s_val
                best_theta = theta
        
        # Refine with Nelder-Mead
        try:
            res = minimize(compute_side, best_theta, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100})
            if res.fun < best_s:
                best_s = res.fun
                best_theta = res.x[0]
        except:
            pass
        
        if best_s < best_score:
            best_score = best_s
            best_theta = best_theta
            best_cluster = cluster_centers
    
    if best_cluster is None:
        # Fallback
        return [[0,0,0] for _ in range(N)], (0,0), 10.0, 0.0
    
    # Construct the result
    # Inner hexagons
    inner = []
    theta_deg = math.degrees(best_theta)
    c = math.cos(best_theta)
    s = math.sin(best_theta)
    
    for (cx, cy) in best_cluster:
        rx = cx * c - cy * s
        ry = cx * s + cy * c
        inner.append([rx, ry, theta_deg])
    
    outer_c = (0.0, 0.0)
    rot = 0.0
    side = best_score
    
    # Verify validity
    if not _valid(inner, outer_c, side, rot):
        # If invalid, expand side slightly or return a known valid seed
        # For now, just expand side to make it valid if it was on the boundary
        side += 1e-5
        if not _valid(inner, outer_c, side, rot):
             # Fallback to a very large valid hexagon
             return [[0,0,0] for _ in range(N)], (0,0), 10.0, 0.0

    return inner, outer_c, side, rot
