import numpy as np
import time

N = 12
DEADLINE = 150.0
t0 = time.time()

def hex_vertices(center, angle_deg=0.0):
    # returns 6 vertices of a unit regular hexagon (side 1) centered at center, rotated by angle_deg
    a = np.deg2rad(angle_deg)
    # base vertices at angles 0,60,...,300 (vertex at 0)
    base = np.array([[np.cos(np.pi*k/3), np.sin(np.pi*k/3)] for k in range(6)])
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s],[s, c]])
    return center + base @ R.T  # (6,2)

def lattice_point(i, j):
    e1 = np.array([1.5, np.sqrt(3)/2])
    e2 = np.array([0.0, np.sqrt(3)])
    return i*e1 + j*e2

def construct():
    # Generate 19 centers for a side-3 hexagon cluster (lattice side 3)
    # Lattice coords (i,j) such that |i|<=2, |j|<=2, |i-j|<=2 ? No.
    # Standard hexagon of side k (in lattice units) contains points (i,j)
    # where max(|i|, |j|, |i-j|) < k?
    # For k=3 (19 points), we want points with lattice distance <= 2 from center.
    # Lattice distance in axial coords (i,j) is max(|i|, |j|, |i-j|)? No, it's (|i| + |j| + |i-j|)/2.
    # Let's just list them manually or generate.
    
    centers = []
    # Ring 0
    centers.append(lattice_point(0,0))
    # Ring 1 (6 points)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0: continue
            # Check if in ring 1
            dist = (abs(i) + abs(j) + abs(i-j))/2
            if dist == 1:
                centers.append(lattice_point(i,j))
    # Ring 2 (12 points)
    for i in range(-2, 3):
        for j in range(-2, 3):
            dist = (abs(i) + abs(j) + abs(i-j))/2
            if dist == 2:
                centers.append(lattice_point(i,j))
                
    # We have 1 + 6 + 12 = 19 centers.
    # We need 12. Pick the first 12.
    # To be safe and symmetric, pick center + 6 neighbors + 5 from ring 2.
    # Or just take the first 12 from the list.
    # The list order:
    # 0: (0,0)
    # 1-6: Ring 1
    # 7-18: Ring 2
    
    selected = centers[:12]
    
    # Outer hexagon
    # Side length 3.0
    # Center (0,0)
    # Angle 0
    
    inner = []
    for c in selected:
        inner.append((float(c[0]), float(c[1]), 0.0))
        
    center = (0.0, 0.0)
    side = 3.0
    angle = 0.0
    
    return inner, center, side, angle
