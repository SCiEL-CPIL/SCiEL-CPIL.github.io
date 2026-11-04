import numpy as np
import math

def construct():
    # The optimal packing for 11 unit hexagons is a subset of the honeycomb lattice
    # consisting of the center, 6 immediate neighbors, and 4 of the 6 second-ring neighbors.
    # Specifically, dropping the second-ring neighbors at 0 and 180 degrees (or any opposite pair)
    # yields a symmetric configuration with minimal bounding hexagon side length.
    
    # 1. Define Centers
    centers = []
    
    # Center hexagon
    centers.append((0.0, 0.0))
    
    # 6 neighbors at distance sqrt(3)
    r1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        centers.append((x, y))
        
    # 4 second-ring hexagons at distance 3
    # We choose angles 60, 120, 240, 300 to maintain symmetry and minimize the bounding box
    # (dropping 0 and 180)
    r2 = 3.0
    for angle_deg in [60, 120, 240, 300]:
        angle = np.deg2rad(angle_deg)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        centers.append((x, y))
        
    n_hex = len(centers)
    
    # 2. Define Orientations
    # To minimize the bounding hexagon side length, each inner hexagon should be oriented
    # such that a flat side faces outward (away from the center).
    # A regular hexagon with a flat side facing angle alpha has its vertices at alpha +/- 30, etc.
    # The "angle" in the output usually defines the orientation of the hexagon.
    # If we define angle 0 as a vertex pointing right (standard math convention for polygons),
    # then a flat side facing right (angle 0) corresponds to a rotation of 30 degrees.
    # Generally, for a hexagon at center angle theta_c, we want a flat side facing theta_c.
    # So the rotation angle should be theta_c + 30 degrees.
    
    inner = []
    for i in range(n_hex):
        cx, cy = centers[i]
        if cx == 0.0 and cy == 0.0:
            # Center hexagon orientation doesn't affect the bounding box much, 
            # but let's align it with the neighbors for aesthetics/consistency.
            # Neighbors are at 0, 60, etc.
            # Let's just pick 0 or 30. 30 is consistent with "flat side outward" for the ring.
            angle_deg = 30.0 
        else:
            theta_c = np.arctan2(cy, cx)
            # Rotate so flat side faces outward
            angle_deg = np.rad2deg(theta_c) + 30.0
            
        inner.append([cx, cy, angle_deg])
        
    # 3. Calculate the exact side length of the minimal enclosing hexagon
    # The cluster is symmetric. The critical width is in the directions of the outer hexagon normals.
    # The max projection from the center is determined by the second-ring hexagons.
    # Distance of center = 3.
    # Max projection of a unit hexagon with flat side facing outward = sqrt(3)/2.
    # Total max projection R = 3 + sqrt(3)/2.
    # The side length S of a regular hexagon with apothem (distance to side) h is S = 2h/sqrt(3).
    # Here h = R.
    # S = 2 * (3 + sqrt(3)/2) / sqrt(3) = (6 + sqrt(3)) / sqrt(3) = 2*sqrt(3) + 1.
    
    side = 2 * np.sqrt(3) + 1
    
    # Add a tiny margin to account for floating point precision in the evaluator
    side += 1e-9
    
    # The outer hexagon center is at (0,0) and rotation 0 (or 30, doesn't matter for side length)
    # The interface asks for: inner, center, side, angle
    # angle is the rotation of the outer hexagon. 0 is fine.
    
    return inner, (0.0, 0.0), side, 0.0

if __name__ == "__main__":
    construct()
