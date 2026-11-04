"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(angle_deg=0.0):
    """Return 6 vertices of a unit regular hexagon rotated by angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append([math.cos(theta), math.sin(theta)])
    return np.array(verts)

def _cluster_vertices(centers, angles):
    """Get all vertices of all hexagons in the cluster."""
    verts_list = []
    for i in range(len(centers)):
        v = _hex_vertices(angles[i]) + centers[i]
        verts_list.append(v)
    return np.vstack(verts_list)

def _min_side_length(vertices, outer_angle_deg):
    """
    Calculate the minimum side length of a regular hexagon with the given rotation
    that contains the given vertices.
    The outer hexagon is centered at the origin.
    """
    a = math.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        normals.append([math.cos(theta), math.sin(theta)])
    normals = np.array(normals) # (6, 2)
    
    # Projections of all vertices onto the 6 normals
    # vertices: (M, 2), normals: (6, 2)
    # proj: (M, 6)
    proj = vertices @ normals.T
    
    # For each normal k, the max projection must be <= S * sqrt(3) / 2
    # and min projection >= -S * sqrt(3) / 2
    # Since we can translate the cluster, we should center it.
    # But here we assume the cluster is centered at (0,0) in the input.
    # The constraint is max(proj[:, k]) <= S * sqrt(3) / 2 
    # AND -min(proj[:, k]) <= S * sqrt(3) / 2
    # So S >= max( max(proj[:, k]), -min(proj[:, k]) ) / (sqrt(3)/2)
    
    max_pos = np.max(proj, axis=0)
    min_neg = -np.min(proj, axis=0)
    required_extent = np.maximum(max_pos, min_neg)
    max_extent = np.max(required_extent)
    
    S = max_extent / (math.sqrt(3) / 2.0)
    return S

def _optimize_cluster_rotation(centers, angles):
    """
    Find the best rotation of the outer hexagon for a fixed cluster configuration.
    """
    vertices = _cluster_vertices(centers, angles)
    
    # Grid search for outer rotation
    # The symmetry of the hexagon is 60 degrees.
    # We search in [0, 60)
    best_S = float('inf')
    best_alpha = 0.0
    
    # High resolution search
    # 60 degrees = pi/3
    # Step size 0.01 degrees
    num_steps = 6000
    for i in range(num_steps):
        alpha_deg = i * (60.0 / num_steps)
        S = _min_side_length(vertices, alpha_deg)
        if S < best_S:
            best_S = S
            best_alpha = alpha_deg
            
    return best_S, best_alpha

def _generate_topologies():
    """
    Generate candidate topologies for the 11 hexagons.
    Returns a list of (centers, angles).
    """
    topologies = []
    # Use a slightly larger distance to ensure strict disjointness if needed
    d = 2 * math.sqrt(3) + 1e-6
    
    # Base 7: Center + 6 neighbors
    base_centers = [[0, 0]]
    base_angles = [0.0]
    for i in range(6):
        theta = i * math.pi / 3.0
        base_centers.append([d * math.cos(theta), d * math.sin(theta)])
        base_angles.append(0.0)
        
    # Second ring positions (12 spots)
    ring2_centers = []
    ring2_angles = []
    for i in range(12):
        theta = i * math.pi / 6.0
        ring2_centers.append([2*d * math.cos(theta), 2*d * math.sin(theta)])
        ring2_angles.append(0.0)
        
    # Topology 1: 0, 3, 6, 9 (Symmetric, 90 deg apart)
    idxs1 = [0, 3, 6, 9]
    c1 = base_centers + [ring2_centers[i] for i in idxs1]
    a1 = base_angles + [ring2_angles[i] for i in idxs1]
    topologies.append((np.array(c1), np.array(a1)))
    
    # Topology 2: 0, 2, 4, 6 (Symmetric, 60 deg apart, touching chain)
    idxs2 = [0, 2, 4, 6]
    c2 = base_centers + [ring2_centers[i] for i in idxs2]
    a2 = base_angles + [ring2_angles[i] for i in idxs2]
    topologies.append((np.array(c2), np.array(a2)))
    
    # Topology 3: 0, 1, 3, 4 (Clustered)
    idxs3 = [0, 1, 3, 4]
    c3 = base_centers + [ring2_centers[i] for i in idxs3]
    a3 = base_angles + [ring2_angles[i] for i in idxs3]
    topologies.append((np.array(c3), np.array(a3)))
    
    # Topology 4: 0, 1, 2, 3 (Consecutive)
    idxs4 = [0, 1, 2, 3]
    c4 = base_centers + [ring2_centers[i] for i in idxs4]
    a4 = base_angles + [ring2_angles[i] for i in idxs4]
    topologies.append((np.array(c4), np.array(a4)))
    
    # Topology 5: 1, 4, 7, 10 (Symmetric, 90 deg apart, shifted)
    idxs5 = [1, 4, 7, 10]
    c5 = base_centers + [ring2_centers[i] for i in idxs5]
    a5 = base_angles + [ring2_angles[i] for i in idxs5]
    topologies.append((np.array(c5), np.array(a5)))
    
    return topologies

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    topologies = _generate_topologies()
    
    best_S = float('inf')
    best_centers = None
    best_angles = None
    best_alpha = 0.0
    
    for c, a in topologies:
        if time.time() > deadline:
            break
            
        S, alpha = _optimize_cluster_rotation(c, a)
        
        if S < best_S:
            best_S = S
            best_centers = c
            best_angles = a
            best_alpha = alpha
            
    # Fallback if something went wrong
    if best_centers is None:
        # Use the first topology
        c, a = topologies[0]
        S, alpha = _optimize_cluster_rotation(c, a)
        best_S = S
        best_centers = c
        best_angles = a
        best_alpha = alpha
        
    # Return the configuration
    # The outer hexagon is rotated by best_alpha.
    # The inner hexagons are at best_centers with best_angles.
    # We need to return (inner, center, side, angle)
    # inner: list of [x, y, angle_deg]
    # center: [x, y] of outer hexagon (0,0)
    # side: best_S
    # angle: best_alpha
    
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]), float(best_angles[i])])
        
    return inner, (0.0, 0.0), float(best_S), float(best_alpha)

# EVOLVE-BLOCK-END
