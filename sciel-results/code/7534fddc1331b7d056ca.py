# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    # Define the 6 normals for the hexagon edges (outward pointing)
    # Vertices are at angles 0, 60, 120...
    # Edges are perpendicular to angles 30, 90, 150...
    angles = np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Vertices of a unit hexagon centered at origin, angle 0
    hex_angles = np.deg2rad(60 * np.arange(6))
    unit_vertices = np.column_stack((np.cos(hex_angles), np.sin(hex_angles)))
    
    def get_vertices(centers):
        # centers: (12, 2)
        # Returns (12, 6, 2)
        return centers[:, np.newaxis, :] + unit_vertices[np.newaxis, :, :]

    def compute_M(centers):
        # Compute max projection of any vertex onto any normal
        verts = get_vertices(centers) # (12, 6, 2)
        # Project onto normals: (12, 6, 6)
        # dots[i, j, k] = dot(verts[i, j], normals[k])
        dots = np.einsum('ijk,kj->ij', verts, normals.T) 
        # Wait, normals.T is (2, 6).
        # verts is (12, 6, 2).
        # einsum 'ijk,kj->ij' sums over k (dim 2 of verts) and j (dim 1 of normals.T which is dim 2 of normals).
        # This results in (12, 6) where the second index is the normal index?
        # Let's verify.
        # verts[i, j, k] * normals.T[k, m] -> sum k. Result [i, j, m]?
        # No, 'ijk,kj->ij' implies the output has indices i, j.
        # But normals.T has indices (2, 6). Let's call them (k, m).
        # So 'ijk,km->ijm' would be (12, 6, 6).
        # I want max over all i, j, m.
        dots = np.einsum('ijk,km->ijm', verts, normals)
        return np.max(dots)

    def compute_overlap_penalty(centers, weight=100.0):
        verts = get_vertices(centers) # (12, 6, 2)
        n_hex = 12
        total_penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                vi = verts[i] # (6, 2)
                vj = verts[j] # (6, 2)
                
                pi = vi @ normals.T # (6, 6)
                pj = vj @ normals.T # (6, 6)
                
                min_i = np.min(pi, axis=0) # (6,)
                max_i = np.max(pi, axis=0) # (6,)
                min_j = np.min(pj, axis=0) # (6,)
                max_j = np.max(pj, axis=0) # (6,)
                
                gap = np.maximum(min_i, min_j) - np.minimum(max_i, max_j)
                max_gap = np.max(gap)
                if max_gap < 0:
                    overlap = -max_gap
                    total_penalty += weight * (overlap ** 2)
        
        return total_penalty

    def objective(x):
        centers = x.reshape(12, 2)
        M = compute_M(centers)
        pen = compute_overlap_penalty(centers)
        return M + pen

    best_s = float('inf')
    best_centers = None
    
    # Run multiple optimizations from different starting points
    for seed in range(20):
        np.random.seed(seed)
        # Initialize with a lattice-like structure to help convergence
        # 4x3 grid roughly
        x0 = np.zeros(24)
        for i in range(12):
            row = i // 4
            col = i % 4
            x0[i*2] = (col - 1.5) * 1.8
            x0[i*2+1] = (row - 1.5) * 1.6
            # Add some noise
            x0[i*2] += np.random.uniform(-0.5, 0.5)
            x0[i*2+1] += np.random.uniform(-0.5, 0.5)
            
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-10})
        
        if res.success:
            centers = res.x.reshape(12, 2)
            # Verify validity
            verts = get_vertices(centers)
            # Check containment
            dots = np.einsum('ijk,km->ijm', verts, normals)
            M = np.max(dots)
            s_candidate = (2.0 / np.sqrt(3.0)) * M
            
            # Check overlap
            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    vi = verts[i]
                    vj = verts[j]
                    pi = vi @ normals.T
                    pj = vj @ normals.T
                    min_i = np.min(pi, axis=0)
                    max_i = np.max(pi, axis=0)
                    min_j = np.min(pj, axis=0)
                    max_j = np.max(pj, axis=0)
                    gap = np.maximum(min_i, min_j) - np.minimum(max_i, max_j)
                    if np.max(gap) < -1e-6: # Overlap
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and s_candidate < best_s:
                best_s = s_candidate
                best_centers = centers

    if best_centers is None:
        # Fallback to a known valid configuration (from prompt, scaled down if possible, but prompt's is valid)
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        s = 8
        return inner_hex_data, outer_hex_data, s

    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = best_centers[:, 0]
    inner_hex_data[:, 1] = best_centers[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
