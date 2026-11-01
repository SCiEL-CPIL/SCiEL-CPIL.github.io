"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, cKDTree
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def get_covering_radius(pts):
        """
        Computes the covering radius via convex hull facets.
        Returns (radius_in_radians, worst_circumcenter, worst_simplex_indices).
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            # Fallback: return a large radius
            return np.pi, np.random.randn(3) / np.linalg.norm(np.random.randn(3)), None

        max_radius = 0.0
        worst_center = None
        worst_simplex = None
        
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
                
            c = n / norm_n
            # Ensure c is on the correct side (pointing outwards)
            if np.dot(c, v1) < 0:
                c = -c
                
            # Calculate angular distance from c to v1
            # cos(theta) = c . v1
            # theta = arccos(c . v1)
            dot_val = np.clip(np.dot(c, v1), -1.0, 1.0)
            radius = np.arccos(dot_val)
            
            if radius > max_radius:
                max_radius = radius
                worst_center = c
                worst_simplex = simplex.copy()
                
        if worst_center is None:
            return np.pi, np.random.randn(3) / np.linalg.norm(np.random.randn(3)), None
            
        return max_radius, worst_center, worst_simplex

    def lloyd_step(pts):
        """
        Lloyd's algorithm step for spherical Voronoi relaxation.
        Moves each point to the centroid of its Voronoi cell (approximated by 
        the centroid of the adjacent hull facets).
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return pts.copy()
            
        # Map vertex index to list of adjacent simplex indices
        vertex_to_simplex = [[] for _ in range(N)]
        for s_idx, simplex in enumerate(hull.simplices):
            for v_idx in simplex:
                vertex_to_simplex[v_idx].append(s_idx)
        
        new_pts = pts.copy()
        
        for i in range(N):
            # Collect vertices of all facets adjacent to point i
            # The centroid of the Voronoi cell on the sphere is approximately
            # the normalized sum of the facet circumcenters (or just the centroid of the neighboring vertices)
            # A more robust approximation for spherical Voronoi centroids is the 
            # normalized sum of the unit vectors to the neighbors.
            # However, using the facet normals (circumcenters) weighted by area is better.
            # Let's use the centroid of the vertices of the adjacent simplices.
            
            neighbors = set()
            for s_idx in vertex_to_simplex[i]:
                for v_idx in hull.simplices[s_idx]:
                    neighbors.add(v_idx)
            
            if not neighbors:
                continue
                
            # Average the positions of the neighbors (and the point itself for stability?)
            # Standard Lloyd: move to centroid of the cell.
            # For a triangulation, the centroid of the cell is related to the dual graph.
            # Approximation: Move towards the average of the facet circumcenters.
            
            facet_centers = []
            for s_idx in vertex_to_simplex[i]:
                simplex = hull.simplices[s_idx]
                v1 = pts[simplex[0]]
                v2 = pts[simplex[1]]
                v3 = pts[simplex[2]]
                
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                c = n / norm_n
                if np.dot(c, pts[i]) < 0:
                    c = -c
                facet_centers.append(c)
            
            if facet_centers:
                avg_center = np.mean(facet_centers, axis=0)
                norm_avg = np.linalg.norm(avg_center)
                if norm_avg > 1e-10:
                    new_pts[i] = avg_center / norm_avg
            else:
                # Fallback: move slightly towards average of neighbors
                neigh_pts = [pts[j] for j in neighbors if j != i]
                if neigh_pts:
                    avg_neigh = np.mean(neigh_pts, axis=0)
                    norm_avg = np.linalg.norm(avg_neigh)
                    if norm_avg > 1e-10:
                        new_pts[i] = avg_neigh / norm_avg
        
        # Normalize all points
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        return new_pts

    def spring_relaxation(pts, steps=100, dt=0.01):
        """
        Force-directed relaxation using repulsive forces.
        F_ij = k * (1/d_ij^2) * (r_i - r_j) / d_ij  (Coulomb-like)
        """
        current = pts.copy()
        k = 1.0
        
        for step in range(steps):
            if time.time() > deadline - 2:
                break
                
            forces = np.zeros_like(current)
            
            # Compute pairwise distances
            # dist^2[i,j]
            diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, 1.0) # Avoid div by 0
            
            dist = np.sqrt(dist_sq)
            
            # Force magnitude: 1 / dist^3 (since F = k/d^2 * dir, dir = diff/d, so F = k*diff/d^3)
            # We want to push apart. 
            # inv_dist_cubed = 1 / dist^3
            inv_dist_cubed = 1.0 / (dist_sq * dist)
            np.fill_diagonal(inv_dist_cubed, 0.0)
            
            # forces[i] = sum_j k * inv_dist_cubed[i,j] * diff[i,j]
            # diff[i,j] = current[i] - current[j]
            # So force on i is sum_j k * inv_dist_cubed[i,j] * (current[i] - current[j])
            # This is repulsive.
            
            # Multiply diff by inv_dist_cubed
            # diff shape (N, N, 3), inv_dist_cubed shape (N, N)
            forces = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
            
            # Update positions
            current += dt * forces
            current /= np.linalg.norm(current, axis=1, keepdims=True)
            
        return current

    # Strategy:
    # 1. Start with Fibonacci sphere (good initial spread).
    # 2. Interleave Lloyd's relaxation (to fix local non-uniformities) and Spring relaxation (to push apart close pairs).
    # 3. Use basin hopping with random kicks to escape local minima.
    
    best_points = None
    best_radius = np.pi
    
    # Initial population
    replicas = []
    for i in range(5):
        if i == 0:
            pts = fibonacci_sphere(N).copy()
        else:
            # Random rotation of Fibonacci
            R = np.random.randn(3, 3)
            R, _ = np.linalg.qr(R)
            pts = fibonacci_sphere(N) @ R.T
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            # Add noise
            noise = np.random.randn(N, 3) * 0.05
            pts += noise
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        replicas.append(pts)
    
    # Main Loop
    kick_mag = 0.05
    while time.time() < deadline - 5:
        for i in range(len(replicas)):
            if time.time() > deadline - 5:
                break
            
            # 1. Perturb
            noise = np.random.randn(N, 3) * kick_mag
            perturbed = replicas[i] + noise
            perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
            
            # 2. Hybrid Relaxation
            current = perturbed.copy()
            
            # Alternate between Lloyd and Spring
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                # Lloyd step
                current = lloyd_step(current)
                # Spring step
                current = spring_relaxation(current, steps=10, dt=0.005)
            
            # 3. Evaluate
            radius, _, _ = get_covering_radius(current)
            
            if radius < best_radius:
                best_radius = radius
                best_points = current.copy()
            
            # 4. Update Replica
            old_radius, _, _ = get_covering_radius(replicas[i])
            if radius < old_radius:
                replicas[i] = current
            else:
                # Keep old
                pass
            
            # Decay kick
            kick_mag *= 0.99
            if kick_mag < 0.005:
                kick_mag = 0.005

    # Final Polish on best point
    if best_points is not None and time.time() < deadline - 2:
        final_pts = best_points.copy()
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            final_pts = lloyd_step(final_pts)
            final_pts = spring_relaxation(final_pts, steps=5, dt=0.001)
        
        final_radius, _, _ = get_covering_radius(final_pts)
        if final_radius < best_radius:
            best_points = final_pts
            best_radius = final_radius

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
