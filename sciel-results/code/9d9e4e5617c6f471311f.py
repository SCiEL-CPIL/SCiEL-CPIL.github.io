import math
import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def project(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def are_disjoint_sat(v1, v2, eps=1e-9):
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        
        return False

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        max_proj = 0.0
        otheta_rad = math.radians(otheta)
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def objective_full(params):
        if time.time() > deadline - 5:
            return 1e9

        centers = []
        for i in range(12):
            centers.append([params[i * 2], params[i * 2 + 1]])
        angles = [params[24 + i] for i in range(12)]
        ocx, ocy = params[36], params[37]
        otheta = params[38]
        
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        # Soft penalty for overlaps based on center distance
        # Minimum distance between centers of two unit hexagons is sqrt(3)
        min_dist = math.sqrt(3)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return side + 10.0 * penalty

    def is_valid(centers, angles):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    return False
        return True

    def generate_hexagonal_cluster():
        """
        Generates a 12-hexagon cluster based on a hexagonal lattice.
        Returns list of (x, y) centers and list of angles.
        """
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            
        ring2_candidates = []
        basis = [u, v, v - u, -u, -v, u - v]
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
                        
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
            
        angles = [30.0] * 12
        return centers, angles

    def spectral_embedding_optimize():
        """
        Uses spectral graph embedding to find a good initial configuration.
        We define a graph topology and optimize edge weights.
        """
        # Topology 1: Hexagonal Lattice Connectivity (1 center, 6 ring1, 5 ring2)
        # Nodes: 0 (center), 1-6 (ring1), 7-11 (ring2)
        # Edges:
        # 0 connected to 1-6
        # 1-6 connected to each other in a ring
        # 7-11 connected to specific ring1 nodes and each other
        
        # Let's use a simpler, more symmetric graph for spectral embedding
        # A 12-node graph that resembles the hexagonal packing.
        
        # Adjacency matrix indices
        # 0: center
        # 1-6: ring 1
        # 7-11: ring 2 (5 nodes)
        
        # Define potential edges
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
            # Ring 2 connections (approximate)
            (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
            (7, 8), (8, 9), (9, 10), (10, 11), (11, 7)
        ]
        
        num_edges = len(edges)
        
        def obj_spectral(weights):
            if time.time() > deadline - 10:
                return 1e9
            
            # Build Laplacian
            L = np.zeros((12, 12))
            for i, (u, v) in enumerate(edges):
                w = weights[i]
                L[u, u] += w
                L[v, v] += w
                L[u, v] -= w
                L[v, u] -= w
            
            # Get eigenvectors
            # We want the 2D embedding. 
            # The smallest eigenvalue is 0 (constant vector).
            # The next two give the 2D coordinates.
            try:
                # Use dense eig for small matrix
                vals, vecs = np.linalg.eigh(L)
                # Sort by eigenvalue
                idx = np.argsort(vals)
                vecs = vecs[:, idx]
                
                # Take eigenvectors for indices 1 and 2
                v1 = vecs[:, 1]
                v2 = vecs[:, 2]
                
                # Center the layout
                v1 -= np.mean(v1)
                v2 -= np.mean(v2)
                
                # Scale to reasonable size
                # The scale factor is part of the optimization? 
                # No, spectral embedding gives relative positions.
                # We need to scale it so that adjacent hexagons are roughly sqrt(3) apart.
                # Let's calculate the average distance between connected nodes in the embedding
                dists = []
                for u, v in edges:
                    dx = v1[u] - v1[v]
                    dy = v2[u] - v2[v]
                    dists.append(math.sqrt(dx*dx + dy*dy))
                
                if not dists or np.mean(dists) == 0:
                    return 1e9
                    
                avg_dist = np.mean(dists)
                target_dist = math.sqrt(3)
                scale = target_dist / avg_dist
                
                centers = []
                for i in range(12):
                    cx = v1[i] * scale
                    cy = v2[i] * scale
                    centers.append([cx, cy])
                    
                # Optimize outer container orientation and position for this fixed layout
                # We can do a quick grid search or Nelder-Mead on outer params
                best_side = float('inf')
                best_otheta = 0
                best_ocx = 0
                best_ocy = 0
                
                # Try a few rotations
                for otheta in np.linspace(0, 30, 7):
                    # Compute side length assuming center at 0,0
                    side = compute_outer_side(centers, [0.0]*12, 0.0, 0.0, otheta)
                    if side < best_side:
                        best_side = side
                        best_otheta = otheta
                        
                # Add penalty for overlaps
                penalty = 0.0
                min_dist = math.sqrt(3)
                for i in range(12):
                    for j in range(i + 1, 12):
                        dx = centers[i][0] - centers[j][0]
                        dy = centers[i][1] - centers[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < min_dist:
                            penalty += (min_dist - dist) ** 2
                            
                return best_side + 10.0 * penalty

            except Exception:
                return 1e9

        # Initial weights
        init_weights = [1.0] * num_edges
        
        # Optimize weights
        res = minimize(obj_spectral, init_weights, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < 1e8:
            # Reconstruct centers from best weights
            weights = res.x
            L = np.zeros((12, 12))
            for i, (u, v) in enumerate(edges):
                w = weights[i]
                L[u, u] += w
                L[v, v] += w
                L[u, v] -= w
                L[v, u] -= w
            
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            vecs = vecs[:, idx]
            
            v1 = vecs[:, 1]
            v2 = vecs[:, 2]
            
            v1 -= np.mean(v1)
            v2 -= np.mean(v2)
            
            dists = []
            for u, v in edges:
                dx = v1[u] - v1[v]
                dy = v2[u] - v2[v]
                dists.append(math.sqrt(dx*dx + dy*dy))
            
            avg_dist = np.mean(dists)
            if avg_dist > 0:
                scale = math.sqrt(3) / avg_dist
                centers = []
                for i in range(12):
                    cx = v1[i] * scale
                    cy = v2[i] * scale
                    centers.append([cx, cy])
                return centers
        return None

    # 1. Generate base configurations
    configs = []
    
    # Config 1: Standard Hex Lattice
    c1, a1 = generate_hexagonal_cluster()
    configs.append((c1, a1))
    
    # Config 2: Spectral Embedding
    random.seed(42)
    c_spectral = spectral_embedding_optimize()
    if c_spectral:
        a_spectral = [30.0] * 12
        configs.append((c_spectral, a_spectral))
        
    # Config 3: Random Start
    c3 = [[random.uniform(-3, 3), random.uniform(-3, 3)] for _ in range(12)]
    a3 = [random.uniform(0, 60) for _ in range(12)]
    configs.append((c3, a3))

    best_global_result = None
    best_global_val = float('inf')

    # Optimize each config
    for centers, angles in configs:
        if time.time() > deadline - 40:
            break
            
        # Convert to param vector
        params = []
        for cx, cy in centers:
            params.append(cx)
            params.append(cy)
        for a in angles:
            params.append(a)
        params.extend([0.0, 0.0, 0.0]) # ocx, ocy, otheta
        
        # Local optimization
        res = minimize(objective_full, params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6})
        
        if res.fun < best_global_val:
            best_global_val = res.fun
            best_global_result = res.x

    if best_global_result is None:
        # Fallback
        c1, a1 = generate_hexagonal_cluster()
        best_global_result = []
        for cx, cy in c1:
            best_global_result.append(cx)
            best_global_result.append(cy)
        for a in a1:
            best_global_result.append(a)
        best_global_result.extend([0.0, 0.0, 0.0])
        best_global_val = objective_full(best_global_result)

    # Extract best parameters
    centers = []
    for i in range(12):
        centers.append([best_global_result[i * 2], best_global_result[i * 2 + 1]])
    angles = [best_global_result[24 + i] for i in range(12)]
    ocx, ocy = best_global_result[36], best_global_result[37]
    otheta = best_global_result[38]
    
    # Final Polish: Grid search on outer parameters only (fast)
    best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
    
    # Refined grid search
    for da in np.linspace(-5, 5, 11):
        for dx in np.linspace(-0.2, 0.2, 5):
            for dy in np.linspace(-0.2, 0.2, 5):
                s = compute_outer_side(centers, angles, ocx + dx, ocy + dy, otheta + da)
                if s < best_side:
                    best_side = s
                    best_ocx, best_ocy, best_otheta = ocx + dx, ocy + dy, otheta + da
                    
    # Verify final validity
    if not is_valid(centers, angles):
        # Fallback to initial lattice if optimization resulted in invalid configuration
        c1, a1 = generate_hexagonal_cluster()
        centers = [[c[0], c[1]] for c in c1]
        angles = a1
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
