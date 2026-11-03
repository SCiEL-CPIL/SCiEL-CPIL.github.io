import math
import time
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix

N = 12
SQRT3 = math.sqrt(3)
DEADLINE = 140.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def time_left():
        return deadline - time.time()

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts_fast(cx, cy, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return unit_verts_0 @ rot.T + np.array([cx, cy])

    def calc_outer_side(params, outer_center=(0.0, 0.0), outer_angle=0.0):
        """
        params: [x0..x11, y0..y11, a0..a11] (36 values)
        Returns min side length of outer hexagon containing all inner hexagons.
        """
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:]
        
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides
        normals = []
        for k in range(6):
            ang_n = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals)
        
        max_proj = 0.0
        oc = np.array(outer_center)
        
        for i in range(N):
            cx, cy = pos[i]
            a_rad = ang_rad[i]
            verts = get_hex_verts_fast(cx, cy, a_rad)
            verts_rel = verts - oc
            projs = verts_rel @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        side = max_proj * 2.0 / SQRT3
        return side

    def sat_overlap_depth(poly1, poly2):
        """
        Returns the minimum penetration depth if poly1 and poly2 overlap.
        Returns 0.0 if they are disjoint or just touching.
        """
        min_overlap = 0.0
        axes = []
        
        # Axes from poly1 edges
        for k in range(6):
            edge = poly1[(k+1)%6] - poly1[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n/norm)
        
        # Axes from poly2 edges
        for k in range(6):
            edge = poly2[(k+1)%6] - poly2[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n/norm)
                
        for axis in axes:
            proj1 = np.dot(poly1, axis)
            proj2 = np.dot(poly2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                return 0.0
            if overlap > min_overlap:
                min_overlap = overlap
                
        return min_overlap

    def calc_penalty(params):
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:]
        
        polys = []
        for i in range(N):
            cx, cy = pos[i]
            verts = get_hex_verts_fast(cx, cy, ang_rad[i])
            polys.append(verts)
            
        penalty = 0.0
        for i in range(N):
            p1 = polys[i]
            for j in range(i+1, N):
                p2 = polys[j]
                
                # Quick bounding box check
                min_x1, min_y1 = np.min(p1, axis=0)
                max_x1, max_y1 = np.max(p1, axis=0)
                min_x2, min_y2 = np.min(p2, axis=0)
                max_x2, max_y2 = np.max(p2, axis=0)
                
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue
                
                depth = sat_overlap_depth(p1, p2)
                if depth > 0:
                    penalty += depth * depth * 10.0
                    
        return penalty

    def objective(params):
        side = calc_outer_side(params)
        penalty = calc_penalty(params)
        return side + penalty

    # --- SPECTRAL GRAPH EMBEDDING APPROACH ---
    # Define a base graph structure for 12 hexagons
    # We'll use a honeycomb-like adjacency pattern
    
    def build_graph(weights):
        """
        Build a weighted adjacency matrix from edge weights.
        weights: list of (i, j, w) tuples
        Returns sparse Laplacian
        """
        A = np.zeros((N, N))
        for i, j, w in weights:
            A[i, j] = w
            A[j, i] = w
        # Laplacian L = D - A
        D = np.diag(A.sum(axis=1))
        L = D - A
        return L

    def spectral_embedding(L, scale=1.0):
        """
        Get positions from eigenvectors of Laplacian.
        Returns (N, 2) array of positions.
        """
        eigvals, eigvecs = np.linalg.eigh(L)
        # Use the 2nd and 3rd smallest eigenvectors (skip trivial first)
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        v1 = eigvecs[:, idx[1]]
        v2 = eigvecs[:, idx[2]]
        pos = np.column_stack([v1, v2])
        pos *= scale
        pos -= np.mean(pos, axis=0)
        return pos

    def generate_spectral_init(edge_weights, scale=1.0):
        """
        Generate initial configuration using spectral embedding.
        edge_weights: list of (i, j, w)
        """
        L = build_graph(edge_weights)
        pos = spectral_embedding(L, scale)
        ang = np.zeros(N)
        return np.concatenate([pos.flatten(), ang])

    # Define several graph structures for 12 nodes
    # 1. Honeycomb lattice pattern (hexagonal tiling subset)
    honeycomb_edges = [
        (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0),
        (3, 4, 1.0), (4, 5, 1.0), (5, 0, 1.0),
        (6, 7, 1.0), (7, 8, 1.0), (8, 9, 1.0),
        (9, 10, 1.0), (10, 11, 1.0), (11, 6, 1.0),
        (0, 6, 1.0), (1, 7, 1.0), (2, 8, 1.0),
        (3, 9, 1.0), (4, 10, 1.0), (5, 11, 1.0)
    ]
    
    # 2. Ring with spokes
    ring_edges = []
    for i in range(N):
        ring_edges.append((i, (i+1)%N, 1.0))
    ring_edges.append((0, 6, 0.5))
    ring_edges.append((3, 9, 0.5))
    
    # 3. Two triangles connected
    tri1_edges = [(0,1,1.0), (1,2,1.0), (2,0,1.0)]
    tri2_edges = [(3,4,1.0), (4,5,1.0), (5,3,1.0)]
    tri3_edges = [(6,7,1.0), (7,8,1.0), (8,6,1.0)]
    tri4_edges = [(9,10,1.0), (10,11,1.0), (11,9,1.0)]
    connect_edges = [(0,3,0.8), (1,4,0.8), (2,5,0.8), (6,9,0.8), (7,10,0.8), (8,11,0.8)]
    cluster_edges = tri1_edges + tri2_edges + tri3_edges + tri4_edges + connect_edges
    
    # 4. Star topology
    star_edges = [(i, 0, 0.5) for i in range(1, N)]
    star_edges += [(i, (i+1)%N, 0.3) for i in range(1, N)]
    
    # 5. Lattice 3x4 grid
    grid_edges = []
    for i in range(3):
        for j in range(4):
            idx = i*4 + j
            if j < 3:
                grid_edges.append((idx, idx+1, 1.0))
            if i < 2:
                grid_edges.append((idx, idx+4, 1.0))
    
    # 6. Complete graph with random weights
    np.random.seed(123)
    random_edges = []
    for i in range(N):
        for j in range(i+1, N):
            if np.random.rand() < 0.4:
                random_edges.append((i, j, np.random.uniform(0.3, 1.0)))
    
    # Generate spectral initializations
    inits = []
    for edges in [honeycomb_edges, ring_edges, cluster_edges, star_edges, grid_edges, random_edges]:
        for scale in [0.8, 1.0, 1.2, 1.5]:
            if time.time() > deadline - 30:
                break
            init = generate_spectral_init(edges, scale)
            inits.append(init)
    
    # Also add some random starts for diversity
    np.random.seed(42)
    for _ in range(5):
        pos = np.random.uniform(-2, 2, (N, 2))
        ang = np.random.uniform(0, 2*math.pi, N)
        inits.append(np.concatenate([pos.flatten(), ang]))

    best_side = float('inf')
    best_params = None
    best_outer_center = [0.0, 0.0]
    best_outer_angle = 0.0

    # --- OPTIMIZE EACH SPECTRAL INITIALIZATION ---
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        # Use Nelder-Mead to optimize from this starting point
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            
            if res.fun < best_side:
                # Verify penalty is small
                pen = calc_penalty(res.x)
                if pen < 1e-3:
                    best_side = calc_outer_side(res.x)
                    best_params = res.x.copy()
        except:
            pass
    
    # --- SECOND PASS: Refine with tighter tolerance ---
    if best_params is not None:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if res.fun < best_side:
                pen = calc_penalty(res.x)
                if pen < 1e-4:
                    best_side = calc_outer_side(res.x)
                    best_params = res.x.copy()
        except:
            pass

    # --- OPTIMIZE OUTER CENTER AND ANGLE ---
    if best_params is not None:
        pos = best_params[:2*N].reshape(N, 2)
        
        def outer_obj(outer_params):
            oc = outer_params[:2]
            oa = outer_params[2]
            side = calc_outer_side(best_params, outer_center=oc, outer_angle=np.degrees(oa))
            return side

        centroid = np.mean(pos, axis=0)
        x0_outer = np.array([centroid[0], centroid[1], 0.0])
        
        try:
            res_outer = minimize(
                outer_obj, 
                x0_outer, 
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res_outer.fun < best_side:
                best_side = res_outer.fun
                best_outer_center = res_outer.x[:2]
                best_outer_angle = np.degrees(res_outer.x[2])
        except:
            pass

    # Fallback
    if best_params is None:
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        ang = np.zeros(N)
        best_params = np.concatenate([pos.flatten(), ang])
        best_side = calc_outer_side(best_params)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0

    # Construct return values
    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], np.degrees(final_ang[i])])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle
