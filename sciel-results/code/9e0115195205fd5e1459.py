"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -1.0
    best_result = None

    # Helper: Evaluate validity and score
    def evaluate_packing(circles):
        """
        circles: (N, 3) array [x, y, r]
        Returns (score, valid)
        Score is sum of radii.
        Valid if all circles are disjoint and fit in a rectangle of perimeter <= 4.
        """
        if len(circles) != N:
            return 0.0, False
        
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Check positive radii
        if np.any(r <= 0):
            return 0.0, False
            
        # Check disjointness
        # Vectorized distance check
        i_idx, j_idx = np.triu_indices(N, k=1)
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        r_sum = r[i_idx] + r[j_idx]
        
        if np.any(dist < r_sum - 1e-8):
            return 0.0, False
            
        # Check bounding box perimeter
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        
        w = max_x - min_x
        h = max_y - min_y
        perimeter = 2 * (w + h)
        
        if perimeter > 4.0 + 1e-8:
            return 0.0, False
            
        return np.sum(r), True

    def normalize_to_perimeter(circles):
        """
        Scales positions and radii such that the bounding box perimeter is exactly 4.
        Returns the scaled circles.
        """
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        
        w = max_x - min_x
        h = max_y - min_y
        p = 2 * (w + h)
        
        if p == 0:
            return circles
            
        scale = 4.0 / p
        
        # Shift to origin (0,0) first
        x_new = (x - min_x) * scale
        y_new = (y - min_y) * scale
        r_new = r * scale
        
        return np.column_stack((x_new, y_new, r_new))

    def solve_contact_system(positions, edges):
        """
        Given fixed positions and a set of edges (contacts), solve for radii r_i
        such that r_i + r_j = dist_ij for all (i, j) in edges.
        This is a linear system: A * r = b.
        We want to maximize the minimum radius, or just find a feasible positive solution.
        Actually, if we fix positions, the radii are determined by the contact graph if it's rigid enough.
        If under-determined, we can add a constraint to maximize min(r) or sum(r).
        
        For simplicity and robustness, we use the following:
        We define the system r_i + r_j = d_ij.
        We add a dummy variable lambda to allow for slack if the system is inconsistent, 
        or we just solve the least squares problem and check if the residual is small and r > 0.
        
        Better approach: 
        Since we start from a grid, the Delaunay edges define the topology.
        We can treat this as a linear programming problem or just a linear system.
        Let's use numpy.linalg.lstsq to find the best fit radii.
        """
        if len(edges) == 0:
            return None
            
        A = np.zeros((len(edges), N))
        b = np.zeros(len(edges))
        
        for k, (i, j) in enumerate(edges):
            A[k, i] = 1.0
            A[k, j] = 1.0
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            b[k] = np.sqrt(dx*dx + dy*dy)
            
        # Solve A r = b
        # We expect a unique solution if the graph is connected and rigid enough.
        # If not, lstsq gives the minimum norm solution.
        try:
            r, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            return None
            
        # Check if the solution is valid (positive radii and small residual)
        if np.any(r <= 1e-6):
            return None
            
        # Check residual
        res = np.sum((A @ r - b)**2)
        if res > 1e-6:
            return None
            
        return r

    def relax_positions(circles, aspect_target):
        """
        Local search to maximize sum of radii while maintaining validity.
        Uses a simple gradient-free method (Nelder-Mead) on the positions only,
        with radii re-solved from the contact graph at each step? 
        No, that's too slow.
        
        Alternative: 
        1. Fix aspect ratio.
        2. Initialize positions on a grid.
        3. Compute Delaunay edges.
        4. Solve for radii.
        5. If valid, try to move points to increase the minimum radius or sum of radii.
        
        Let's use a different strategy for the local search:
        We will optimize the positions x, y and the radii r jointly using Nelder-Mead,
        but with a strong penalty for violation of the contact constraints derived from the initial topology.
        Actually, the prompt suggests "solve for radii using a linear programming formulation... decouples combinatorial structure".
        
        Let's stick to the structural idea:
        1. Generate a candidate topology (set of edges).
        2. Solve for optimal positions and radii that satisfy r_i + r_j = d_ij for edges, and d_ij >= r_i + r_j for non-edges.
        
        This is a non-linear system for positions.
        
        Let's use a simpler, robust hybrid:
        1. Start with a grid.
        2. Run a "Lloyd's algorithm" style relaxation where we push circles apart until they touch neighbors defined by Delaunay.
        3. Then, use Nelder-Mead to fine-tune the aspect ratio and positions to maximize the minimum radius (which scales the whole packing).
        """
        
        pass # Not using this directly, see main loop

    # Main Strategy:
    # 1. Generate diverse initial topologies using grid perturbations.
    # 2. For each topology:
    #    a. Initialize positions on a grid fitting in a 1x1 square (perimeter 4).
    #    b. Compute Delaunay triangulation.
    #    c. Use the Delaunay edges as the contact graph.
    #    d. Optimize positions and aspect ratio to maximize the "minimum radius" achievable by solving the linear system for radii.
    #       Objective: Maximize min(r_i) subject to r_i + r_j <= d_ij for all i,j.
    #       This is a linear program if positions are fixed.
    #       But positions are variable.
    #       
    #    Let's simplify: 
    #    We will use Nelder-Mead to optimize the POSITIONS and ASPECT RATIO.
    #    The objective function will:
    #    1. Calculate the Delaunay neighbors of the current positions.
    #    2. Solve the linear system r_i + r_j = d_ij for the edges.
    #    3. If the solution has positive radii, return -min(r) (we want to maximize min r).
    #    4. If the solution is invalid (negative r or inconsistent), return a large penalty.
    #    
    #    This effectively searches for a configuration where the "bottleneck" radius is maximized.
    #    Since all circles scale together in a maximal packing, maximizing min(r) is equivalent to maximizing sum(r) for a fixed topology.

    def objective_structural(x_flat, aspect):
        """
        x_flat: [x1, y1, ..., xN, yN]
        aspect: width/height ratio
        """
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        positions = x_flat.reshape(N, 2)
        
        # Check bounds
        # We don't strictly enforce bounds here, but the Delaunay/LP will fail if circles are far apart?
        # No, we need to ensure circles are inside the rectangle.
        # If a circle is outside, the "packing" is invalid.
        # However, the linear system for radii doesn't know about the box.
        # We must ensure the resulting circles fit in the box.
        
        # 1. Compute Delaunay
        try:
            tri = Delaunay(positions)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        edge = (int(simplex[i]), int(simplex[j]))
                        if edge[0] > edge[1]:
                            edge = (edge[1], edge[0])
                        edges.add(edge)
        except Exception:
            return 1e6 # Penalty for degenerate geometry
            
        if len(edges) < N - 1:
            return 1e6 # Not connected
            
        # 2. Solve for radii
        A = np.zeros((len(edges), N))
        b = np.zeros(len(edges))
        
        edge_list = list(edges)
        for k, (i, j) in enumerate(edge_list):
            A[k, i] = 1.0
            A[k, j] = 1.0
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            b[k] = np.sqrt(dx*dx + dy*dy)
            
        try:
            r, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except:
            return 1e6
            
        # Check if radii are positive
        if np.any(r <= 1e-6):
            return 1e6
            
        # Check if the circles fit in the rectangle [0, w] x [0, h]
        # We need to translate the configuration to fit in the box.
        # The box is fixed size w x h.
        # We can shift the positions.
        # Let's find the minimal bounding box of the circles (x-r, x+r, y-r, y+r).
        min_x = np.min(positions[:, 0] - r)
        max_x = np.max(positions[:, 0] + r)
        min_y = np.min(positions[:, 1] - r)
        max_y = np.max(positions[:, 1] + r)
        
        w_box = max_x - min_x
        h_box = max_y - min_y
        
        # If the bounding box of the circles is larger than the rectangle, it's invalid.
        if w_box > w + 1e-6 or h_box > h + 1e-6:
            return 1e6
            
        # If it fits, we can shift it to [0, w] x [0, h].
        # The score is the sum of radii (or min radius).
        # Since the topology is fixed and the system is homogeneous, scaling the positions scales r.
        # We are optimizing positions within the fixed aspect ratio box.
        # To maximize r, we should push the circles to the boundaries.
        
        # The objective is to maximize the minimum radius.
        # If the configuration fits with room to spare, we can scale up the whole thing?
        # No, the positions are fixed in the objective.
        # We should encourage the configuration to "fill" the box.
        # Let's return -min(r) + penalty_for_not_touching_boundaries.
        
        # Actually, if we just return -min(r), the optimizer might shrink the circles to the center.
        # We need to force them to be as large as possible.
        # This is achieved by the fact that if we scale up positions, r scales up.
        # But positions are constrained by the box? 
        # My objective doesn't constrain positions to the box explicitly, it only checks if the *resulting* circles fit.
        # If I move points closer, r decreases. If I move them further, r increases, but they might not fit in the box.
        # So the optimizer will naturally push them apart until they hit the box limit.
        
        # Let's add a small penalty if they don't touch the boundaries to ensure we use the full perimeter.
        # But for now, -min(r) is a good proxy.
        
        return -np.min(r)

    # Multi-start search
    num_starts = 10
    random.seed(42)
    np.random.seed(42)
    
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
            
        # 1. Generate initial positions
        # Grid with random jitter
        side = 5 # 5x5 grid = 25 spots, use 21
        # Or 4x6? 24 spots.
        # Let's try a few grid shapes
        grid_shapes = [(5, 5), (4, 6), (3, 7)]
        rows, cols = grid_shapes[start_idx % len(grid_shapes)]
        
        # Aspect ratio initial guess
        # If grid is 5x5, aspect ~ 1.0
        # If 4x6, aspect ~ 1.5
        aspect_init = cols / rows
        
        # Create grid points in [0, 1] x [0, 1]
        positions = np.zeros((N, 2))
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= N:
                    break
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                # Add jitter
                x += np.random.uniform(-0.05, 0.05)
                y += np.random.uniform(-0.05, 0.05)
                # Keep in [0.1, 0.9] to avoid boundary issues initially
                x = np.clip(x, 0.1, 0.9)
                y = np.clip(y, 0.1, 0.9)
                positions[count] = [x, y]
                count += 1
        
        # Scale positions to fit in an initial box of aspect aspect_init
        # We will let the optimizer handle the scaling/aspect, but we need a good start.
        # Let's fix the aspect ratio for this run to simplify.
        aspect_fixed = aspect_init
        
        # Flatten positions for Nelder-Mead
        x0 = positions.flatten()
        
        # Run Nelder-Mead
        # We optimize 2N variables (positions). Aspect is fixed.
        # This is much lower dimensional than optimizing r as well.
        
        try:
            res = minimize(
                objective_structural, 
                x0, 
                args=(aspect_fixed,), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
            
            x_opt = res.x
            positions_opt = x_opt.reshape(N, 2)
            
            # Re-evaluate to get valid circles
            # We need to reconstruct the radii and check validity
            w = 2.0 * aspect_fixed / (1.0 + aspect_fixed)
            h = 2.0 / (1.0 + aspect_fixed)
            
            # Compute Delaunay again on optimal positions
            try:
                tri = Delaunay(positions_opt)
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i+1, 3):
                            edge = (int(simplex[i]), int(simplex[j]))
                            if edge[0] > edge[1]:
                                edge = (edge[1], edge[0])
                            edges.add(edge)
            except:
                continue
                
            if len(edges) < N - 1:
                continue
                
            A = np.zeros((len(edges), N))
            b = np.zeros(len(edges))
            edge_list = list(edges)
            for k, (i, j) in enumerate(edge_list):
                A[k, i] = 1.0
                A[k, j] = 1.0
                dx = positions_opt[i, 0] - positions_opt[j, 0]
                dy = positions_opt[i, 1] - positions_opt[j, 1]
                b[k] = np.sqrt(dx*dx + dy*dy)
                
            try:
                r, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            if np.any(r <= 1e-6):
                continue
                
            # Construct candidate circles
            circles = np.column_stack((positions_opt, r))
            
            # Normalize to perimeter 4
            # The optimizer tried to fit them in w x h, but let's ensure strict validity
            # and scale if necessary.
            min_x = np.min(circles[:, 0] - circles[:, 2])
            max_x = np.max(circles[:, 0] + circles[:, 2])
            min_y = np.min(circles[:, 1] - circles[:, 2])
            max_y = np.max(circles[:, 1] + circles[:, 2])
            
            w_box = max_x - min_x
            h_box = max_y - min_y
            p = 2 * (w_box + h_box)
            
            if p > 4.0 + 1e-6:
                # Scale down to fit
                scale = 4.0 / p
                circles[:, 0] = (circles[:, 0] - min_x) * scale
                circles[:, 1] = (circles[:, 1] - min_y) * scale
                circles[:, 2] = circles[:, 2] * scale
            else:
                # Shift to [0, w] x [0, h]
                circles[:, 0] = circles[:, 0] - min_x
                circles[:, 1] = circles[:, 1] - min_y
                # Optionally scale up to fill the box if there's slack?
                # If p < 4, we can scale up.
                if p < 4.0 - 1e-6:
                    scale = 4.0 / p
                    circles[:, 0] *= scale
                    circles[:, 1] *= scale
                    circles[:, 2] *= scale
                    # Re-shift to ensure min is 0?
                    # Scaling from (min_x, min_y) which is now 0,0 keeps min at 0.
                    # But max might exceed w, h?
                    # w_box * scale = w_box * (4/p) = w_box * (2/(w_box+h_box)) = 2*w_box/(w_box+h_box)
                    # The new width is w_new = 2*w_box/(w_box+h_box).
                    # The new height is h_new = 2*h_box/(w_box+h_box).
                    # Perimeter is 2*(w_new + h_new) = 4. Correct.
            
            score, valid = evaluate_packing(circles)
            
            if valid and score > best_score:
                best_score = score
                best_result = circles.copy()
                
        except Exception:
            continue

    if best_result is None:
        # Fallback: simple grid
        side = 5
        res = np.zeros((N, 3))
        r = 0.1
        for i in range(N):
            row = i // 5
            col = i % 5
            res[i, 0] = (col + 0.5) * 0.2
            res[i, 1] = (row + 0.5) * 0.2
            res[i, 2] = r
        # Check if valid
        s, v = evaluate_packing(res)
        if v:
            return res
        # If even fallback is invalid (should not happen), return tiny circles
        res[:, 2] = 0.01
        return res

    return best_result
# EVOLVE-BLOCK-END
