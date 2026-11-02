import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0
EPS = 1e-9
MIN_DIST_SQ = 4.0  # (2*apothem)^2 = distance between centers for non-overlap

def time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

def hex_hex_dist_sq(p1, a1, p2, a2):
    """Squared minimum distance between two unit hexagons.
    p1, p2 are (x,y) centers; a1, a2 are rotation angles in radians.
    Returns the squared distance between the two hexagons (0 if overlapping/touching).
    """
    # The distance between two regular hexagons equals the minimum over all
    # vertex-vertex, vertex-edge, edge-edge pairs. For regular hexagons of the
    # same size, the distance can be computed via support functions.
    # Two convex polygons are at distance d if and only if d is the minimum
    # distance between any pair of points on the polygons.
    # For efficiency, use the fact that for two regular hexagons, the minimum
    # distance is the minimum over the 6x6=36 vertex pairs and 6x6=36
    # vertex-edge distances.
    # Actually, the cleanest approach: compute the 6 vertices of each hexagon,
    # then compute min distance between the two convex polygons.
    
    def hex_vertices(center, angle):
        cx, cy = center
        verts = []
        for i in range(6):
            a = angle + i * math.pi / 3.0
            verts.append((cx + math.cos(a), cy + math.sin(a)))
        return verts
    
    v1 = hex_vertices(p1, a1)
    v2 = hex_vertices(p2, a2)
    
    # Compute minimum distance between two convex polygons
    # Use the separating axis theorem in reverse: min distance
    min_d2 = float('inf')
    
    # Check all vertex-vertex pairs
    for v in v1:
        for w in v2:
            dx = v[0] - w[0]
            dy = v[1] - w[1]
            d2 = dx*dx + dy*dy
            if d2 < min_d2:
                min_d2 = d2
    
    # Check vertex-edge pairs (v from poly1 to edges of poly2, and vice versa)
    def point_to_segment_sq(px, py, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay
        len2 = dx*dx + dy*dy
        if len2 < 1e-20:
            return (px-ax)**2 + (py-ay)**2
        t = ((px-ax)*dx + (py-ay)*dy) / len2
        t = max(0.0, min(1.0, t))
        projx = ax + t*dx
        projy = ay + t*dy
        return (px-projx)**2 + (py-projy)**2
    
    for i in range(6):
        ax, ay = v1[i]
        bx, by = v1[(i+1) % 6]
        for w in v2:
            d2 = point_to_segment_sq(w[0], w[1], ax, ay, bx, by)
            if d2 < min_d2:
                min_d2 = d2
        
        ax, ay = v2[i]
        bx, by = v2[(i+1) % 6]
        for v in v1:
            d2 = point_to_segment_sq(v[0], v[1], ax, ay, bx, by)
            if d2 < min_d2:
                min_d2 = d2
    
    return min_d2

def check_feasibility(positions, angles):
    """Check if all hexagons are non-overlapping.
    positions: list of (x,y), angles: list of rotation angles.
    Returns True if feasible.
    """
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            d2 = hex_hex_dist_sq(positions[i], angles[i], positions[j], angles[j])
            if d2 < -1e-9:  # overlapping
                return False
    return True

def enclosing_hexagon_side(positions, center, beta):
    """Compute the side length of the minimum enclosing regular hexagon
    with given center and rotation beta (in radians)."""
    cx, cy = center
    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    pts = np.array(positions)
    pts_centered = pts - np.array([cx, cy])
    projs = pts_centered @ normals.T
    D = np.max(np.abs(projs))
    return (D + APOTHEM) / APOTHEM

def compute_fitness(positions, angles):
    """Compute the fitness (enclosing hexagon side length) for a configuration.
    Returns (S, center) where S is the side length and center is the optimal center.
    """
    pts = np.array(positions)
    
    # Find optimal center: minimize max projection
    # The optimal center for a given beta is the one that minimizes the
    # maximum absolute projection. This is a linear programming problem.
    # For a hexagon, the optimal center is the center of the minimum enclosing
    # hexagon, which can be found by solving:
    # min D s.t. -D <= proj_k(center) <= D for all k, all points
    
    # Use a simple approach: the optimal center is approximately the
    # center of the bounding box in each of the 6 directions.
    # More precisely, for each direction k, the center should be at the
    # midpoint of the min and max projections.
    # The optimal center minimizes the maximum over k of (max_proj_k - min_proj_k)/2.
    
    # For a given beta, the optimal center (cx, cy) satisfies:
    # For each k: cx*cos(ak) + cy*sin(ak) = (max_proj_k + min_proj_k) / 2
    # where proj_k = x*cos(ak) + y*sin(ak)
    
    # This is a system of 6 equations in 2 unknowns. We can solve it
    # by least squares, but the optimal solution is actually the one that
    # minimizes the maximum deviation.
    
    # Simpler approach: use the fact that for the minimum enclosing hexagon,
    # the center is the average of the 6 "supporting" lines.
    # Let's just do a small local optimization.
    
    best_S = float('inf')
    best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
    
    # Try several candidate centers
    candidate_centers = [
        (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))),
        (float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))),
    ]
    
    # Add centers from pairwise midpoints of extreme points
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            candidate_centers.append(((pts[i,0]+pts[j,0])/2, (pts[i,1]+pts[j,1])/2))
    
    for cx, cy in candidate_centers[:50]:
        S = enclosing_hexagon_side(positions, (cx, cy), 0.0)
        if S < best_S:
            best_S = S
            best_center = (cx, cy)
    
    return best_S, best_center

def repair(positions, angles, max_iters=20):
    """Repair overlapping pairs by pushing them apart.
    Returns repaired positions and angles.
    """
    positions = [list(p) for p in positions]
    angles = list(angles)
    n = len(positions)
    
    for _ in range(max_iters):
        if not time_ok():
            break
        fixed = True
        for i in range(n):
            for j in range(i+1, n):
                d2 = hex_hex_dist_sq(tuple(positions[i]), angles[i], tuple(positions[j]), angles[j])
                if d2 < 1e-9:
                    fixed = False
                    # Compute the direction to push apart
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-12:
                        # Random direction
                        ang = np.random.uniform(0, 2*math.pi)
                        dx, dy = math.cos(ang), math.sin(ang)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    # Push apart by a small amount
                    push = 0.1
                    positions[i][0] -= push * dx
                    positions[i][1] -= push * dy
                    positions[j][0] += push * dx
                    positions[j][1] += push * dy
        if fixed:
            break
    
    return [tuple(p) for p in positions], angles

def initial_population():
    """Generate initial population of configurations."""
    pop = []
    
    # Start with hexagonal lattice configurations
    # Standard hexagonal packing
    base_positions = []
    for q, r in [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        base_positions.append((x, y))
    
    for _ in range(20):
        # Perturb the base configuration
        positions = []
        angles = []
        for x, y in base_positions:
            positions.append((x + np.random.normal(0, 0.5), y + np.random.normal(0, 0.5)))
            angles.append(np.random.uniform(0, math.pi/3))
        positions, angles = repair(positions, angles)
        pop.append((positions, angles))
    
    # Cross shape
    base_cross = []
    for q, r in [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        base_cross.append((x, y))
    
    for _ in range(10):
        positions = []
        angles = []
        for x, y in base_cross:
            positions.append((x + np.random.normal(0, 0.5), y + np.random.normal(0, 0.5)))
            angles.append(np.random.uniform(0, math.pi/3))
        positions, angles = repair(positions, angles)
        pop.append((positions, angles))
    
    # Random configurations
    for _ in range(10):
        positions = []
        angles = []
        # Place in a circle of radius 3
        for i in range(N):
            ang = 2 * math.pi * i / N + np.random.normal(0, 0.2)
            r = 2.0 + np.random.uniform(0, 1.5)
            positions.append((r * math.cos(ang), r * math.sin(ang)))
            angles.append(np.random.uniform(0, math.pi/3))
        positions, angles = repair(positions, angles)
        pop.append((positions, angles))
    
    return pop

def mutate(positions, angles, mutation_strength=0.5):
    """Apply mutation to a configuration."""
    positions = [list(p) for p in positions]
    angles = list(angles)
    
    # Mutate a random subset of hexagons
    n_mutate = np.random.randint(1, 4)
    for _ in range(n_mutate):
        idx = np.random.randint(0, N)
        positions[idx][0] += np.random.normal(0, mutation_strength)
        positions[idx][1] += np.random.normal(0, mutation_strength)
        angles[idx] += np.random.normal(0, 0.1)
        angles[idx] = ((angles[idx] % (math.pi/3)) + math.pi/3) % (math.pi/3)
    
    # Also try global rotation
    if np.random.random() < 0.1:
        global_rot = np.random.normal(0, 0.05)
        for i in range(N):
            x, y = positions[i]
            new_x = x * math.cos(global_rot) - y * math.sin(global_rot)
            new_y = x * math.sin(global_rot) + y * math.cos(global_rot)
            positions[i] = [new_x, new_y]
            angles[i] = ((angles[i] + global_rot) % (math.pi/3) + math.pi/3) % (math.pi/3)
    
    positions = [tuple(p) for p in positions]
    positions, angles = repair(positions, angles)
    return positions, angles

def crossover(pos1, ang1, pos2, ang2):
    """Crossover between two configurations."""
    positions = []
    angles = []
    for i in range(N):
        if np.random.random() < 0.5:
            positions.append(pos1[i])
            angles.append(ang1[i])
        else:
            positions.append(pos2[i])
            angles.append(ang2[i])
    positions, angles = repair(positions, angles)
    return positions, angles

def optimize_center_and_rotation(positions):
    """For a fixed set of positions, optimize the enclosing hexagon center and rotation."""
    pts = np.array(positions)
    
    best_S = float('inf')
    best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
    best_beta = 0.0
    
    # Coarse search over beta
    for i in range(13):
        beta = i * (math.pi / 6) / 12
        
        # Find optimal center for this beta
        # The optimal center minimizes the maximum absolute projection
        # This is equivalent to finding the center of the minimum enclosing
        # hexagon with fixed orientation.
        
        # Use the fact that the optimal center is the one that equalizes
        # the maximum positive and negative projections in each direction.
        
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # For each direction k, the projection of point i is:
        # proj_k(i) = x_i * nx_k + y_i * ny_k
        # The range is max(proj_k) - min(proj_k)
        # The center should be at the midpoint: (max + min) / 2
        
        # The optimal center minimizes the maximum over k of (max_k - min_k) / 2
        # subject to the center being consistent across all k.
        
        # Simple approach: try several centers and pick the best
        candidate_centers = [
            (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))),
        ]
        
        # Add centers from extreme point pairs
        for k in range(6):
            projs = pts @ normals[k]
            idx_max = np.argmax(projs)
            idx_min = np.argmin(projs)
            candidate_centers.append(
                ((pts[idx_max, 0] + pts[idx_min, 0]) / 2,
                 (pts[idx_max, 1] + pts[idx_min, 1]) / 2)
            )
        
        for cx, cy in candidate_centers:
            S = enclosing_hexagon_side(positions, (cx, cy), beta)
            if S < best_S:
                best_S = S
                best_center = (cx, cy)
                best_beta = beta
    
    # Refine with local optimization
    try:
        def objective(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            return enclosing_hexagon_side(positions, (cx, cy), beta)
        
        cx0, cy0 = best_center
        beta0 = best_beta
        
        starts = [(cx0, cy0, beta0)]
        for dcx in [-0.2, 0.2]:
            for dcy in [-0.2, 0.2]:
                for db in [0.02, -0.02]:
                    b = ((beta0 + db) % (math.pi/6) + math.pi/6) % (math.pi/6)
                    starts.append((cx0 + dcx, cy0 + dcy, b))
        
        for s in starts[:8]:
            if not time_ok():
                break
            try:
                res = minimize(objective, s, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < best_S:
                    best_S = res.fun
                    best_center = (float(res.x[0]), float(res.x[1]))
                    best_beta = ((float(res.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
            except Exception:
                pass
    except Exception:
        pass
    
    return best_S, best_center, best_beta

def construct():
    if not time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Initialize population
    population = initial_population()
    
    # Evaluate initial population
    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_center = None
    best_beta = 0.0
    
    for positions, angles in population:
        if not time_ok():
            break
        S, center, beta = optimize_center_and_rotation(positions)
        if S < best_S:
            best_S = S
            best_positions = positions
            best_angles = angles
            best_center = center
            best_beta = beta
    
    # Genetic algorithm loop
    gen = 0
    pop_size = len(population)
    
    while time_ok():
        gen += 1
        
        # Evaluate all individuals
        scores = []
        for positions, angles in population:
            S, center, beta = optimize_center_and_rotation(positions)
            scores.append(S)
            if S < best_S:
                best_S = S
                best_positions = positions
                best_angles = angles
                best_center = center
                best_beta = beta
        
        # Sort by score (lower is better)
        sorted_idx = np.argsort(scores)
        population = [population[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        
        # Elitism: keep top 20%
        elite_size = max(2, pop_size // 5)
        new_population = population[:elite_size]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if not time_ok():
                break
            
            # Selection: tournament selection
            def tournament_select():
                idx1 = np.random.randint(0, elite_size + 5)
                idx2 = np.random.randint(0, elite_size + 5)
                if idx1 < len(population) and idx2 < len(population):
                    if idx1 == idx2:
                        return population[idx1]
                    s1 = scores[idx1] if idx1 < len(scores) else float('inf')
                    s2 = scores[idx2] if idx2 < len(scores) else float('inf')
                    return population[idx1] if s1 < s2 else population[idx2]
                return population[np.random.randint(0, len(population))]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover
            if np.random.random() < 0.7:
                child = crossover(parent1[0], parent1[1], parent2[0], parent2[1])
            else:
                child = (list(parent1[0]), list(parent1[1]))
            
            # Mutation
            mutation_strength = 0.5 / (1 + gen * 0.05)  # Decreasing mutation
            child = mutate(child[0], child[1], mutation_strength)
            
            new_population.append(child)
        
        population = new_population
        
        # Occasionally add random immigrants
        if gen % 10 == 0 and time_ok():
            for _ in range(3):
                positions = []
                angles = []
                for i in range(N):
                    ang = 2 * math.pi * i / N + np.random.normal(0, 0.3)
                    r = 2.0 + np.random.uniform(0, 2.0)
                    positions.append((r * math.cos(ang), r * math.sin(ang)))
                    angles.append(np.random.uniform(0, math.pi/3))
                positions, angles = repair(positions, angles)
                population.append((positions, angles))
        
        # Limit population size
        if len(population) > pop_size:
            population = population[:pop_size]
    
    # Final polish
    if best_positions is not None and time_ok():
        S, center, beta = optimize_center_and_rotation(best_positions)
        if S < best_S:
            best_S = S
            best_center = center
            best_beta = beta
    
    # Prepare output
    if best_positions is None:
        # Fallback: hexagonal lattice
        positions = []
        for q, r in [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions.append((x, y))
        best_positions = positions
        best_angles = [0.0] * N
        best_S, best_center, best_beta = optimize_center_and_rotation(positions)
    
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    inner = []
    for i in range(N):
        x, y = best_positions[i]
        inner.append([float(x), float(y), float(best_angles[i] * 180.0 / math.pi)])
    
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
