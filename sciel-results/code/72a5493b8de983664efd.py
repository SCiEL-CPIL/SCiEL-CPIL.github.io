"""Circle packing in unit square: pack 26 circles, maximize sum of radii.

Strategy: Voronoi-based packing.
- Represent the packing by N seed points (centers).
- The radius of each circle is the distance from its seed to the nearest
  "obstacle", where obstacles are the square boundaries and the other seeds.
- Specifically, for seed i, r_i = min(
      min(x_i, 1-x_i, y_i, 1-y_i),  # Distance to walls
      min_{j != i} (dist(i, j) / 2) # Distance to other seeds (assuming equal growth logic? No, distinct radii)
  )
  
  Actually, for distinct radii in a Voronoi diagram, the radius of circle i
  is the distance from seed i to the nearest edge of its Voronoi cell.
  The edges of the Voronoi cell are perpendicular bisectors between seed i and
  its neighbors, and the square boundaries.
  
  The distance from seed i to the bisector between i and j is dist(i,j)/2.
  However, the Voronoi cell is the intersection of half-planes.
  The radius is the distance from the seed to the boundary of the intersection.
  
  A simpler, robust approach for optimization:
  Define r_i such that the circles are disjoint and inside the box.
  We can maximize sum(r_i) subject to:
  r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
  r_i + r_j <= dist(i, j)
  
  This is the original formulation. The "Voronoi" insight is that at the
  optimum, many constraints are tight.
  
  Let's try a different structural change: **Simulated Annealing on the 
  configuration space with a specialized move set**.
  
  Wait, the prompt asks for a MATERIALLY DIFFERENT strategy.
  The current program uses grid init + local search + gradient polish.
  
  New Strategy: **Direct Optimization of Radii and Centers with a 
  "Repulsion" Potential and Wall Attraction**.
  
  Actually, let's look at the "Voronoi" idea more closely.
  If we fix the centers, the maximal radii are determined by the Voronoi diagram.
  So we can optimize ONLY the centers (x, y) for 26 circles.
  Objective: Maximize sum( r_i(x, y) ) where r_i is the inradius of the 
  Voronoi cell of point i.
  
  The inradius of a Voronoi cell is the distance from the site to the nearest 
  edge of the cell.
  For a site p_i, the nearest edge is either:
  1. The square boundary (if p_i is close to a wall).
  2. The perpendicular bisector between p_i and some p_j.
  
  The distance from p_i to the bisector of p_i and p_j is ||p_i - p_j|| / 2.
  BUT, this is only valid if the intersection of the half-planes actually
  forms a cell that includes that edge segment.
  
  A simpler, differentiable approximation:
  r_i = min( min_wall_dist_i, min_{j != i} (dist(i, j) / 2) )
  
  This is a lower bound on the true Voronoi inradius? 
  No, if three points form a triangle, the Voronoi vertex is the circumcenter.
  The distance from the site to the circumcenter can be larger than dist/2 
  if the angle is obtuse? No, in a triangle, the circumradius R = a / (2 sin A).
  If A is small, R is large.
  The Voronoi cell boundary is formed by bisectors.
  
  Let's use the property:
  Maximize sum(r_i) s.t. 
  r_i <= wall_clearance_i
  r_i + r_j <= dist(i, j)
  
  This is a linear programming problem if centers are fixed.
  If we optimize centers, it's non-convex.
  
  Let's try **Particle Swarm Optimization (PSO)** or **Differential Evolution** 
  on the full (x, y, r) space, but with a penalty-based fitness that is 
  highly sensitive to violations. This is a "black-box" approach, different 
  from the gradient-based local search.
  
  Actually, the current program is quite good (1.1778). To beat it, we need 
  to escape local minima better.
  
  Let's try **Basin Hopping**.
  Basin Hopping:
  1. Start with a configuration.
  2. Local Optimize (use the existing gradient polish or a fast local search).
  3. Perturb the configuration (random kick).
  4. Local Optimize again.
  5. Accept if better (or with Metropolis criterion).
  
  This explicitly explores the landscape of local minima.
  
  Let's implement Basin Hopping using the existing `_gradient_polish` as the 
  local optimizer, but with a more aggressive perturbation schedule.
  
  Wait, the current program already does multi-start and local search.
  Basin Hopping is essentially multi-start with annealing on the basin.
  
  Let's try a **different representation**: 
  **Ordering-based packing**.
  Sort circles by radius. Place them one by one.
  This is hard to optimize directly.
  
  How about **Simulated Annealing** on the positions only, where radii are 
  derived from the Voronoi diagram?
  
  Let's implement **Voronoi-based SA**.
  1. State: 26 points (x, y).
  2. Energy: -sum(r_i) where r_i is the inradius of the Voronoi cell.
  3. Move: Perturb one point.
  4. Calculate new energy.
  5. Accept/Reject.
  
  To calculate the inradius of a Voronoi cell efficiently:
  We need the nearest neighbor distance to any other point (dist/2) and 
  distance to walls.
  BUT, as noted, the Voronoi cell is a polygon. The inradius is the distance 
  to the nearest edge.
  For a point p_i, the candidate edges are:
  - The 4 walls.
  - The bisectors with all other points.
  
  The distance to the bisector of i and j is dist(i, j) / 2.
  Is the closest edge always one of these?
  Yes, the Voronoi cell is the intersection of half-planes H(i, j) = {x | dist(x, i) <= dist(x, j)}.
  The boundary of the cell is a subset of the lines forming the bisectors.
  The distance from i to the cell boundary is the minimum distance from i to 
  any point on the boundary.
  Since the boundary is composed of line segments from the bisectors, and i 
  is inside the cell, the minimum distance is the minimum over all j of 
  dist(i, j)/2?
  
  Counter-example:
  Three points A, B, C forming an equilateral triangle.
  Voronoi cell of A is a hexagon (truncated by walls).
  The vertices of the cell are the circumcenters of the triangles.
  The distance from A to the circumcenter of ABC is R = side / sqrt(3).
  The distance from A to the bisector of AB is side / 2.
  Since 1/sqrt(3) ≈ 0.577 > 0.5, the circumcenter is FURTHER than the 
  bisector midpoint.
  So the minimum distance is indeed to the bisector segment, not the vertex.
  
  Wait, the Voronoi cell of A is bounded by the bisector of AB and AC.
  The "corner" of the cell is the intersection of bisector(AB) and bisector(AC), 
  which is the circumcenter.
  The distance from A to the boundary is the minimum distance to any point 
  on the boundary.
  The boundary consists of segments.
  The closest point on a line segment to A is either the projection of A onto 
  the line, or an endpoint.
  The projection of A onto bisector(AB) is the midpoint of AB.
  Distance = dist(A, B) / 2.
  The endpoint is the circumcenter. Distance = R.
  Since R > dist(A, B)/2, the minimum distance to the bisector line is 
  dist(A, B)/2.
  
  So, r_i = min( wall_dist_i, min_{j != i} dist(i, j) / 2 ).
  
  This is a very simple formula!
  Let's verify.
  If this formula holds, we can optimize 52 variables (x, y) with a simple 
  objective function.
  
  Let's check if this is always true.
  The Voronoi cell V_i = { x | ||x - p_i|| <= ||x - p_j|| for all j }.
  This is the intersection of half-planes.
  The distance from p_i to the boundary of V_i is the inradius.
  The boundary is a polygon.
  The distance from a point inside a convex polygon to its boundary is the 
  minimum distance to any edge.
  Each edge lies on a bisector line L(i, j).
  The distance from p_i to the line L(i, j) is ||p_i - p_j|| / 2.
  The edge is a segment of L(i, j).
  The closest point on the segment to p_i is the projection of p_i onto the 
  line, IF the projection falls within the segment.
  The projection of p_i onto the bisector of i and j is the midpoint M_ij.
  Does M_ij lie on the edge of the Voronoi cell?
  The edge exists if the half-planes intersect in a way that includes M_ij.
  Actually, the Voronoi edge between i and j exists if there is a point x such 
  that ||x - p_i|| = ||x - p_j|| and ||x - p_k|| >= ||x - p_i|| for all k.
  
  If M_ij is in the Voronoi cell of i (and j), then the distance is dist/2.
  If M_ij is "cut off" by another point k, then the boundary is further away?
  No, if M_ij is cut off, it means the edge is shorter.
  The closest point on the shortened edge to p_i might be an endpoint (Voronoi vertex).
  A Voronoi vertex is the circumcenter of i, j, k.
  Distance from p_i to circumcenter of i, j, k.
  Is this distance always >= dist(i, j)/2?
  In triangle i, j, k, let R be circumradius.
  R = a / (2 sin A).
  dist(i, j) = a.
  dist(i, j)/2 = a/2.
  R >= a/2 iff sin A <= 1, which is always true.
  So the distance to the vertex is >= distance to the midpoint of the side.
  
  Therefore, the minimum distance to the boundary is ALWAYS achieved at the 
  projection onto one of the bisector lines, i.e., at the midpoint of the 
  segment connecting i to some j.
  
  So, **r_i = min( min_wall_dist_i, min_{j != i} dist(i, j) / 2 )** is EXACT.
  
  This is a huge simplification.
  We can optimize only the centers (x, y) for 26 circles.
  Objective: Maximize Sum( min( wall_dist_i, min_j dist(i, j)/2 ) ).
  
  This function is non-smooth (min), but we can smooth it or use subgradient 
  methods, or simply use a robust optimizer like Nelder-Mead or COBYLA, or 
  even a simple hill-climbing with random restarts.
  
  Given the low dimensionality (52 vars), we can use **Differential Evolution** 
  or **Particle Swarm** or just **Basin Hopping with Nelder-Mead**.
  
  Let's implement **Basin Hopping** with **Nelder-Mead** local optimization 
  on the 52-dimensional center space.
  
  Steps:
  1. Initialize 26 centers randomly.
  2. Local Optimize using Nelder-Mead to maximize the smooth approximation 
     of the sum of radii.
     To make it smooth, we can use a soft-min:
     soft_min(a, b) = -(1/beta) * log( exp(-beta a) + exp(-beta b) )
     Or just use the hard min and let Nelder-Mead handle it (it's derivative-free).
  3. Perturb centers.
  4. Repeat.
  
  This is structurally different:
  - Variable space: 52D (centers only) vs 78D (centers + radii).
  - Objective: Derived from geometry (Voronoi inradius) vs explicit radius variables.
  - Optimizer: Global (Basin Hopping) + Local (Nelder-Mead) vs Gradient-based (L-BFGS).
  
  Let's code this.
"""

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

def _compute_radii(centers):
    """
    Given centers (N, 2), compute the maximal radii for each circle
    based on the Voronoi inradius formula.
    r_i = min( wall_dist_i, min_{j != i} dist(i, j) / 2 )
    """
    n = len(centers)
    radii = np.zeros(n)
    
    # Precompute pairwise distances
    # dist[i, j]
    dx = centers[:, 0, np.newaxis] - centers[:, 0][np.newaxis, :]
    dy = centers[:, 1, np.newaxis] - centers[:, 1][np.newaxis, :]
    dist = np.sqrt(dx * dx + dy * dy)
    
    for i in range(n):
        # Wall distance
        x, y = centers[i]
        wall_d = min(x, 1.0 - x, y, 1.0 - y)
        
        # Min distance to other centers / 2
        # dist[i, j] for j != i
        d_others = dist[i, :].copy()
        d_others[i] = np.inf
        min_d = np.min(d_others) / 2.0
        
        radii[i] = min(wall_d, min_d)
        
    return radii

def _objective(centers):
    """
    Objective to MINIMIZE: -sum(radii)
    """
    radii = _compute_radii(centers)
    return -np.sum(radii)

def _basin_hopping(centers, deadline):
    """
    Basin Hopping algorithm.
    1. Local Optimize (Nelder-Mead)
    2. Perturb
    3. Local Optimize
    4. Accept/Reject
    """
    n = len(centers)
    current_centers = centers.copy()
    current_radii = _compute_radii(current_centers)
    current_obj = -np.sum(current_radii)
    
    best_centers = current_centers.copy()
    best_obj = current_obj
    
    rng = np.random.RandomState(42)
    
    # Local optimization using Nelder-Mead
    def local_optimize(x0):
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'fatol': 1e-10, 'xatol': 1e-10})
            return res.x
        except Exception:
            return x0
    
    # Perturbation function
    def perturb(x, step):
        return x + rng.uniform(-step, step, size=x.shape)
    
    # Basin Hopping loop
    max_iterations = 50
    step_size = 0.05
    
    for i in range(max_iterations):
        if time.time() > deadline - 5:
            break
            
        # 1. Local Optimize current
        opt_centers = local_optimize(current_centers)
        opt_radii = _compute_radii(opt_centers)
        opt_obj = -np.sum(opt_radii)
        
        if opt_obj < best_obj:
            best_obj = opt_obj
            best_centers = opt_centers.copy()
            
        # 2. Perturb
        pert_centers = perturb(opt_centers, step_size)
        # Clip to box
        pert_centers[:, 0] = np.clip(pert_centers[:, 0], 0.01, 0.99)
        pert_centers[:, 1] = np.clip(pert_centers[:, 1], 0.01, 0.99)
        
        # 3. Local Optimize perturbed
        new_centers = local_optimize(pert_centers)
        new_radii = _compute_radii(new_centers)
        new_obj = -np.sum(new_radii)
        
        # 4. Accept/Reject (Metropolis)
        # We want to minimize obj, so lower is better.
        # delta = new_obj - current_obj (where current_obj is from step 1)
        delta = new_obj - opt_obj
        if delta < 0:
            # Better, accept
            current_centers = new_centers
            current_obj = new_obj
        else:
            # Worse, accept with probability exp(-delta / T)
            # T is temperature. Start high, decay.
            T = 0.01 * (0.9 ** i)
            if T > 1e-6:
                if rng.random() < np.exp(-delta / T):
                    current_centers = new_centers
                    current_obj = new_obj
        
        # Adapt step size
        # If we accepted, maybe increase step? If rejected, decrease?
        # Simple decay
        step_size *= 0.95
        
    return best_centers, _compute_radii(best_centers)

def _differential_evolution_init(deadline):
    """
    Use Differential Evolution to find a good starting point.
    """
    # Bounds for centers
    bounds = [(0.01, 0.99) for _ in range(2 * N)]
    
    # DE is slow, so use it only if we have time
    # Actually, let's just use random inits and basin hopping.
    # DE might be too slow for 52 vars.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centers = None
    best_radii = None
    best_score = 0.0
    
    # Multiple random starts
    rng = np.random.RandomState(123)
    
    # Try a few different initialization strategies
    inits = []
    
    # 1. Random
    for _ in range(3):
        centers = rng.uniform(0.1, 0.9, size=(N, 2))
        inits.append(centers)
        
    # 2. Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_centers = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                grid_centers.append([x, y])
                count += 1
    inits.append(np.array(grid_centers))
    
    # 3. Hexagonal-ish
    hex_centers = []
    for i in range(6):
        for j in range(5):
            if len(hex_centers) < N:
                x = 0.1 + i * 0.15 + (0.075 if j % 2 else 0)
                y = 0.1 + j * 0.15
                if x < 0.9 and y < 0.9:
                    hex_centers.append([x, y])
    if len(hex_centers) == N:
        inits.append(np.array(hex_centers))
    
    # Run Basin Hopping on each init
    for init in inits:
        if time.time() > deadline - 20:
            break
            
        c, r = _basin_hopping(init, deadline)
        score = np.sum(r)
        
        if score > best_score:
            best_score = score
            best_centers = c
            best_radii = r
    
    # Final polish: 
    # The Voronoi radius formula gives the MAXIMAL radius for fixed centers.
    # So the result is already optimal for those centers.
    # We just need to format the output.
    
    if best_centers is None:
        # Fallback
        best_centers = np.array([[0.5, 0.5]] * N)
        best_radii = np.array([0.01] * N)
    
    # Construct result array (N, 3)
    result = np.zeros((N, 3))
    result[:, 0] = best_centers[:, 0]
    result[:, 1] = best_centers[:, 1]
    result[:, 2] = best_radii
    
    # Validation and clipping
    for i in range(N):
        x, y, r = result[i]
        # Ensure inside box
        r = min(r, x, 1.0 - x, y, 1.0 - y)
        r = max(r, 0.001)
        result[i, 2] = r
        result[i, 0] = np.clip(x, r, 1.0 - r)
        result[i, 1] = np.clip(y, r, 1.0 - r)
        
    # Final check for overlaps (should be none by construction, but float errors?)
    # The formula r_i = min(dist/2) ensures dist(i, j) >= r_i + r_j?
    # r_i <= dist(i, j)/2 and r_j <= dist(i, j)/2 => r_i + r_j <= dist(i, j).
    # Yes, it's guaranteed.
    
    return result

# EVOLVE-BLOCK-START
# (The code above is the complete program)
# EVOLVE-BLOCK-END
