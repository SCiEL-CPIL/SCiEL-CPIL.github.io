"""
Heilbronn triangles: maximize the smallest triangle area formed by 11 points 
in an equilateral triangle.

Strategy:
1. Structural LP Approach: Fix the orientation of all triples by permuting the 
   points. For a fixed permutation, the condition "area >= t" becomes a set of 
   linear inequalities if we assume a consistent orientation (e.g., all CCW).
   We solve a Linear Program to maximize t.
2. Refinement: The LP solution is a local optimum for that permutation. We use 
   a gradient-based local search (soft-min approximation) to polish the result, 
   potentially allowing for small orientation flips if beneficial, or just 
   refining the boundaries.
"""

import numpy as np
import time
from scipy.optimize import linprog, minimize
import itertools

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2.0

# Triangle vertices for reference
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute all triples (i, j, k) with i < j < k
_TRIPLES = []
def _init_triples():
    global _TRIPLES
    if not _TRIPLES:
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    _TRIPLES.append((i, j, k))
_init_triples()

def _min_triangle_area(points):
    """Compute the smallest area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in _TRIPLES:
        pi = points[i]
        pj = points[j]
        pk = points[k]
        # Cross product magnitude / 2
        area = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0]))
        if area < min_area:
            min_area = area
    return min_area

def _solve_lp_for_ordering(ordering, deadline):
    """
    Solve the LP problem for a specific ordering of points.
    We assume that for indices i < j < k in the ordering, the signed area 
    (cross product) is positive.
    
    Variables: x_0, y_0, ..., x_10, y_10, t
    Total variables: 2*N + 1
    
    Objective: Maximize t -> Minimize -t
    c = [0, 0, ..., 0, -1] (last element is -1 for t)
    
    Constraints:
    1. Point in triangle:
       y_i >= 0
       y_i - SQRT3*x_i <= 0  (y <= sqrt(3)x)
       y_i + SQRT3*(x_i - 1) <= 0 (y <= -sqrt(3)(x-1))
       
    2. Area constraints:
       For each triple (i, j, k) in the ORIGINAL indices, we need to check 
       their positions in the ORDERING.
       Let pos[i] be the index of point i in the ordering.
       We require the points to be in CCW order according to the ordering.
       This means for any triple of points (a, b, c) from the set {0..N-1},
       if we sort them by their position in `ordering`, say p_a < p_b < p_c,
       then the signed area of (a, b, c) should be >= 2*t (since area = 0.5*cross).
       
       Wait, the constraint is on the set of points. 
       Let the points be P_0, ..., P_{N-1}.
       Let the ordering be a permutation sigma of 0..N-1.
       We enforce that for any i, j, k such that sigma^{-1}(i) < sigma^{-1}(j) < sigma^{-1}(k),
       the cross product (P_j - P_i) x (P_k - P_i) >= 2*t.
       
       Note: This assumes the "ordering" defines the cyclic order or the topological sort.
       Actually, a total order on points doesn't uniquely define the sign of the cross product 
       for all triples unless the points are in convex position and the order is consistent with 
       the boundary. However, for the interior, we can just enforce the sign based on the permutation.
       
       Let's define the constraint:
       For any triple (i, j, k) of point indices:
       Let r_i, r_j, r_k be their ranks in the `ordering`.
       Sort them by rank: let a, b, c be the point indices such that rank(a) < rank(b) < rank(c).
       Then we require:
       (x_b - x_a)(y_c - y_a) - (y_b - y_a)(x_c - x_a) >= 2*t
       
       This is linear in x, y, t.
    """
    # Variable vector: [x_0, y_0, ..., x_N, y_N, t]
    n_vars = 2 * N + 1
    
    # Objective: Minimize -t
    c = np.zeros(n_vars)
    c[-1] = -1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    A_ub = []
    b_ub = []
    
    # 1. Triangle Constraints
    # y_i >= 0  =>  -y_i <= 0
    # y_i <= SQRT3 * x_i  =>  -SQRT3*x_i + y_i <= 0
    # y_i <= -SQRT3 * (x_i - 1)  =>  SQRT3*x_i + y_i <= SQRT3
    
    for i in range(N):
        # -y_i <= 0
        row = np.zeros(n_vars)
        row[2*i + 1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
        # -SQRT3*x_i + y_i <= 0
        row = np.zeros(n_vars)
        row[2*i] = -SQRT3
        row[2*i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
        # SQRT3*x_i + y_i <= SQRT3
        row = np.zeros(n_vars)
        row[2*i] = SQRT3
        row[2*i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(SQRT3)
        
    # 2. Area Constraints
    # Map point index to its rank in the ordering
    rank = [0] * N
    for pos, pt_idx in enumerate(ordering):
        rank[pt_idx] = pos
        
    for i, j, k in _TRIPLES:
        # Determine the order of i, j, k in the `ordering`
        # We want the points sorted by their rank in the ordering
        pts = [i, j, k]
        pts.sort(key=lambda x: rank[x])
        a, b, c = pts
        
        # Constraint: (x_b - x_a)(y_c - y_a) - (y_b - y_a)(x_c - x_a) >= 2*t
        # => (x_b - x_a)(y_c - y_a) - (y_b - y_a)(x_c - x_a) - 2*t >= 0
        # => -[(x_b - x_a)(y_c - y_a) - (y_b - y_a)(x_c - x_a) - 2*t] <= 0
        # => (y_b - y_a)(x_c - x_a) - (x_b - x_a)(y_c - y_a) + 2*t <= 0
        
        # Expand:
        # (y_b x_c - y_b x_a - y_a x_c + y_a x_a) - (x_b y_c - x_b y_a - x_a y_c + x_a y_a) + 2t <= 0
        # y_b x_c - y_b x_a - y_a x_c + y_a x_a - x_b y_c + x_b y_a + x_a y_c - x_a y_a + 2t <= 0
        # y_a x_a and -x_a y_a cancel out.
        
        row = np.zeros(n_vars)
        
        # Coeffs for x_a, y_a:
        # From - y_a x_c: row[2*a+1] += -1 (coeff of x_c is 1, so term is -y_a * x_c) -> Wait, variables are separate.
        # Let's just build the linear combination directly.
        
        # Term: (y_b - y_a)(x_c - x_a) - (x_b - x_a)(y_c - y_a)
        
        # y_b * x_c
        row[2*b+1] += 1.0  # y_b
        row[2*c]   += 1.0  # x_c
        
        # - y_b * x_a
        row[2*b+1] += 0.0 # already added
        row[2*a]   += -1.0 # x_a
        
        # - y_a * x_c
        row[2*a+1] += -1.0 # y_a
        row[2*c]   += 0.0  # x_c (already added)
        
        # + y_a * x_a
        row[2*a+1] += 0.0
        row[2*a]   += 1.0
        
        # - x_b * y_c
        row[2*b]   += -1.0 # x_b
        row[2*c+1] += -1.0 # y_c
        
        # + x_b * y_a
        row[2*b]   += 0.0
        row[2*a+1] += 1.0
        
        # + x_a * y_c
        row[2*a]   += 1.0
        row[2*c+1] += 0.0
        
        # - x_a * y_a
        row[2*a]   += 0.0
        row[2*a+1] += -1.0
        
        # + 2*t
        row[-1] += 2.0
        
        A_ub.append(row)
        b_ub.append(0.0)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds
    # x_i, y_i in [0, 1] (loose bounds, triangle constraints are tighter)
    # t >= 0
    bounds = [(0, 1)] * (2 * N) + [(0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x = res.x
            points = np.zeros((N, 2))
            points[:, 0] = x[0::2]
            points[:, 1] = x[1::2]
            t_val = x[-1]
            return points, t_val
        else:
            return None, 0.0
    except Exception:
        return None, 0.0

def _soft_min_objective(params, beta):
    """
    Objective for gradient refinement.
    params: flat array [x0, y0, ..., x10, y10]
    """
    points = params.reshape(N, 2)
    # Clip to triangle to ensure validity
    # This is a soft clip, better to use projection if we were doing constrained opt, 
    # but SLSQP handles bounds. For gradient calc, we assume points are inside.
    
    areas = np.zeros(len(_TRIPLES))
    for idx, (i, j, k) in enumerate(_TRIPLES):
        pi = points[i]
        pj = points[j]
        pk = points[k]
        cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        areas[idx] = 0.5 * abs(cross)
        
    # Soft min
    if len(areas) == 0:
        return 0.0
    max_area = np.max(areas)
    # Use log-sum-exp for stability
    # soft_min = -1/beta * log(sum(exp(-beta * areas)))
    # To avoid underflow/overflow, subtract max
    neg_beta_areas = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(neg_beta_areas)))
    soft_min = -max_area - (1.0 / beta) * log_sum
    
    return -soft_min

def _refine_with_gradient(points, deadline):
    """
    Refine points using SLSQP with soft-min objective.
    """
    x0 = points.flatten()
    
    # Constraints for SLSQP: Points inside triangle
    # y_i >= 0
    # y_i <= SQRT3 * x_i
    # y_i <= -SQRT3 * (x_i - 1)
    
    constraints = []
    for i in range(N):
        # y_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        # y_i - SQRT3*x_i <= 0  => SQRT3*x_i - y_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: SQRT3*x[2*i] - x[2*i+1]})
        # -SQRT3*(x_i - 1) - y_i >= 0 => SQRT3 - SQRT3*x_i - y_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: SQRT3 - SQRT3*x[2*i] - x[2*i+1]})
        
    bounds = [(0, 1)] * (2 * N)
    
    best_points = points.copy()
    best_score = _min_triangle_area(points)
    
    # Try a few beta values
    for beta in [20, 50, 100]:
        if time.time() > deadline:
            break
            
        def obj(x):
            return _soft_min_objective(x, beta)
            
        try:
            res = minimize(
                obj,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if res.success:
                new_points = res.x.reshape(N, 2)
                new_score = _min_triangle_area(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
                    x0 = res.x
        except Exception:
            continue
            
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 20s safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -float('inf')
    
    # 1. Generate candidate orderings
    # We will try random permutations and a few heuristic ones.
    orderings = []
    
    # Heuristic 1: Random
    for _ in range(10):
        orderings.append(list(rng.permutation(N)))
        
    # Heuristic 2: Radial sort from centroid
    # Points in a good configuration are often roughly evenly spaced.
    # Let's just use random permutations for now, maybe add a "grid-like" permutation later if time permits.
    
    # 2. Solve LP for each ordering
    for idx, ordering in enumerate(orderings):
        if time.time() > deadline - 20:
            break
            
        points, t_val = _solve_lp_for_ordering(ordering, deadline)
        
        if points is not None:
            # Evaluate true score
            score = _min_triangle_area(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
    # Fallback if LP failed or score is very low
    if best_points is None or best_score < 0.01:
        # Simple random initialization
        best_points = np.zeros((N, 2))
        for i in range(N):
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            best_points[i, 0] = u
            best_points[i, 1] = v * SQRT3 # Wait, v is barycentric? 
            # Let's use the standard random point in triangle generation
            # x = u, y = v * sqrt(3) * (1-u) ... no
            # Standard: x = u, y = v * sqrt(3) * (1 - u) is for right triangle?
            # For equilateral:
            # Let u, v be uniform [0,1]. If u+v > 1, flip.
            # x = u + 0.5*v
            # y = v * sqrt(3)/2
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            best_points[i, 0] = u + 0.5 * v
            best_points[i, 1] = v * H
        best_score = _min_triangle_area(best_points)

    # 3. Refine the best solution found
    if time.time() < deadline - 5:
        refined_points = _refine_with_gradient(best_points, deadline)
        refined_score = _min_triangle_area(refined_points)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score

    return best_points
