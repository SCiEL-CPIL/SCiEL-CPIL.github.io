"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Formulate the problem as a Linear Program (LP).
   The area of a triangle formed by points $p_i, p_j, p_k$ is given by:
   $2 \cdot \text{Area} = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|$
   
   The function $f(p) = \min_{\{i,j,k\}} 2 \cdot \text{Area}(p_i, p_j, p_k)$ is the minimum of a set of linear functions (after fixing the sign/orientation for each triplet based on an initial guess).
   Maximizing the minimum of linear functions is a standard LP problem.
   
   We use `scipy.optimize.linprog` with the HiGHS solver.
   
2. Multi-start strategy:
   - Generate diverse initial configurations (Grid, Random, Greedy).
   - For each configuration, determine the orientation (sign of cross product) of every triplet.
   - Solve the LP to maximize $t$ subject to:
     a) $\sum x_{coord} \cdot \text{coeff} \ge t$ for all triplets (with fixed signs)
     b) Points remain inside the triangle (linear inequalities)
   - Keep the best result.
   
3. Polish with a local search if time permits, but the LP solution is typically very strong for this problem size.
"""
import numpy as np
import time
import random
from scipy.optimize import linprog

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
M = len(_triplets)  # 165 triplets

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_triplets[:, 0]]
    p_j = pts[_triplets[:, 1]]
    p_k = pts[_triplets[:, 2]]
    
    # 2 * Area = |cross(p_j - p_i, p_k - p_i)|
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    """
    pts = []
    # Start with corners for stability? Or random?
    # Let's use a mix.
    p0 = np.array([0.5, S3H/3]) 
    pts.append(p0)
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        # Sample many candidates to find a good spot
        for _ in range(2000):
            cand = np.array(_random_point_in_triangle(rng))
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def _structured_init():
    """Create a structured initialization based on known Heilbronn-like configurations."""
    # A rough grid or known good config
    # Corners + midpoints + center-ish
    pts = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.25, 0.0], [0.75, 0.0], [0.5, S3H/2], # Midpoints of edges
        [0.25, S3H/2], [0.75, S3H/2],
        [0.5, S3H/3],
        [0.333, S3H/3], [0.666, S3H/3]
    ]
    return np.array(pts)

def _solve_lp(pts_init):
    """
    Solve the LP for a fixed orientation (sign) of the cross products.
    
    Variables: x_1, y_1, ..., x_N, y_N, t
    Total variables: 2*N + 1 = 23
    
    Objective: Maximize t -> Minimize -t
    c = [0, 0, ..., 0, -1]
    
    Constraints:
    1. Area constraints:
       For each triplet (i, j, k), let s_{ijk} be the sign of the cross product in the initial config.
       We want: s_{ijk} * (x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)) >= t
       
       Expanding:
       s_{ijk} * (x_i y_j - x_i y_k + x_j y_k - x_j y_i + x_k y_i - x_k y_j) >= t
       
       This is NOT linear in the variables because of terms like x_i y_j.
       
       WAIT. The area formula $2 \text{Area} = | \det(p_i, p_j, p_k) |$ where $p_i = (x_i, y_i, 1)$ in homogeneous coords.
       The determinant is:
       $x_i(y_j - y_k) - y_i(x_j - x_k) + (x_j y_k - x_k y_j)$
       
       This is linear in the coordinates $x_i, y_i$ if we view it as:
       $A_{ijk} \cdot v \ge t$
       where $v$ is the vector of all coordinates.
       
       Let's verify linearity.
       Term: $x_i y_j$. This is a product of two variables. It is NOT linear.
       
       Ah, the standard formulation for Heilbronn usually fixes the triangle and optimizes points.
       Is the area linear in the point coordinates?
       Area of triangle with vertices $(x_1, y_1), (x_2, y_2), (x_3, y_3)$:
       $0.5 | x_1(y_2 - y_3) + x_2(y_3 - y_1) + x_3(y_1 - y_2) |$
       
       Yes, this expression is LINEAR in the variables $x_1, y_1, x_2, y_2, x_3, y_3$.
       It does not involve products of variables like $x_1 y_1$. It involves products of a variable and a constant (or another variable from a *different* term in the sum, but looking at the structure:
       $x_1 y_2$ is a product of two variables.
       
       Let's re-read carefully.
       $x_1(y_2 - y_3) = x_1 y_2 - x_1 y_3$.
       This contains $x_1 y_2$.
       
       If the variables are $x_1, y_1, x_2, y_2, x_3, y_3$, then $x_1 y_2$ is a bilinear term.
       Therefore, the constraint $Area \ge t$ is NOT linear in the Cartesian coordinates.
       
       However, if we fix the *positions* of 2 points, it is linear in the 3rd.
       Or if we use Barycentric coordinates? No, same issue.
       
       So, simple LP on Cartesian coordinates does NOT work because the area function is bilinear (quadratic), not linear.
       
       **Correction**:
       The strategy "Formulate the problem as a sequence of linear programs" in the prompt might be a trap or I need to be clever.
       
       Wait, look at inspiration 1/2/3. They all use NLP.
       The directive says: "Formulate the problem as a sequence of linear programs (LPs) ... leveraging the fact that the objective and constraints are linear in the point coordinates for fixed barycentric representations."
       
       Is it linear in barycentric?
       Let $p_i = \sum_{v=1}^3 \lambda_{iv} V_v$.
       The area of triangle $p_i p_j p_k$ is:
       $\frac{1}{2} | \det(p_i, p_j, p_k) |$
       Since $p_i$ are linear combinations of fixed vertices $V_v$, the determinant is linear in the coefficients $\lambda_{iv}$.
       
       Proof:
       $p_i = \lambda_{i1} V_1 + \lambda_{i2} V_2 + \lambda_{i3} V_3$
       The map $p \to \det(p, q, r)$ is linear in $p$ for fixed $q, r$.
       So $\det(p_i, p_j, p_k)$ is linear in the coordinates of $p_i$, which are linear in $\lambda_{i1}, \lambda_{i2}, \lambda_{i3}$.
       Therefore, the area is a linear function of the barycentric coordinates $\lambda$.
       
       So, we can define variables $z_{iv}$ for $i=1..N, v=1..3$.
       Constraints:
       1. $\sum_v z_{iv} = 1$ for all $i$.
       2. $z_{iv} \ge 0$ for all $i, v$.
       3. Area constraints: $L_{ijk}(z) \ge t$ where $L_{ijk}$ is the linear function of $z$ representing $2 \times \text{Area}$.
       
       This IS a Linear Program!
       
       Variables: $3N + 1 = 34$ variables.
       $z_{11}, z_{12}, z_{13}, z_{21}, ..., z_{11,3}, t$.
       
       We need to construct the matrix $A$ for the constraints.
       
       Let's implement this.
    """
    start_time = time.time()
    if time.time() - start_time > 0.1: # Quick check, though this is called per start
        pass
        
    # Determine orientation signs from initial points
    # Calculate 2*Area for each triplet
    p_i = pts_init[_triplets[:, 0]]
    p_j = pts_init[_triplets[:, 1]]
    p_k = pts_init[_triplets[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    # Sign is sign of cross. If cross is 0, pick arbitrary (say +1)
    signs = np.sign(cross)
    signs[signs == 0] = 1.0
    
    # Construct LP
    # Variables:
    # 0 to 3N-1: barycentric coords z_{i,v}
    # 3N: t
    
    n_vars = 3 * N + 1
    t_idx = 3 * N
    
    # Objective: Minimize -t
    c = np.zeros(n_vars)
    c[t_idx] = -1.0
    
    # Inequality constraints: A_ub x <= b_ub
    # We want: L_{ijk}(z) * sign_{ijk} - t >= 0
    # => -L_{ijk}(z) * sign_{ijk} + t <= 0
    
    # L_{ijk}(z) is the signed double area.
    # $2 \text{Area} = \det(p_i, p_j, p_k)$
    # $p_i = \sum_v z_{iv} V_v$
    # $\det(\sum_v z_{iv} V_v, p_j, p_k) = \sum_v z_{iv} \det(V_v, p_j, p_k)$
    # This is linear in $z_{i1}, z_{i2}, z_{i3}$.
    
    # We need $\det(V_v, p_j, p_k)$ for $v=1,2,3$.
    # But $p_j$ and $p_k$ also depend on $z$.
    # $\det(p_i, p_j, p_k)$ is multilinear.
    # $\det(\sum_v z_{iv} V_v, \sum_w z_{jw} V_w, \sum_u z_{ku} V_u)$
    # $= \sum_{v,w,u} z_{iv} z_{jw} z_{ku} \det(V_v, V_w, V_u)$
    
    # This is CUBIC in z!
    # $\det(V_1, V_2, V_3)$ is a constant.
    # The term is $z_{i1} z_{j2} z_{k3} \det(V_1, V_2, V_3) + \dots$
    
    # So the area is NOT linear in the barycentric coordinates either, unless we fix two points.
    # It is a degree 3 polynomial in the barycentric coordinates.
    
    # So the LP approach on barycentric coords is invalid for the full problem.
    
    # However, we can use a **Sequence of LPs** (Sequential Linear Programming).
    # 1. Fix all points except $p_k$.
    # 2. The area of any triangle involving $p_k$ is linear in the coordinates of $p_k$.
    # 3. The area of any triangle NOT involving $p_k$ is constant.
    # 4. Maximize $t$ subject to:
    #    - $Area(p_i, p_j, p_k) \ge t$ (linear in $p_k$)
    #    - $Area(p_i, p_j, p_l) \ge t$ (constant, must be satisfied)
    #    - $p_k$ inside triangle.
    # 5. Update $p_k$.
    # 6. Repeat for all $k$.
    
    # This is effectively a coordinate descent method where each step is an LP.
    # Since the subproblems are LPs, they can be solved exactly and fast.
    
    # Let's implement this "Coordinate Descent with LP steps".
    
    pts = np.array(pts_init, dtype=float)
    
    # Pre-compute constants for the LPs
    # Vertices of the big triangle
    V = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H]
    ])
    
    # For each point k, we need to solve:
    # Maximize t
    # Subject to:
    # 1. For all (i, j, k): $2 Area(p_i, p_j, p_k) \ge t$
    #    $2 Area(p_i, p_j, p_k) = \det(p_i, p_j, p_k) = \det(p_i, p_j, p_k)$
    #    Let $p_i = (x_i, y_i)$, $p_j = (x_j, y_j)$, $p_k = (x_k, y_k)$.
    #    $2 A = x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)$
    #    $= x_i y_j - x_i y_k + x_j y_k - x_j y_i + x_k y_i - x_k y_j$
    #    $= (x_i y_j - x_j y_i) + y_k(x_j - x_i) + x_k(y_i - y_j)$
    #    
    #    Let $C_{ij} = x_i y_j - x_j y_i$ (constant for fixed i, j)
    #    Let $A_{ij} = y_j - y_i$
    #    Let $B_{ij} = x_j - x_i$
    #    
    #    $2 A_{ijk} = C_{ij} + B_{ij} y_k + A_{ij} x_k$
    #    
    #    So the constraint is:
    #    $B_{ij} y_k + A_{ij} x_k + C_{ij} \ge t$
    #    
    #    This is linear in $x_k, y_k, t$.
    
    # 2. For all (i, j, l) where $l \neq k$:
    #    $2 A_{ijl} \ge t$
    #    This is a constraint on $t$ only: $t \le 2 A_{ijl}$ (constant)
    
    # 3. $p_k$ inside triangle:
    #    $x_k \ge 0, y_k \ge 0$
    #    $y_k \le \sqrt{3} x_k$
    #    $y_k \le \sqrt{3} (1 - x_k)$
    
    # Variables for sub-LP: $x_k, y_k, t$ (3 variables)
    
    # We will iterate through all points k, solving the LP for each.
    
    max_iter = 20 # Number of full sweeps
    tol = 1e-6
    
    for _sweep in range(max_iter):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        prev_min_area = _min_triangle_area(pts)
        
        for k in range(N):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Build LP for point k
            # Variables: x_k, y_k, t
            # Indices: 0: x_k, 1: y_k, 2: t
            
            c_sub = np.array([0.0, 0.0, -1.0]) # Minimize -t
            
            A_ub = []
            b_ub = []
            
            # Constraints from triangles involving k
            # (i, j, k)
            for i in range(N):
                if i == k: continue
                for j in range(i + 1, N):
                    if j == k: continue
                    
                    # Calculate coefficients for x_k, y_k
                    xi, yi = pts[i]
                    xj, yj = pts[j]
                    
                    C = xi * yj - xj * yi
                    A_coeff = yj - yi # Coeff of x_k
                    B_coeff = xj - xi # Coeff of y_k
                    
                    # Constraint: A_coeff * x_k + B_coeff * y_k + C >= t
                    # => A_coeff * x_k + B_coeff * y_k - t >= -C
                    # => -A_coeff * x_k - B_coeff * y_k + t <= C  (for linprog <=)
                    
                    A_ub.append([-A_coeff, -B_coeff, 1.0])
                    b_ub.append(C)
                    
            # Constraints from triangles NOT involving k
            # (i, j, l) with i,j,l != k
            min_area_others = float('inf')
            for i in range(N):
                if i == k: continue
                for j in range(i + 1, N):
                    if j == k: continue
                    for l in range(j + 1, N):
                        if l == k: continue
                        
                        xi, yi = pts[i]
                        xj, yj = pts[j]
                        xl, yl = pts[l]
                        
                        # 2 * Area
                        area2 = abs(xi*(yj - yl) + xj*(yl - yi) + xl*(yi - yj))
                        if area2 < min_area_others:
                            min_area_others = area2
                        
            # Constraint: t <= min_area_others
            # => t - min_area_others <= 0
            # => 0*x + 0*y + 1*t <= min_area_others
            if min_area_others < float('inf'):
                A_ub.append([0.0, 0.0, 1.0])
                b_ub.append(min_area_others)
                
            # Triangle boundary constraints for (x_k, y_k)
            # x_k >= 0 => -x_k <= 0
            A_ub.append([-1.0, 0.0, 0.0])
            b_ub.append(0.0)
            
            # y_k >= 0 => -y_k <= 0
            A_ub.append([0.0, -1.0, 0.0])
            b_ub.append(0.0)
            
            # y_k <= sqrt(3) x_k => -sqrt(3) x_k + y_k <= 0
            A_ub.append([-np.sqrt(3), 1.0, 0.0])
            b_ub.append(0.0)
            
            # y_k <= sqrt(3) (1 - x_k) => sqrt(3) x_k + y_k <= sqrt(3)
            A_ub.append([np.sqrt(3), 1.0, 0.0])
            b_ub.append(np.sqrt(3))
            
            # Bounds for x_k, y_k (redundant with inequalities but good for solver)
            # t can be negative? No, areas are positive. But let's allow wide bounds.
            bounds = [(0, 1), (0, S3H), (0, None)]
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            try:
                res = linprog(c_sub, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    x_new = res.x[0]
                    y_new = res.x[1]
                    # t_new = res.x[2]
                    
                    # Update point k
                    pts[k] = [x_new, y_new]
                    
            except Exception:
                pass
        
        # Check convergence
        curr_min_area = _min_triangle_area(pts)
        if abs(curr_min_area - prev_min_area) < tol:
            # Optional: break if no improvement
            pass
            
    return pts

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Coordinate Descent with Linear Programming steps.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    initializations = []
    
    # 1. Structured (Grid-like)
    p_struct = _structured_init()
    initializations.append(p_struct)
    
    # 2. Greedy
    try:
        p_greedy = _greedy_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
        
    # 3. Random starts
    for _ in range(5):
        p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        p_rand = _project_to_triangle(p_rand)
        initializations.append(p_rand)
        
    # 4. Perturbed structured
    p_struct_pert = _structured_init() + 0.1 * np.random.normal(size=(N, 2))
    p_struct_pert = _project_to_triangle(p_struct_pert)
    initializations.append(p_struct_pert)
    
    for s, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        try:
            # Run the CD-LP optimization
            opt_pts = _solve_lp(init_pts)
            opt_pts = _project_to_triangle(opt_pts)
            
            if _check_inside(opt_pts):
                score = _min_triangle_area(opt_pts)
                if score > best_score:
                    best_score = score
                    best_pts = opt_pts.copy()
                    
        except Exception as e:
            # Fallback to initial if LP fails
            if _check_inside(init_pts):
                score = _min_triangle_area(init_pts)
                if score > best_score:
                    best_score = score
                    best_pts = init_pts.copy()
                    
    # If no optimization succeeded, fall back to the best initial guess
    if best_pts is None:
        for init_pts in initializations:
            if _check_inside(init_pts):
                score = _min_triangle_area(init_pts)
                if score > best_score:
                    best_score = score
                    best_pts = init_pts.copy()
                    
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
