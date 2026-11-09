# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

_TOL = 1e-7
_N = 10

def _is_feasible(centers, sides, tol=_TOL):
    """Check all constraints: inside unit square, positive sides, no overlap."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _solve_lp(topology):
    """
    Solve the LP for a given topology.
    topology: list of tuples (i, j, direction)
      direction: 0 = i left of j, 1 = i right of j, 2 = i below j, 3 = i above j
    Variables: x[0..9] (cx), x[10..19] (cy), x[20..29] (sides)
    """
    n = _N
    # Objective: maximize sum(sides) -> minimize -sum(sides)
    c = np.zeros(3 * n)
    c[2 * n:3 * n] = -1.0
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints:
    # cx - s/2 >= 0  => -cx + 0.5*s <= 0
    # cx + s/2 <= 1  =>  cx - 0.5*s <= 1
    # cy - s/2 >= 0  => -cy + 0.5*s <= 0
    # cy + s/2 <= 1  =>  cy - 0.5*s <= 1
    
    for i in range(n):
        cx_idx = i
        cy_idx = n + i
        s_idx = 2 * n + i
        
        # -cx + 0.5*s <= 0
        row = np.zeros(3 * n)
        row[cx_idx] = -1.0
        row[s_idx] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # cx - 0.5*s <= 1
        row = np.zeros(3 * n)
        row[cx_idx] = 1.0
        row[s_idx] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # -cy + 0.5*s <= 0
        row = np.zeros(3 * n)
        row[cy_idx] = -1.0
        row[s_idx] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # cy - 0.5*s <= 1
        row = np.zeros(3 * n)
        row[cy_idx] = 1.0
        row[s_idx] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)

    # Pairwise constraints based on topology
    for i, j, direction in topology:
        cx_i, cy_i, s_i = i, n + i, 2 * n + i
        cx_j, cy_j, s_j = j, n + j, 2 * n + j
        
        row = np.zeros(3 * n)
        
        if direction == 0: # i left of j: x_j - s_j/2 >= x_i + s_i/2 => x_i + 0.5*s_i - x_j + 0.5*s_j <= 0
            row[cx_i] = 1.0
            row[s_i] = 0.5
            row[cx_j] = -1.0
            row[s_j] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
        elif direction == 1: # i right of j: x_i - s_i/2 >= x_j + s_j/2 => -x_i + 0.5*s_i + x_j - 0.5*s_j <= 0
            row[cx_i] = -1.0
            row[s_i] = 0.5
            row[cx_j] = 1.0
            row[s_j] = -0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
        elif direction == 2: # i below j: y_j - s_j/2 >= y_i + s_i/2 => y_i + 0.5*s_i - y_j + 0.5*s_j <= 0
            row[cy_i] = 1.0
            row[s_i] = 0.5
            row[cy_j] = -1.0
            row[s_j] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
        elif direction == 3: # i above j: y_i - s_i/2 >= y_j + s_j/2 => -y_i + 0.5*s_i + y_j - 0.5*s_j <= 0
            row[cy_i] = -1.0
            row[s_i] = 0.5
            row[cy_j] = 1.0
            row[s_j] = -0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: s_i > 0 (use small epsilon), cx, cy in [0,1]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((1e-6, 1.0)) # s
    bounds = tuple(bounds)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            centers = np.zeros((n, 2))
            sides = np.zeros(n)
            for i in range(n):
                centers[i, 0] = res.x[i]
                centers[i, 1] = res.x[n + i]
                sides[i] = res.x[2 * n + i]
            # Check feasibility with tolerance
            if _is_feasible(centers, sides):
                return centers, sides, -res.fun
            else:
                # Sometimes LP solution violates due to numerical issues or strict tol
                # Try to repair? Or just reject.
                # Let's try to return it and let the caller decide, but _is_feasible is strict.
                # If it fails, it's likely due to the tol in the check vs the LP constraint (which is strict equality in limit).
                # Actually, LP enforces non-overlap strictly (gap >= 0).
                # The check uses gap >= tol. So LP solution should pass if gap >= 0.
                # If gap is 0, check: abs(dx) < t => 0 < (s_i+s_j)/2 - tol. True. So it fails.
                # We need gap >= tol.
                # We can add tol to the LP constraints.
                # Let's modify LP to include tol.
                pass
        else:
            return None
    except Exception:
        return None
        
    return None

def _solve_lp_with_tol(topology, tol=1e-7):
    """
    Solve LP with tolerance added to separation constraints.
    """
    n = _N
    c = np.zeros(3 * n)
    c[2 * n:3 * n] = -1.0
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints (with tol? No, boundary is [0,1]. Check allows [-tol, 1+tol].
    # So LP should allow [0,1] and we rely on check?
    # If LP gives cx=0, s=0.1, left edge = -0.05. Check: -0.05 < -1e-7. Fail.
    # So we must enforce boundary in LP with tol.
    # cx - s/2 >= -tol => -cx + 0.5*s <= tol
    # cx + s/2 <= 1+tol => cx - 0.5*s <= 1+tol
    
    for i in range(n):
        cx_idx = i
        cy_idx = n + i
        s_idx = 2 * n + i
        
        row = np.zeros(3 * n)
        row[cx_idx] = -1.0
        row[s_idx] = 0.5
        A_ub.append(row)
        b_ub.append(tol)
        
        row = np.zeros(3 * n)
        row[cx_idx] = 1.0
        row[s_idx] = -0.5
        A_ub.append(row)
        b_ub.append(1.0 + tol)
        
        row = np.zeros(3 * n)
        row[cy_idx] = -1.0
        row[s_idx] = 0.5
        A_ub.append(row)
        b_ub.append(tol)
        
        row = np.zeros(3 * n)
        row[cy_idx] = 1.0
        row[s_idx] = -0.5
        A_ub.append(row)
        b_ub.append(1.0 + tol)

    # Pairwise constraints with tol
    # Separation >= tol
    # e.g. i left of j: x_j - s_j/2 >= x_i + s_i/2 + tol
    # => x_i + 0.5*s_i - x_j + 0.5*s_j <= -tol
    
    for i, j, direction in topology:
        cx_i, cy_i, s_i = i, n + i, 2 * n + i
        cx_j, cy_j, s_j = j, n + j, 2 * n + j
        
        row = np.zeros(3 * n)
        
        if direction == 0: # i left of j
            row[cx_i] = 1.0
            row[s_i] = 0.5
            row[cx_j] = -1.0
            row[s_j] = 0.5
            A_ub.append(row)
            b_ub.append(-tol)
            
        elif direction == 1: # i right of j
            row[cx_i] = -1.0
            row[s_i] = 0.5
            row[cx_j] = 1.0
            row[s_j] = -0.5
            A_ub.append(row)
            b_ub.append(-tol)
            
        elif direction == 2: # i below j
            row[cy_i] = 1.0
            row[s_i] = 0.5
            row[cy_j] = -1.0
            row[s_j] = 0.5
            A_ub.append(row)
            b_ub.append(-tol)
            
        elif direction == 3: # i above j
            row[cy_i] = -1.0
            row[s_i] = 0.5
            row[cy_j] = 1.0
            row[s_j] = -0.5
            A_ub.append(row)
            b_ub.append(-tol)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = []
    for i in range(n):
        bounds.append((-tol, 1.0 + tol)) # cx
        bounds.append((-tol, 1.0 + tol)) # cy
        bounds.append((1e-6, 1.0)) # s
    bounds = tuple(bounds)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            centers = np.zeros((n, 2))
            sides = np.zeros(n)
            for i in range(n):
                centers[i, 0] = res.x[i]
                centers[i, 1] = res.x[n + i]
                sides[i] = res.x[2 * n + i]
            if _is_feasible(centers, sides):
                return centers, sides, -res.fun
            else:
                # Debug: print why it failed?
                return None
        else:
            return None
    except Exception:
        return None

def _generate_topology_from_solution(centers, sides):
    """
    Determine the relative positions of squares based on their centers.
    Returns a topology list.
    """
    n = _N
    topology = []
    for i in range(n):
        for j in range(i + 1, n):
            # Determine if i is left/right of j or below/above j
            # We need to pick the one that is "more true" or simply pick one.
            # If they overlap in X, they must be separated in Y.
            # If they don't overlap in X, they can be separated in X.
            
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            
            # Check if they are separated in X
            sep_x = abs(dx) - (sides[i] + sides[j]) / 2
            sep_y = abs(dy) - (sides[i] + sides[j]) / 2
            
            # If sep_x > 0, we can use X separation.
            # If sep_y > 0, we can use Y separation.
            # If both > 0, pick one (e.g. X).
            # If one is < 0, we MUST use the other.
            
            if sep_x >= 0 and sep_y >= 0:
                # Prefer X if dx > 0 else Y? Or just X.
                if dx >= 0:
                    topology.append((i, j, 0)) # i left of j
                else:
                    topology.append((i, j, 1)) # i right of j
            elif sep_x >= 0:
                if dx >= 0:
                    topology.append((i, j, 0))
                else:
                    topology.append((i, j, 1))
            elif sep_y >= 0:
                if dy >= 0:
                    topology.append((i, j, 2)) # i below j
                else:
                    topology.append((i, j, 3)) # i above j
            else:
                # Overlapping? Should not happen if feasible.
                # Pick arbitrary
                topology.append((i, j, 0))
    return topology

def _generate_random_topology():
    """
    Generate a random topology.
    """
    n = _N
    topology = []
    for i in range(n):
        for j in range(i + 1, n):
            direction = np.random.randint(4)
            topology.append((i, j, direction))
    return topology

def _generate_grid_topology():
    """
    Generate a topology based on a grid layout.
    """
    n = _N
    # 4x4 grid, pick 10 positions
    positions = []
    for r in range(4):
        for c in range(4):
            if len(positions) < n:
                positions.append((r, c))
    # Shuffle to make it interesting? No, keep it structured.
    # Assign squares to these grid cells.
    # If two squares are in the same row, they are separated in X.
    # If in the same column, separated in Y.
    # If in different rows and cols, pick X separation if col diff > row diff?
    
    topology = []
    for i in range(n):
        for j in range(i + 1, n):
            r_i, c_i = positions[i]
            r_j, c_j = positions[j]
            
            if c_i < c_j:
                topology.append((i, j, 0)) # i left
            elif c_i > c_j:
                topology.append((i, j, 1)) # i right
            elif r_i < r_j:
                topology.append((i, j, 2)) # i below
            else:
                topology.append((i, j, 3)) # i above
    return topology

def squares_in_square10() -> tuple:
    deadline = time.time() + 580
    
    best_centers = None
    best_sides = None
    best_score = 0.0
    
    # Initial topologies
    topologies = []
    topologies.append(_generate_grid_topology())
    
    # Add a few random ones
    for _ in range(20):
        topologies.append(_generate_random_topology())
        
    # Add some "structured" random ones
    # Sort by x, then y
    idx = np.arange(_N)
    np.random.shuffle(idx)
    # This is just random permutation
    
    # Iterate
    while time.time() < deadline:
        for topo in topologies:
            if time.time() > deadline:
                break
            res = _solve_lp_with_tol(topo)
            if res:
                centers, sides, score = res
                if score > best_score:
                    best_score = score
                    best_centers = centers
                    best_sides = sides
                    # Generate new topologies from this solution
                    new_topo = _generate_topology_from_solution(centers, sides)
                    topologies.append(new_topo)
                    
                    # Also generate perturbations of this topology
                    for _ in range(5):
                        perturbed = []
                        for (i, j, d) in new_topo:
                            if np.random.rand() < 0.1:
                                d = np.random.randint(4)
                            perturbed.append((i, j, d))
                        topologies.append(perturbed)
                        
        # Remove old topologies to keep list size manageable
        if len(topologies) > 100:
            topologies = topologies[-100:]
            
    if best_centers is None:
        # Fallback
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(10, side, dtype=np.float64)
    else:
        best_centers = best_centers.astype(np.float64)
        best_sides = best_sides.astype(np.float64)
        
    return best_centers, best_sides
# EVOLVE-BLOCK-END
