# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from scipy.optimize import linprog

def squares_in_square10() -> tuple:
    """
    Pack 10 axis-aligned squares in the unit square to maximize sum of side lengths.
    
    Combined Strategy:
    1. Generate candidate topologies (combinatorial assignments of relative positions).
    2. For each topology, solve a Linear Program (LP) to find the optimal geometry.
    3. Use the best LP solution as a seed for a local refinement phase (push-apart + grow)
       to handle numerical slack or potential local improvements near the LP optimum.
    4. Also run a few pure heuristic starts (push-apart based) for robustness.
    
    This leverages the global optimality of LP for fixed topologies (from inspiration)
    and the robust local search machinery of the current program.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    tol = 1e-7
    
    def check_valid(centers, sides):
        """Check if configuration is valid. Returns (valid, violation_amount)."""
        if np.any(sides <= 0):
            return False, 1.0
        if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(sides)):
            return False, 1.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False, 1.0
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False, 1.0
        max_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if t <= 0:
                    continue
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                if dx < t and dy < t:
                    overlap = min(t - dx, t - dy)
                    max_overlap = max(max_overlap, overlap)
        if max_overlap > 0:
            return False, max_overlap
        return True, 0.0

    def score(centers, sides):
        """Return sum of sides if valid, else -inf."""
        valid, _ = check_valid(centers, sides)
        if valid:
            return np.sum(sides)
        return -np.inf

    def push_apart(centers, sides, iterations=200, strength=1.0):
        """
        Force-based relaxation: push overlapping squares apart and keep them in bounds.
        """
        centers = centers.copy().astype(np.float64)
        sides = sides.copy().astype(np.float64)
        
        for it in range(iterations):
            if time.time() > deadline:
                break
                
            forces = np.zeros_like(centers)
            
            # Overlap forces
            for i in range(n):
                for j in range(i+1, n):
                    t = (sides[i] + sides[j]) / 2
                    dx = centers[i,0] - centers[j,0]
                    dy = centers[i,1] - centers[j,1]
                    
                    if abs(dx) < t and abs(dy) < t:
                        # Overlapping: push apart along axis of least penetration
                        pen_x = t - abs(dx)
                        pen_y = t - abs(dy)
                        
                        if pen_x < pen_y:
                            # Push along x
                            if dx >= 0:
                                forces[i, 0] += strength * pen_x
                                forces[j, 0] -= strength * pen_x
                            else:
                                forces[i, 0] -= strength * pen_x
                                forces[j, 0] += strength * pen_x
                        else:
                            # Push along y
                            if dy >= 0:
                                forces[i, 1] += strength * pen_y
                                forces[j, 1] -= strength * pen_y
                            else:
                                forces[i, 1] -= strength * pen_y
                                forces[j, 1] += strength * pen_y
            
            # Boundary forces
            for i in range(n):
                s = sides[i]
                half = s / 2
                # Left boundary
                if centers[i,0] - half < 0:
                    forces[i, 0] += strength * (half - centers[i,0])
                # Right boundary
                if centers[i,0] + half > 1:
                    forces[i, 0] -= strength * (centers[i,0] + half - 1)
                # Bottom boundary
                if centers[i,1] - half < 0:
                    forces[i, 1] += strength * (half - centers[i,1])
                # Top boundary
                if centers[i,1] + half > 1:
                    forces[i, 1] -= strength * (centers[i,1] + half - 1)
            
            # Apply forces with damping
            damping = 0.5
            centers += damping * forces
            
            # Project to feasible region (hard constraints)
            for i in range(n):
                s = sides[i]
                half = s / 2
                centers[i, 0] = np.clip(centers[i, 0], half, 1 - half)
                centers[i, 1] = np.clip(centers[i, 1], half, 1 - half)
        
        return centers, sides

    def grow_squares(centers, sides, max_iter=500):
        """
        Try to grow squares while maintaining non-overlap.
        Greedy: at each step, pick the square with most room to grow and grow it.
        """
        centers = centers.copy().astype(np.float64)
        sides = sides.copy().astype(np.float64)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            best_i = -1
            best_growth = 0.0
            
            for i in range(n):
                # How much can we grow square i?
                # Limited by boundaries
                room_left = centers[i,0] - sides[i]/2
                room_right = 1 - centers[i,0] - sides[i]/2
                room_bottom = centers[i,1] - sides[i]/2
                room_top = 1 - centers[i,1] - sides[i]/2
                boundary_limit = min(room_left, room_right, room_bottom, room_top)
                
                # Limited by other squares
                other_limit = np.inf
                for j in range(n):
                    if i == j:
                        continue
                    t_current = (sides[i] + sides[j]) / 2
                    dx = abs(centers[i,0] - centers[j,0])
                    dy = abs(centers[i,1] - centers[j,1])
                    
                    # If already separated in x
                    if dx >= t_current:
                        limit_x = (2*dx - sides[i] - sides[j]) / 2
                        other_limit = min(other_limit, max(0, limit_x))
                    # If already separated in y
                    if dy >= t_current:
                        limit_y = (2*dy - sides[i] - sides[j]) / 2
                        other_limit = min(other_limit, max(0, limit_y))
                    # If overlapping in both, can't grow
                    else:
                        other_limit = 0
                        break
                
                total_limit = min(boundary_limit, other_limit)
                if total_limit > best_growth:
                    best_growth = total_limit
                    best_i = i
            
            if best_i < 0 or best_growth < 1e-8:
                break
            
            # Grow by a fraction of the limit
            growth = best_growth * 0.5
            sides[best_i] += growth
        
        return centers, sides

    def local_refine(centers, sides, max_iter=1000):
        """
        Local refinement: try small perturbations to improve sum of sides.
        """
        centers = centers.copy().astype(np.float64)
        sides = sides.copy().astype(np.float64)
        current_score = np.sum(sides)
        
        rng = np.random.default_rng(42)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try moving each square slightly
            for i in range(n):
                for _ in range(3):
                    dx = rng.normal(0, 0.005)
                    dy = rng.normal(0, 0.005)
                    ds = rng.normal(0, 0.003)
                    
                    new_centers = centers.copy()
                    new_sides = sides.copy()
                    new_centers[i, 0] += dx
                    new_centers[i, 1] += dy
                    new_sides[i] += ds
                    new_sides[i] = max(new_sides[i], 0.001)
                    
                    # Project to bounds
                    for j in range(n):
                        h = new_sides[j] / 2
                        new_centers[j, 0] = np.clip(new_centers[j, 0], h, 1-h)
                        new_centers[j, 1] = np.clip(new_centers[j, 1], h, 1-h)
                    
                    valid, _ = check_valid(new_centers, new_sides)
                    if valid:
                        new_score = np.sum(new_sides)
                        if new_score > current_score + 1e-10:
                            centers = new_centers
                            sides = new_sides
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            
            # Also try growing all squares slightly
            test_sides = sides * 1.001
            valid, _ = check_valid(centers, test_sides)
            if valid:
                sides = test_sides
                current_score = np.sum(sides)
        
        return centers, sides, current_score

    def solve_topology(x_order, y_order):
        """
        Solve LP for a given topology.
        x_order: permutation of indices representing the order of squares by their x-center?
        Actually, for LP, we need to define which side of the pair is 'separating'.
        
        Let's use a simpler heuristic for topology generation:
        We will define the topology by a set of binary choices for each pair (i,j).
        Choice 0: i is left of j
        Choice 1: i is right of j
        Choice 2: i is below j
        Choice 3: i is above j
        
        For efficiency, we will generate random topologies and solve LP.
        """
        num_vars = 3 * n
        
        # Objective: maximize sum(s_i) -> minimize -sum(s_i)
        c = np.zeros(num_vars)
        for i in range(n):
            c[3*i + 2] = -1.0
            
        A_list = []
        b_list = []
        
        # 1. Bounds
        for i in range(n):
            # -x_i + 0.5 s_i <= 0
            row = np.zeros(num_vars)
            row[3*i] = -1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(0.0)
            
            # x_i + 0.5 s_i <= 1
            row = np.zeros(num_vars)
            row[3*i] = 1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(1.0)
            
            # -y_i + 0.5 s_i <= 0
            row = np.zeros(num_vars)
            row[3*i+1] = -1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(0.0)
            
            # y_i + 0.5 s_i <= 1
            row = np.zeros(num_vars)
            row[3*i+1] = 1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(1.0)
            
        # 2. Non-overlap constraints based on topology
        # We need to pass in the topology choices
        # Let's assume the function receives a list of choices for each pair
        pass # Placeholder, actual implementation below

    def solve_topology_with_choices(choices):
        """
        choices: list of length n*(n-1)/2, each value 0..3
        """
        num_vars = 3 * n
        
        c = np.zeros(num_vars)
        for i in range(n):
            c[3*i + 2] = -1.0
            
        A_list = []
        b_list = []
        
        # 1. Bounds
        for i in range(n):
            # -x_i + 0.5 s_i <= 0
            row = np.zeros(num_vars)
            row[3*i] = -1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(0.0)
            
            # x_i + 0.5 s_i <= 1
            row = np.zeros(num_vars)
            row[3*i] = 1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(1.0)
            
            # -y_i + 0.5 s_i <= 0
            row = np.zeros(num_vars)
            row[3*i+1] = -1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(0.0)
            
            # y_i + 0.5 s_i <= 1
            row = np.zeros(num_vars)
            row[3*i+1] = 1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(1.0)
            
        # 2. Non-overlap constraints
        pair_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                dir = choices[pair_idx]
                row = np.zeros(num_vars)
                if dir == 0: # i left of j
                    row[3*i] = 1.0
                    row[3*i+2] = 0.5
                    row[3*j] = -1.0
                    row[3*j+2] = 0.5
                elif dir == 1: # i right of j
                    row[3*i] = -1.0
                    row[3*i+2] = 0.5
                    row[3*j] = 1.0
                    row[3*j+2] = -0.5
                elif dir == 2: # i below j
                    row[3*i+1] = 1.0
                    row[3*i+2] = 0.5
                    row[3*j+1] = -1.0
                    row[3*j+2] = 0.5
                else: # dir == 3, i above j
                    row[3*i+1] = -1.0
                    row[3*i+2] = 0.5
                    row[3*j+1] = 1.0
                    row[3*j+2] = -0.5
                    
                A_list.append(row)
                b_list.append(0.0)
                pair_idx += 1
                
        A_ub = np.array(A_list)
        b_ub = np.array(b_list)
        
        bounds = []
        for i in range(n):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((1e-6, 1)) # s
            
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                x = res.x
                centers = np.zeros((n, 2))
                sides = np.zeros(n)
                for i in range(n):
                    centers[i, 0] = x[3*i]
                    centers[i, 1] = x[3*i+1]
                    sides[i] = x[3*i+2]
                
                if check_valid(centers, sides)[0]:
                    return centers, sides
        except Exception:
            pass
            
        return None, None

    def generate_topology_heuristic(base_centers, base_sides):
        """
        Generate a topology choice based on a feasible configuration.
        For each pair, pick the axis with the largest separation relative to the sum of radii.
        """
        choices = []
        for i in range(n):
            for j in range(i+1, n):
                dx = base_centers[i,0] - base_centers[j,0]
                dy = base_centers[i,1] - base_centers[j,1]
                t = (base_sides[i] + base_sides[j]) / 2
                
                # If separated in x
                if abs(dx) > abs(dy):
                    if dx > 0:
                        choices.append(0) # i left of j
                    else:
                        choices.append(1) # i right of j
                else:
                    if dy > 0:
                        choices.append(3) # i above j
                    else:
                        choices.append(2) # i below j
        return choices

    def init_2_large_8_small():
        """Two large squares (side ~0.4) and 8 small squares."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        s_large = 0.45
        centers[0] = [s_large/2, 0.5]
        centers[1] = [1 - s_large/2, 0.5]
        sides[0] = s_large
        sides[1] = s_large
        
        s_small = 0.12
        for i in range(4):
            centers[2+i] = [0.125 + i*0.25, 0.9]
            sides[2+i] = s_small
        for i in range(4):
            centers[6+i] = [0.125 + i*0.25, 0.1]
            sides[6+i] = s_small
        
        return centers, sides

    def init_3_3_4():
        """3 large, 3 medium, 4 small."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        s_large = 0.32
        for i in range(3):
            centers[i] = [(i+0.5)*s_large, 0.5]
            sides[i] = s_large
        
        s_med = 0.18
        for i in range(3):
            centers[3+i] = [(i+0.5)*0.33, 0.85]
            sides[3+i] = s_med
        
        s_small = 0.12
        for i in range(4):
            centers[6+i] = [(i+0.5)*0.25, 0.1]
            sides[6+i] = s_small
        
        return centers, sides

    def init_corner_cluster():
        """Cluster of squares in corners."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        s_corner = 0.25
        centers[0] = [s_corner/2, s_corner/2]
        centers[1] = [1-s_corner/2, s_corner/2]
        centers[2] = [s_corner/2, 1-s_corner/2]
        centers[3] = [1-s_corner/2, 1-s_corner/2]
        for i in range(4):
            sides[i] = s_corner
        
        s_edge = 0.15
        centers[4] = [0.5, s_edge/2]
        centers[5] = [0.5, 1-s_edge/2]
        centers[6] = [s_edge/2, 0.5]
        centers[7] = [1-s_edge/2, 0.5]
        for i in range(4, 8):
            sides[i] = s_edge
        
        s_center = 0.12
        centers[8] = [0.45, 0.5]
        centers[9] = [0.55, 0.5]
        for i in range(8, 10):
            sides[i] = s_center
        
        return centers, sides

    def init_staggered():
        """Staggered arrangement."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        col_width = 1.0 / 5
        for i in range(n):
            col = i % 5
            row = i // 5
            s = 0.18
            cx = (col + 0.5) * col_width
            if row == 1:
                cx += col_width * 0.2
            cy = 0.25 if row == 0 else 0.75
            centers[i] = [cx, cy]
            sides[i] = s
        
        return centers, sides

    def init_random_valid(seed):
        """Random valid configuration."""
        rng = np.random.default_rng(seed)
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        for i in range(n):
            placed = False
            for _ in range(200):
                s = rng.uniform(0.08, 0.22)
                cx = rng.uniform(s/2, 1-s/2)
                cy = rng.uniform(s/2, 1-s/2)
                
                valid = True
                for j in range(i):
                    t = (s + sides[j]) / 2
                    if abs(cx - centers[j,0]) < t and abs(cy - centers[j,1]) < t:
                        valid = False
                        break
                
                if valid:
                    centers[i] = [cx, cy]
                    sides[i] = s
                    placed = True
                    break
            
            if not placed:
                s = 0.05
                cx = s/2 + 0.01
                cy = s/2 + 0.01
                centers[i] = [cx, cy]
                sides[i] = s
        
        return centers, sides

    # Generate initializations
    inits = [
        init_2_large_8_small(),
        init_3_3_4(),
        init_corner_cluster(),
        init_staggered(),
    ]
    for seed in [1, 2, 3, 4, 5]:
        inits.append(init_random_valid(seed))
    
    random.shuffle(inits)

    # Phase 1: Heuristic Search (Push-Apart + Grow + Refine)
    for idx, (c_init, s_init) in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Phase 1: Push apart to resolve overlaps
        c, s = push_apart(c_init, s_init, iterations=300, strength=0.8)
        
        # Phase 2: Grow squares
        c, s = grow_squares(c, s, max_iter=300)
        
        # Phase 3: Local refinement
        c, s, sc = local_refine(c, s, max_iter=500)
        
        valid, _ = check_valid(c, s)
        if valid:
            sc = np.sum(s)
            if sc > best_score:
                best_score = sc
                best_centers = c.copy()
                best_sides = s.copy()
                print(f"  Heuristic Init {idx}: score={sc:.6f}", file=sys.stderr)
        else:
            for scale in [0.999, 0.995, 0.99, 0.98, 0.95, 0.9]:
                test_s = s * scale
                valid2, _ = check_valid(c, test_s)
                if valid2:
                    sc2 = np.sum(test_s)
                    if sc2 > best_score:
                        best_score = sc2
                        best_centers = c.copy()
                        best_sides = test_s.copy()
                    break

    # Phase 2: Topology-Guided LP Search
    # We will try to solve LPs for topologies derived from the best heuristic solutions
    # and also for random topologies.
    
    # First, use the best heuristic solution to define a topology
    if best_centers is not None:
        choices = generate_topology_heuristic(best_centers, best_sides)
        c_lp, s_lp = solve_topology_with_choices(choices)
        if c_lp is not None:
            sc_lp = np.sum(s_lp)
            if sc_lp > best_score:
                best_score = sc_lp
                best_centers = c_lp.copy()
                best_sides = s_lp.copy()
                print(f"  LP from Best Heuristic: score={sc_lp:.6f}", file=sys.stderr)

    # Try a few random topologies
    num_lp_trials = 20
    for trial in range(num_lp_trials):
        if time.time() > deadline:
            break
            
        # Generate random topology
        choices = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
        
        # Sometimes bias towards "grid-like" topologies? 
        # For now, just random.
        
        c_lp, s_lp = solve_topology_with_choices(choices)
        if c_lp is not None:
            sc_lp = np.sum(s_lp)
            if sc_lp > best_score:
                best_score = sc_lp
                best_centers = c_lp.copy()
                best_sides = s_lp.copy()
                print(f"  Random LP Trial {trial}: score={sc_lp:.6f}", file=sys.stderr)
                
                # Refine the LP solution with local search
                c_ref, s_ref, sc_ref = local_refine(c_lp, s_lp, max_iter=200)
                valid, _ = check_valid(c_ref, s_ref)
                if valid:
                    if sc_ref > best_score:
                        best_score = sc_ref
                        best_centers = c_ref.copy()
                        best_sides = s_ref.copy()
                        print(f"  Refined LP Trial {trial}: score={sc_ref:.6f}", file=sys.stderr)

    # Final fallback
    if best_centers is None:
        side = 1.0 / 4.0
        centers = np.zeros((n, 2))
        sides = np.full(n, side)
        for i in range(n):
            col = i % 4
            row = i // 4
            centers[i] = [(col+0.5)*side, (row+0.5)*side]
        best_centers = centers
        best_sides = sides

    return best_centers, best_sides

# EVOLVE-BLOCK-END
