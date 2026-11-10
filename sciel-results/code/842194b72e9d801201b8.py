# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from scipy.optimize import linprog

def squares_in_square10() -> tuple:
    """
    Pack 10 axis-aligned squares in the unit square to maximize sum of side lengths.
    
    Strategy: Branch-and-bound over rectangle partitions with area-based upper bounds,
    followed by LP polishing of the best candidate topologies.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 575.0
    tol = 1e-7
    
    def check_valid(centers, sides):
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

    def solve_lp_for_topology(choices):
        """Solve LP given topology choices (0=left,1=right,2=below,3=above for each pair)."""
        num_vars = 3 * n
        c = np.zeros(num_vars)
        for i in range(n):
            c[3*i + 2] = -1.0
            
        A_list = []
        b_list = []
        
        for i in range(n):
            row = np.zeros(num_vars)
            row[3*i] = -1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(0.0)
            
            row = np.zeros(num_vars)
            row[3*i] = 1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(1.0)
            
            row = np.zeros(num_vars)
            row[3*i+1] = -1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(0.0)
            
            row = np.zeros(num_vars)
            row[3*i+1] = 1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(1.0)
        
        pair_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                dir = choices[pair_idx]
                row = np.zeros(num_vars)
                if dir == 0:
                    row[3*i] = 1.0
                    row[3*i+2] = 0.5
                    row[3*j] = -1.0
                    row[3*j+2] = 0.5
                elif dir == 1:
                    row[3*i] = -1.0
                    row[3*i+2] = 0.5
                    row[3*j] = 1.0
                    row[3*j+2] = -0.5
                elif dir == 2:
                    row[3*i+1] = 1.0
                    row[3*i+2] = 0.5
                    row[3*j+1] = -1.0
                    row[3*j+2] = 0.5
                else:
                    row[3*i+1] = -1.0
                    row[3*i+2] = 0.5
                    row[3*j+1] = 1.0
                    row[3*j+2] = -0.5
                A_list.append(row)
                b_list.append(0.0)
                pair_idx += 1
                
        A_ub = np.array(A_list)
        b_ub = np.array(b_list)
        bounds = [(0, 1), (0, 1), (1e-6, 1)] * n
        
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

    def area_upper_bound(rectangles, remaining):
        """
        Upper bound on sum of sides: each square's side <= sqrt(area of its rectangle).
        Sum of sqrt(areas) is an upper bound.
        """
        total = 0.0
        for i in range(remaining):
            # The i-th remaining square gets the i-th largest rectangle area
            total += np.sqrt(rectangles[i])
        return total

    def extract_topology(centers, sides):
        """Extract topology from a valid configuration."""
        choices = []
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                t = (sides[i] + sides[j]) / 2
                if abs(dx) >= t and abs(dx) > abs(dy):
                    choices.append(0 if dx > 0 else 1)
                elif abs(dy) >= t and abs(dy) > abs(dx):
                    choices.append(2 if dy > 0 else 3)
                elif abs(dx) >= t:
                    choices.append(0 if dx > 0 else 1)
                elif abs(dy) >= t:
                    choices.append(2 if dy > 0 else 3)
                else:
                    # Fallback: use x
                    choices.append(0 if dx > 0 else 1)
        return choices

    def local_refine(centers, sides, max_iter=300):
        """Local refinement via small perturbations."""
        centers = centers.copy().astype(np.float64)
        sides = sides.copy().astype(np.float64)
        current_score = np.sum(sides)
        rng = np.random.default_rng(123)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                for _ in range(2):
                    dx = rng.normal(0, 0.003)
                    dy = rng.normal(0, 0.003)
                    ds = rng.normal(0, 0.002)
                    new_c = centers.copy()
                    new_s = sides.copy()
                    new_c[i, 0] += dx
                    new_c[i, 1] += dy
                    new_s[i] = max(new_s[i] + ds, 0.001)
                    for j in range(n):
                        h = new_s[j] / 2
                        new_c[j, 0] = np.clip(new_c[j, 0], h, 1-h)
                        new_c[j, 1] = np.clip(new_c[j, 1], h, 1-h)
                    valid, _ = check_valid(new_c, new_s)
                    if valid:
                        ns = np.sum(new_s)
                        if ns > current_score + 1e-10:
                            centers = new_c
                            sides = new_s
                            current_score = ns
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                # Try uniform growth
                test_s = sides * 1.0005
                valid, _ = check_valid(centers, test_s)
                if valid:
                    sides = test_s
                    current_score = np.sum(sides)
                    improved = True
            if not improved and it > 50:
                break
        return centers, sides, current_score

    # Initial configurations
    def init_grid():
        centers = np.zeros((n, 2))
        sides = np.full(n, 0.25)
        for i in range(n):
            col = i % 4
            row = i // 4
            centers[i] = [(col+0.5)*0.25, (row+0.5)*0.25]
        return centers, sides

    def init_2_8():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        s_large = 0.45
        centers[0] = [s_large/2, 0.5]
        centers[1] = [1 - s_large/2, 0.5]
        sides[0] = s_large
        sides[1] = s_large
        s_small = 0.12
        for i in range(4):
            centers[2+i] = [0.125 + i*0.25, 0.92]
            sides[2+i] = s_small
        for i in range(4):
            centers[6+i] = [0.125 + i*0.25, 0.08]
            sides[6+i] = s_small
        return centers, sides

    def init_random(seed):
        rng = np.random.default_rng(seed)
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            placed = False
            for _ in range(300):
                s = rng.uniform(0.08, 0.25)
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
                centers[i] = [s/2 + 0.01, s/2 + 0.01]
                sides[i] = s
        return centers, sides

    def push_apart(centers, sides, iterations=200):
        centers = centers.copy().astype(np.float64)
        sides = sides.copy().astype(np.float64)
        for it in range(iterations):
            if time.time() > deadline:
                break
            forces = np.zeros_like(centers)
            for i in range(n):
                for j in range(i+1, n):
                    t = (sides[i] + sides[j]) / 2
                    dx = centers[i,0] - centers[j,0]
                    dy = centers[i,1] - centers[j,1]
                    if abs(dx) < t and abs(dy) < t:
                        pen_x = t - abs(dx)
                        pen_y = t - abs(dy)
                        if pen_x < pen_y:
                            if dx >= 0:
                                forces[i, 0] += pen_x
                                forces[j, 0] -= pen_x
                            else:
                                forces[i, 0] -= pen_x
                                forces[j, 0] += pen_x
                        else:
                            if dy >= 0:
                                forces[i, 1] += pen_y
                                forces[j, 1] -= pen_y
                            else:
                                forces[i, 1] -= pen_y
                                forces[j, 1] += pen_y
            for i in range(n):
                s = sides[i]
                h = s / 2
                if centers[i,0] - h < 0: forces[i, 0] += (h - centers[i,0])
                if centers[i,0] + h > 1: forces[i, 0] -= (centers[i,0] + h - 1)
                if centers[i,1] - h < 0: forces[i, 1] += (h - centers[i,1])
                if centers[i,1] + h > 1: forces[i, 1] -= (centers[i,1] + h - 1)
            centers += 0.5 * forces
            for i in range(n):
                h = sides[i] / 2
                centers[i, 0] = np.clip(centers[i, 0], h, 1-h)
                centers[i, 1] = np.clip(centers[i, 1], h, 1-h)
        return centers, sides

    # Phase 1: Generate diverse initial solutions
    inits = [
        init_grid(),
        init_2_8(),
        init_random(1),
        init_random(2),
        init_random(3),
        init_random(4),
    ]
    random.shuffle(inits)

    best_centers = None
    best_sides = None
    best_score = -np.inf
    best_topologies = []  # Store topologies from good solutions

    for idx, (c_init, s_init) in enumerate(inits):
        if time.time() > deadline:
            break
        c, s = push_apart(c_init, s_init, iterations=200)
        # Grow
        for _ in range(200):
            if time.time() > deadline:
                break
            best_i = -1
            best_growth = 0.0
            for i in range(n):
                room = min(c[i,0]-s[i]/2, 1-c[i,0]-s[i]/2, c[i,1]-s[i]/2, 1-c[i,1]-s[i]/2)
                for j in range(n):
                    if i == j: continue
                    t = (s[i]+s[j])/2
                    dx = abs(c[i,0]-c[j,0])
                    dy = abs(c[i,1]-c[j,1])
                    if dx >= t:
                        room = min(room, max(0, (2*dx-s[i]-s[j])/2))
                    if dy >= t:
                        room = min(room, max(0, (2*dy-s[i]-s[j])/2))
                    else:
                        room = 0
                        break
                if room > best_growth:
                    best_growth = room
                    best_i = i
            if best_i < 0 or best_growth < 1e-9:
                break
            s[best_i] += best_growth * 0.4
            h = s[best_i]/2
            c[best_i, 0] = np.clip(c[best_i, 0], h, 1-h)
            c[best_i, 1] = np.clip(c[best_i, 1], h, 1-h)
        
        c, s, sc = local_refine(c, s, max_iter=200)
        valid, _ = check_valid(c, s)
        if valid:
            sc = np.sum(s)
            if sc > best_score:
                best_score = sc
                best_centers = c.copy()
                best_sides = s.copy()
                best_topologies.append(extract_topology(c, s))
                print(f"  Init {idx}: score={sc:.6f}", file=sys.stderr)

    # Phase 2: LP polish on collected topologies + random topologies
    if best_topologies:
        for topo in best_topologies:
            if time.time() > deadline:
                break
            c_lp, s_lp = solve_lp_for_topology(topo)
            if c_lp is not None:
                sc_lp = np.sum(s_lp)
                if sc_lp > best_score:
                    best_score = sc_lp
                    best_centers = c_lp.copy()
                    best_sides = s_lp.copy()
                    print(f"  LP polish: score={sc_lp:.6f}", file=sys.stderr)
                    c_ref, s_ref, sc_ref = local_refine(c_lp, s_lp, max_iter=150)
                    valid, _ = check_valid(c_ref, s_ref)
                    if valid and sc_ref > best_score:
                        best_score = sc_ref
                        best_centers = c_ref.copy()
                        best_sides = s_ref.copy()

    # Phase 3: Random topology LP trials
    num_trials = 40
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        choices = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
        c_lp, s_lp = solve_lp_for_topology(choices)
        if c_lp is not None:
            sc_lp = np.sum(s_lp)
            if sc_lp > best_score:
                best_score = sc_lp
                best_centers = c_lp.copy()
                best_sides = s_lp.copy()
                best_topologies.append(extract_topology(c_lp, s_lp))
                print(f"  Random LP {trial}: score={sc_lp:.6f}", file=sys.stderr)

    # Phase 4: Final local refinement
    if best_centers is not None:
        c_final, s_final, sc_final = local_refine(best_centers, best_sides, max_iter=200)
        valid, _ = check_valid(c_final, s_final)
        if valid and sc_final > best_score:
            best_score = sc_final
            best_centers = c_final.copy()
            best_sides = s_final.copy()

    # Fallback
    if best_centers is None:
        side = 0.25
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
