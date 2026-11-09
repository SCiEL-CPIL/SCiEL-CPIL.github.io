# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def squares_in_square10() -> tuple:
    """
    Optimizes the placement of 10 squares in a unit square to maximize the sum of their side lengths.
    Uses Simulated Annealing with heterogeneous shelf-based initializations,
    smooth soft-min penalty, and a high-precision final polish phase.
    """
    n = 10
    start_time = time.time()
    max_time = 590.0  # Safety margin before 600s limit

    def get_bounds(x):
        cx = x[:n]
        cy = x[n:2*n]
        s = x[2*n:3*n]
        return cx, cy, s

    def check_validity(cx, cy, s, tol=1e-7):
        if np.any(cx < -tol) or np.any(cx > 1 + tol):
            return False
        if np.any(cy < -tol) or np.any(cy > 1 + tol):
            return False
        if np.any(s < -tol):
            return False
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                min_dx = (s[i] + s[j]) / 2.0 - tol
                min_dy = (s[i] + s[j]) / 2.0 - tol
                if dx < min_dx and dy < min_dy:
                    return False
        return True

    def smooth_penalty(cx, cy, s, beta=50.0):
        """
        Smooth penalty using soft-min for overlap detection.
        For each pair, the overlap in x is max(0, (s_i+s_j)/2 - |dx|).
        We use a smooth approximation: (1/beta)*log(1 + exp(beta*val))
        The total penalty is the product (in log-space) of x-overlap and y-overlap for each pair,
        which approximates intersection area smoothly.
        """
        penalty = 0.0
        # Boundary violations (smooth)
        # cx - s/2 >= 0  => violation = max(0, s/2 - cx)
        v = s/2 - cx
        penalty += np.sum(np.logaddexp(0, beta * v) / beta)
        v = cx + s/2 - 1
        penalty += np.sum(np.logaddexp(0, beta * v) / beta)
        v = s/2 - cy
        penalty += np.sum(np.logaddexp(0, beta * v) / beta)
        v = cy + s/2 - 1
        penalty += np.sum(np.logaddexp(0, beta * v) / beta)

        # Overlap: for each pair, penalty += smooth_overlap_x * smooth_overlap_y
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                min_d = (s[i] + s[j]) / 2.0
                # Overlap in x: max(0, min_d - dx)
                val_x = min_d - dx
                val_y = min_d - dy
                # Smooth max(0, val)
                smx = np.logaddexp(0, beta * val_x) / beta
                smy = np.logaddexp(0, beta * val_y) / beta
                # Product approximates area of intersection
                penalty += smx * smy
        return penalty

    def objective(x):
        cx, cy, s = get_bounds(x)
        if np.any(s <= 0):
            return 1e10
        penalty = smooth_penalty(cx, cy, s)
        score = np.sum(s)
        # Combined: minimize penalty, maximize score
        return 1e5 * penalty - score

    def generate_heterogeneous_shelf(seed_val):
        """
        Generate a heterogeneous shelf-based initialization.
        The idea: divide the unit square into horizontal shelves.
        Each shelf has a height equal to the largest square in that shelf.
        Within a shelf, squares are placed side by side.
        
        Strategy: Try various shelf configurations.
        """
        np.random.seed(seed_val)
        
        # Configuration 1: 2 shelves, one with 1 large square, one with 9 small
        # Large square: side ~ 0.45, small: side ~ 0.1
        # Shelf 1 height = 0.45, Shelf 2 height = 0.1, total = 0.55 (fits in 1)
        # Actually let's think more carefully.
        
        # Better: 1 large square (side L) + 9 small squares (side s)
        # Place large in one corner, 9 small fill the rest.
        # If large is at bottom-left: [0,L]x[0,L]
        # Remaining: [L,1]x[0,1] and [0,L]x[L,1]
        # [L,1]x[0,1] has width (1-L), height 1
        # [0,L]x[L,1] has width L, height (1-L)
        
        # Try L = 0.4, then remaining areas:
        # Right strip: 0.6 x 1
        # Top strip: 0.4 x 0.6
        # In right strip (0.6 wide, 1 tall): can fit 6 squares of side 0.2 in 2 rows x 3 cols? No, 0.2*3=0.6, 0.2*5=1.0
        # Actually 0.6/0.2 = 3 columns, 1.0/0.2 = 5 rows => 15 squares, but we only need 9
        # Let's use s = 0.15: 0.6/0.15 = 4, 1.0/0.15 = 6.67 => 4x6 = 24, plenty
        # Top strip: 0.4 x 0.6: 0.4/0.15=2.67, 0.6/0.15=4 => 2x4=8
        # Total available: 24+8=32, we need 9. Easy.
        
        # Let's parameterize: L = large side, s = small side
        # We want to maximize L + 9*s subject to packing constraints.
        
        # Simple template: L=0.35, s=0.13
        # Right strip: 0.65 wide. 0.65/0.13 = 5 cols. 1.0/0.13 = 7.7 rows. 5*7=35 slots.
        # Top strip: 0.35 wide, 0.65 tall. 0.35/0.13=2.7, 0.65/0.13=5. 2*5=10 slots.
        # Total 45 slots, need 9. Fine.
        
        # Let's just create a valid configuration and let SA optimize.
        
        configs = []
        
        # Config A: 1 large + 9 small in shelves
        L = 0.35
        s = 0.13
        cx_list = [L/2]
        cy_list = [L/2]
        s_list = [L]
        # Place 9 small squares in the right strip [L,1]x[0,1]
        # 3 columns x 3 rows of small squares
        for row in range(3):
            for col in range(3):
                cx_list.append(L + s/2 + col * s)
                cy_list.append(s/2 + row * s)
                s_list.append(s)
        configs.append((np.array(cx_list), np.array(cy_list), np.array(s_list)))
        
        # Config B: 2 large + 8 small
        L1 = 0.3
        L2 = 0.3
        s = 0.12
        # Place L1 at bottom-left, L2 at bottom-right
        cx_list = [L1/2, 1 - L2/2]
        cy_list = [L1/2, L2/2]
        s_list = [L1, L2]
        # 8 small in top area [0,1]x[max(L1,L2), 1] = [0,1]x[0.3, 1], height 0.7
        # 4 cols x 2 rows
        for row in range(2):
            for col in range(4):
                cx_list.append(s/2 + col * s)
                cy_list.append(0.3 + s/2 + row * s)
                s_list.append(s)
        configs.append((np.array(cx_list), np.array(cy_list), np.array(s_list)))
        
        # Config C: 5 shelves of 2 squares each
        s = 0.18
        cx_list = []
        cy_list = []
        s_list = []
        for row in range(5):
            for col in range(2):
                cx_list.append(s/2 + col * s)
                cy_list.append(s/2 + row * s)
                s_list.append(s)
        configs.append((np.array(cx_list), np.array(cy_list), np.array(s_list)))
        
        # Config D: Random valid-ish
        cx = np.random.rand(n) * 0.8 + 0.1
        cy = np.random.rand(n) * 0.8 + 0.1
        s = np.random.rand(n) * 0.15 + 0.08
        configs.append((cx, cy, s))
        
        # Config E: 3x3 grid of 1/3 + 1 small
        s_big = 0.31
        s_small = 0.1
        cx_list = []
        cy_list = []
        s_list = []
        for row in range(3):
            for col in range(3):
                cx_list.append(s_big/2 + col * s_big)
                cy_list.append(s_big/2 + row * s_big)
                s_list.append(s_big)
        # 10th square in a gap? The 3x3 grid of 0.31 has total 0.93, gap of 0.07.
        # Place small square in center gap area
        cx_list.append(0.5)
        cy_list.append(0.5)
        s_list.append(s_small)
        configs.append((np.array(cx_list), np.array(cy_list), np.array(s_list)))
        
        # Config F: Heterogeneous rows - 1 row of 5 small, 1 row of 5 slightly larger
        s_top = 0.15
        s_bot = 0.19
        cx_list = []
        cy_list = []
        s_list = []
        for col in range(5):
            cx_list.append(s_top/2 + col * s_top)
            cy_list.append(s_top/2)
            s_list.append(s_top)
        for col in range(5):
            cx_list.append(s_bot/2 + col * s_bot)
            cy_list.append(s_top + s_bot/2)
            s_list.append(s_bot)
        configs.append((np.array(cx_list), np.array(cy_list), np.array(s_list)))
        
        # Config G: 2x5 grid with varying sizes
        cx_list = []
        cy_list = []
        s_list = []
        for row in range(2):
            for col in range(5):
                sz = 0.18 + 0.02 * np.random.randn()
                sz = max(0.1, min(0.22, sz))
                cx_list.append(sz/2 + col * 0.2)
                cy_list.append(sz/2 + row * 0.2)
                s_list.append(sz)
        configs.append((np.array(cx_list), np.array(cy_list), np.array(s_list)))
        
        # Pick a config based on seed
        idx = seed_val % len(configs)
        cx, cy, s = configs[idx]
        return np.array([cx, cy, s]).T.flatten()

    best_score = -np.inf
    best_x = None
    
    # Run multiple restarts with different initializations
    num_restarts = 15
    steps_per_restart = 40000
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        x = generate_heterogeneous_shelf(r)
        current_score = objective(x)
        
        # Simulated Annealing with adaptive temperature
        T = 0.5
        T_min = 1e-7
        cooling = 0.9995
        
        # Estimate initial penalty to set T
        cx, cy, s = get_bounds(x)
        init_pen = smooth_penalty(cx, cy, s)
        if init_pen > 0:
            T = max(T, init_pen * 50)
            
        no_improve = 0
        for step in range(steps_per_restart):
            if time.time() - start_time > max_time:
                break
                
            # Propose move - occasionally move multiple params
            if np.random.rand() < 0.1:
                # Move 2-3 params
                num_moves = np.random.randint(2, 4)
                idxs = np.random.choice(3*n, size=num_moves, replace=False)
                x_new = x.copy()
                for idx in idxs:
                    delta = np.random.normal(0, 0.03)
                    x_new[idx] += delta
                    if idx >= 2*n:
                        x_new[idx] = max(0.01, x_new[idx])
                    else:
                        x_new[idx] = np.clip(x_new[idx], 0, 1)
            else:
                idx = np.random.randint(0, 3*n)
                delta = np.random.normal(0, 0.04)
                x_new = x.copy()
                x_new[idx] += delta
                if idx >= 2*n:
                    x_new[idx] = max(0.01, x_new[idx])
                else:
                    x_new[idx] = np.clip(x_new[idx], 0, 1)
                
            new_score = objective(x_new)
            
            # Acceptance
            delta_score = new_score - current_score
            if delta_score < 0 or (T > T_min and np.random.rand() < np.exp(-delta_score / T)):
                x = x_new
                current_score = new_score
                no_improve = 0
                
                # Track best valid
                cx, cy, s = get_bounds(x)
                if check_validity(cx, cy, s):
                    val = np.sum(s)
                    if val > best_score:
                        best_score = val
                        best_x = x.copy()
            else:
                no_improve += 1
                
            T *= cooling
            
    # High-precision final polish
    if best_x is not None:
        cx, cy, s = get_bounds(best_x)
        current_val = np.sum(s)
        
        # Multi-stage polish: progressively smaller steps
        for stage in range(4):
            step_size = 0.01 * (0.5 ** stage)
            iters = 2000 if stage < 3 else 3000
            for _ in range(iters):
                if time.time() - start_time > max_time:
                    break
                idx = np.random.randint(0, 3*n)
                if idx >= 2*n:
                    # Bias towards increasing sides
                    delta = np.random.normal(0.002 * (0.5 ** stage), step_size * 0.5)
                else:
                    delta = np.random.normal(0, step_size * 0.3)
                    
                x_new = best_x.copy()
                x_new[idx] += delta
                if idx >= 2*n:
                    x_new[idx] = max(0.005, x_new[idx])
                else:
                    x_new[idx] = np.clip(x_new[idx], 0.001, 0.999)
                    
                cx_n, cy_n, s_n = get_bounds(x_new)
                if check_validity(cx_n, cy_n, s_n):
                    new_val = np.sum(s_n)
                    if new_val > current_val:
                        best_x = x_new
                        current_val = new_val

    if best_x is None:
        # Fallback
        k = 4
        side = 0.25
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(10, side, dtype=np.float64)
        return centers, sides

    cx, cy, s = get_bounds(best_x)
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    # Final safety check
    if not check_validity(cx, cy, sides):
        k = 4
        side = 0.25
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(10, side, dtype=np.float64)
        
    return centers, sides

# EVOLVE-BLOCK-END
