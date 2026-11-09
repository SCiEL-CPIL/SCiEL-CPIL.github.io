# EVOLVE-BLOCK-START
import time
import numpy as np

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize
    the sum of their side lengths, subject to non-overlapping interiors.
    
    Strategy:
    1. Variable representation: centers (cx, cy) and sides (s).
    2. Hard Boundary Clamping: Centers are clamped to [s/2, 1-s/2] at every step,
       ensuring boundary constraints are always satisfied without penalty terms.
    3. Simulated Annealing: Stochastic global search that accepts non-improving moves
       to escape local minima. The energy function penalizes overlaps quadratically.
    4. Multi-start: Multiple random initial configurations to cover different topologies.
    5. Deterministic Polish: A final local search (coordinate descent) to tighten
       the solution and maximize side lengths where possible.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 610  # 20s margin

    def get_energy(centers, sides):
        """
        Calculate the energy (penalty) of the configuration.
        Lower energy is better.
        Energy = sum of overlap violations.
        Overlap violation for pair (i, j):
        If dx < t and dy < t, where t = (s_i + s_j)/2 - tol,
        then violation = (t - dx) * (t - dy) or similar.
        We use a smooth approximation of the overlap area or a squared penalty.
        Actually, the constraint is: NOT (dx < t AND dy < t).
        Violation exists if dx < t AND dy < t.
        Let vx = t - dx, vy = t - dy.
        If vx > 0 and vy > 0, they overlap.
        Penalty = max(0, vx) * max(0, vy) ? This is 0 if they just touch.
        But we want to push them apart.
        Let's use penalty = max(0, vx)^2 + max(0, vy)^2? No, that penalizes separation in one axis even if separated in the other.
        
        Correct logic:
        If they overlap in x (vx > 0) AND in y (vy > 0), it's a violation.
        The "amount" of violation can be measured by how much they need to move to separate.
        Minimal separation distance in x to resolve: vx.
        Minimal separation distance in y to resolve: vy.
        We can penalize min(vx, vy) if both > 0.
        Or simply: penalty = max(0, vx) * max(0, vy). This is 0 if either is <= 0.
        This creates a "valley" where they can slide past each other.
        However, for SA, we want a strong repulsion.
        Let's use: penalty = max(0, vx)^2 + max(0, vy)^2 is wrong because if vx=1, vy=-1, penalty=1 but they don't overlap.
        
        Correct Overlap Penalty:
        If vx > 0 and vy > 0:
            They overlap. The area of overlap is vx * vy.
            We want to minimize this.
            So penalty = max(0, vx) * max(0, vy).
            This is 0 if no overlap.
            This is smooth enough for SA (we don't need gradients).
        """
        energy = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                vx = t - dx
                vy = t - dy
                if vx > 0 and vy > 0:
                    # Overlap exists
                    # Penalty proportional to overlap area
                    energy += vx * vy
        return energy

    def is_feasible(centers, sides):
        """Strict feasibility check."""
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
                return False
            if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        return True

    def clamp_centers(centers, sides):
        """Ensure centers are such that squares stay within [0,1]^2."""
        for i in range(n):
            s = sides[i]
            min_c = s / 2.0
            max_c = 1.0 - s / 2.0
            if min_c > max_c:
                # Square too big? This shouldn't happen if we limit side growth.
                # But if it does, clamp to center.
                centers[i, 0] = 0.5
                centers[i, 1] = 0.5
            else:
                centers[i, 0] = np.clip(centers[i, 0], min_c, max_c)
                centers[i, 1] = np.clip(centers[i, 1], min_c, max_c)
        return centers

    def simulated_annealing(centers0, sides0, max_iter=20000, T_start=1.0, T_end=1e-5, alpha=0.9995):
        """
        Run SA to minimize overlap energy while maximizing side lengths.
        We actually want to maximize sum(sides) subject to energy=0.
        Since we clamp boundaries, we can focus on overlaps.
        However, if we just minimize energy, we might shrink sides to 0.
        So we need to balance.
        
        Better approach for SA:
        State: centers, sides.
        Move: Perturb centers and/or sides.
        Cost: Energy (overlap penalty).
        But we want to maximize sides.
        
        Let's use a composite cost:
        Cost = Energy - lambda * sum(sides)?
        No, if Energy > 0, we must fix it.
        
        Standard approach for packing:
        1. Start with small squares.
        2. Grow squares and move them to find a feasible packing.
        3. Once feasible, try to grow further.
        
        Let's simplify:
        We will run SA to minimize Energy.
        But we also want large sides.
        If we start with reasonable sides (e.g., 0.2), and only allow side changes that don't increase Energy significantly, we can grow them.
        
        Alternative:
        Use Cost = Energy + max(0, target_sum - sum(sides))? No.
        
        Let's stick to: Minimize Energy.
        But we want the solution with the largest sum(sides) among those with Energy=0.
        
        Strategy:
        1. Initialize with random sides in [0.1, 0.3].
        2. Run SA to minimize Energy.
        3. During SA, we allow moves that change sides.
           - If a move increases side and keeps Energy low, it's good.
           - If a move decreases side to reduce Energy, it's allowed if Temperature is high.
        
        Let's define Cost = Energy.
        But we want to bias towards larger sides.
        We can add a small penalty for small sides?
        Or, we can run SA on a fixed set of sides? No, sides are variables.
        
        Let's try: Cost = Energy - 0.1 * sum(sides).
        This encourages larger sides, but if Energy blows up, it dominates.
        """
        centers = centers0.copy()
        sides = sides0.copy()
        current_energy = get_energy(centers, sides)
        
        best_centers = centers.copy()
        best_sides = sides.copy()
        best_energy = current_energy
        best_sum_sides = np.sum(sides)
        
        T = T_start
        rng = np.random.default_rng(12345)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Generate a move
            # 1. Move a random square
            idx = rng.integers(0, n)
            move_type = rng.integers(0, 2) # 0: move center, 1: change side
            
            new_centers = centers.copy()
            new_sides = sides.copy()
            
            if move_type == 0:
                # Perturb center
                step_size = 0.05 * (1.0 - 0.5 * it / max_iter)
                new_centers[idx, 0] += rng.normal(0, step_size)
                new_centers[idx, 1] += rng.normal(0, step_size)
            else:
                # Perturb side
                # Allow growth and shrinkage
                step_size = 0.02
                new_sides[idx] *= np.exp(rng.normal(0, step_size))
                # Limit side to reasonable range
                new_sides[idx] = np.clip(new_sides[idx], 0.01, 0.9)
            
            # Clamp centers for new state
            new_centers = clamp_centers(new_centers, new_sides)
            
            new_energy = get_energy(new_centers, new_sides)
            
            # Acceptance criterion
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                centers = new_centers
                sides = new_sides
                current_energy = new_energy
                
                # Track best feasible solution
                if current_energy < 1e-6: # Consider it feasible
                    sum_sides = np.sum(sides)
                    if sum_sides > best_sum_sides:
                        best_sum_sides = sum_sides
                        best_centers = centers.copy()
                        best_sides = sides.copy()
                        best_energy = current_energy
            
            T *= alpha
            if T < T_end:
                T = T_end
                
        return best_centers, best_sides

    def local_polish(centers, sides, max_iter=500):
        """
        Deterministic coordinate descent to tighten the solution.
        Try to increase side lengths one by one if possible.
        Try to move centers to resolve any residual overlaps.
        """
        centers = centers.copy()
        sides = sides.copy()
        rng = np.random.default_rng(54321)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            improved = False
            # Try to increase sides
            for i in range(n):
                # Try small increase
                new_s = sides[i] * 1.01
                if new_s > 1.0: continue
                new_sides = sides.copy()
                new_sides[i] = new_s
                new_centers = clamp_centers(centers, new_sides)
                if is_feasible(new_centers, new_sides):
                    centers = new_centers
                    sides = new_sides
                    improved = True
            
            # Try to move centers to allow further growth or fix overlaps
            # If not feasible, we need to move things.
            # But we assume we start from a feasible state from SA.
            # If SA returns infeasible, this polish might fail.
            # Let's check feasibility first.
            if not is_feasible(centers, sides):
                 # Try to fix by moving
                 for i in range(n):
                     for _ in range(10):
                         new_centers = centers.copy()
                         new_centers[i, 0] += rng.normal(0, 0.01)
                         new_centers[i, 1] += rng.normal(0, 0.01)
                         new_centers = clamp_centers(new_centers, sides)
                         if is_feasible(new_centers, sides):
                             centers = new_centers
                             improved = True
                             break
            
            if not improved:
                # Try random kicks to escape local optimum in polish
                if rng.random() < 0.1:
                    idx = rng.integers(0, n)
                    new_centers = centers.copy()
                    new_centers[idx, 0] += rng.normal(0, 0.05)
                    new_centers[idx, 1] += rng.normal(0, 0.05)
                    new_centers = clamp_centers(new_centers, sides)
                    if is_feasible(new_centers, sides):
                        centers = new_centers
                        improved = True
                        
            if not improved and is_feasible(centers, sides):
                 # Stuck, try shrinking a bit to allow rearrangement?
                 # Or just break.
                 pass

        return centers, sides

    # Main Search Loop
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    # Multiple starts
    # Start 1: Grid-like
    # Start 2: Random
    # Start 3: Random with larger sides
    
    rng_main = np.random.default_rng(999)
    
    for run in range(20): # 20 SA runs
        if time.time() > deadline - 60: # Leave time for polish
            break
            
        # Initialize
        if run == 0:
            # Grid 2x5
            side = 0.2
            cx, cy = np.meshgrid(np.linspace(side/2, 1-side/2, 5), np.linspace(side/2, 1-side/2, 2))
            centers0 = np.column_stack((cx.flatten(), cy.flatten()))
            sides0 = np.full(n, side)
        elif run == 1:
            # Grid 5x2
            side = 0.2
            cx, cy = np.meshgrid(np.linspace(side/2, 1-side/2, 2), np.linspace(side/2, 1-side/2, 5))
            centers0 = np.column_stack((cx.flatten(), cy.flatten()))
            sides0 = np.full(n, side)
        else:
            # Random
            sides0 = rng_main.uniform(0.15, 0.30, n)
            centers0 = np.zeros((n, 2))
            for i in range(n):
                s = sides0[i]
                centers0[i, 0] = rng_main.uniform(s/2, 1-s/2)
                centers0[i, 1] = rng_main.uniform(s/2, 1-s/2)
            
        # Run SA
        # We need to be careful with the time per SA run.
        # 20 runs * ~20s = 400s.
        # Let's reduce iterations or runs.
        # 10 runs * 30s = 300s.
        
        max_iter_sa = 10000
        c, s = simulated_annealing(centers0, sides0, max_iter=max_iter_sa)
        
        # Polish
        if is_feasible(c, s):
            c_polished, s_polished = local_polish(c, s, max_iter=100)
            if is_feasible(c_polished, s_polished):
                score = np.sum(s_polished)
                if score > best_score:
                    best_score = score
                    best_centers = c_polished
                    best_sides = s_polished
        else:
            # Try to fix infeasible SA result with aggressive polish
            # This is risky, but worth a try if no feasible found yet
            if best_score == -np.inf:
                 c_fixed, s_fixed = local_polish(c, s, max_iter=200)
                 if is_feasible(c_fixed, s_fixed):
                     score = np.sum(s_fixed)
                     if score > best_score:
                         best_score = score
                         best_centers = c_fixed
                         best_sides = s_fixed

    # Fallback
    if best_centers is None:
        # Return a known valid grid
        side = 0.2
        cx, cy = np.meshgrid(np.linspace(side/2, 1-side/2, 5), np.linspace(side/2, 1-side/2, 2))
        best_centers = np.column_stack((cx.flatten(), cy.flatten()))
        best_sides = np.full(n, side)

    return best_centers, best_sides

# EVOLVE-BLOCK-END
