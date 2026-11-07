"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven Differential Evolution for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Reference point for hypervolume calculation (from task description)
        ref_point = np.array([3.0, 5.0, 7.0])
        
        eval_count = 0
        # Store all evaluated points to calculate final hypervolume
        all_points = []
        
        # Population size
        pop_size = 30
        
        # Initialize population
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        
        rng = np.random.RandomState(42)
        
        # Initial random sampling
        for i in range(pop_size):
            x = lo + (hi - lo) * rng.random(dim)
            f = np.array(problem(x))
            eval_count += 1
            F[i] = f
            all_points.append(x)
            
        def calculate_hv(points, ref):
            """Calculate hypervolume indicator for a set of points in 3D."""
            if len(points) == 0:
                return 0.0
            
            # Filter out points dominated by reference or worse in any dimension
            # We only care about points that are better than ref in ALL objectives
            # Since we are minimizing, "better" means < ref
            valid_points = []
            for p in points:
                if np.all(p < ref):
                    valid_points.append(p)
            
            if not valid_points:
                return 0.0
            
            # For 3D, we can use a simple sorting algorithm
            # Sort by first objective
            valid_points.sort(key=lambda x: x[0])
            
            # Remove dominated points (non-dominated set)
            nds = []
            for p in valid_points:
                dominated = False
                for q in nds:
                    if np.all(q <= p) and np.any(q < p):
                        dominated = True
                        break
                if not dominated:
                    # Remove points in nds dominated by p
                    nds = [q for q in nds if not (np.all(p <= q) and np.any(p < q))]
                    nds.append(p)
            
            if not nds:
                return 0.0
            
            # Calculate volume
            # Sort by f1
            nds.sort(key=lambda x: x[0])
            
            vol = 0.0
            prev_x = ref[0]
            # We integrate from the "worst" valid point towards the reference point
            # Actually, standard HV calculation:
            # Sum over each point in NDS of the volume of the box it "owns"
            
            # Let's use a simpler recursive or slice method for 3D
            # Slice along f1 axis
            # Sort NDS by f1
            # For each point i, the contribution is (f1[i+1] - f1[i]) * HV_2D(remaining points, ref[1:])
            # But we need to be careful with the reference point.
            
            # Let's implement a standard 3D HV algorithm
            # 1. Sort by f1
            # 2. Iterate from "worst" (largest f1) to "best" (smallest f1)
            #    Wait, usually we iterate from best to worst or vice versa.
            #    Let's sort ascending f1.
            #    The volume is the sum of boxes.
            
            # A robust 3D HV calculation:
            # 1. Filter points < ref
            # 2. Find NDS
            # 3. Sort NDS by f1
            # 4. For each point i in NDS (sorted by f1):
            #    The "width" in f1 is (f1[i+1] - f1[i]) if i < len-1, else (ref[0] - f1[i])
            #    Wait, this is only valid if the points form a staircase.
            #    Actually, for NDS sorted by f1, f2 and f3 are decreasing.
            #    So the "slice" at f1[i] has a 2D hypervolume defined by the points j >= i
            #    projected onto f2, f3.
            
            # Let's use a simpler approximation or a known library if available? No, std lib only.
            # Let's implement a basic 3D HV.
            
            # Sort by f1
            nds.sort(key=lambda x: x[0])
            
            total_hv = 0.0
            # We will calculate the 2D HV for the "suffix" of the sorted list
            # and multiply by the delta in f1.
            
            # Precompute 2D HV for suffixes?
            # n is small (< 30), so O(n^3) is fine.
            
            for i in range(len(nds)):
                # Width in f1
                if i == len(nds) - 1:
                    w1 = ref[0] - nds[i][0]
                else:
                    w1 = nds[i+1][0] - nds[i][0]
                
                # 2D HV for points i..end with ref (ref[1], ref[2])
                # Points are nds[i], ..., nds[end]
                # They are sorted by f1, but for 2D HV we need to sort by f2.
                # Actually, since they are NDS in 3D, they are also NDS in 2D projection?
                # Not necessarily, but we can just calculate 2D HV for the subset.
                
                # Extract 2D points
                pts_2d = [(p[1], p[2]) for p in nds[i:]]
                
                # Calculate 2D HV
                # Filter < ref_2d
                ref_2d = (ref[1], ref[2])
                valid_2d = [p for p in pts_2d if p[0] < ref_2d[0] and p[1] < ref_2d[1]]
                
                if not valid_2d:
                    hv_2d = 0.0
                else:
                    # Find NDS in 2D
                    # Sort by f2
                    valid_2d.sort(key=lambda x: x[0])
                    nds_2d = []
                    for p in valid_2d:
                        dominated = False
                        for q in nds_2d:
                            if q[0] <= p[0] and q[1] <= p[1]:
                                dominated = True
                                break
                        if not dominated:
                            nds_2d = [q for q in nds_2d if not (p[0] <= q[0] and p[1] <= q[1])]
                            nds_2d.append(p)
                    
                    # Calculate volume
                    # Sort by f2
                    nds_2d.sort(key=lambda x: x[0])
                    hv_2d = 0.0
                    prev_y = ref_2d[1]
                    # Iterate from "worst" f2 to "best" f2?
                    # No, sort ascending f2.
                    # The "height" in f3 is determined by the current point's f3.
                    # The "width" in f2 is determined by the next point's f2.
                    # Wait, if sorted by f2 ascending, f3 is descending (for NDS).
                    # So for point i, the box height is (ref[1] - f2[i])? No.
                    
                    # Let's visualize:
                    # Points (f2, f3). Ref (R2, R3).
                    # Sort by f2.
                    # Point 0 has smallest f2. It dominates in f2.
                    # The region "owned" by Point 0 in f2 is [f2[0], f2[1]].
                    # The height in f3 is (R3 - f3[0]).
                    # So volume += (f2[1] - f2[0]) * (R3 - f3[0]).
                    
                    for j in range(len(nds_2d)):
                        if j == len(nds_2d) - 1:
                            w2 = ref_2d[0] - nds_2d[j][0]
                        else:
                            w2 = nds_2d[j+1][0] - nds_2d[j][0]
                        
                        h3 = ref_2d[1] - nds_2d[j][1]
                        
                        if w2 > 0 and h3 > 0:
                            hv_2d += w2 * h3
                
                total_hv += w1 * hv_2d
            
            return total_hv

        def local_search(x, f, max_iter=5):
            """Simple local search to improve a single point."""
            current_x = x.copy()
            current_f = f.copy()
            
            # Simple random perturbation
            for _ in range(max_iter):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Perturb one or two dimensions
                n_perturb = 2
                idx = rng.choice(dim, n_perturb, replace=False)
                new_x = current_x.copy()
                for i in idx:
                    # Small step
                    step = (hi[i] - lo[i]) * 0.05
                    new_x[i] += rng.uniform(-step, step)
                    new_x[i] = np.clip(new_x[i], lo[i], hi[i])
                
                new_f = np.array(problem(new_x))
                eval_count += 1
                all_points.append(new_x)
                
                # If new point is better (Pareto improvement or HV improvement?)
                # For local search, we accept if it dominates or is non-dominated and better in one obj
                # Simple: Accept if it dominates current_f
                if np.all(new_f <= current_f) and np.any(new_f < current_f):
                    current_x = new_x
                    current_f = new_f
                # Or if it is non-dominated but closer to ideal?
                # Let's stick to dominance for simplicity in this tight budget.
                
            return current_x, current_f

        # Main DE Loop
        # Population is X, F
        # We maintain a global archive of non-dominated points found so far
        # For simplicity, we just keep the current population and use it for selection.
        # We will use the entire history `all_points` for the final HV, 
        # but for selection in DE, we use the current population.
        
        # To make DE work well, we need to select parents from the population.
        # We can use a simple tournament based on hypervolume contribution or just random.
        # Given the small budget, random selection with a diversity bias might be good.
        
        # Let's run DE until budget or time runs out.
        
        # Initial HV calculation for tracking
        # Note: calculate_hv is expensive if called every generation.
        # We will call it every few generations or use a proxy.
        
        # Proxy: Sum of individual distances to ideal point (0,0,0) or similar.
        # Or just random selection.
        
        while eval_count < self.budget and time.time() < deadline:
            # Create offspring
            # For each individual in population, create a mutant
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select 3 distinct parents different from i
                # Simple random selection
                parents = rng.choice(pop_size, 3, replace=False)
                # Ensure i is not in parents? DE usually allows it, but distinct is better.
                # If i is in parents, we might have issues. Let's just pick 3 random.
                
                p1 = X[parents[0]]
                p2 = X[parents[1]]
                p3 = X[parents[2]]
                
                # Differential Evolution Mutation
                # v = p1 + F * (p2 - p3)
                F_factor = 0.8
                v = p1 + F_factor * (p2 - p3)
                
                # Crossover
                # Binary crossover with rate CR
                CR = 0.5
                j_rand = rng.randint(dim)
                
                u = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = X[i][j]
                
                # Clip to bounds
                u = np.clip(u, lo, hi)
                
                # Evaluate
                f_u = np.array(problem(u))
                eval_count += 1
                all_points.append(u)
                
                # Selection
                # If u dominates X[i], replace X[i]
                # If X[i] dominates u, keep X[i]
                # Otherwise, keep the one with better hypervolume contribution?
                # For simplicity: Keep u if it is non-dominated w.r.t X[i] and better in at least one objective?
                # Or just random?
                # Let's use a simple rule: Replace if u dominates X[i].
                # To maintain diversity, we can also replace if X[i] dominates u? No, that's bad.
                # Let's use a "better" function based on distance to ideal point (0,0,0) as a tie-breaker.
                
                if np.all(f_u <= F[i]) and np.any(f_u < F[i]):
                    # u dominates X[i]
                    new_X[i] = u
                    new_F[i] = f_u
                elif np.all(F[i] <= f_u) and np.any(F[i] < f_u):
                    # X[i] dominates u
                    new_X[i] = X[i]
                    new_F[i] = F[i]
                else:
                    # Non-dominated
                    # Choose the one closer to the origin (ideal point)
                    # Or just keep the old one to avoid disruption?
                    # Let's keep the one with smaller sum of objectives (proxy for closeness to ideal)
                    if np.sum(f_u) < np.sum(F[i]):
                        new_X[i] = u
                        new_F[i] = f_u
                    else:
                        new_X[i] = X[i]
                        new_F[i] = F[i]
            
            # Update population
            X = new_X
            F = new_F
            
            # Periodic Local Search on the best individual (based on sum of objectives)
            if eval_count < self.budget and time.time() < deadline:
                # Find best individual
                best_idx = np.argmin(np.sum(F, axis=1))
                x_best = X[best_idx]
                f_best = F[best_idx]
                
                # Perform local search
                x_ls, f_ls = local_search(x_best, f_best, max_iter=10)
                
                # If local search found a better point, update population
                if np.all(f_ls <= f_best) and np.any(f_ls < f_best):
                    X[best_idx] = x_ls
                    F[best_idx] = f_ls
                # Also add to all_points (already done in local_search)
                
        # Return the list of all evaluated points
        # The evaluator will record the objective vectors returned by problem(x)
        # Wait, the task says: "Every returned objective vector is recorded"
        # This implies the evaluator calls problem(x) and records the result.
        # My code calls problem(x) and stores x in all_points.
        # But the evaluator records the *return value* of problem(x).
        # So I don't need to return the points. The evaluator handles the recording.
        # However, the task says: "the return value of __call__ is ignored."
        # So I can just return None or an empty list.
        
        return []

# EVOLVE-BLOCK-END
