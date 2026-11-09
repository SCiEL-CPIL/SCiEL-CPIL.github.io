"""Black-box continuous optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for Büche-Rastrigin (BBOB f4) using Halton global initialization
    and Nelder-Mead local refinement, with a final fine-tuning phase.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin for 300s limit
        rng = np.random.default_rng(12345)
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget or check_deadline():
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Global Exploration using Halton Sequence ---
        # Halton sequences provide low-discrepancy coverage, which is excellent
        # for finding promising basins in multimodal functions like Rastrigin.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def generate_halton_points(n, dim):
            points = np.zeros((n, dim))
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(n):
                for j in range(dim):
                    points[i, j] = halton(i + 1, primes[j])
            return points

        # Use 60 calls for initial exploration
        n_initial = 60
        halton_points = generate_halton_points(n_initial, dim)
        # Map [0, 1]^d to [-5, 5]^d
        halton_points = halton_points * 10.0 - 5.0
        
        # Add origin and a few random points for extra diversity
        extra_points = [np.zeros(dim)]
        extra_points.extend(rng.uniform(-5, 5, size=(5, dim)))
        
        all_initial_points = np.vstack([halton_points, np.array(extra_points)])
        # Shuffle to avoid bias in order (though Halton is already well-distributed)
        rng.shuffle(all_initial_points)

        candidates = []
        
        for x in all_initial_points:
            fx = evaluate(x)
            if fx is None:
                break
            candidates.append((fx, x.copy()))

        if best_x is None:
            return None, np.inf

        # Sort candidates by function value
        candidates.sort(key=lambda item: item[0])
        
        # --- Phase 2: Local Refinement using Nelder-Mead ---
        # Select top K candidates for deep optimization
        # With 500 budget and 65 used, we have ~435 left.
        # Nelder-Mead in 5D uses (dim+1) points in simplex.
        # Each iteration uses 1-3 function calls.
        # We can afford to optimize top 3-5 candidates.
        
        top_k = 5
        top_candidates = candidates[:top_k]
        
        # Allocate remaining budget
        remaining_budget = budget - calls_made
        # Reserve some budget for potential random restarts if NM fails
        reserve_calls = 20
        nm_budget = max(0, remaining_budget - reserve_calls)
        
        # Distribute budget among top candidates.
        # Give more budget to the best candidate.
        # Weights: 50%, 30%, 10%, 5%, 5%
        weights = [0.5, 0.3, 0.1, 0.05, 0.05]
        if len(top_candidates) < 5:
             # Adjust weights if fewer candidates
             weights = [1.0 / len(top_candidates)] * len(top_candidates)

        def nelder_mead(x0, max_calls):
            nonlocal calls_made, best_f, best_x, deadline
            
            if check_deadline() or calls_made >= budget:
                return
            
            # Initialize simplex
            simplex = [x0.copy()]
            simplex_f = []
            
            fx0 = evaluate(x0)
            if fx0 is None:
                return
            simplex_f.append(fx0)
            
            # Add perturbed points to build simplex
            # Perturb each dimension
            for i in range(dim):
                if check_deadline() or calls_made >= budget:
                    return
                perturbation = x0.copy()
                # If x0[i] is 0, step size 0.5, else 0.5 * sign(x0[i])
                step = 0.5 if x0[i] >= 0 else -0.5
                perturbation[i] += step
                perturbation = np.clip(perturbation, -5, 5)
                
                fp = evaluate(perturbation)
                if fp is None:
                    return
                simplex.append(perturbation)
                simplex_f.append(fp)
                
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            # Main NM loop
            while calls_made < budget and not check_deadline():
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                
                # Reflection
                x_reflect = centroid + alpha * (centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                
                if check_deadline() or calls_made >= budget:
                    break
                f_reflect = evaluate(x_reflect)
                if f_reflect is None:
                    break
                
                if f_reflect < best_f:
                    best_f = f_reflect
                    best_x = x_reflect.copy()
                
                if simplex_f[0] <= f_reflect < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                elif f_reflect < simplex_f[0]:
                    # Expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    
                    if check_deadline() or calls_made >= budget:
                        break
                    f_expand = evaluate(x_expand)
                    if f_expand is None:
                        break
                    
                    if f_expand < best_f:
                        best_f = f_expand
                        best_x = x_expand.copy()
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                else:
                    # Contraction
                    # Inside contraction if f_reflect < f_worst, else outside
                    if f_reflect < f_worst:
                        x_contract = centroid + 0.5 * (x_reflect - centroid)
                    else:
                        x_contract = centroid - 0.5 * (x_worst - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    
                    if check_deadline() or calls_made >= budget:
                        break
                    f_contract = evaluate(x_contract)
                    if f_contract is None:
                        break
                    
                    if f_contract < best_f:
                        best_f = f_contract
                        best_x = x_contract.copy()
                    
                    if f_contract < min(f_reflect, f_worst):
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if check_deadline() or calls_made >= budget:
                                return
                            f_shrink = evaluate(simplex[i])
                            if f_shrink is None:
                                return
                            simplex_f[i] = f_shrink
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = simplex[i].copy()
        
        # Run Nelder-Mead on top candidates
        for i, (fx, x) in enumerate(top_candidates):
            if check_deadline() or calls_made >= budget:
                break
            
            # Calculate budget for this candidate
            if i < len(weights):
                w = weights[i]
            else:
                w = 1.0 / len(top_candidates)
            
            alloc = int(nm_budget * w)
            if i == len(top_candidates) - 1:
                # Give remaining budget to last candidate
                alloc = budget - calls_made - reserve_calls
            
            if alloc > 10:
                nelder_mead(x, alloc)
        
        # --- Phase 3: Final Fine-Tuning ---
        # Use remaining budget for a local random walk around the best point
        if calls_made < budget and best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            stall_count = 0
            max_stall = 30
            
            while calls_made < budget and not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                    stall_count = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.9
                    stall_count += 1
                
                if stall_count > max_stall:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
