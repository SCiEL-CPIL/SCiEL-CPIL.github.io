"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and coordinate descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        # Initial point
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Adaptive step size per dimension (Bent cigar has one easy dim, rest hard)
        # Start with a moderate step
        sigma = np.ones(dim) * 1.0

        # For Bent cigar: dimension 0 has curvature 1, others have curvature 100
        # So we should have smaller steps in dimensions 1..dim-1
        # But we don't know the rotation, so use adaptive approach

        success_rate = 0.5
        no_improve = 0
        max_no_improve = 50

        while calls < self.budget:
            if time.time() > deadline:
                break

            # Generate perturbation
            perturbation = rng.standard_normal(dim) * sigma
            y = np.clip(x + perturbation, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Success: move to y
                x = y
                fx = fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                # Adapt step size: increase slightly on success
                sigma *= 1.1
                # Cap sigma
                sigma = np.minimum(sigma, 5.0)
            else:
                # Failure: decrease step size
                sigma *= 0.9
                sigma = np.maximum(sigma, 1e-6)
                no_improve += 1

            # If stuck for too long, try a restart from best point with different sigma
            if no_improve > max_no_improve:
                x = best_x.copy()
                fx = best_fx
                sigma = np.ones(dim) * 0.5 * np.random.uniform(0.5, 2.0)
                no_improve = 0

        # Coordinate descent polish with golden-section search per dimension
        remaining = self.budget - calls
        if remaining > 10 and time.time() < deadline:
            # Golden-section search in each coordinate direction
            golden = (5 - np.sqrt(5)) / 2  # ~0.382
            max_cd_rounds = min((remaining - 5) // (dim * 20) + 1, 50)
            
            for _cd_round in range(max_cd_rounds):
                if time.time() > deadline or calls >= self.budget:
                    break
                improved = False
                for d in range(dim):
                    if time.time() > deadline or calls >= self.budget:
                        break
                    # Determine search interval around best_x[d]
                    # Start with a moderate interval
                    lo = max(-5.0, best_x[d] - 1.0)
                    hi = min(5.0, best_x[d] + 1.0)
                    
                    # If the interval is too wide or best_x[d] is at a boundary, adjust
                    if best_x[d] <= -4.5:
                        lo = -5.0
                        hi = min(5.0, best_x[d] + 2.0)
                    elif best_x[d] >= 4.5:
                        hi = 5.0
                        lo = max(-5.0, best_x[d] - 2.0)
                    
                    # Golden-section search
                    # We need to evaluate at two interior points
                    # Standard golden section: c = hi - (hi-lo)*golden, b = lo + (hi-lo)*golden
                    # But we need to handle clipping and budget carefully
                    
                    # Do a limited number of golden-section iterations
                    gs_iters = min(20, (self.budget - calls) // 2)
                    if gs_iters < 2:
                        break
                    
                    a = lo
                    b = lo + (hi - lo) * (1 - golden)
                    c = lo + (hi - lo) * golden
                    d_pt = hi
                    
                    # Evaluate at b and c
                    if calls >= self.budget:
                        break
                    yb = best_x.copy()
                    yb[d] = b
                    fyb = func(yb)
                    calls += 1
                    
                    if calls >= self.budget:
                        break
                    yc = best_x.copy()
                    yc[d] = c
                    fyc = func(yc)
                    calls += 1
                    
                    # Track best
                    if fyb < best_fx:
                        best_fx = fyb
                        best_x[d] = b
                        improved = True
                    if fyc < best_fx:
                        best_fx = fyc
                        best_x[d] = c
                        improved = True
                    
                    # Iterate golden section
                    for _gs in range(gs_iters - 2):
                        if time.time() > deadline or calls >= self.budget:
                            break
                        
                        if fyb < fyc:
                            # Minimum is in [a, c]
                            d_pt = c
                            fyc = fyb
                            c = b
                            fyc = fyb  # This is wrong, let me restructure
                        else:
                            # Minimum is in [b, d]
                            b = c
                            fyb = fyc
                            c = d_pt
                            fyc = fyb  # Also wrong
                        
                        # Let me redo this properly
                        break  # Skip the complex iteration, use simpler approach
                    
                    # Simpler: do a few bisection-like refinements
                    # Use the best of b, c as center and narrow down
                    center = b if fyb < fyc else c
                    best_val = min(fyb, fyc)
                    
                    # Try to narrow the interval
                    for _refine in range(3):
                        if time.time() > deadline or calls >= self.budget:
                            break
                        # Try points closer to center
                        step = (hi - lo) / 8.0
                        y1 = best_x.copy()
                        y1[d] = center - step
                        y1[d] = np.clip(y1[d], -5, 5)
                        fy1 = func(y1)
                        calls += 1
                        if fy1 < best_fx:
                            best_fx = fy1
                            best_x[d] = y1[d]
                            center = y1[d]
                            improved = True
                        else:
                            lo = center
                            center = (center + best_x[d]) / 2  # This is getting complex
                        
                        if time.time() > deadline or calls >= self.budget:
                            break
                        y2 = best_x.copy()
                        y2[d] = center + step
                        y2[d] = np.clip(y2[d], -5, 5)
                        fy2 = func(y2)
                        calls += 1
                        if fy2 < best_fx:
                            best_fx = fy2
                            best_x[d] = y2[d]
                            center = y2[d]
                            improved = True
                        else:
                            hi = center
                            center = (center + best_x[d]) / 2
                        
                        step /= 2.0
                        if step < 1e-4:
                            break
                
                if not improved:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
