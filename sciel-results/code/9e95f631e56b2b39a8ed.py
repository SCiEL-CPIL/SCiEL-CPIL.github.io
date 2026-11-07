"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    1. For each dimension, evaluate the positive and negative boundary points (5 and -5) with other coordinates at 0.
    2. Determine the sign of the gradient for that dimension by comparing the two values.
    3. Construct the optimal corner based on the determined signs.
    4. Evaluate the corner.
    5. Use remaining budget to re-evaluate the best point (or just stop, but re-evaluating is safe).
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290  # Safety margin

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_point(x):
            nonlocal best_x, best_f, calls_made
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)
            return fx

        signs = np.zeros(dim)
        
        # Phase 1: Determine signs for each dimension using pairwise boundary comparisons
        for i in range(dim):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Evaluate +5 on axis i
            x_pos = np.zeros(dim)
            x_pos[i] = 5.0
            f_pos = eval_point(x_pos)
            
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Evaluate -5 on axis i
            x_neg = np.zeros(dim)
            x_neg[i] = -5.0
            f_neg = eval_point(x_neg)
            
            # Determine sign
            # If f_pos < f_neg, then 5*c_i < -5*c_i => c_i < 0.
            # If c_i < 0, we want x_i to be +5 to minimize c_i*x_i.
            # So signs[i] = 1.
            # If f_pos > f_neg, then c_i > 0.
            # We want x_i to be -5.
            # So signs[i] = -1.
            if f_pos < f_neg:
                signs[i] = 1.0
            else:
                signs[i] = -1.0

        # Phase 2: Evaluate the estimated optimal corner
        if calls_made < budget and time.time() < deadline:
            corner = signs * 5.0
            # Ensure corner is within bounds (it is, by construction)
            eval_point(corner)

        # Phase 3: Fill remaining budget
        if calls_made < budget and time.time() < deadline:
            best_pt = best_x.copy()
            while calls_made < budget:
                if time.time() > deadline:
                    break
                eval_point(best_pt)

        return best_x, best_f
# EVOLVE-BLOCK-END
