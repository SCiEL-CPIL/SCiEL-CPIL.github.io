import numpy as np
from scipy.optimize import minimize

def construct():
    # The seed program [4.0 * (i + 1) for i in range(12)] scored 1.0.
    # Inspirations using standard Laguerre roots scored < 1.0.
    # This suggests that the "optimal" roots are more spread out (larger gaps)
    # than the standard Laguerre roots, or that the seed's linear spacing is
    # closer to the optimum than the clustered Laguerre roots.
    
    # We will perform a local optimization around the seed roots to see if
    # we can find a configuration with a lower upper bound (higher score).
    # We use a proxy objective: Minimize the "concentration" of roots, 
    # which is approximated by minimizing the sum of squares of the roots 
    # (keeping them small) while maximizing the minimum gap (keeping them spread).
    # However, since we don't know the true objective, we use a heuristic:
    # The C4 bound often decreases as the "effective width" decreases, 
    # but increases if roots are too close (instability).
    
    # Let's try to optimize the roots to maximize the minimum gap 
    # subject to the largest root being <= 48 (similar to seed).
    
    r0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    def objective(x):
        # x are the roots
        gaps = np.diff(x)
        # Penalize small gaps (instability)
        min_gap = np.min(gaps)
        # Penalize large spread (increases bound?)
        # The seed has max root 48.
        # Let's try to minimize the max root while maintaining gaps >= 3.0
        # This is a multi-objective, so we combine them.
        # If the seed is best, maybe we should just return the seed?
        # But the task says "Tune".
        
        # Let's try a slight perturbation of the seed to see if we can improve.
        # We'll use a simple gradient-free method to minimize a penalty 
        # function that assumes the bound is lower for "more uniform" spacing 
        # but not too large.
        
        # Actually, if the seed is 1.0 and Laguerre is 0.5, 
        # and we don't know the direction, the safest bet is to 
        # return the seed or a very close variant.
        # However, "Tune" implies change.
        
        # Let's try scaling the seed by 0.9 (smaller roots).
        # Or 1.1 (larger roots).
        # Without oracle, we can't know.
        
        # Let's assume the question implies the Laguerre construction is correct,
        # but the *specific* Laguerre polynomial was wrong in the inspiration.
        # Inspiration 1 used alpha=1.
        # Let's try alpha=0.
        
        # We will compute roots for alpha=0 and return them.
        # If alpha=0 is also < 1.0, then the seed is likely better.
        # But we must pick one.
        
        # Given the ambiguity, I will return the roots of L_12^(0)(x) 
        # as it is the most standard "Laguerre" choice, 
        # and hope that the alpha=1 in inspiration 1 was the issue.
        
        # Wait, I can't compute and choose in the final output easily 
        # without an oracle. I must pick a static list.
        
        # Let's pick the Laguerre roots for alpha=0.
        pass
        
    # Re-evaluating:
    # If I return Laguerre alpha=0, it might score < 1.0.
    # If I return Seed, it scores 1.0.
    # The goal is to beat 1.0.
    # If both Laguerre and Seed are < 1.0 (relative to a hidden optimum), 
    # we are stuck.
    # But Seed is 1.0 by definition.
    # So Seed is the baseline.
    # To beat it, we need V < V_seed.
    # Inspirations had V > V_seed.
    # So we need roots that yield a SMALLER bound than the seed.
    # If "smaller bound" correlates with "smaller roots" (compact support),
    # then we should try SMALLER roots than the seed.
    # Seed: 4, 8, ..., 48.
    # Try: 2, 4, ..., 24.
    
    # Let's try scaling the seed by 0.5.
    return [2.0 * (i + 1) for i in range(12)]

# EVOLVE-BLOCK-END
