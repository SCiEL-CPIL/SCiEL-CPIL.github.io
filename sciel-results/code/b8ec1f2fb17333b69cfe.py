"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Strategy: Deterministic Thompson Sampling
    # Instead of using a fixed exploration constant (like UCB) or an analytic expectation (like EI),
    # we draw a sample from the posterior distribution N(mu, sigma^2).
    # We use a fixed random seed to ensure the sampling process is deterministic 
    # for a given posterior state, avoiding side-effects on the global RNG state 
    # and ensuring reproducibility across the evaluation seeds.
    # This approach implicitly balances exploration and exploitation: 
    # high sigma leads to larger deviations from mu, encouraging exploration of uncertain regions.
    
    # Create a local random generator with a fixed seed to ensure determinism
    # without interfering with the global numpy random state.
    rng = np.random.default_rng(42)
    
    # Draw samples from the posterior: f ~ N(mu, sigma^2)
    # sigma is standard deviation
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # The acquisition score is the sampled value itself.
    # Higher sampled value means we "believe" the function might be higher there,
    # so we evaluate there to potentially find a new best (or confirm high value).
    # Note: We are minimizing, so normally we'd look for low values.
    # However, acquisition functions typically return a score where HIGHER is better to select.
    # In minimization, we often negate the function or use Expected Improvement.
    # If we sample f, and we want to minimize, we should pick the candidate with the LOWEST sample?
    # No, the prompt says: "Return 2000 finite scores, higher meaning evaluate next".
    # And "f_best is the lowest observed value".
    # If we sample f, and f is the objective value (to be minimized),
    # a low sample is "good".
    # So we should return -samples?
    # Let's check the logic.
    # If we use EI, we compute E[I(x)], which is high when f(x) is likely < f_best.
    # If we use UCB, we compute mu + kappa*sigma. For minimization, we usually use LCB (Lower Confidence Bound) or just UCB on the negative function.
    # The previous program returned `mu + kappa * sigma`.
    # If the function is to be minimized, `mu + kappa * sigma` is an upper bound.
    # Selecting the max of an upper bound is weird for minimization unless we are looking for "surprises" (high values) to update the GP?
    # No, standard BO for minimization uses:
    # 1. EI (Expected Improvement): High where f is likely < f_best.
    # 2. UCB: Usually applied to the *negative* function, or LCB for minimization: mu - kappa*sigma.
    # The previous program used `mu + kappa * sigma`. If this is for minimization, it selects points where the function is likely HIGH.
    # This would be terrible for finding the minimum.
    # UNLESS the "mu" and "sigma" are for the *negative* of the function?
    # The prompt says: "mu and sigma ... are in objective units". "minimizes weierstrass".
    # So mu is the mean of the objective.
    # If the previous program used `mu + kappa * sigma` and got 0.04, it was likely exploring high-value regions, which is wrong for minimization.
    # The parent program (EI) got 1.0. EI is correct for minimization.
    # So, for Thompson Sampling in minimization:
    # We sample f ~ N(mu, sigma^2).
    # We want to find the minimum.
    # So we should select the candidate with the LOWEST sample.
    # But the return value must be "higher meaning evaluate next".
    # So we should return `-samples`.
    
    # Wait, let's re-read the previous program's comment.
    # "score = mu + kappa * sigma"
    # If the previous program was bad, maybe it was because it was doing UCB for maximization on a minimization problem?
    # Or maybe the evaluator negates the function?
    # "minimizes weierstrass 1d ... minimum value 0".
    # "f_best is the lowest observed value".
    # "Return ... higher meaning evaluate next".
    # If I return `mu + kappa * sigma`, I am picking high mu + high sigma.
    # This is the Upper Bound.
    # For minimization, we want the Lower Bound (LCB): `mu - kappa * sigma`.
    # Or EI.
    # The parent (EI) worked (1.0).
    # The child (UCB max) failed (0.04).
    # This confirms that maximizing `mu + kappa * sigma` is bad for minimization.
    
    # So, for Thompson Sampling:
    # Sample f.
    # We want the minimum.
    # So we prefer low f.
    # Score to return (higher is better) should be `-f`.
    # So return `-samples`.
    
    return -samples

# EVOLVE-BLOCK-END
