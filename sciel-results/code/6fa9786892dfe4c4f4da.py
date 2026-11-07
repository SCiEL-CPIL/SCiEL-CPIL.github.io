import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a hybrid of Expected Improvement (EI) and Lower Confidence Bound (LCB)
    with adaptive exploration.
    """
    # Ensure inputs are numpy arrays of correct shape
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure finite values
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Avoid division by zero in EI calculation
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # --- Component 1: Expected Improvement (EI) ---
    # EI for minimization:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the CDF of the standard normal, and phi is the PDF.
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in extreme tails
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is always >= 0. Higher EI is better.
    
    # --- Component 2: Lower Confidence Bound (LCB) ---
    # LCB(x) = mu(x) - kappa * sigma(x)
    # We want to minimize LCB, so higher score is -LCB = -mu + kappa * sigma.
    # Adaptive kappa: start high, decrease as we get closer to the global minimum.
    # Since we don't know the true minimum, we use f_best as a proxy.
    # A simple adaptive scheme: kappa = base + decay * (f_best - lower_bound_estimate)
    # For Hartmann 6D, min is approx -3.32. Let's assume a lower bound of -4.0.
    # However, f_best changes. Let's use a fixed but slightly aggressive kappa for now,
    # or adapt based on the spread of sigma.
    
    # Let's try an adaptive kappa that depends on the current uncertainty.
    # If sigma is large, we want to explore more.
    # kappa = 2.0 + 1.0 * (sigma / max_sigma) ? No, kappa is a scalar.
    
    # Let's stick to a strong exploration constant but normalize the terms.
    # The scales of EI and LCB might differ.
    # EI is in objective units.
    # LCB score (-mu + kappa*sigma) is in objective units.
    
    # Let's use a weighted sum.
    # weights: w_ei, w_lcb
    # score = w_ei * ei + w_lcb * (-mu + kappa * sigma)
    
    # To make it robust, let's normalize ei and lcb_score to [0, 1] or similar?
    # Or just use a fixed weight.
    # Inspiration 3 used a hybrid.
    
    # Let's try: score = ei + alpha * (-mu + beta * sigma)
    # But EI already contains exploration.
    # Pure EI can be too greedy. Pure LCB can be too random.
    
    # Let's use a "PI" (Probability of Improvement) blended with LCB?
    # PI(x) = Phi((f_best - mu(x))/sigma(x))
    # This is bounded [0,1]. EI is unbounded.
    
    # Let's try a different approach:
    # Use EI as the primary driver, but add a small exploration bonus based on sigma.
    # score = ei + gamma * sigma
    
    # Or, use the LCB formulation but with a time-varying kappa.
    # Since we don't have the step count, we can't easily adapt kappa by time.
    # But we can adapt kappa based on f_best.
    # If f_best is very good (close to -3.32), reduce exploration.
    # If f_best is poor, increase exploration.
    
    # Hartmann 6D min is -3.32237.
    # Let's assume a reasonable lower bound of -4.0.
    # progress = (f_best - (-4.0)) / (initial_guess - (-4.0)) ?
    # We don't know initial_guess.
    
    # Let's try a fixed hybrid that worked well in inspiration 3 (score 1.15) vs current 1.17.
    # Current is pure LCB with kappa=3.
    # Inspiration 3 was hybrid EI + LCB.
    # Maybe pure LCB is slightly better? Or maybe the kappa was tuned?
    
    # Let's try to improve the LCB by using a more sophisticated exploration term.
    # "Thompson Sampling" style? No, we only have mu and sigma.
    
    # Let's try: score = -mu + kappa * sigma
    # But adjust kappa based on the ratio of sigma to the range of mu?
    
    # Let's try an adaptive kappa:
    # kappa = 2.0 + 2.0 * (sigma / (np.max(sigma) + 1e-8))
    # This makes exploration stronger where uncertainty is high relative to the batch.
    # But kappa is a scalar in standard LCB. If we make it vector, it's a local LCB.
    
    # Let's try: score = -mu + kappa_local * sigma
    # where kappa_local = 3.0 (fixed)
    
    # Alternative: Use the "Knowledge Gradient" approximation? Too complex.
    
    # Let's try a blend of EI and LCB scores, normalized.
    # Normalize ei to [0, 1] by dividing by max(ei) + eps
    # Normalize lcb_score to [0, 1]
    
    # Actually, let's look at the failure mode.
    # Hartmann 6D is multimodal.
    # With 30 evals, we need to find the global basin.
    # Greedy EI might get stuck in a local optimum if the GP model is confident there.
    # LCB encourages exploration.
    
    # Let's try a slightly lower kappa to balance exploitation, but ensure finite scores.
    # Current: kappa=3.0.
    # Try kappa=2.5?
    
    # Or, let's use a "soft" EI that is more robust.
    
    # Let's try the hybrid from inspiration 3 again, but tuned.
    # score = w1 * ei_norm + w2 * lcb_norm
    
    # Let's calculate EI and LCB scores.
    # ei is already calculated.
    lcb_score = -mu + 3.0 * sigma
    
    # Normalize both to have zero mean and unit variance?
    # This removes scale dependency.
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-8
    ei_norm = (ei - ei_mean) / ei_std
    
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score) + 1e-8
    lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    # Weighted sum
    w_ei = 0.5
    w_lcb = 0.5
    
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
