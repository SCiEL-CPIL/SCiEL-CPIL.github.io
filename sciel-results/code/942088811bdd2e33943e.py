"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_TIMEOUT_LIMIT = 325.0  # 300s limit + 5s buffer, but task says 330s kill. Use 325 for safety.

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _TIMEOUT_LIMIT:
        # Return a deterministic fallback (e.g., based on mu) to avoid crashes/exceptions
        # Higher score for lower mu (assuming minimization)
        return -mu / (np.max(-mu) - np.min(-mu) + 1e-9)

    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and has a small floor to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)

    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-normalized EI
    ei_cost = ei / cost

    # Infer number of evaluations from cost
    # Cost starts at 5 uniform points. Each point costs 1 + mean(x_scaled).
    # Mean cost is roughly 1.5. So n_evals ~ total_cost / 1.5.
    # We don't have total_cost directly, but we can estimate the "age" of the search.
    # The prompt says "While total cost is below 30".
    # We can estimate n_evals by looking at the distribution of costs? No, cost is per-candidate.
    # Actually, the `cost` argument is the cost of *evaluating* each candidate.
    # We don't know how many have been evaluated.
    # However, the GP is updated after each evaluation. 
    # A common heuristic is to use a fixed beta or one that adapts to the magnitude of sigma or f_best.
    # Let's try an adaptive beta based on the standard deviation of the observations? 
    # We don't have the observations.
    
    # Alternative: Use a fixed but tuned beta, or one that depends on the "confidence" of the GP.
    # The UCB term is mu + beta * sigma.
    # We want to combine EI and UCB.
    # Score = w * EI_cost + (1-w) * UCB_cost
    
    # Let's estimate the progress. 
    # If we assume the loop runs until cost > 30, and each eval adds ~1.5 cost, 
    # we will have about 30/1.5 = 20 evaluations.
    # Early on (few evals), we want more exploration (higher beta).
    # Later, we want exploitation (lower beta).
    
    # Since we can't track eval count globally easily without state, 
    # we can use the magnitude of f_best or sigma as a proxy?
    # Or just use a constant beta that worked well in inspiration 1/3.
    # Inspiration 1 (1.2193) used UCB with adaptive beta.
    # Inspiration 3 (1.5828) used EI with robust sigma floor and optional UCB blend.
    
    # Let's try a blend: 
    # UCB_score = (mu + beta * sigma) / cost
    # Note: For minimization, lower is better. 
    # Standard UCB for minimization: mu - beta * sigma.
    # Let's use: ucb_min = mu - beta * sigma
    
    # Let's estimate beta.
    # If we assume ~20 evals total, and we are at step t, beta could be ~ 1/sqrt(t) or similar.
    # Without t, let's use a heuristic based on the relative scale of sigma.
    # Or simply a fixed beta. 
    # Let's try beta = 0.5 initially.
    
    # Wait, the "adaptive beta" in inspiration 1 might have been adaptive to the *data*.
    # We can estimate the "exploration" needed based on the variance of mu?
    # Let's try a blend where the weight of UCB decreases as f_best gets closer to 0?
    # No, that's risky.
    
    # Let's stick to a robust blend:
    # Score = EI_cost + lambda * UCB_min_cost
    # Let's set lambda to a small value to add exploration to EI.
    # EI is already an exploration/exploitation balance.
    # Adding UCB might double-count exploration.
    
    # Alternative Strategy from Inspiration 3: "EI with robust sigma floor and optional UCB blend"
    # Let's try: 
    # 1. Calculate EI.
    # 2. Calculate UCB_min = mu - beta * sigma.
    # 3. Normalize both to [0,1] range? Or just combine.
    
    # Let's try a dynamic beta that starts high and decays.
    # We can estimate the number of evaluations `n` by assuming the current `f_best` is the result of `n` evals.
    # This is not directly possible.
    
    # Let's look at the cost array. The cost is deterministic for each x.
    # We can't infer n from the cost array alone.
    
    # Let's use a fixed beta = 0.1 and see if the blend helps.
    # Or, use the "Probability of Improvement" (PI) which is simpler?
    # PI = Phi(z). Cost-normalized: PI / cost.
    # EI is generally better.
    
    # Let's try a different acquisition: Knowledge Gradient? Too complex.
    
    # Let's refine the EI calculation with a "robust" sigma.
    # The prompt says "sigma (a standard deviation)".
    # Sometimes sigma can be very small, making z huge and EI spike.
    # The floor 1e-8 handles that.
    
    # Let's try adding a small UCB component to encourage exploring areas with high uncertainty 
    # even if EI is low (e.g., far from f_best but high sigma).
    
    beta = 0.5
    ucb_min = mu - beta * sigma
    
    # Normalize EI and UCB to be comparable?
    # EI is in objective units. UCB is in objective units.
    # Cost is in cost units.
    # So EI/cost and UCB/cost are in objective/cost units.
    
    # Let's blend them:
    # score = 0.8 * ei_cost + 0.2 * (ucb_min / cost)
    
    # Wait, UCB_min can be very negative (good). EI is always positive (improvement over f_best).
    # If mu is much lower than f_best, EI is large.
    # If mu is high but sigma is high, UCB_min might be lower than mu.
    
    # Let's try:
    # score = ei_cost + 0.1 * (sigma / cost)  # Encourage exploration of high sigma areas
    
    # This is a simple heuristic. Let's implement this "EI + Exploration Bonus".
    
    exploration_bonus = sigma / cost
    score = ei_cost + 0.1 * exploration_bonus
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
