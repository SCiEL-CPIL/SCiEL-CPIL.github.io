import numpy as np
from scipy.stats import norm
import time

# Track call index to adapt strategy over the optimization loop
_call_count = [0]

def acquisition(mu, sigma, f_best):
    _call_count[0] += 1
    step = _call_count[0]
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Strategy:
    # For the first few steps, use a more exploratory UCB to ensure we don't get stuck
    # in a local minimum far from 0.
    # For later steps, switch to Expected Improvement (EI) which balances exploration
    # and exploitation more finely, or pure exploitation if very close.
    
    # Let's use a hybrid approach:
    # 1. Calculate Expected Improvement (EI)
    # 2. Calculate UCB with a beta that decays over time
    
    # EI for minimization:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # UCB for minimization: -mu + beta * sigma
    # We want to start with some exploration and then focus on exploitation.
    # The length scale is 0.2 on [0,1] scaled, which is quite small, meaning
    # the GP is very wiggly unless the noise is low. But noise is 1e-6.
    # This implies the GP fits the data very tightly.
    # With only 30 evals, we are in a regime where the GP is highly uncertain
    # in unexplored regions.
    
    # Let's try a decaying UCB.
    # Beta starts high and decays.
    # Step 1-5: Beta ~ 1.0 (high exploration)
    # Step 6-15: Beta ~ 0.5
    # Step 16+: Beta ~ 0.1 (near exploitation)
    
    if step <= 5:
        beta = 1.0
    elif step <= 10:
        beta = 0.5
    elif step <= 20:
        beta = 0.2
    else:
        beta = 0.05
        
    ucb = -mu + beta * sigma_safe
    
    # Combine or choose?
    # Often, EI is better than UCB for minimization when f_best is known.
    # However, UCB with tuned beta is robust.
    # Let's look at the inspirations.
    # Inspiration 1 (score 1.0) used pure exploitation? No, it said "uses numpy, scipy; defines acquisition".
    # The current program (score 1.08) uses pure exploitation (-mu).
    # The child p23 (score 1.0) used EI.
    # The child p22 (score 1.08) used pure exploitation.
    # Wait, p22 is the parent of p23. p22 had score 1.08. p23 had score 1.0.
    # So EI (p23) was worse than pure exploitation (p22) in that unpaired comparison?
    # Or is the score "higher is better"?
    # "Scoring (lower is better): mean over the seeds of the simple regret..."
    # "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # So score > 1.0 is better.
    # Current program score: 1.0832.
    # p23 (EI) score: 1.0. This is WORSE than 1.0832.
    # p22 (Exploitation) score: 1.0832.
    # So pure exploitation is currently beating EI.
    
    # Why?
    # With Matern-5/2 and length scale 0.2 (scaled), the GP is very flexible.
    # In the early stages, the uncertainty is high everywhere except near the 5 initial points.
    # Pure exploitation moves to the lowest mean.
    # If the initial points are spread out, the lowest mean might be close to the global minimum (0).
    # EI might be pulled towards areas of high uncertainty (high sigma) which could be far from 0.
    # Since the domain is [-5, 5] and the minimum is at 0, if the GP hasn't seen 0,
    # the mean might be biased by the prior (which is 0 in standardized space? No, standardized targets).
    # The prior mean is 0 in standardized space. The actual function is x^2.
    # Standardized targets: (y - mean) / std.
    # If y are large (e.g., 25), the standardized values are small.
    # The GP fits these.
    
    # Let's try a slightly different approach:
    # Use EI but with a "bonus" for being close to the center? No, we don't have coordinates.
    # We only have mu and sigma.
    
    # Let's try a mix of EI and UCB.
    # Or perhaps the issue is that EI is too conservative.
    # Let's try UCB with a moderate beta, e.g., 0.5 or 1.0, which might explore more.
    # The inspiration mentioned: "UCB with beta=0.5 provides a different exploration-exploitation balance... Measured: 1 -> 14.95"
    # Wait, that observation says "1 -> 14.95". Does that mean the score jumped to 14.95?
    # "Measured: 1 -> 14.95. (strategy of a program that improved island 0)"
    # If the score is "seed/value", and value is regret, then high score means low regret.
    # If a program scored 14.95, that's a massive improvement over 1.08.
    # Let's check the context. "UCB with beta=0.5... Measured: 1 -> 14.95"
    # This suggests UCB with beta=0.5 is the winner.
    
    # Let's implement UCB with beta=0.5.
    # scores = -mu + 0.5 * sigma
    
    # Let's verify the direction.
    # Minimize f.
    # We want to select x that minimizes f(x).
    # GP predicts mu(x) ~ f(x).
    # So we want low mu.
    # UCB for minimization: -mu + beta * sigma.
    # Maximizing this score means minimizing mu and maximizing sigma.
    
    # Let's try this.
    
    scores = -mu + 0.5 * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
