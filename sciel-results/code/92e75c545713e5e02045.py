"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# We use a simple state to track if we are in the early phase (exploration)
# or late phase (exploitation). Since we don't have explicit step count,
# we can infer it from the number of evaluations if we had it, but we only have mu/sigma.
# However, the problem states the loop runs until 30 evaluations.
# We can approximate progress by looking at the spread of sigma or just use a fixed strategy.
# A robust approach for multi-modal functions like Styblinski-Tang is to use
# a "batch" of points. Since we must return 2000 scores for 2000 candidates,
# and the evaluator picks the argmax, we can simulate a batch by assigning
# high scores to a diverse set of top candidates.

# Strategy: Thompson Sampling with a diversity penalty.
# 1. Sample a function value from the posterior: f_sample = mu + sigma * z
# 2. To encourage diversity, we don't just pick the single best.
#    Instead, we can view this as: for each candidate x_i, we want to know
#    if it is likely to be the global minimum among a batch of K points.
#    However, the evaluator only picks ONE point per step.
#    So, if we return a score for each point, and the evaluator picks the max,
#    we effectively pick one point.
#    
#    To get diversity in the *sequence* of evaluations, we need the acquisition
#    to change over time or to be sensitive to past evaluations.
#    But we don't have past evaluations explicitly, only f_best.
#    
#    Alternative: Use a "virtual batch" Thompson Sampling.
#    Imagine we are choosing a batch of size B. The score for point i is the
#    probability that it is the best in the batch, or simply its rank.
#    If we simply return the sampled value, we pick the single best sample.
#    This is standard TS.
#    
#    To improve on standard TS for multi-modal functions, we can add a small
#    random offset to the sampled values to break ties and encourage exploring
#    different modes if the posterior is flat.
#    
#    However, a more structural change is to use **Knowledge Gradient** (KG) or
#    **Expected Hypervolume Improvement** (EHI) approximations, but they are slow.
#    
#    Let's try a **Hybrid EI with a dynamic beta** based on the uncertainty
#    distribution. If sigma is high everywhere, explore. If sigma is low, exploit.
#    
#    Actually, the most effective simple structural change for multi-modal 1D
#    is to use **Thompson Sampling** directly. It naturally balances exploration
#    and exploitation. The current program uses EI.
#    
#    Let's implement Thompson Sampling:
#    score_i = mu_i + sigma_i * z_i  where z_i ~ N(0, 1)
#    We want to maximize the probability of improvement, so we minimize the function.
#    So we want the score to be LOW? No, the contract says "higher meaning evaluate next".
#    So we should return -f_sample? Or just f_sample and the evaluator minimizes?
#    Contract: "evaluates the candidate with the highest score".
#    So if we want to minimize f, we should return -f_sample.
#    
#    Let's refine:
#    1. Generate a random z vector.
#    2. Compute sampled_values = mu + sigma * z.
#    3. Return -sampled_values.
#    
#    Why is this better than EI?
#    EI is deterministic given mu, sigma, f_best. TS is stochastic.
#    In multi-modal landscapes, EI tends to get stuck in the first found basin
#    if f_best is already good. TS has a non-zero probability of jumping to
#    another basin if the posterior variance is high there.
#    
#    To make it even better, we can use a **normalized** TS score to prevent
#    numerical issues, but raw TS is usually fine.
#    
#    Let's add a small exploration bonus to the TS score to ensure we don't
#    just pick the single best sample if there are many similar ones.
#    Actually, standard TS picks the best sample. That is fine.
#    
#    One issue: The evaluator picks the argmax of the scores we return.
#    If we return -sampled_values, it picks the lowest sampled value.
#    This is correct.
#    
#    Let's verify the "higher meaning evaluate next" constraint.
#    Yes, return scores such that the best candidate has the highest score.
#    Since we minimize f, best candidate has lowest f.
#    So score = -f_sampled.
#    
#    We must ensure finite scores.

def acquisition(mu, sigma, f_best):
    # Generate a single random vector for Thompson Sampling
    # Using a fixed seed would make it deterministic, but we want stochasticity
    # to escape local minima. The evaluator runs multiple seeds, so this is fine.
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior
    z = np.random.randn(mu.shape[0])
    sampled_values = mu + sigma * z
    
    # We want to minimize the function, so we want the lowest sampled value.
    # The evaluator picks the highest score.
    # So we return the negative of the sampled values.
    scores = -sampled_values
    
    # To ensure stability and avoid any potential NaN/Inf issues (though unlikely with randn),
    # we can clip the scores.
    # The problem states "Return 2000 finite scores".
    # np.random.randn is finite. mu and sigma are finite.
    # So scores are finite.
    
    return scores
# EVOLVE-BLOCK-END
