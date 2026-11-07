"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Weierstrass 1D parameters (standard definition)
# f(x) = sum_{k=0}^{inf} a^k cos(pi b^k (2x + 1))
# a=0.5, b=10 is common; minimum is at x=0 (and x=±1 in [-1,1] -> x=0 in [-0.5,0.5] if scaled)
# Actually, for x in [-0.5, 0.5], the standard Weierstrass function f(x) = sum a^k cos(pi b^k x) 
# or similar. The task says "weierstrass 1d on [-0.5, 0.5]^1, minimum value 0".
# A common formulation that has minimum 0 at x=0 is:
# f(x) = sum_{k=0}^{K} a^k (1 - cos(pi b^k x)) or similar.
# However, we don't need to know the function to write the acquisition function,
# but knowing the structure might help. The standard Weierstrass function is:
# W(x) = sum_{n=0}^{N} a^n cos(b^n pi x)
# with 0 < a < 1, b odd integer, ab > 1 + 3pi/2.
# The minimum of cos is -1, so the minimum of W is sum a^n (-1) = -1/(1-a).
# But the task says "minimum value 0". This implies the function is shifted.
# e.g., f(x) = W(x) - W_min, or f(x) = sum a^n (1 - cos(...)).
# Regardless, the GP is fit to the observed values. The acquisition function
# operates on the GP posterior (mu, sigma) and f_best.

# The key is to design a good acquisition function.
# The current one is Expected Improvement (EI).
# The directive suggests Knowledge Gradient (KG).
# KG(x) = E[ max(f_best - min(f(x), f_best), 0) ]
#      = E[ max(f_best - f(x), 0) ]  if we assume f_best is the global min so far?
# Actually, KG is defined as the expected improvement in the estimate of the global minimum.
# Let f_best be the current best observed value.
# Let f_new be the value at the new point x.
# The new best will be min(f_best, f_new).
# The improvement in the estimate of the global minimum (assuming f_best is the current best)
# is max(f_best - min(f_best, f_new), 0) = max(f_best - f_new, 0).
# Wait, this is exactly EI!
# EI(x) = E[ max(f_best - f(x), 0) | x, D ]
# where f(x) ~ N(mu(x), sigma^2(x)).
# So EI is a form of KG if we only care about improving the current best.
# However, true KG considers the posterior distribution of the global minimum over the *entire* domain.
# KG(x) = E[ max( f_best - min_{y in domain} f(y), 0 ) ] - f_best
#        = E[ max( f_best - min( f_best, min_{y != x} f(y), f(x) ), 0 ) ] - f_best
# This is complex to compute exactly.
# A common approximation is to use the GP posterior to estimate the distribution of the minimum.
# Or, a simpler variant: "Knowledge Gradient" often refers to the expected reduction in the
# posterior standard deviation of the minimum, or the expected improvement in the best observed value.
# Given the constraints (300s limit, 2000 candidates), we need an efficient approximation.
# One efficient approximation for KG is to sample from the GP posterior and compute the minimum,
# but that's expensive.
# Another approach: Use the "Expected Hypervolume Improvement" or similar, but for minimization.
# Let's stick to a robust variant of EI or a simple KG approximation.
# The prompt mentions "Combining EI with UCB balances exploitation and exploration better".
# UCB (Upper Confidence Bound) for minimization:
# UCB(x) = mu(x) - beta * sigma(x)
# We want to *minimize* the function, so we want low mu and high sigma (exploration).
# So the acquisition score should be *high* when we want to evaluate.
# If we define the score as something we *maximize*, then:
# Score(x) = - (mu(x) - beta * sigma(x)) = -mu(x) + beta * sigma(x)
# Or, we can use a combination of EI and UCB.
# Let's try a weighted combination:
# Score(x) = w_ei * EI(x) + w_ucb * UCB_score(x)
# But EI and UCB have different scales.
# Alternatively, we can use a "Probability of Improvement" (PI) or "Thompson Sampling".
# Let's try a simple KG approximation:
# The expected improvement in the global minimum estimate.
# If we assume the current best is the global minimum, then EI is correct.
# But if there are other regions with lower values, we should explore them.
# A simple way to encourage exploration of unexplored regions is to add a penalty for
# being close to observed points, or to use the GP's predictive variance.
# Let's try a "Modified EI" that incorporates a bonus for high uncertainty, but in a
# way that is more principled than just adding sigma.
# Actually, the standard EI already incorporates uncertainty through sigma.
# The issue with EI is that it can be too exploitative.
# Let's try "Expected Improvement with a floor" or "EI with a scaling factor".
# Or, let's try the "Knowledge Gradient" approximation by Chaudhuri et al. (2010).
# But that might be too complex.
# Let's try a simpler structural change: **Thompson Sampling**.
# In Thompson Sampling, we sample a function from the GP posterior and pick the point
# that minimizes that sample.
# However, we are given mu and sigma for 2000 candidates. We can sample from
# f(x) ~ N(mu(x), sigma(x)^2) for each candidate, but we need to account for correlations.
# Since we don't have the covariance matrix between candidates, we can't easily sample
# a full function. We can only sample independently, which is not correct.
# So Thompson Sampling is not directly applicable here without the covariance matrix.
# Let's go back to EI and UCB.
# The observation says "Combining EI with UCB ... Measured: 1 -> 1.032".
# So a combination might be better.
# Let's define the score as:
# Score(x) = EI(x) + lambda * sigma(x)
# where lambda is a small constant to encourage exploration.
# Or, Score(x) = EI(x) / (sigma(x) + epsilon) ? No, that would reduce exploration.
# Let's try: Score(x) = EI(x) + beta * sigma(x)
# We need to choose beta. Since EI is in objective units and sigma is in objective units,
# they are comparable.
# Let's set beta = 0.1 or something small.
# But the directive says "do not micro-tune constants".
# So we need a structural change that doesn't rely on tuning a constant.
# How about using the **Probability of Improvement (PI)**?
# PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
# This is a probability, so it's bounded between 0 and 1.
# EI is unbounded.
# We could combine PI and EI, but again, scaling issue.
# Let's try a different structural change: **Use the GP posterior to estimate the
# location of the global minimum more directly.**
# For example, we can compute the expected value of the minimum over the 2000 candidates:
# E[min(f(x_i))]
# and then use the difference between f_best and this expected minimum as a guide.
# But this is a single number, not a score for each candidate.
# We need a score for each candidate.
# Let's reconsider the **Knowledge Gradient (KG)** approximation.
# A common approximation is:
# KG(x) ≈ E[ max(f_best - f(x), 0) ] - E[ max(f_best - f(x'), 0) ]
# where x' is the best candidate among the others.
# This is still complex.
# Let's try a simpler idea: **Use the predictive mean and variance to define a
# "regret" score that is more sensitive to the global structure.**
# For example, we can compute the "Expected Regret" as:
# ER(x) = E[ max(f(x) - f_min, 0) ]
# where f_min is the true global minimum. But we don't know f_min.
# We can approximate f_min by the minimum of the predictive means:
# f_min_approx = min(mu(x_i))
# Then ER(x) = E[ max(f(x) - f_min_approx, 0) ]
# This is similar to EI but with a different baseline.
# If f_min_approx is lower than f_best, then this might encourage exploration.
# Let's try this.
# f_min_approx = min(mu)
# z = (f_min_approx - mu) / sigma
# Score(x) = (f_min_approx - mu) * Phi(z) + sigma * phi(z)
# This is EI with f_best replaced by f_min_approx.
# If f_min_approx < f_best, this will have higher scores for points with low mu,
# even if they are not better than f_best.
# This could encourage exploration of regions with low predictive means.
# Let's try this structural change.
# We need to be careful: if f_min_approx is much lower than f_best, the EI will be large
# for all points, which might not be discriminative.
# We can normalize by the range of mu or something.
# But let's first try the simple version.

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute the minimum of the predictive means
    f_min_approx = np.min(mu)
    
    # Compute EI with f_min_approx as the baseline
    z = (f_min_approx - mu) / sigma
    ei = (f_min_approx - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # We want to maximize the score, so higher is better.
    # EI is higher when f_min_approx - mu is large (i.e., mu is small) and sigma is large.
    # This should encourage exploration of regions with low predictive means.
    
    # However, if f_min_approx is very low, EI might be dominated by the term
    # (f_min_approx - mu) * Phi(z). For points with mu close to f_min_approx,
    # z is close to 0, Phi(z) is close to 0.5, so EI is close to 0.5 * (f_min_approx - mu) + ...
    # For points with mu much larger than f_min_approx, z is large negative, Phi(z) is close to 0,
    # so EI is close to 0.
    # So EI is highest for points with mu close to f_min_approx and large sigma.
    # This seems reasonable.
    
    # But we also want to ensure that we don't ignore f_best entirely.
    # If f_best is much lower than f_min_approx, then f_min_approx is a poor estimate.
    # In that case, we should fall back to standard EI with f_best.
    # Let's blend the two:
    # If f_best < f_min_approx, use standard EI.
    # Otherwise, use the modified EI.
    # Or, we can use the minimum of f_best and f_min_approx as the baseline.
    # baseline = min(f_best, f_min_approx)
    # z = (baseline - mu) / sigma
    # ei = (baseline - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # This is a simple structural change: use the best available estimate of the global minimum
    # as the baseline for EI.
    
    baseline = min(f_best, np.min(mu))
    z = (baseline - mu) / sigma
    ei = (baseline - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    return ei
# EVOLVE-BLOCK-END
