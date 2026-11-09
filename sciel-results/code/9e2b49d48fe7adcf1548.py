"""Rewritten program using a different acquisition family: cost-aware upper confidence bound (UCB) with exploration decay."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB: we want to maximize the expected improvement in the
    # minimization sense, penalized by cost. We use a variant that combines
    # expected improvement with an exploration bonus, but scaled by cost.
    #
    # Since we are minimizing, lower mu is better. The standard UCB for minimization
    # would be: mu - beta * sigma (we prefer points with low predicted mean and high uncertainty).
    # However, to be cost-aware, we divide by cost.
    #
    # We also incorporate a term that favors points where the predicted improvement
    # over f_best is large. A robust approach is to use the "probability of improvement"
    # or "expected improvement" but adjusted for cost.
    #
    # Let's use a hybrid: EI per unit cost, but with a small UCB-like exploration term
    # to avoid getting stuck in local optima. Actually, the seed uses EI/cost. Let's try
    # a different approach: use the "knowledge gradient" or a simpler "cost-aware UCB".
    #
    # For minimization, UCB = mu - beta * sigma. We want to maximize this (since lower is better,
    # but UCB gives an upper bound on the true value; for minimization, we want a lower bound,
    # so it's actually a lower confidence bound: LCB = mu + beta * sigma? No, for minimization,
    # we want to explore where the true value might be low. So we use mu - beta * sigma as a
    # "confidence lower bound" on the true minimum? Actually, for minimization, the acquisition
    # function should favor points where the predicted value is low and/or uncertainty is high.
    #
    # Let's use: score = (f_best - mu) / cost - beta * sigma / cost
    # This is like EI without the norm.cdf/pdf, but with an exploration term.
    #
    # Alternatively, use the "probability of improvement" per cost:
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Then score = PI / cost
    #
    # Let's try a combination: use EI but with a different scaling. The seed uses EI/cost.
    # Let's try UCB per cost: (mu - beta * sigma) / cost, but since we want to minimize,
    # we want to maximize the negative of this? No, let's think: if mu is low and sigma is high,
    # mu - beta * sigma is very low, which is good for minimization. So we want to maximize
    # (mu - beta * sigma) / cost? But cost is positive, so dividing by cost penalizes expensive points.
    # However, mu - beta * sigma can be negative, and we want it to be as low as possible.
    # So we should maximize -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost.
    #
    # Let's use: score = (beta * sigma - mu) / cost, with beta = 1.0
    # This is a cost-aware UCB for minimization.
    
    # Add a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB-like score for minimization: we want to maximize (beta * sigma - mu) / cost
    beta = 1.0
    score = (beta * sigma - mu) / cost_safe
    
    # Ensure the output is finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
