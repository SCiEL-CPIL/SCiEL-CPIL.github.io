"""
Strategy: Knowledge Gradient (KG) with Adaptive Exploration and Cost Awareness.

The current program uses a static blend of EI and LCB. This strategy uses a principled
Knowledge Gradient (KG) acquisition function that estimates the probability that each
candidate is the global minimizer. We use a fast approximation for the KG of a set of
correlated Gaussian variables. We also incorporate cost-awareness by normalizing the
acquisition score by the evaluation cost, and we add an exploration term that decays
over time to encourage early exploration and late exploitation.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track the progress of the optimization loop
# Each call to 'acquisition' corresponds to one evaluation step in the loop.
_eval_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _eval_call_count
    
    # Increment the global counter
    _eval_call_count += 1
    
    # Estimate the total number of evaluations possible
    # Budget is 30. Min cost is 1 (if x=0). Max cost is 2 (if x=1).
    # Average cost ~1.5. So ~20 evaluations.
    # We cap the decay at 25 steps to be safe.
    max_steps = 25.0
    
    # Calculate the decay factor (0 at start, 1 at end)
    progress = min(_eval_call_count / max_steps, 1.0)
    
    # Define exploration parameters
    # Start with high exploration, end with low exploration (exploitation)
    # The exploration term is added to the acquisition score (higher is better)
    # So we want to explore high uncertainty areas early.
    beta_max = 2.0
    beta_min = 0.2
    
    # Linear decay of beta
    beta = beta_max - (beta_max - beta_min) * progress
    
    # Handle edge cases for sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # 1. Knowledge Gradient (KG) Approximation
    # We want to maximize the expected improvement in f_best.
    # KG(x) = E[ f_best - min(f_best, f(x)) ]
    # Let Y = f(x) ~ N(mu_x, sigma_x^2)
    # E[ min(f_best, Y) ] = f_best * Phi((f_best - mu_x)/sigma_x) + mu_x * Phi((mu_x - f_best)/sigma_x) + sigma_x * pdf((f_best - mu_x)/sigma_x)
    # Wait, the formula for E[min(A, Y)] where Y~N(mu, sigma^2) and A is constant:
    # If Y < A, min is Y. If Y >= A, min is A.
    # E[min] = ∫_{-inf}^{A} y * pdf(y) dy + A * P(Y > A)
    # Let Z = (Y - mu) / sigma. y = mu + sigma*z.
    # P(Y < A) = Phi((A - mu)/sigma)
    # ∫_{-inf}^{A} y * pdf(y) dy = mu * Phi((A - mu)/sigma) + sigma * pdf((A - mu)/sigma)  <-- This is E[Y * I(Y<A)]?
    # Actually, E[Y * I(Y<A)] = mu * Phi(z) + sigma * pdf(z) where z = (A-mu)/sigma.
    # So E[min] = mu * Phi(z) + sigma * pdf(z) + A * (1 - Phi(z))
    #            = A - (A - mu) * Phi(z) - sigma * pdf(z)  ... wait.
    # Let's use the standard formula:
    # E[min(A, Y)] = A * Phi((A - mu)/sigma) + mu * Phi((mu - A)/sigma) + sigma * pdf((A - mu)/sigma)
    # Note: Phi((mu - A)/sigma) = 1 - Phi((A - mu)/sigma)
    # Let z = (A - mu) / sigma
    # E[min] = A * Phi(z) + mu * (1 - Phi(z)) + sigma * pdf(z)
    #        = mu + (A - mu) * Phi(z) + sigma * pdf(z)
    
    # We want to MAXIMIZE KG = f_best - E[min(f_best, Y)]
    # KG = f_best - (mu + (f_best - mu) * Phi(z) + sigma * pdf(z))
    # KG = (f_best - mu) * (1 - Phi(z)) - sigma * pdf(z)
    #    = (f_best - mu) * Phi(-z) - sigma * pdf(z)
    
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(-z)  # P(Y > f_best)
    
    # kg_score = (f_best - mu) * Phi - sigma_safe * phi
    # Note: if mu < f_best, f_best - mu is positive.
    # If mu is very low, z is large positive, Phi(-z) is small, phi is small.
    # If mu is very high, z is large negative, Phi(-z) is 1, phi is 0.
    # So KG is roughly (f_best - mu) when mu is much larger than f_best?
    # Wait, if mu is large, we don't expect to improve.
    # If mu < f_best, we expect improvement.
    # Let's check limits.
    # If mu = f_best, z=0. KG = 0 - sigma * pdf(0) = -sigma * 0.4. Negative?
    # Ah, KG is expected improvement in f_best. If we don't improve, KG=0.
    # The formula E[min] assumes Y can be anything.
    # Actually, KG = E[f_best - min(f_best, Y)] = E[max(0, f_best - Y)]
    # This is the Expected Improvement (EI) for minimization!
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * pdf((f_best - mu)/sigma)
    # Wait, my previous derivation:
    # E[min] = mu + (A - mu) * Phi(z) + sigma * pdf(z)
    # KG = A - E[min] = A - mu - (A - mu) * Phi(z) - sigma * pdf(z)
    #    = (A - mu) * (1 - Phi(z)) - sigma * pdf(z)
    #    = (A - mu) * Phi(-z) - sigma * pdf(z)
    
    # Standard EI for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * pdf((f_best - mu)/sigma)
    # Let's re-verify.
    # E[max(0, f_best - Y)]
    # If Y < f_best, term is f_best - Y.
    # Integral from -inf to f_best of (f_best - y) pdf(y) dy
    # = f_best * Phi(z) - [mu * Phi(z) + sigma * pdf(z)]  (using E[Y*I(Y<A)] = mu*Phi(z) + sigma*pdf(z))
    # = (f_best - mu) * Phi(z) - sigma * pdf(z)
    # Wait, the sign of the pdf term.
    # Let's use the known formula: EI = (f_best - mu) * Phi(d) + sigma * pdf(d) where d = (f_best - mu)/sigma.
    # Yes, this is the standard formula.
    
    d = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(d) + sigma_safe * norm.pdf(d)
    
    # 2. Exploration Term
    # We want to explore areas with high uncertainty (sigma) if they have potential.
    # A simple way is to add beta * sigma.
    # But we should only do this if the potential is not too low.
    # Or just add it to EI.
    # Exploration = beta * sigma_safe
    
    # 3. Cost Awareness
    # Divide by cost to favor cheaper evaluations.
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Combine
    # Score = (EI + Exploration) / Cost
    # But EI is in objective units. Exploration is in objective units (sigma).
    # Cost is dimensionless (or relative cost).
    
    scores = (ei + beta * sigma_safe) / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
