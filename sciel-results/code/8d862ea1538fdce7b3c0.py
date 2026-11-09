"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Thompson Sampling-like approach.
    
    Instead of maximizing Expected Improvement (EI), which can be locally greedy,
    we use a surrogate score that mimics the minimizer of a sample from the GP posterior.
    
    For a GP with mean mu and standard deviation sigma, a sample y is drawn from N(mu, sigma^2).
    The minimizer of y is approximately equivalent to minimizing mu - c * sigma, where c is a constant.
    A common choice is c = 1.0 or slightly higher to encourage exploration.
    Here we use c = 1.25 to provide a stronger exploration incentive than pure exploitation (c=0)
    but less than pure random exploration. This is a robust default for smooth functions like Beale.
    
    Since we must return deterministic scores for the same inputs (to ensure reproducibility in the 
    evaluator's loop which calls acquisition once per batch of 2000 candidates), we cannot 
    actually draw random samples inside this function if we want strict determinism across calls 
    with identical inputs. However, the "score" returned is used to select the best candidate.
    
    Wait, the contract says: "For each seed... draws 2000 fresh uniform candidates, computes the 
    posterior there, calls acquisition and evaluates the candidate with the highest score".
    
    If I use a fixed heuristic like mu - 1.25 * sigma, it is deterministic.
    This is known as the "Lower Confidence Bound" (LCB) for minimization.
    LCB(x) = mu(x) - beta * sigma(x).
    Higher score = better (since we want to minimize the function, we want low mu and high sigma).
    So Score = - (mu - beta * sigma) = -mu + beta * sigma.
    
    The current program uses EI + beta * sigma.
    EI is roughly (f_best - mu) * Phi(z) + sigma * phi(z).
    When mu < f_best, EI is positive.
    LCB is -mu + beta * sigma.
    
    Let's analyze the difference.
    EI focuses on improvement over f_best.
    LCB focuses on the lower bound of the function value.
    
    For the Beale function, which has a unique global minimum at 0, LCB is often more robust 
    against getting stuck in local minima because it doesn't depend on f_best as directly 
    in its shape (except through the GP fit itself).
    
    Let's implement LCB with a beta that adapts slightly or is fixed.
    A fixed beta of 1.0 to 2.0 is standard.
    Let's try beta = 1.5.
    
    Score = -mu + 1.5 * sigma
    
    This is materially different from EI because:
    1. It does not saturate when mu is much lower than f_best (EI becomes 0 if mu << f_best, 
       effectively ignoring those points, whereas LCB still rewards low mu).
    2. It treats exploration (sigma) as a linear bonus, whereas EI's dependence on sigma 
       is non-linear and bounded by the scale of improvement.
    
    To ensure we handle the "higher score means evaluate next" requirement:
    We want to select points with LOW mu and HIGH sigma.
    So Score = -mu + beta * sigma.
    
    Let's refine beta.
    If beta is too high, we explore too much.
    If beta is too low, we exploit too much.
    Given 30 evaluations, we have very few chances.
    Early on, sigma is large everywhere.
    Later, sigma is large only in unexplored regions.
    
    A common adaptive beta for LCB in Bayesian Optimization is beta_t = log(t) or similar,
    but with only 30 steps, a fixed value is often safer.
    
    Let's use beta = 1.0.
    Score = -mu + 1.0 * sigma
    
    This is equivalent to Thompson Sampling with a fixed "risk" parameter.
    
    One potential issue: if mu is very negative (unlikely for Beale which is >= 0), 
    the score would be very high.
    
    Let's add a small safety check for numerical stability.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: we want to minimize mu - beta * sigma
    # So the score to maximize is: -mu + beta * sigma
    beta = 1.25
    
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
