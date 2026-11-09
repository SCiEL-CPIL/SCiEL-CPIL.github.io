"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We don't know the exact step count, but the problem states 30 evaluations total.
# We can use a fixed exploration constant that is generally effective for small budgets.
# A common choice is beta = 2.0 or similar. 
# Alternatively, we can estimate the step count if we had access to it, but we don't.
# However, we can just use a static UCB.
# Let's use beta = 2.0 which is a standard choice for UCB in BO.
# To be slightly more adaptive, we could use a beta that decreases, but without 
# knowing the step, a fixed beta is safer.
# Let's try beta = 1.5 to 2.0. Let's stick with 2.0.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    UCB = mu + beta * sigma
    
    Since we are minimizing, we want to maximize the acquisition score.
    Higher UCB means higher potential to find a lower value (if sigma is high, 
    it's uncertain, so we explore; if mu is low, it's good, so we exploit).
    
    Wait, UCB for minimization is typically:
    UCB(x) = mu(x) - beta * sigma(x) ? 
    No, standard UCB for maximization is mu + beta * sigma.
    For minimization, we want to find low values.
    If we assume the function values are "good" when low, 
    we can define the acquisition as:
    A(x) = mu(x) - beta * sigma(x)
    This encourages regions with low mean (good) and high variance (uncertain).
    
    However, the prompt says "higher meaning evaluate next".
    So if I return `mu - beta * sigma`, a very low mu (good) gives a very negative score.
    This would be interpreted as "low priority" if the evaluator just picks the max.
    
    Let's re-read carefully: "Return 2000 finite scores, higher meaning evaluate next".
    
    If I use UCB for minimization:
    We want to maximize the probability of finding a new minimum.
    The "Upper Confidence Bound" for the *minimum* is actually the *Lower* Confidence Bound for the value?
    
    Let's think about the distribution.
    f(x) ~ N(mu, sigma^2).
    We want to find x that minimizes f(x).
    
    Standard EI for minimization:
    EI(x) = E[max(0, f_best - f(x))]
    This is positive when f(x) is likely to be lower than f_best.
    
    Standard UCB for minimization:
    Often defined as:
    UCB_min(x) = mu(x) - beta * sigma(x)
    
    If we return `mu - beta * sigma`, 
    - If mu is very low (e.g., -1.8), score is low (e.g., -1.8).
    - If mu is high (e.g., 1.0) and sigma is high (e.g., 1.0), score is 1.0 - 2.0 = -1.0.
    - If mu is high (e.g., 1.0) and sigma is low (e.g., 0.1), score is 1.0 - 0.2 = 0.8.
    
    The evaluator picks the candidate with the *highest* score.
    So `mu - beta * sigma` would prefer high mu (bad) and low sigma (exploit known bad areas?).
    This is the opposite of what we want.
    
    We want to prefer low mu (good) and high sigma (explore).
    So we should return a score that is high when mu is low and sigma is high.
    
    Let's invert the logic.
    Score = - (mu - beta * sigma) = -mu + beta * sigma.
    
    Let's check:
    - If mu is very low (-1.8), -mu is high (1.8).
    - If sigma is high, beta * sigma is high.
    - So high score corresponds to low mu and high sigma.
    - This matches the requirement "higher meaning evaluate next" for a minimization problem.
    
    So, Acquisition Score = -mu + beta * sigma.
    
    Let's verify with EI.
    EI is always >= 0. It is high when f_best - mu is large (mu is low) and sigma is large.
    EI effectively combines -mu and sigma.
    
    UCB with Score = -mu + beta * sigma is a linear combination.
    EI is non-linear (involves CDF and PDF).
    
    Is UCB better?
    UCB is simpler and can be more aggressive in exploration if beta is tuned well.
    EI can be "lazy" if the GP is confident (low sigma) but wrong (local min).
    
    Let's use beta = 2.0.
    """
    beta = 2.0
    # Ensure sigma is non-negative and handle division by zero if we were normalizing, 
    # but here we just add.
    # sigma is already standard deviation.
    scores = -mu + beta * sigma
    
    # Safety check: ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
