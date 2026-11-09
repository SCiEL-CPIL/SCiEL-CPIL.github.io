"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Pure UCB-based acquisition function.
    
    Strategy:
    - Use Upper Confidence Bound (UCB) for minimization.
    - We want to find points with low function values.
    - The "optimistic" lower bound of the function value is mu - kappa * sigma.
    - We want to maximize the potential for finding a low value, so we maximize
      the negative of this lower bound? No, standard UCB for minimization is:
      Minimize LCB = mu - kappa * sigma.
      But we need to return scores where HIGHER is better.
      So we return -LCB = -mu + kappa * sigma.
      
    Wait, let's re-read the directive carefully:
    "Use a Pure Upper Confidence Bound (UCB) strategy by returning the score $-\mu - \kappa \sigma$"
    
    Let's check the math for minimization.
    If we want to minimize f(x), and we have a GP posterior N(mu, sigma^2).
    The probability that f(x) < f_best is high if mu is low and sigma is large.
    UCB (Upper Confidence Bound) usually refers to the upper bound of the reward in maximization problems.
    In minimization, the analogous bound for "potential improvement" or "optimism" is the Lower Confidence Bound (LCB) of the function value:
    LCB = mu - kappa * sigma.
    We want to select the point with the lowest LCB.
    Since the interface requires returning scores where HIGHER is better, we should return -LCB.
    -LCB = -(mu - kappa * sigma) = -mu + kappa * sigma.
    
    However, the directive explicitly says: "returning the score $-\mu - \kappa \sigma$".
    Let's analyze what $-\mu - \kappa \sigma$ does.
    If mu is low (good), -mu is high (good score).
    If sigma is high (uncertain), -kappa*sigma is low (bad score).
    This penalizes uncertainty. This is an Exploitation-only or Pessimistic strategy.
    This contradicts the stated goal of "aggressively targets regions of high posterior variance".
    
    Let's re-read the directive text again.
    "Strategy to try in this change: Use a Pure Upper Confidence Bound (UCB) strategy by returning the score $-\mu - \kappa \sigma$ with a high exploration constant $\kappa=5.0$, which aggressively targets regions of high posterior variance..."
    
    There is a contradiction in the directive's text between the formula $-\mu - \kappa \sigma$ and the description "targets regions of high posterior variance".
    $-\mu - \kappa \sigma$ penalizes high sigma.
    $-\mu + \kappa \sigma$ rewards high sigma.
    
    Usually, in BO for minimization:
    1. EI: Exploitative, favors low mu, small sigma.
    2. UCB (for max reward): mu + kappa * sigma.
    3. LCB (for min cost): mu - kappa * sigma. We minimize this.
       To convert to a "maximize score" interface: Score = - (mu - kappa * sigma) = -mu + kappa * sigma.
       
    If I follow the formula $-\mu - \kappa \sigma$, it behaves like a "Pessimistic LCB" or similar, which is bad for exploration.
    If I follow the description "targets regions of high posterior variance", I must use $-\mu + \kappa \sigma$.
    
    Given the context of "escaping local minima" and "high exploration constant", the intent is clearly to reward uncertainty. The formula in the prompt text might have a typo (minus instead of plus) or I am misinterpreting "UCB".
    Let's look at the standard definitions.
    UCB is typically for maximization: $a(x) = \mu(x) + \kappa \sigma(x)$.
    If we map minimization to maximization by defining reward $r(x) = -f(x)$, then $\mu_r = -\mu_f$ and $\sigma_r = \sigma_f$.
    Then UCB for reward is $\mu_r + \kappa \sigma_r = -\mu_f + \kappa \sigma_f$.
    This matches $-\mu + \kappa \sigma$.
    
    The directive says: "returning the score $-\mu - \kappa \sigma$".
    If I use $-\mu - \kappa \sigma$, I am effectively using UCB on the cost function directly? No, UCB on cost would be $\mu + \kappa \sigma$ (Upper bound of cost). We want to minimize cost, so we'd want the Lower bound of cost $\mu - \kappa \sigma$.
    
    Let's assume the text description "aggressively targets regions of high posterior variance" is the primary intent, and the formula has a sign error for the sigma term. I will implement the strategy that rewards uncertainty: $-\mu + \kappa \sigma$.
    
    Wait, let's look at the previous program. It used `score = -mu + kappa * sigma_safe`. That is the standard optimistic LCB (or UCB on negative function).
    The directive says "DIFFERENT strategy... not refining the current program".
    The current program uses dynamic kappa.
    The directive suggests "Pure UCB... with a high exploration constant kappa=5.0".
    
    If I use $-\mu + 5.0 \sigma$, it is structurally similar to the current program but with a fixed, much higher kappa.
    Is that "materially different"? It's a parameter change, but a large one.
    However, the directive explicitly names the formula $-\mu - \kappa \sigma$.
    Let's consider if there is a different interpretation.
    Maybe the user wants me to try the *penalty* version to see if it fails? No, "Idea target... design the new strategy so that its result answers this open question".
    
    Let's look at Inspiration 1 (score 1.1513). It likely used a better exploration strategy.
    
    Let's stick to the most robust interpretation of "UCB for minimization with high exploration":
    Score = -mu + kappa * sigma.
    
    Why? Because "targets regions of high posterior variance" is the explicit goal. The formula in the text is likely a typo for the standard UCB-on-reward form.
    
    However, to be safe and strictly follow the "Pure UCB" label which often implies the standard form $\mu + \kappa \sigma$ (for maximization), and since we are minimizing, we transform to $-\mu + \kappa \sigma$.
    
    Let's implement $-\mu + 5.0 \sigma$.
    
    Wait, the current program (score 1.0) already does this but with dynamic kappa (0.5 to 3.0).
    Changing to fixed kappa=5.0 is a significant shift to much more aggressive exploration.
    
    Let's double check if "Pure UCB" implies something else.
    In some contexts, UCB is just $\mu + \kappa \sigma$. If we treat the problem as maximizing $-f$, then $\mu_{-f} = -\mu_f$.
    So UCB is $-\mu_f + \kappa \sigma_f$.
    
    I will proceed with $-\mu + 5.0 \sigma$.
    
    Deadline guard: The function is very fast (vectorized numpy ops). No explicit time check needed for 2000 points, but I will add a basic safety check just in case.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 0.0)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # High exploration constant
    kappa = 5.0
    
    # Score = -mu + kappa * sigma
    # This rewards low mean (good for minimization) and high variance (good for exploration)
    score = -mu + kappa * sigma
    
    # Ensure output is finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
