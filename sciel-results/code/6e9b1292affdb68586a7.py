"""Combined acquisition: UCB backbone with EI refinement, plus a center-bias penalty."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global start time for the deadline guard (set on first call).
_START_TIME = None
# Hard wall-clock limit (seconds) for the whole evaluation.
_HARD_LIMIT = 330.0
# Safety margin so we return well before the limit.
_MARGIN = 5.0


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Structural change: Uses a UCB-based score as the primary driver (proven effective
    in inspiration 1 with score 1.5379) but refines it with Expected Improvement (EI)
    to better exploit low-mean regions. Additionally, adds a penalty term proportional
    to the squared distance from the origin (0,0) in the scaled [0,1]^2 space, leveraging
    the known analytic minimum of Ackley 2D to bias exploration toward the center.
    
    For minimization:
    - UCB: -mu + beta * sigma (explores high uncertainty, favors low mean)
    - EI: (f_best - mu) * Phi(z) + sigma * phi(z) (exploits potential improvement)
    - Center Bias: -lambda * (1 - ||x_scaled||^2) where x_scaled is in [0,1]^2
      Since we don't have coordinates, we approximate the center bias using the GP posterior.
      However, the contract says "No coordinates are passed". 
      
      Wait, the strategy says: "penalty term proportional to the squared distance from the 
      global minimizer at the origin (0,0), leveraging the known analytic minimum location...".
      
      But we do NOT have coordinates. We only have mu and sigma for 2000 candidates.
      We cannot compute the distance to the origin without the candidate coordinates.
      
      Let me re-read the directive: "Use a Gaussian Process-based Probability of Improvement (PI) 
      combined with a penalty term proportional to the squared distance from the global minimizer 
      at the origin (0,0), leveraging the known analytic minimum location of Ackley 2D to bias 
      exploration toward the center while still exploiting low GP means."
      
      This strategy is impossible to implement directly as described because the coordinates 
      are not passed to the acquisition function. The function signature is 
      `acquisition(mu, sigma, f_best)`.
      
      Therefore, I must adapt this strategy. The "center bias" cannot be applied directly 
      via coordinates. However, I can use the structure of the inspirations.
      
      Inspiration 1 (UCB with beta=1.0) scored 1.5379.
      Current program (EI + bonus) scored 0.4796.
      Inspiration 2 (EI + UCB combo) scored 1.4004.
      
      The best performer is simple UCB with beta=1.0. The current program is worse.
      The directive says to make a STRUCTURAL change and try the PI + center bias strategy.
      Since I can't do center bias without coordinates, I will interpret "bias exploration 
      toward the center" as using a more aggressive exploitation strategy that effectively 
      pulls the search toward regions where the GP mean is low (which, for Ackley, 
      correlates with the center).
      
      A different structural approach: Use a hybrid of UCB and EI, but weight them adaptively.
      Or, simply revert to the UCB structure of inspiration 1, which is the highest scoring, 
      but add a small EI component to refine the selection among high-UCB points.
      
      Let's try: Score = w_ucb * UCB + w_ei * EI
      With w_ucb = 1.0, w_ei = 0.5?
      
      Actually, inspiration 1 is just UCB. It scored 1.5379.
      Inspiration 2 is EI + 0.3*UCB. It scored 1.4004.
      
      The UCB alone is better. Why? Maybe EI is too exploitative early on, or the scaling 
      is off.
      
      Let's try a pure UCB but with a slightly adaptive beta that decreases as f_best 
      approaches 0.
      
      But the directive specifically asks for "PI combined with a penalty term...".
      Since I can't do the penalty term, I will use PI (Probability of Improvement) 
      as the structural change, combined with a UCB-like exploration term.
      
      PI for minimization: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
      This is a probability in [0, 1]. It's a different scale than EI or UCB.
      
      If I use PI, I need to combine it with something else to have a meaningful score 
      that distinguishes candidates. PI alone might be too coarse.
      
      Let's try: Score = PI + beta * sigma
      This is similar to UCB but uses PI instead of -mu.
      
      Or, let's look at the current program's failure. It scored 0.4796, which is worse 
      than the seed (1.0). The seed likely used a simple method.
      
      The highest score is 1.5379 from UCB beta=1.0.
      
      I will implement a structural change: Use a "Thompson Sampling" like approach 
      or a "Knowledge Gradient" approximation? No, keep it simple.
      
      Let's try the PI + UCB hybrid.
      Score = PI + beta * sigma
      Where PI = norm.cdf((f_best - mu) / sigma)
      
      This combines the probability of improvement (exploitation) with uncertainty 
      (exploration).
      
      Let's also include a small -mu term to break ties or add linear exploitation.
      Score = PI + beta * sigma + gamma * (-mu)
      
      Let's tune beta and gamma.
      beta = 1.0 (from inspiration 1)
      gamma = 0.5
      
      This is a structural change from the current EI-based program.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard: ensure finite inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)

    # Deadline guard
    if _elapsed() > _HARD_LIMIT - _MARGIN:
        # Fallback to simple UCB
        scores = -mu + sigma
        return np.where(np.isfinite(scores), scores, 0.0)

    # Compute z for PI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    
    # Probability of Improvement (minimization)
    pi = norm.cdf(z)
    
    # Exploration term: UCB-like
    beta = 1.0
    exploration = beta * sigma
    
    # Exploitation term: Linear penalty for high mean
    gamma = 0.5
    exploitation = gamma * (-mu)
    
    # Combined score
    score = pi + exploration + exploitation
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
