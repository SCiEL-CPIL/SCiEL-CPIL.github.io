import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Quantile-based acquisition: 0.1st percentile (Q0.1)
    # The 0.1st percentile of a Normal(mu, sigma) distribution is mu + sigma * norm.ppf(0.1)
    # norm.ppf(0.1) is approximately -1.28155
    # We want to maximize the probability of improvement, which correlates with
    # targeting the lower tail. A higher score should mean "evaluate next".
    # Since we are minimizing, a lower predicted value is better.
    # The Q0.1 value represents a "pessimistic" (good for minimization) estimate of the function value.
    # To convert this to a score where higher is better, we can simply use -Q0.1.
    # Alternatively, we can just return the quantile value and let the caller minimize it?
    # No, the contract says "Return 2000 finite scores, higher meaning evaluate next".
    # So if we predict a low value (good), we want a high score.
    # Score = - (mu + sigma * norm.ppf(0.1))
    # This is equivalent to -mu - sigma * norm.ppf(0.1) = -mu + sigma * 1.28155
    
    # Let's verify the logic:
    # If sigma is large, the lower tail is very low (very good), so -Q0.1 is very high (very good score).
    # If mu is low, the lower tail is low, so -Q0.1 is high.
    # This behaves like a UCB with a fixed beta of ~1.28.
    
    # Is this materially different from the current UCB with adaptive beta?
    # The current program uses beta = 0.5 + 2.5 * tanh(0.1 * (f_best - 3.0)).
    # At start (f_best ~ high), beta ~ 3.0.
    # As f_best approaches 3.0, beta drops to 0.5.
    # The quantile approach uses a fixed beta of 1.28155.
    
    # To make it more distinct and potentially better, we can use an adaptive quantile level.
    # Early on, we want to explore the far tail (very low quantile, e.g., 0.01).
    # Late on, we want to exploit near the mean (quantile closer to 0.5).
    # Let's use a quantile level q that adapts based on the gap to the minimum.
    
    # Gap = f_best - 3.0
    # If gap is large, we are far from the minimum. We should explore aggressively.
    # Aggressive exploration in minimization means targeting very low values, i.e., very low quantiles.
    # Low quantile = high |z-score| for the negative tail.
    # z = norm.ppf(q). If q=0.01, z ~ -2.33.
    # If q=0.5, z = 0.
    
    # Let's map the gap to the quantile level.
    # gap = 0 -> q = 0.5 (pure mean/exploitation)
    # gap = inf -> q = 0.01 (aggressive exploration)
    
    # Use a smooth transition.
    # q = 0.5 - 0.49 * exp(-gap / scale) ?
    # Or simpler: use the z-score directly.
    
    # Let's stick to the prompt's specific suggestion: "specifically the 0.1st percentile... targeting the lower tail".
    # The prompt suggests a fixed 0.1 quantile. Let's try that first as it is a distinct structural choice from the adaptive UCB/EI blend.
    # However, a fixed 1.28 beta might be too conservative compared to the initial beta=3.0 of the parent.
    # But the parent's beta drops to 0.5 quickly.
    
    # Let's implement the fixed 0.1 quantile as requested by the strategy description.
    # Score = - (mu + sigma * norm.ppf(0.1))
    
    z_01 = norm.ppf(0.1)  # ~ -1.28155
    
    # Quantile value
    q_01 = mu + sigma * z_01
    
    # Score: higher is better. Since we minimize, lower q_01 is better.
    scores = -q_01
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
