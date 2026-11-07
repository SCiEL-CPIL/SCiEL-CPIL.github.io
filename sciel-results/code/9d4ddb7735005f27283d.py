# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    n = mu.size

    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 10:
        # Return a neutral score if we are running out of time
        # Using zeros is safe as it doesn't favor any point specifically,
        # but usually we want to exploit the best known point if possible.
        # However, without coordinates, we can't. So zeros (random choice effectively)
        # or maybe just return the current best index? No, we don't have indices.
        # Just return zeros to be safe.
        return np.zeros(n, dtype=float)

    # Sanitize inputs
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    f_best = float(f_best)
    sigma_safe = np.maximum(sigma, 1e-12)

    # 1. Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # 2. Lower Confidence Bound (LCB) for minimization
    # We want to minimize, so we look for low predicted values with high uncertainty.
    # LCB = mu - beta * sigma
    # We want to maximize the "score", so we can use -LCB or just LCB if we are minimizing?
    # Wait, the task says "Return 2000 finite scores, higher meaning evaluate next".
    # So we want a high score for points we want to evaluate.
    # For minimization, we want low f(x).
    # EI is high when we expect a large drop from f_best.
    # LCB (mu - beta*sigma) is low when we expect a low value.
    # So we should use -LCB as the score? Or just EI?
    # Let's check Inspiration 1: "pure Lower Confidence Bound (LCB) with a high fixed exploration coefficient (beta=5.0)".
    # If they used LCB as the score, they must have meant the score is such that lower LCB is better?
    # No, "higher meaning evaluate next".
    # If I use score = -LCB, then low LCB (good) -> high score.
    # If I use score = EI, high EI (good) -> high score.
    
    # The previous program used: score = ei + beta * sigma_safe
    # This is effectively a mix of EI and a pure exploration term (sigma).
    # Note that -LCB = -mu + beta*sigma.
    # EI is roughly related to -mu when f_best is fixed.
    
    # Let's try a different structural approach suggested by the stagnation directive.
    # "GP-UCB with time-varying beta_t provides theoretically grounded exploration... while adding normalized EI maintains exploitation pressure"
    # This was measured to improve score from 1.0 to 6.744.
    # The current score is 1.27. The best inspiration is 6.79.
    # The diff shows the child (6.79) removed the complex mix and likely simplified or changed the beta strategy.
    
    # Let's look at the diff between parent (1.27) and child (6.79).
    # The child code is not fully visible, but the description says "pure Lower Confidence Bound (LCB) with a high fixed exploration coefficient (beta=5.0)".
    # Wait, the description says "whereas the | uses numpy, time; defines acquisition".
    # And "Insiration 1 (score 6.7904) uses a pure Lower Confidence Bound (LCB) with a high fixed exploration coefficient (beta=5.0)".
    
    # If pure LCB with beta=5.0 works so well, let's try that.
    # Score = -LCB = -(mu - beta * sigma) = -mu + beta * sigma.
    # But we also have EI.
    # The "GP-UCB + normalized EI" strategy got 6.744.
    # The "Pure LCB" strategy got 6.79.
    
    # Let's try to combine them intelligently or just switch to the pure LCB strategy with a tuned beta.
    # However, the directive says "Make a STRUCTURAL change".
    # The current code is a mix of EI and sigma.
    # A pure LCB is structurally different from a mix? It's simpler.
    
    # Let's try a time-varying beta for LCB, as suggested by the knowledge cell.
    # beta_t = beta_0 * sqrt(log(t)) ?
    # We have _call_count.
    # t = _call_count
    # beta_t = 5.0 * np.sqrt(np.log(t + 1)) / np.sqrt(np.log(30)) ? No, just scale it.
    
    # Actually, let's look at the "Pure LCB" inspiration again.
    # If beta=5.0 is fixed, it might be over-exploring late in the game.
    # Let's try a decay.
    
    # But wait, the best score (6.79) used "pure LCB with high fixed exploration coefficient".
    # Let's try to replicate that but with a slight decay to avoid missing the optimum in the last few steps.
    
    # Score = -mu + beta * sigma
    # Let's try beta that starts high and decays.
    # t ranges from 1 to 25 (approx, since 5 initial + 20 steps).
    # Actually, _call_count starts at 1.
    
    # Let's try:
    # beta = 5.0 * (1.0 / (1.0 + 0.1 * (_call_count - 1)))
    # Or simpler:
    # beta = 5.0
    
    # Let's stick to the inspiration that worked: Pure LCB with beta=5.0.
    # But to make it a "structural change" from the current EI+sigma mix, I will implement it cleanly.
    # The current code is: score = ei + beta * sigma_safe
    # This is NOT pure LCB. It's EI + exploration.
    # Pure LCB is: score = -mu + beta * sigma
    
    # Let's try that.
    
    # Also, consider that f_best is used in EI but not in LCB (except implicitly in the choice of beta if it were adaptive).
    # If I use pure LCB, I ignore f_best entirely for the score calculation (other than potentially adapting beta).
    
    # Let's try a hybrid that is structurally distinct: 
    # Use EI for exploitation, but weight it heavily against a UCB/LCB term.
    # Or just switch to the known good "Pure LCB" approach.
    
    # Let's try:
    # score = -mu + beta * sigma
    # with beta = 5.0
    
    # But is 5.0 optimal? The inspiration says it got 6.79.
    # The current program got 1.27.
    # So switching to this structure is a big jump.
    
    # Let's also consider the "GP-UCB + normalized EI" which got 6.744.
    # score = -mu + beta_t * sigma + alpha * EI_normalized
    # This is more complex.
    
    # Given the stagnation, let's go with the simpler, higher-scoring structure from the inspiration.
    # I will implement Pure LCB with a slightly adaptive beta to ensure we don't over-explore at the very end.
    
    # t = _call_count
    # We have 25 calls total.
    # beta_t = 5.0 * (1.0 - 0.5 * (t / 25.0)) ? 
    # At t=1, beta=4.9. At t=25, beta=2.5.
    # This ensures we still explore but less so.
    
    # Actually, let's just use the fixed beta=5.0 first to match the inspiration exactly, as it's a proven high score.
    
    beta = 5.0
    score = -mu + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
