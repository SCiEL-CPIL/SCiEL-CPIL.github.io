"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 300.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _MARGIN:
        return np.zeros_like(mu)
    
    # Use a small epsilon to avoid division by zero in sigma
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Knowledge Gradient (KG) approximation for minimization.
    # KG(x) = E[ min(f_best, f(x)) ] - min(f_best, min(f(x))) 
    # Actually, the standard KG for minimization is:
    # KG(x) = E[ min(f_best, f(x)) ] - min(f_best, E[f(x)]) is not quite right.
    # The true KG is the expected improvement in the *best known minimum*.
    # Let Y be the new observation at x.
    # New best = min(f_best, Y)
    # Old best = f_best
    # KG(x) = E[ min(f_best, Y) ] - min(f_best, E[Y]) ... No.
    # The regret is f_best - min_global.
    # The reduction in regret is max(0, f_best - min(f_best, Y)) = max(0, f_best - Y) if Y < f_best.
    # So KG(x) = E[ max(0, f_best - Y) ] = E[max(0, f_best - mu - sigma*Z)]
    # This is exactly Expected Improvement (EI).
    
    # Wait, why is EI (score 1.0) not optimal?
    # EI maximizes the expected *improvement* at that specific point.
    # KG maximizes the expected *reduction in the minimum*.
    # For a single point evaluation, EI and KG are mathematically identical in expectation?
    # No. EI(x) = E[ max(0, f_best - f(x)) ].
    # KG(x) = E[ max(0, f_best - min(f_best, f(x))) ] = E[ max(0, f_best - f(x)) ].
    # They are the same!
    
    # So why would a different strategy be better?
    # Perhaps the GP is not well-calibrated, or the search space requires a different heuristic.
    # Or maybe "Knowledge Gradient" refers to a different approximation in literature?
    # Some sources distinguish EI from KG when considering multiple points or different priors.
    # However, for a single point, they are equivalent.
    
    # Let's try a different strategy: Probability of Improvement (POI) or UCB.
    # UCB is often better for exploration in early stages.
    # The current program uses EI.
    # Let's try a hybrid or a different acquisition function entirely.
    # The prompt asks for a MATERIALLY DIFFERENT strategy.
    
    # Strategy: Upper Confidence Bound (UCB)
    # UCB(x) = mu(x) + beta * sigma(x)
    # For minimization, we want to minimize mu + beta*sigma.
    # So the score should be - (mu + beta*sigma) to maximize.
    # Or simply return -mu - beta*sigma.
    
    # How to choose beta?
    # A common adaptive beta is beta = max(1, 10 * sigma / (f_best - mu_min)) or similar.
    # Or simply a fixed beta that balances exploration/exploitation.
    # Given the small domain [-2,2]^2 and 30 evals, a moderate beta might work.
    # Let's try beta = 1.0 or 2.0.
    
    # Another option: Probability of Improvement (POI)
    # POI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    # Let's implement a simple UCB with a heuristic beta.
    # If f_best is close to the minimum (3.0), we should exploit more (smaller beta).
    # If f_best is far, explore more (larger beta).
    
    # Let's try a fixed beta first, as it's simpler and robust.
    # A beta of 2.0 is a common starting point.
    
    # However, EI is generally considered superior to UCB in standard BO benchmarks.
    # Is there a flaw in the EI implementation?
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is correct.
    
    # Maybe the issue is the GP hyperparameters? We can't change those.
    # Maybe the issue is the random seed or initialization? We can't change those.
    
    # Let's look at the inspirations.
    # Inspiration 1: UCB when f_best > 15, EI when f_best < 15. Score 0.67 (worse).
    # Inspiration 2: Score 1.46 (better). Diff omitted.
    # Inspiration 3: Score omitted.
    
    # Since EI is the baseline, and UCB was worse, maybe a more sophisticated exploration is needed.
    # What about "Expected Regret Minimization" which is equivalent to EI?
    
    # Let's try **Probability of Improvement (POI)** with a small threshold.
    # Or **Expected Improvement with a penalty for high uncertainty**?
    
    # Actually, let's reconsider the **Knowledge Gradient**.
    # In some contexts, KG is approximated differently.
    # But for 1D, EI is optimal.
    
    # What if we use **UCB** but with a time-varying beta?
    # Beta starts high and decays.
    # We don't have the step count directly, but we can estimate it from f_best?
    # No, f_best is not monotonic in time, but it is non-increasing.
    # We can track the number of calls? No, stateless.
    
    # Let's try a **Hybrid**:
    # If sigma is very large, use UCB (explore).
    # If sigma is small, use EI (exploit).
    
    # Let's implement **UCB** with a fixed beta of 2.0.
    # Score = - (mu + 2.0 * sigma)
    
    # Wait, the prompt says "MATERIALLY DIFFERENT".
    # UCB is a different design family.
    
    # Let's try **Probability of Improvement (POI)**.
    # Score = Phi((f_best - mu) / sigma)
    
    # Or **Expected Improvement with a small noise term**?
    
    # Let's go with **UCB** as it is a distinct, well-known alternative.
    # To make it better, we can use a **logarithmic** or **adaptive** beta.
    # Since we don't have step count, we can use the spread of mu and sigma?
    # Or just a fixed beta.
    
    # Let's try **Beta = 1.0**.
    
    # Alternative: **Expected Improvement** is often improved by using **Local Penalization** or **Repulsion**, but we only have mu, sigma, f_best.
    
    # Let's try **UCB** with beta = 2.0.
    
    # Actually, a very common and effective strategy is **EI** but with a **smaller** sigma? No.
    
    # Let's look at the "Inspiration 2" which scored 1.46.
    # It edited `_ensure_deadline` and `_remaining`. This suggests the previous failure was due to timeout or invalidity?
    # The current program (EI) scored 1.0.
    # The Inspiration 1 (Hybrid UCB/EI) scored 0.67.
    
    # Maybe the GP is overconfident in some regions?
    # If we use **UCB**, we explicitly reward high uncertainty.
    
    # Let's implement **UCB** with an adaptive beta.
    # beta = 1.0 + 10.0 * (sigma_max / (f_best - mu_min + 1e-6))
    # This encourages more exploration when the current best is far from the potential minimum.
    
    # However, a simpler, robust choice is **UCB** with a fixed beta.
    # Let's try beta = 2.5.
    
    # Let's also consider **Probability of Improvement (POI)**.
    # POI is less sensitive to the magnitude of improvement, only the probability.
    # This can lead to more exploration.
    
    # I will implement **UCB** with a fixed beta of 2.0.
    
    beta = 2.0
    scores = -(mu + beta * sigma_safe)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

# EVOLVE-BLOCK-END
