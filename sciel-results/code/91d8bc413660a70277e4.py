"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_hermite
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 295.0
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # 1. Compute Expected Improvement (EI) for each candidate
    # EI_i = E[max(f_best - f(x_i), 0)]
    # Using standard normal CDF/PDF
    z_ei = (f_best - mu) / sigma
    ei_score = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # 2. Compute Expected Regret (ER) for the set
    # The acquisition value for candidate i in an ER framework is often defined as:
    # A_i = E[ min_{j} max(f_best - f(x_j), 0) | D ]
    # However, a common and efficient approximation that captures the "best of set" logic
    # while being computationally tractable for 2000 points is to compute the probability
    # that candidate i is the minimum regret point, weighted by its improvement.
    
    # A more robust structural change: Use the "Best-of-Set" Expected Improvement.
    # Instead of just EI_i, we consider the probability that x_i is the best point 
    # among all candidates, given the GP posterior.
    
    # Exact ER calculation is O(N^2) or worse. 
    # A faster structural alternative that is different from simple EI/UCB:
    # Use "Probability of Improvement" (PI) combined with a global scaling factor 
    # that accounts for the spread of the posterior, or simply use a higher-order 
    # quadrature based EI which is more accurate in tails?
    
    # Let's try a different structural approach: 
    # **Max-Value Entropy Search (MVES) approximation** or **Expected Hypervolume Improvement (EHI)**?
    # EHI is complex.
    
    # Let's look at the inspirations. Inspiration 2 (score 1.0) used a "single, robust EI".
    # The current program (score 0.5) uses max(EI, LCB).
    # The failure of the current program suggests the LCB term is harmful or the max operation is suboptimal.
    # The success of Inspiration 2 suggests pure EI is better.
    
    # Structural Change: Instead of a heuristic combination, let's implement **Expected Improvement 
    # with a dynamic exploration bonus based on the local density of candidates**? 
    # No, we don't have coordinates.
    
    # Let's try **Expected Regret via Gauss-Hermite Quadrature** for a more accurate EI calculation.
    # While standard EI is O(N), the accuracy in the tails might be the issue.
    # However, for 2000 points, standard EI is very fast.
    
    # Let's reconsider the "Different Design Family".
    # What if we use **Knowledge Gradient (KG)**?
    # KG_i = E[ max_j EI_j - EI_i | D ]
    # This is O(N^2). For N=2000, 4 million operations. In Python/NumPy, this might be slow 
    # if not vectorized perfectly, but 4e6 float ops is < 1s usually.
    # Let's estimate: 
    # We need to compute the matrix of EIs? No, KG requires joint expectations.
    # Approximate KG: 
    # A_i = E[ max_{j} (f_best - f_j) - (f_best - f_i) ]
    # This is hard to compute exactly without joint distributions.
    
    # Alternative Structural Change: **Thompson Sampling** style?
    # No, we must return scores for all 2000 to pick the max.
    
    # Let's look at the "Stagnation" directive: "Make a STRUCTURAL change".
    # The current program mixes EI and LCB. 
    # Inspiration 2 (Pure EI) scored 1.0. 
    # Inspiration 1 (Pure EI?) scored 1.0.
    # The current program (Max EI, LCB) scored 0.5.
    
    # It is highly likely that the LCB term is causing the drop in performance.
    # Why? LCB = mu - kappa * sigma. Maximizing -mu + kappa * sigma.
    # If sigma is high, LCB score is high. This forces exploration.
    # If EI is used, it balances exploitation (f_best - mu) and exploration (sigma) naturally 
    # through the normal CDF/PDF.
    # The "Max" operation might pick LCB in regions where EI is low but sigma is high, 
    # leading to overly aggressive exploration or ignoring good exploitation opportunities.
    
    # Therefore, the most logical structural change is to **abandon the hybrid heuristic** 
    # and move to a **pure, but more sophisticated, exploitation-focused acquisition**.
    # Or, move to **Expected Regret (ER)** which is the theoretically correct objective 
    # for minimizing the minimum value.
    
    # Let's implement an approximation of **Expected Regret (ER)** that is efficient.
    # A_i = E[ min_j max(f_best - f_j, 0) ]
    # This is difficult.
    
    # Let's try **Expected Improvement with a "Confidence" penalty**? No, that's LCB again.
    
    # Let's try **Probability of Improvement (PI)**?
    # PI_i = P(f(x_i) < f_best) = Phi((f_best - mu_i) / sigma_i)
    # PI is purely exploitation-oriented (ignores sigma magnitude in the score, only sign).
    # It might be too greedy.
    
    # Let's try **Upper Confidence Bound (UCB) for Maximization** but applied to -f?
    # That's what LCB is.
    
    # What if we use **Expected Improvement** but with a **different kernel assumption**? 
    # No, we are given mu and sigma.
    
    # Let's look at the score 0.4997 vs 1.0.
    # The drop is significant.
    # The current program: `np.maximum(ei_score, exploration_score)`
    # `exploration_score = -mu + 2.0 * sigma`
    
    # If `mu` is close to `f_best` (good), `ei` is small. `exploration_score` is `-f_best + 2*sigma`.
    # If `sigma` is large, `exploration_score` is large.
    # This forces the algorithm to pick points with high sigma even if mu is not very low.
    # In Hartmann 3D, which has multiple local minima, this might be good?
    # But it scored 0.5. Pure EI scored 1.0.
    
    # So, Pure EI is better.
    # How to make a structural change that is *different* from Pure EI but potentially better?
    # Or, simply return to Pure EI but with a **more accurate calculation**?
    
    # Standard EI uses `norm.cdf` and `norm.pdf`.
    # For very large |z|, `norm.pdf(z)` becomes 0, and `norm.cdf(z)` becomes 0 or 1.
    # If z is very negative (mu >> f_best), EI -> 0.
    # If z is very positive (mu << f_best), EI -> f_best - mu.
    
    # Let's try **Expected Improvement with a "Tail" correction** or **Gauss-Hermite Quadrature** 
    # for the EI calculation itself.
    # This is a "different modelling family" for the acquisition computation.
    
    # EI(x) = integral_{-inf}^{f_best} (f_best - y) p(y|x) dy
    # Let y = mu + sigma * t
    # EI = sigma * integral_{-inf}^{z} (z - t) phi(t) dt  (where z = (f_best-mu)/sigma)
    # This is the standard formula.
    
    # Is there a different acquisition function that is structurally different?
    # **Knowledge Gradient (KG)** is the gold standard for "best of set".
    # Let's implement an **Approximate Knowledge Gradient**.
    # A_i = E[ max_j EI_j - EI_i ]
    # We can approximate the joint distribution by assuming independence?
    # If independent, E[max_j EI_j] is not trivial.
    
    # However, there is a simpler structural change: **Use the "Best" EI from a subset?**
    # No, we must return scores for all.
    
    # Let's try **Expected Hypervolume Improvement (EHI)**? Too complex.
    
    # Let's try **Probability of Being the Minimum** (PBM)?
    # PBM_i = P(f(x_i) < f(x_j) for all j != i)
    # This is related to KG.
    # If we assume independent Gaussians:
    # P(f_i < f_j) = Phi((mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2))
    # P(f_i < all others) = product_{j != i} P(f_i < f_j)
    # This is O(N^2).
    # For N=2000, 4 million operations.
    # Let's check the time budget.
    # 300s limit.
    # NumPy vectorized O(N^2) for N=2000:
    # 2000x2000 matrix = 4e6 elements.
    # Creation and product: ~10-50 ms in NumPy.
    # This is very fast.
    
    # Strategy: **Probability of Improvement (PI)** is too greedy.
    # **Expected Improvement (EI)** is good.
    # **Knowledge Gradient (KG)** is better for global optimization because it accounts for the fact 
    # that picking one point makes the others less valuable (information gain).
    # Approximate KG via PBM (Probability of Being the Minimum) is a known heuristic.
    # Actually, the acquisition function for KG is:
    # A_i = E[ max_{j} (f_best - f_j) | D ] - E[ f_best - f_i | D ]
    # A_i = E[ min_j (f_j - f_best) | D ] + (f_best - mu_i) ... wait.
    # A_i = E[ min_j max(f_best - f_j, 0) | D ] - E[ max(f_best - f_i, 0) | D ]
    # The second term is just EI_i.
    # The first term is the Expected Regret of the set.
    # This is constant for all i!
    # So A_i = Constant - EI_i?
    # No, the first term depends on the *entire* set, but it is the same for all i in the set.
    # Wait, the definition of KG is:
    # A_i = E[ max_{j} (f_best - f_j) | D, x_i ] - E[ max_{j} (f_best - f_j) | D ]
    # The first term changes because observing x_i updates the posterior for all j.
    # Without observing x_i, we can't compute the first term easily.
    
    # However, there is a simpler metric: **Expected Improvement of the Best Point (EIBP)**?
    
    # Let's go back to the observation: Pure EI scored 1.0.
    # The current program (Max EI, LCB) scored 0.5.
    # The most robust structural change is to **remove the LCB component** and stick to a **pure exploitation-driven acquisition** but perhaps using a **more robust variant of EI** like **Expected Improvement with a noise floor** or **EI with a different scaling**.
    
    # But the directive says "Different design or modelling family".
    # Pure EI is the same family as the current program's EI component.
    
    # Let's try **Thompson Sampling** implemented as a score?
    # No, we need deterministic scores.
    
    # Let's try **Upper Confidence Bound (UCB)** but with a **dynamic kappa**?
    # No, that's tuning.
    
    # Let's try **Expected Regret (ER)** using **Gauss-Hermite Quadrature** on the joint distribution?
    # Too expensive.
    
    # How about **Probability of Improvement (PI)**?
    # PI is structurally different from EI (no magnitude of improvement, just probability).
    # It is more exploratory in a sense? No, it's more exploitative in terms of "which one is best".
    # Actually, PI can be flat in regions where sigma is high and mu is far from f_best.
    
    # Let's try **Expected Improvement** but **normalized by sigma**?
    # z-score? No.
    
    # Let's look at the "Inspiration 1" and "Inspiration 2".
    # Both scored 1.0.
    # Inspiration 2: "Single, robust EI".
    # Inspiration 1: "Initial program...".
    
    # It seems Pure EI is the winner.
    # To make a structural change, I will implement **Expected Improvement** but using **Gauss-Hermite Quadrature** for the integral, which is a different *numerical method* (modelling family of the computation) and can be more stable in extreme tails.
    # Although standard EI is fine, this satisfies the "different approach" requirement technically.
    # Or, I can implement **Knowledge Gradient (KG)** approximation.
    
    # Let's try **KG Approximation** via **Probability of Being the Minimum (PBM)**.
    # PBM_i = P(f_i < f_j, forall j!=i)
    # Assuming independence:
    # PBM_i = prod_{j!=i} Phi((mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2))
    # This is O(N^2).
    # For N=2000, this is 4e6 operations.
    # Let's implement this and see if it's fast enough.
    # If it is, it's a structurally different acquisition function (KG-like).
    
    # Time check:
    # 2000x2000 matrix creation: ~10ms.
    # Division and Sqrt: ~10ms.
    # CDF: ~10ms.
    # Product along axis: ~10ms.
    # Total: < 100ms.
    # This is well within the 300s budget.
    
    # Let's implement PBM.
    
    # Calculate pairwise differences and variances
    # mu shape (N,), sigma shape (N,)
    # mu_diff = mu[:, None] - mu[None, :]  # (N, N)
    # sigma_sq_sum = sigma[:, None]**2 + sigma[None, :]**2 # (N, N)
    # z = mu_diff / np.sqrt(sigma_sq_sum)
    # probs = norm.cdf(z)
    # # Set diagonal to 1 (P(f_i < f_i) is undefined, treat as 1 for product)
    # np.fill_diagonal(probs, 1.0)
    # p_bm = np.prod(probs, axis=1)
    
    # This PBM score is higher for points that are likely to be the minimum.
    # It naturally balances mu (lower is better) and sigma (lower is better for certainty, but if mu is very low, high sigma is okay).
    # Actually, if sigma is high, the probability of being the minimum drops unless mu is very low.
    # This is a very robust "Best of Set" metric.
    
    # Let's try this.
    
    N = len(mu)
    if N == 0:
        return np.array([])
        
    # Vectorized PBM calculation
    # mu_i - mu_j
    mu_diff = mu[:, None] - mu[None, :]
    # sigma_i^2 + sigma_j^2
    sigma_sq_sum = sigma[:, None]**2 + sigma[None, :]**2
    
    # Avoid division by zero
    sigma_sq_sum = np.maximum(sigma_sq_sum, 1e-16)
    
    z = mu_diff / np.sqrt(sigma_sq_sum)
    
    # Probability that i is better than j
    probs = norm.cdf(z)
    
    # The probability that i is the best is the product of probabilities that i is better than all j
    # We exclude j=i.
    # Set diagonal to 1.
    np.fill_diagonal(probs, 1.0)
    
    # Compute product
    # Log sum for numerical stability?
    # log_p_bm = sum(log(probs), axis=1)
    # p_bm = exp(log_p_bm)
    # But probs can be 0.
    # np.prod is usually fine for N=2000.
    
    p_bm = np.prod(probs, axis=1)
    
    # Ensure finite
    p_bm = np.nan_to_num(p_bm, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return p_bm

# EVOLVE-BLOCK-END
