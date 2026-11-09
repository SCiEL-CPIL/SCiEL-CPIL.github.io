"""Cost-aware acquisition for three hump camel 2d BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Precompute nothing heavy; keep state minimal.
_deadline = time.time() + 320.0  # safety margin under 330s kill
_initialized = False
_init_time = None


def _ensure_init():
    global _initialized, _init_time
    if not _initialized:
        _init_time = time.time()
        _initialized = True


def acquisition(mu, sigma, f_best, cost):
    """
    mu, sigma, cost: arrays of shape (2000,)
    Returns 2000 finite scores (higher = evaluate next).
    """
    _ensure_init()

    # Deadline guard
    if time.time() > _deadline:
        # Return a simple, fast fallback
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Clip sigma to avoid division by zero or negative
    sigma_safe = np.maximum(sigma, 1e-6)

    # z-score for expected improvement
    z = (f_best - mu) / sigma_safe

    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Cost-aware: divide by cost. Cost is 1 + mean(x_scaled), in [1, 2].
    # Use a power to slightly penalize high-cost points more aggressively.
    # cost^0.5 is a milder penalty than cost^1, allowing some exploration
    # at higher cost if EI is very high.
    # Actually, the finding says strict linear cost penalty (EI/Cost) worked.
    # Let's stick with EI / cost but add a small exploration term based on
    # sigma to ensure we don't get stuck, but the finding said removing
    # exploration bonus helped. So let's stick to pure EI/Cost.
    
    # However, pure EI/Cost can be zero if mu >= f_best. 
    # In BO, we usually want to explore if sigma is high.
    # But the finding explicitly states: "Removing the explicit exploration bonus 
    # and using a strict linear cost penalty (EI/Cost) forces the algorithm to 
    # prioritize high-efficiency evaluations... Measured: 0.8542 -> 1."
    
    # So the current program is already EI/Cost. 
    # Why did it score 1.0? It's the baseline.
    # The directive says: "STAGNATION: ... Make a STRUCTURAL change".
    # And "FRESH DESIGN: ... Write a new design from scratch".
    
    # The current program is:
    # z = (f_best - mu) / (sigma + 1e-9)
    # return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost
    
    # This is EI / cost.
    
    # What structural change can we make?
    # 1. Use a different acquisition function, e.g., Probability of Improvement (PI) / cost?
    #    PI is more aggressive, less exploration.
    # 2. Use Upper Confidence Bound (UCB) / cost?
    #    UCB = mu + beta * sigma. But we are minimizing.
    #    Minimization UCB: mu - beta * sigma.
    #    Score = (f_best - (mu - beta * sigma)) / cost?
    #    Or just (mu - beta * sigma) / cost? No, higher score means evaluate next.
    #    For minimization, we want low mu and high sigma.
    #    A common cost-aware UCB is: (mu - beta * sigma) / cost. But this can be negative.
    #    We need higher = evaluate next.
    #    So maybe: (f_best - mu + beta * sigma) / cost?
    #    This is similar to EI but with a linear sigma term instead of the non-linear norm.pdf term.
    #    EI has a non-linear exploration bonus that decays as sigma increases.
    #    UCB has a linear exploration bonus.
    
    # Let's try a hybrid or a different formulation.
    # The three hump camel function is multimodal.
    # EI is good for exploitation.
    # The cost structure penalizes points far from the origin (in scaled space).
    # x_scaled = (x + 5) / 10.
    # cost = 1 + mean(x_scaled) = 1 + (x1 + x2 + 10) / 20 = 1.5 + (x1+x2)/20.
    # So cost is lowest when x1+x2 is most negative, i.e., near (-5, -5).
    # But the minimum of three hump camel is near (0, 0) in the original space?
    # Wait, the task says: "minimizes three hump camel 2d on [-5, 5]^2 (min 0)".
    # Standard three hump camel: f(x,y) = 2x^2 - 1.05x^4 + x^6/6 + xy + y^2.
    # Minimum is at (0,0) with f(0,0)=0.
    # But the instances are shifted: f_k(x) = f(x - s_k).
    # s_k is a uniform point in the inner 80% of each range.
    # Inner 80% of [-5, 5] is [-4, 4].
    # So the minimum is moved to s_k, which is in [-4, 4]^2.
    # The cost is 1 + mean(x_scaled). x_scaled is in [0, 1].
    # cost = 1 + (x1_scaled + x2_scaled)/2.
    # x_scaled = (x + 5)/10.
    # So cost depends on the actual coordinates x, not the shifted ones.
    # The evaluator scores 2000 new uniform candidates with acquisition.
    # The candidates are uniform in [-5, 5]^2.
    # So cost is known for each candidate.
    
    # The key insight: cost is correlated with position.
    # Points near (-5, -5) have low cost. Points near (5, 5) have high cost.
    # The minimum is at s_k, which is in [-4, 4]^2.
    # So the minimum is never in the corner (-5, -5) or (5, 5), but can be anywhere in [-4, 4]^2.
    # If s_k is near (-4, -4), the minimum is close to the low-cost region.
    # If s_k is near (4, 4), the minimum is close to the high-cost region.
    
    # EI/cost is a good baseline.
    # What if we use a more aggressive exploitation strategy?
    # Or what if we use a different kernel or GP? No, we only have mu, sigma, f_best, cost.
    # We cannot change the GP.
    
    # Let's try Probability of Improvement (PI) divided by cost.
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    # This is a step function, more aggressive than EI.
    # It ignores the magnitude of improvement, only the probability.
    # In a tight budget, this might be better? Or worse?
    # EI considers the expected gain, PI considers the probability of any gain.
    # For a multimodal function, EI might be better because it seeks large gains.
    
    # Let's try a different angle: the cost is 1 + mean(x_scaled).
    # We can rewrite the score as EI / (1 + mean(x_scaled)).
    # This is what the current program does.
    
    # What if we use a lower bound on cost? No, cost is given.
    
    # Let's look at the "STAGNATION" directive again.
    # "Make a STRUCTURAL change: a different design or modelling family, a different optimization or search method, or a different representation of the problem."
    # Since we can't change the GP or the search loop, we can only change the acquisition function.
    # Is there a different acquisition function that is structurally different?
    # 1. EI / cost (current)
    # 2. PI / cost
    # 3. UCB / cost: (mu - beta * sigma) / cost? No, for minimization, we want to minimize mu - beta * sigma.
    #    So score = (f_best - (mu - beta * sigma)) / cost?
    #    Or simply: (f_best - mu + beta * sigma) / cost.
    #    This is a linear combination of exploitation (f_best - mu) and exploration (beta * sigma).
    #    EI is non-linear.
    #    Let's try UCB with beta = 1.0.
    #    Score = (f_best - mu + sigma) / cost.
    #    This gives a linear exploration bonus.
    #    EI gives a non-linear exploration bonus that peaks at a certain sigma.
    #    For small sigma, EI ~ (f_best - mu) * 0.5 + sigma * 0.4.
    #    For large sigma, EI ~ sigma * 0.4.
    #    UCB gives sigma * 1.0.
    #    So UCB explores more aggressively.
    
    # Let's try UCB with beta = 0.5.
    # Score = (f_best - mu + 0.5 * sigma) / cost.
    
    # Or maybe a hybrid: EI + alpha * sigma, divided by cost.
    # But the finding said removing exploration bonus helped.
    # So maybe pure EI/cost is best.
    
    # Wait, the finding says: "Removing the explicit exploration bonus and using a strict linear cost penalty (EI/Cost) forces the algorithm to prioritize high-efficiency evaluations... Measured: 0.8542 -> 1."
    # This implies that the previous version had an exploration bonus, and removing it improved the score.
    # The current program is EI/Cost, which has no explicit exploration bonus (other than the implicit one in EI).
    # So the current program is the improved one.
    
    # How can we improve further?
    # Maybe the cost penalty is too strong or too weak?
    # Cost is in [1, 2].
    # Dividing by cost is equivalent to multiplying by 1/cost, which is in [0.5, 1].
    # So the score is reduced by a factor of 0.5 to 1.
    # This is a mild penalty.
    # What if we use a stronger penalty, e.g., cost^2?
    # Score = EI / cost^2.
    # This would heavily favor low-cost points.
    # But if the minimum is at a high-cost point, we might miss it.
    # However, the minimum is in [-4, 4]^2, so the cost is at most 1 + (4+5+4+5)/20 = 1 + 18/20 = 1.9.
    # And at least 1 + (-4+5-4+5)/20 = 1 + 2/20 = 1.1.
    # So cost is in [1.1, 1.9] for the region containing the minimum.
    # The ratio of max to min cost is 1.9/1.1 ~ 1.7.
    # So the penalty is not extreme.
    
    # Let's try a different approach: use the cost to weight the exploration.
    # But we don't have access to the coordinates, only the cost.
    
    # What if we use a different norm for the z-score?
    # No, the GP is fixed.
    
    # Let's try PI/cost.
    # PI is more aggressive, might find the minimum faster if it's close.
    # But it might get stuck in local minima.
    
    # Let's try a mixture: 0.5 * EI/cost + 0.5 * PI/cost.
    # This balances exploitation and exploration.
    
    # Or, let's try to use the cost to adjust the exploration.
    # For low-cost points, we can afford to explore more.
    # For high-cost points, we should exploit more.
    # So, Score = (EI + alpha * sigma * (1 / cost)) / cost?
    # No, that's not right.
    
    # Let's stick to the finding: strict linear cost penalty (EI/Cost) is best.
    # So the current program is likely optimal or near-optimal.
    # But the directive says to make a structural change.
    # Maybe the structural change is to use a different acquisition function that is more suitable for the cost structure.
    
    # Let's try UCB with beta = 0.5.
    # Score = (f_best - mu + 0.5 * sigma) / cost.
    
    # Or, let's try to use the cost to scale the sigma.
    # Score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z) * (1 / cost)
    # This is not EI/cost, but a modified EI.
    
    # Let's try the simplest structural change: use PI instead of EI.
    # PI = norm.cdf(z).
    # Score = PI / cost.
    
    # Let's try UCB.
    # Score = (f_best - mu + 1.0 * sigma) / cost.
    
    # I will try UCB with beta = 1.0.
    # This is a structural change from EI (non-linear) to UCB (linear).
    
    z = (f_best - mu) / sigma_safe
    ucb = f_best - mu + 1.0 * sigma_safe
    score = ucb / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
