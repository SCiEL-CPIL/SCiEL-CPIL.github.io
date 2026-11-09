"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We need to track the number of evaluations to adapt kappa.
# However, the function signature is acquisition(mu, sigma, f_best).
# We don't have direct access to the step count in the signature.
# But we can infer it or use a heuristic based on the spread of f_best or just a fixed schedule?
# The task says: "at each step ... until 30 evaluations including the first 5".
# Total evaluations = 30.
# Steps = 25 (since 5 initial).
# We can't easily pass the step index unless we use a global variable or infer it.
# Let's try to infer the number of evaluations seen so far.
# Actually, the GP is refit at each step. The `mu` and `sigma` come from the GP.
# There is no explicit `n_evals` argument.
# However, we can store state in a global variable if allowed, but usually these are stateless.
# Let's look at the prompt: "Keep the function names and signatures of the seed program."
# Signature: acquisition(mu, sigma, f_best)
# If we cannot pass step count, we might need a fixed kappa or a heuristic.
# But the directive says: "dynamically increasing exploration constant... kappa scales with the square root of the number of evaluations".
# If we can't know n_evals, we can't do it perfectly.
# Wait, is there a way?
# The GP posterior depends on the data. The data size is n_evals.
# We don't have access to the data points, only mu and sigma at 2000 candidates.
# Maybe we can estimate n_evals from the variance structure? Unlikely to be robust.
# Alternative: Use a global counter. Is that allowed?
# "Python standard library, NumPy and SciPy are available."
# Global state is generally allowed in Python unless restricted.
# Let's try using a global variable to count calls.
# BUT, the evaluator might run multiple seeds in parallel or sequentially?
# "For each seed 0, 1 and 2 the evaluator starts..."
# If they run sequentially, a global counter works. If parallel, it's messy.
# Usually, these evaluators run seeds sequentially.
# However, relying on global state is fragile if the function is re-imported or called in a different context.
# Let's look at the constraints. "No coordinates are passed."
# Is there any other way?
# Maybe the "number of evaluations" can be approximated by how "tight" the sigma is? No.
# Let's assume sequential execution and use a global counter, resetting it if we detect a new run?
# How to detect a new run? We don't know when the loop ends.
# Actually, the prompt implies a single function call per step.
# If I use a global counter, I need to reset it. When?
# I can't reset it easily.
# What if I just use a fixed schedule that mimics sqrt(n) for n=1..30?
# Or, I can use a heuristic: the number of evaluations is roughly proportional to the "age" of the optimization.
# Let's try a different approach: Use a fixed kappa that is known to work well for Hartmann 3D with 30 evals.
# Common values for UCB kappa in BO for small budgets are around 1.0 to 2.0.
# But the directive specifically asks for "dynamically increasing... scales with sqrt(n)".
# If I cannot implement the dynamic part safely, I should stick to a robust fixed value or a simple heuristic.
# However, the directive is a "Strategy to try". It's a suggestion.
# "Make a STRUCTURAL change".
# Let's look at the "Knowledge from this run":
# "EI with small exploration bonus (0.1*sigma) ... Measured: 1 -> 0.5917."
# This suggests that adding a small sigma term to EI (or similar) helped.
# UCB is mu - kappa * sigma.
# Let's try UCB with a kappa that starts high and decreases? Or increases?
# Standard UCB often uses a constant or slowly increasing beta.
# For small budgets, high exploration early is good.
# If I use a global counter, I can implement the sqrt(n) scaling.
# Let's try the global counter approach. It's the only way to truly follow the directive's specific mechanism.
# I will add a safeguard: if the counter exceeds 30, reset it (assuming a new run or just cap it).
# Actually, if the evaluator runs 3 seeds sequentially, the counter will go 1..25, then 26..50, etc.
# I should reset the counter if I see a "restart" pattern.
# How to detect restart? The initial 5 points are uniform random.
# The f_best will jump down significantly at the start of a new seed if the previous seed ended with a worse f_best?
# No, f_best is the lowest observed value *in the current run*.
# So at the start of a new seed, f_best will be the min of the first 5 random points.
# This is typically much worse (higher) than the final f_best of the previous run.
# So if f_best increases significantly compared to the previous call, it might be a new run.
# But we don't store the previous f_best.
# Let's just use a global counter and assume sequential runs. I'll cap the counter at 30 to be safe, or just let it grow.
# If it grows, kappa = sqrt(n). For n=50, kappa=7. That's too high.
# So I MUST reset.
# Heuristic for reset: If the current f_best is significantly higher than the minimum f_best seen in the last few calls?
# Or simpler: The evaluator says "For each seed...". It likely calls the function 25 times per seed.
# Total 75 calls.
# If I just use a fixed kappa, I avoid the statefulness issue.
# Let's re-read the directive: "Strategy to try... Use UCB... with dynamically increasing...".
# It's a strategy to try, not a hard constraint if it's impossible to implement cleanly.
# But "Make a STRUCTURAL change".
# Let's try a hybrid: Use UCB with a fixed kappa that is tuned for this problem, OR use the global counter with a reset heuristic.
# Let's try the global counter with a reset heuristic based on f_best.
# Store last_f_best. If current f_best > last_f_best * 1.5 (or some margin), assume new run.
# This is risky.
# Alternative: Just use a fixed kappa. What is a good fixed kappa for Hartmann 3D with 30 evals?
# Literature suggests kappa around 1.0-2.0 for UCB in 3D.
# Let's try kappa = 1.5.
# Or, let's look at the "EI with 0.1*sigma" insight.
# EI is mu - max(f_best - mu, 0) ... wait, EI for minimization is max(f_best - mu, 0) ... no.
# Standard EI (minimization): E[max(f_best - f(x), 0)] = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu)/sigma.
# This is what the current code does.
# The insight says "EI with small exploration bonus (0.1*sigma)" improved the score.
# This implies: score = EI + 0.1 * sigma? Or EI - 0.1 * sigma?
# "exploration bonus" usually means adding a term that favors high uncertainty.
# In minimization, high sigma means potentially lower f(x).
# So adding sigma encourages exploration.
# So score = EI + beta * sigma.
# The current code is pure EI.
# The insight says adding 0.1*sigma helped.
# UCB is mu - kappa * sigma.
# These are different forms.
# Let's try the UCB form as directed, but maybe with a fixed kappa if dynamic is too risky.
# However, the directive is specific about the dynamic part.
# Let's try to implement the dynamic part with a global counter and a simple reset logic.
# If the counter reaches 25, reset to 1? No, we don't know exactly when it resets.
# Let's assume the evaluator resets the process or imports fresh? No, it's a loop.
# Let's try a different structural change: **PI (Probability of Improvement)** or **Thompson Sampling**?
# Thompson Sampling is a different design family.
# But the directive specifically suggests UCB with dynamic kappa.
# Let's stick to UCB but use a fixed kappa that approximates the average of the dynamic schedule.
# Average sqrt(n) for n=1..25 is roughly sqrt(25/3) ~ 2.9? No, integral of sqrt(x) is (2/3)x^1.5.
# Mean of sqrt(1..25) is approx 2.5.
# So kappa ~ 2.5?
# Let's try UCB with kappa = 2.0.
# And add a time guard as requested.

_global_step = 0
_global_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _global_step, _global_last_f_best
    
    # Heuristic reset detection
    if _global_last_f_best is not None:
        # If f_best jumps up significantly, assume new seed
        if f_best > _global_last_f_best * 1.2 + 0.1:
            _global_step = 0
    else:
        _global_step = 0
        
    _global_step += 1
    _global_last_f_best = f_best
    
    # Cap step to avoid huge kappa if reset fails
    n_evals = min(_global_step, 30)
    
    # Dynamic kappa
    # Start with high exploration, decrease? Or increase?
    # Directive: "kappa scales with the square root of the number of evaluations"
    # This implies increasing exploration? That's unusual for BO, usually exploration decreases.
    # "balance initial exploration against later exploitation" -> usually means start high, end low.
    # But "scales with sqrt(n)" means it grows.
    # Maybe it means the *uncertainty* part is weighted more as we learn more?
    # Or maybe the directive implies kappa = c * sqrt(n).
    # If kappa increases, we explore more later. This is counter-intuitive but let's follow the directive.
    # Wait, "balance initial exploration against later exploitation" usually means we want MORE exploration initially.
    # If kappa increases, we explore MORE later. That contradicts the goal of exploitation later.
    # Unless... the "exploration constant" in UCB is actually a penalty?
    # UCB = mu - kappa * sigma.
    # High kappa -> penalize high sigma -> exploit low uncertainty.
    # Low kappa -> allow high sigma -> explore.
    # So if we want initial exploration, we want LOW kappa initially.
    # If kappa scales with sqrt(n), it increases.
    # So we start with low exploration (high exploitation) and end with high exploration?
    # That seems wrong.
    # Perhaps the directive means kappa = c / sqrt(n)?
    # Or maybe I am misinterpreting "exploration constant".
    # In some formulations, UCB = mu + kappa * sigma (for maximization).
    # For minimization: mu - kappa * sigma.
    # If kappa is small, we pick high sigma (explore).
    # If kappa is large, we pick low sigma (exploit).
    # To balance initial exploration, we want small kappa initially.
    # To balance later exploitation, we want large kappa later.
    # So kappa should INCREASE.
    # So kappa = c * sqrt(n) makes sense if we want to shift from exploration to exploitation.
    # Wait, if kappa increases, the penalty for sigma increases, so we AVOID high sigma.
    # So high kappa = exploitation.
    # Low kappa = exploration.
    # So starting with low kappa (n=1) and ending with high kappa (n=25) means:
    # Start: explore. End: exploit.
    # This matches the goal "balance initial exploration against later exploitation".
    # So the formula kappa = c * sqrt(n) is correct for this purpose.
    
    # Choose c.
    # At n=1, kappa = c.
    # At n=25, kappa = 5c.
    # We want kappa to be in a reasonable range, e.g., 0.5 to 5.0.
    # Let c = 0.5.
    # n=1: kappa=0.5 (explore)
    # n=25: kappa=2.5 (exploit)
    
    c = 0.5
    kappa = c * np.sqrt(n_evals)
    
    # Calculate UCB score
    # Handle zero sigma
    sigma_safe = np.where(sigma < 1e-8, 1e-8, sigma)
    scores = mu - kappa * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

# EVOLVE-BLOCK-END
