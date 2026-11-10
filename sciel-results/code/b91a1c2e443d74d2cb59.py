"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Structural Change: Hybrid Cost-Effective Improvement (CEI) with Probability of Improvement (POI)
# The current program uses pure POI/cost (score 1.4723).
# Inspiration 1 uses EI / (cost * (1 + cost)) (score 1.4391).
# The cell finding states: "Switching from Expected Improvement to Cost-Adjusted UCB provides a more robust balance... UCB's explicit uncertainty term helps navigate the search space more effectively under budget constraints."
# However, the current program is actually POI-based, not UCB. The "1.4723" score is associated with the POI/cost implementation in the provided "CURRENT PROGRAM".
# Wait, the prompt says CURRENT PROGRAM (score 1.4723) is the POI/cost one.
# The cell finding mentions a program that improved island 6 from 1.0 to 1.472 using "Cost-Adjusted UCB".
# Let's look at the inspirations.
# Inspiration 1 (1.4391): EI / (cost * (1 + cost)).
# Inspiration 2 (1.2554): POI / sqrt(cost).
# Inspiration 3 (1.2554): Similar to Insp 1.
#
# The current program (POI / cost) is the best so far (1.4723).
# The directive says: "STAGNATION... Make a STRUCTURAL change... do not micro-tune constants."
# "COMBINE the strategy of the current program with the strongest ideas of the inspirations above into ONE coherent program -- take the structural idea of one and the refinement machinery of the other, do not just concatenate."
#
# Current Strategy: POI / cost.
# Strongest Inspiration: Inspiration 1 (EI / (cost * (1 + cost))) is the highest scoring inspiration (1.4391).
# However, EI is often smoother and more exploitative than POI. POI is more binary (probability of beating best).
# A structural change could be to use a **Hybrid Acquisition** that blends the *magnitude* of improvement (EI) with the *probability* of improvement (POI), while maintaining a strong cost-awareness structure.
#
# Let's look at the "Cost-Adjusted UCB" mention in the findings.
# UCB = mu + beta * sigma.
# Cost-Adjusted UCB for minimization: (f_best - mu) / cost + beta * sigma / cost?
# Or maybe: (f_best - mu + beta * sigma) / cost?
#
# Let's try to combine the **EI structure** (which captures magnitude) from Inspiration 1 with the **POI robustness** (which captures probability) from the Current Program, using a **quadratic cost penalty** from Inspiration 1 which seems effective (1.4391 vs 1.2554 for linear/sqrt cost).
#
# Proposed Structure:
# Score = (w_ei * EI + w_poi * POI) / (cost * (1 + cost))
#
# Why?
# 1. EI captures the expected gain. If mu is far below f_best, EI is large.
# 2. POI captures the likelihood. If sigma is large, POI might be high even if EI is moderate.
# 3. The denominator cost * (1 + cost) strongly penalizes expensive evaluations, which is crucial given the budget constraint (30 total cost).
# 4. This is a structural change from pure POI/cost to a blended metric with a different cost penalty shape.
#
# Let's refine the blend.
# In early stages, exploration is key. POI helps explore.
# In later stages, exploitation is key. EI helps exploit.
# We don't know the stage.
# However, EI is zero when mu >= f_best and sigma is small. POI is 0 in that case too.
# When mu < f_best, EI > 0 and POI > 0.5.
#
# Let's try:
# Score = (EI + 0.5 * POI) / (cost * (1 + cost))
#
# Actually, let's look at the scores again.
# POI / cost: 1.4723
# EI / (cost * (1 + cost)): 1.4391
# POI / sqrt(cost): 1.2554
#
# The current best (POI/cost) is better than Insp 1.
# This suggests that linear cost penalty is better than quadratic for this task, OR that POI is a better signal than EI.
# If POI/cost is best, maybe we should stick to POI but change the cost structure?
# No, the directive says "do not micro-tune constants". Changing cost^1 to cost^2 is a structural change in the penalty function.
# But if POI/cost is 1.47 and EI/quad_cost is 1.44, maybe the penalty is the issue?
#
# Let's try a **Cost-Discounted Expected Improvement** but with a **softer** cost penalty than quadratic, perhaps geometric mean?
# Or, let's try the **UCB** approach mentioned in the cell findings which reportedly improved a lineage to 1.472.
# Wait, the current program IS the 1.4723 one. And it's POI/cost.
# The cell finding says "Switching from Expected Improvement to Cost-Adjusted UCB provides... 1 -> 1.472".
# This implies the 1.472 score might be associated with a UCB variant in the history, but the current code provided is POI.
# Perhaps the "Current Program" label is misleading or the POI/cost implementation effectively acts as a robust UCB-like explorer?
#
# Let's assume the POI/cost is the current state of the art for this lineage.
# To improve, we need a structural change.
# Idea: **Thompson Sampling** style? No, we return scores for 2000 points, not a sample.
# Idea: **Pareto Front** of Improvement vs Cost?
#
# Let's look at **Inspiration 1** again. It uses `EI / (cost * (1 + cost))`.
# Let's look at **Current** again. It uses `POI / cost`.
#
# What if we use **EI** but with the **linear cost** penalty of the current program?
# Score = EI / cost.
# This is the standard Cost-Aware EI. The comments in the current program say: "The seed was likely standard EI / cost... The current program (1.138) is an improvement over the seed (1.0)."
# Wait, the current program in the prompt is labeled "score 1.4723".
# But the comments inside the current program say: "The previous code (score 1.138) used... Let's try POI / cost."
# So the evolution went: Seed (EI/cost?) -> 1.0 -> Something (1.138) -> POI/cost (1.4723).
# So POI/cost is the winner so far.
#
# Why is POI/cost better than EI/cost?
# EI is sensitive to the magnitude of (f_best - mu). If f_best is very low, EI becomes small.
# POI is bounded [0,1]. It focuses on the probability of improvement.
#
# Structural Change: **Hybrid EI-POI with Adaptive Cost Penalty**.
# Let's try to combine the *strength* of EI (magnitude) with the *robustness* of POI (probability).
# And use a cost penalty that is intermediate between linear and quadratic?
# Or, use **UCB** as suggested by the cell finding?
# UCB for minimization: $f_{ucb} = \mu - \beta \sigma$.
# We want to minimize. So we want low $\mu$ and high $\sigma$ (if we believe it might be low).
# Actually, standard UCB for minimization is $\mu - \beta \sigma$.
# Acquisition score (higher is better): $(f_{best} - \mu + \beta \sigma) / cost$?
# Or just $(\mu - \beta \sigma)$? No, we want to find points lower than f_best.
# So score = $(f_{best} - (\mu - \beta \sigma)) / cost = (f_{best} - \mu + \beta \sigma) / cost$.
# This is a linear UCB.
#
# The cell finding says "Cost-Adjusted UCB... 1 -> 1.472".
# If the current program is POI/cost and scores 1.4723, and the finding says UCB got 1.472, they might be similar or the finding refers to a different program.
# However, POI is not UCB.
# Let's try **UCB / cost**.
# Score = (f_best - mu + beta * sigma) / cost.
# What beta?
# In BO, beta is often $\sqrt{2 \log(t)}$ or a constant like 1.0-2.0.
# Since we don't know t, let's use a constant.
# But the directive says "do not micro-tune constants".
# Using a fixed beta is a structural choice.
#
# Let's compare:
# 1. POI / cost (Current, 1.4723)
# 2. EI / cost (Seed/Previous, ~1.0-1.1)
# 3. EI / (cost*(1+cost)) (Insp 1, 1.4391)
# 4. POI / sqrt(cost) (Insp 2, 1.2554)
#
# POI/cost is the best.
# Let's try **UCB / cost**.
# UCB = (f_best - mu) / sigma * sigma + beta * sigma? No.
# UCB score = (f_best - mu + beta * sigma) / cost.
# This is linear in mu and sigma.
# POI is non-linear (CDF).
# EI is non-linear (CDF + PDF).
#
# Let's try a **Structural Change to UCB**.
# The cell finding explicitly links UCB to the 1.472 score.
# Maybe the "Current Program" (POI) is actually a variation that converged to similar performance, but UCB might be more robust or allow for further tuning via beta?
# Or maybe the "Current Program" *is* the UCB program but mislabeled in my analysis?
# No, the code is clearly `poi = norm.cdf(z); scores = poi / cost`.
#
# Let's assume the Cell Finding is a hint that UCB is a good structural family.
# Let's implement **Cost-Adjusted UCB**.
# Score = (f_best - mu + beta * sigma) / cost.
# We need to choose beta.
# A common choice in budgeted BO is to scale beta with the remaining budget or just use a moderate value.
# Let's use beta = 1.0. This is a standard, non-tuned constant (it's the definition of 1-sigma UCB).
#
# Why might this beat POI/cost?
# POI saturates at 1.0. If mu is very low, POI is 1.0 for many points.
# UCB continues to prefer lower mu (since -mu is in the score).
# So UCB distinguishes between "very good" and "good" points, whereas POI treats them all as "likely to improve".
# This could lead to faster convergence (lower simple regret).
#
# Let's try this.
#
# EVOLVE-BLOCK-END

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero (though not strictly needed for UCB form, good for safety)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Structural Change: Cost-Adjusted UCB
    # Score = (f_best - mu + beta * sigma) / cost
    # beta = 1.0 (1-sigma exploration)
    # This balances exploitation (minimizing mu) and exploration (maximizing sigma)
    # while strongly penalizing high cost.
    
    beta = 1.0
    
    # Calculate UCB value for minimization: we want low f(x).
    # f_ucb = mu - beta * sigma
    # We want to maximize the potential improvement: f_best - f_ucb
    # Improvement = f_best - (mu - beta * sigma) = f_best - mu + beta * sigma
    
    improvement_potential = f_best - mu + beta * sigma
    
    # Normalize by cost
    scores = improvement_potential / cost
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return current scores
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
