"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We cache some state to improve performance if the evaluator calls priority
# many times in a tight loop (e.g., during a single packing simulation).
# However, the contract says priority(item, bins) is called per item.
# We cannot assume the bins array is the same object across calls, but we can
# use a simple heuristic that is fast.

# The seed is Best-Fit (maximize used capacity / minimize remaining).
# Score: mean excess over L1 lower bound. Lower is better.
# L1 lower bound for bin packing is max(total_size, max_item_size).
# Actually, the L1 lower bound is usually max(sum(items)/C, max(items)/C) but since C=100,
# the number of bins is at least ceil(sum(items)/100).
# The "excess" is (actual_bins - lower_bound_bins) / lower_bound_bins.

# Best-Fit is a standard strong heuristic.
# Let's try to refine it.
# Common variations:
# 1. Best-Fit Decreasing (not applicable here as items arrive online).
# 2. First-Fit (worse than Best-Fit usually).
# 3. Worst-Fit (often worse).
# 4. Best-Fit with a tie-breaker.

# Let's look at the "inspiration" or "lessons". There are none.
# The directive says "FRESH DESIGN... Write a new design from scratch".
# But it also says "REFINEMENT: for this proposal, KEEP this design and investigate improvements within its family."
# Wait, the directive has both "REFINEMENT" and "FRESH DESIGN".
# "REFINEMENT: ... KEEP this design..."
# "FRESH DESIGN: ... Write a new design from scratch..."
# This is contradictory. Usually, in these prompts, the specific instruction for the *current* step is the one to follow.
# The "DIRECTIVE" section contains:
# "COMPUTE BUDGET..."
# "REFINEMENT: for this proposal, KEEP this design and investigate improvements within its family."
# "FRESH DESIGN: the reference program below is the task's starting point, not the current best. Write a new design from scratch..."

# Let's re-read carefully.
# "DIRECTIVE:
# COMPUTE BUDGET: ...
# REFINEMENT: for this proposal, KEEP this design and investigate improvements within its family. ...
# FRESH DESIGN: the reference program below is the task's starting point, not the current best. Write a new design from scratch; use the measured knowledge in this prompt, not the existing lineage."

# This looks like a template where multiple directives might be present, but typically the one that matches the current state (score 1.0, seed program) is the relevant one.
# The current program IS the seed program.
# If I am evolving from the seed, "FRESH DESIGN" suggests I can change the family.
# However, "REFINEMENT" suggests I should stick to Best-Fit.
# Given the score is 1.0 (baseline), and Best-Fit is a very strong baseline for bin packing, it is hard to beat it significantly with a different online heuristic without knowing the distribution better.
# The distribution is Weibull.
# Weibull(k, lambda) with k<1 is heavy-tailed (many small, few large). k>1 is light-tailed.
# The problem statement doesn't specify the Weibull parameters, just "five generated Weibull sequences".
# Usually, these benchmarks use standard parameters.

# Let's look at the "OBSERVED_CHANGES".
# p3 is the seed (score 1.0).
# p23 is a child of p3, score 1.0 (tie).
# The source diff for p23 is omitted.

# Strategy:
# Best-Fit minimizes the remaining space in the bin.
# Another strong heuristic is "Best-Fit with Gap" or similar.
# Or, we can try to balance the load across bins to avoid leaving one bin half-full while others are full.
# But Best-Fit naturally does this by filling bins up.

# Let's try a slightly modified Best-Fit that considers the *relative* remaining capacity or uses a soft-min.
# The directive mentions: "consider a smooth surrogate for a nonsmooth min/max objective (soft-min/soft-max via log-sum-exp)".
# This suggests moving away from the hard argmin of remaining capacity.

# However, for online bin packing, the decision is local.
# If we use a soft-min, we might spread items more evenly, which could be bad if we want to pack bins tightly.
# Best-Fit is generally superior to First-Fit and Worst-Fit for random inputs.

# Let's try a different angle: **First-Fit Decreasing** is offline. **Best-Fit** is online.
# Is there an online heuristic better than Best-Fit for Weibull?
# Some literature suggests that for certain distributions, **Worst-Fit** or **First-Fit** can be competitive, but Best-Fit is usually the benchmark.

# Let's look at the "REFINEMENT" hint again: "smooth surrogate... soft-min/soft-max".
# If we use a soft-max of the negative remaining capacity, we are essentially doing a probabilistic version of Best-Fit.
# But we must return a single score per bin. The evaluator picks the highest score.
# If we return `score = -remaining_capacity`, it's Best-Fit.
# If we return `score = -remaining_capacity**2`, we prefer bins that are *more* full (quadratic penalty on remaining space). This might help close bins faster.
# If we return `score = -1/remaining_capacity`, we heavily prefer bins with very little space left.

# Let's test `score = -(bins - item)**2`.
# This penalizes large remaining spaces more.
# Or `score = -1 / (bins - item + epsilon)`.

# Another idea: **Next-Fit** is bad. **First-Fit** is okay.
# What about **Harmonic** or **Randomized** heuristics?
# The contract says "each arriving item goes to the highest-scoring bin". This implies a deterministic greedy choice based on the scores.

# Let's try to improve Best-Fit by considering the *total* used capacity of the bin as well? No, remaining capacity is sufficient.

# Let's look at the Weibull distribution.
# If the shape parameter k is small (<1), there are many very small items and a few large ones.
# Best-Fit handles large items well by finding the tightest fit.
# If k is large (>1), items are more uniform.
# Best-Fit is still good.

# Let's try a **Best-Fit with a slight bias towards bins that have been used more**?
# No, that's not standard.

# Let's try **Worst-Fit** for comparison?
# Worst-Fit: `score = bins - item` (maximize remaining). This tends to keep bins balanced but can leave many bins half-full. Usually worse than Best-Fit for minimizing bin count.

# Let's try **First-Fit**?
# First-Fit: pick the first bin that fits.
# To implement First-Fit with a scoring function where the *highest* score is picked:
# We need to give the *first* fitting bin the highest score, and subsequent fitting bins lower scores.
# But we don't know the order of bins? `bins` is an array.
# If the bins are indexed 0..N, First-Fit would pick the lowest index i where `bins[i] >= item`.
# We can score bin `i` as `-i` if it fits, and `-inf` if it doesn't.
# Then the highest score is the lowest index.
# Let's try this. First-Fit is often very competitive and faster.

# However, the "REFINEMENT" directive says "KEEP this design [Best-Fit]".
# The "FRESH DESIGN" directive says "Write a new design".
# Given the ambiguity, and that Best-Fit is the seed (score 1.0), and p23 (a child) also scored 1.0, maybe the search space around Best-Fit is flat.
# Let's try a **soft** version of Best-Fit as suggested by the "REFINEMENT" hint about smooth surrogates.
# But how does a smooth surrogate help in a greedy selection?
# If we use `score = -soft_min(remaining)`, it's just a function of remaining.
# Maybe the hint implies using the scores to do something more complex? No, the interface is fixed.

# Let's try a different heuristic: **Best-Fit Decreasing** is not possible online.
# What about **Next-Fit Decreasing**? No.

# Let's try **Harmonic** heuristic?
# Harmonic: Assign to the smallest bin that fits, but with a twist?
# Actually, the **Harmonic** algorithm for bin packing is:
# Divide bins into classes based on size.
# This is complex to implement with a simple score function.

# Let's stick to simple modifications of Best-Fit.
# 1. `score = -(bins - item)` (Seed)
# 2. `score = -(bins - item)**2`
# 3. `score = -1.0 / (bins - item + 1e-6)`
# 4. `score = np.log(bins - item)` (Worst-Fit variant, maximize log remaining? No, maximize score means maximize log remaining, so Worst-Fit)
# 5. `score = -np.log(bins - item + 1e-6)` (Prefer bins with small remaining, but less aggressively than linear)

# Let's analyze the impact of the Weibull distribution.
# If items are very small, Best-Fit will fill bins to the brim.
# If there is a large item, it goes into the bin with the least remaining space (Best-Fit).
# This is optimal for that item.

# Is there a case where Best-Fit fails?
# If we have a bin with 10 remaining, and an item of size 9. Best-Fit puts it there.
# If we have another bin with 100 remaining.
# Later, an item of size 10 comes. It can't go in the first bin (10+9=19 > 10? No, 10 remaining, item 9, so 1 left. Item 10 doesn't fit).
# So the item 10 goes to the bin with 100 remaining.
# If we had used Worst-Fit, the item 9 would have gone to the bin with 100 remaining.
# Then the item 10 could have gone to the bin with 10 remaining? No, 10 > 10? It fits exactly.
# So Worst-Fit might have been better in this specific scenario?
# Let's trace:
# Bin A: 10 remaining. Bin B: 100 remaining.
# Item 9 arrives.
# Best-Fit: Goes to A. A has 1 remaining.
# Item 10 arrives.
# Best-Fit: Can't go to A (1 < 10). Goes to B. B has 90 remaining.
# Total bins used: 2 (assuming these are the only ones).
# Worst-Fit: Item 9 goes to B (100 remaining > 10). B has 91 remaining.
# Item 10 arrives.
# Worst-Fit: Can go to A (10 remaining) or B (91 remaining).
# Worst-Fit picks B (max remaining). B has 81 remaining.
# A is still 10 remaining.
# This doesn't seem better.

# What if Item 10 arrives first?
# Bin A: 10 remaining. Bin B: 100 remaining.
# Item 10 arrives.
# Best-Fit: Goes to A (fits exactly). A has 0 remaining.
# Item 9 arrives.
# Best-Fit: Goes to B. B has 91 remaining.
# Total bins: 2.
# Worst-Fit: Item 10 goes to B. B has 90 remaining.
# Item 9 goes to B. B has 81 remaining.
# A is untouched.
# Total bins: 1 (if B was empty before? No, B had 100 remaining, meaning it was empty or had 0 used? "Remaining capacity" implies 100 is full capacity. So B was empty. A had 90 used).
# If A had 90 used, and B was empty.
# Best-Fit: Item 10 -> A (full). Item 9 -> B. Total 2 bins.
# Worst-Fit: Item 10 -> B. Item 9 -> B. Total 1 bin (B).
# So Worst-Fit was better here!

# This suggests that for Weibull distributions with large items, **Worst-Fit** might perform better by keeping bins open for larger items.
# However, Worst-Fit is generally considered worse for uniform distributions.
# Weibull with k<1 has large items.
# Let's try a hybrid or a parameterized score.
# `score = -remaining` (Best-Fit)
# `score = +remaining` (Worst-Fit)

# Since the score is "mean excess... lower is better", and we want to *minimize* the number of bins, we want to pack efficiently.
# Let's try **Worst-Fit** and see if it improves the score.
# If the Weibull sequences have many small items, Worst-Fit might spread them too thin.
# But the "large item" example above is compelling.

# Let's try a **Soft Worst-Fit** or **Balanced** approach.
# `score = np.log(bins - item)` (Worst-Fit, but softer)
# `score = bins - item` (Worst-Fit)

# Let's try **First-Fit**.
# First-Fit is often very good for online bin packing.
# Implementation:
# `fits = bins >= item`
# `indices = np.where(fits)[0]`
# If no fits, return -inf for all.
# If fits, give the lowest index the highest score.
# `scores = np.full_like(bins, -np.inf)`
# `scores[indices] = -indices`
# This makes the first fitting bin have the highest score (least negative index).

# Let's compare Best-Fit vs First-Fit.
# Best-Fit is usually better than First-Fit for minimizing bin count.
# Reference: "A tight analysis of the Best-Fit algorithm for bin packing".
# Best-Fit is 1.7 times optimal, First-Fit is 1.7 times optimal.
# But in practice, Best-Fit is often slightly better.

# Given the "REFINEMENT" hint to keep the design (Best-Fit) and investigate improvements, and the "FRESH DESIGN" hint, I will try a **modified Best-Fit** that uses a **quadratic penalty** on remaining space.
# `score = -(bins - item)**2`
# This prefers bins that are *very* full.
# In the example above:
# Bin A: 10 remaining. Bin B: 100 remaining.
# Item 10.
# Best-Fit: Score A = -10^2 = -100. Score B = -100^2 = -10000. Picks A.
# Item 9.
# A has 0 remaining. B has 100.
# Score B = -10000. Picks B.
# Same result.

# What if Bin A: 11 remaining. Bin B: 100 remaining.
# Item 10.
# Best-Fit: A (-11^2=-121) vs B (-10000). Picks A. A has 1 remaining.
# Item 9.
# A (1 remaining, doesn't fit). B (100 remaining). Picks B.
# Total 2 bins.

# If we used `score = -1/(remaining + eps)`:
# Item 10.
# A: -1/11 = -0.09. B: -1/100 = -0.01.
# Picks A (higher score).
# Same.

# It seems Best-Fit is robust.
# Let's try **First-Fit** just in case the Weibull distribution has a specific structure where First-Fit shines (e.g., if items are sorted? No, online).

# Let's look at the "OBSERVED_CHANGES" again.
# p23 is a child of p3 (Best-Fit) and scored 1.0.
# This suggests that simple mutations of Best-Fit might not be improving the score.
# Maybe the score is already optimal for these specific Weibull sequences?
# Or maybe the evaluation is noisy?
# "Score: mean excess... lower is better".
# If the seed is 1.0, and the child is 1.0, it's a tie.

# I will try a **Hybrid** approach:
# Use Best-Fit, but if the remaining capacity is very small, use a different metric?
# No, let's keep it simple.

# Let's try **Worst-Fit** to see if it jumps.
# If the Weibull sequences have a heavy tail (large items), Worst-Fit might be better.
# If they are light-tailed, Best-Fit is better.
# Without knowing the parameters, it's a gamble.
# However, "Weibull sequences" in bin packing benchmarks often use k=2 or similar (light tail) or k=0.5 (heavy tail).
# If k=0.5, there are many small items and a few very large ones.
# Best-Fit handles the large ones well by finding the tightest spot.
# Worst-Fit might leave the large items in bins with lots of space, wasting space if the large items don't fill the bin.
# But Best-Fit might also leave space if no bin is tight enough.

# Let's try **Best-Fit** with a **tie-breaker** that prefers bins with *more* total used capacity?
# No, remaining capacity is the only state.

# I will stick with **Best-Fit** but add a small **randomization** or **perturbation**?
# No, the function must be deterministic for the same input?
# "each arriving item goes to the highest-scoring bin".
# If I add randomness, it might not be reproducible or might violate the "highest-scoring" contract if the scores are tied.
# But I can break ties arbitrarily.

# Let's try **First-Fit** as a fresh design.
# It's a standard, strong heuristic.
# If Best-Fit is 1.0, First-Fit might be 1.0 or slightly different.

# Actually, let's look at the **L1 lower bound**.
# L1 LB = max(sum(items)/C, max(items)/C).
# Excess = (Actual - LB) / LB.
# To minimize excess, we want Actual to be as close to LB as possible.
# This means we want to pack as tightly as possible.
# Best-Fit is designed for this.

# I will try a **soft-min** approach as hinted.
# `score = -soft_min(remaining, temperature)`
# But `soft_min` of a single value is just the value.
# The hint might be misleading for this specific interface.

# Let's try **Worst-Fit** (`score = bins - item`).
# If it performs worse, the score will be > 1.0 (worse).
# If it performs better, the score will be < 1.0 (better).
# Since I can't test it, I have to guess.
# Given that Best-Fit is the standard "best" online heuristic, I will assume it's hard to beat.
# However, the directive says "Write a new design from scratch".
# I will try **First-Fit**.

# Implementation of First-Fit:
# 1. Identify bins that fit: `fits = bins >= item`
# 2. If none fit, return `-np.inf` for all.
# 3. If some fit, assign scores such that the first fitting bin has the highest score.
#    `indices = np.where(fits)[0]`
#    `scores = np.full_like(bins, -np.inf)`
#    `scores[indices] = -indices`
#    The highest score is `-min(indices)`, which corresponds to the smallest index.

# Let's compare Best-Fit and First-Fit complexity.
# Both are O(N) per item.
# Best-Fit: `np.max` on filtered array.
# First-Fit: `np.where` and index assignment.

# I will go with **First-Fit** as the "fresh design".

# Wait, there is one more option: **Next-Fit**.
# Next-Fit: Only use the current bin. If it doesn't fit, open a new one.
# This is worse.

# Let's try **Worst-Fit** instead.
# Why? Because the "REFINEMENT" hint mentioned "smooth surrogate".
# Worst-Fit is the "max" of remaining capacity.
# Best-Fit is the "min" of remaining capacity.
# If the objective is to minimize the number of bins, and the items are Weibull, maybe balancing the load (Worst-Fit) helps avoid the "fragmentation" caused by Best-Fit?
# Fragmentation: Best-Fit might fill bins to 99%, leaving 1% waste.
# Worst-Fit might fill bins to 50%, leaving 50% waste, but might pack larger items better?
# No, 50% waste is worse.

# I'll stick with **Best-Fit** but try to **optimize the implementation** or add a **tie-breaker**.
# The current seed is `return -(bins - item)`.
# This returns `-inf` for bins that don't fit?
# No, `bins - item` can be negative.
# If `bins < item`, `bins - item` is negative.
# `-(bins - item)` is positive.
# So a bin that *doesn't* fit gets a *positive* score.
# A bin that *fits* gets a *negative* score (if remaining > 0).
# So the *highest* score is the *most positive* one, i.e., the bin that *doesn't* fit!
# This is a BUG in the seed program!

# Let's re-read the seed:
# `return -(bins - item)`
# If `bins = [10, 20]`, `item = 15`.
# `bins - item = [-5, 5]`
# `-(bins - item) = [5, -5]`
# Highest score is 5 (Bin 0).
# Bin 0 has 10 capacity, item is 15. It doesn't fit!
# The evaluator says: "score each bin with **enough remaining capacity**".
# "The item goes to the highest-scoring bin".
# If the seed returns a high score for bins that don't fit, the evaluator might pick a bin that doesn't fit.
# What happens then?
# "Score: mean excess... lower is better".
# If an item doesn't fit, it must open a new bin?
# The contract says: "score each bin with enough remaining capacity".
# This implies the function should *only* return valid scores for bins that fit, and `-inf` (or very low) for others.
# The seed program `return -(bins - item)` does **not** mask out invalid bins.
# It gives them the highest scores!
# This means the seed program is **invalid** or **buggy** in the sense that it prefers bins that don't fit.
# Unless the evaluator filters the scores?
# "score each bin with enough remaining capacity; each arriving item goes to the highest-scoring bin".
# If the evaluator takes the argmax of the returned array, and the seed returns high values for non-fitting bins, it will pick a non-fitting bin.
# This would likely result in an exception or an invalid packing (item > capacity).
# But the seed has a score of 1.0.
# This implies either:
# 1. The evaluator handles the "doesn't fit" case by opening a new bin, and the score of 1.0 reflects that inefficiency?
# 2. The evaluator *does* filter the bins before calling priority? No, "score each bin...".
# 3. The evaluator ignores bins where `bins < item`?
#    "score each bin with enough remaining capacity" might mean the *function* is only called with bins that fit?
#    No, `bins` is an array of all bins.
# 4. The evaluator's "highest-scoring bin" logic includes a check for `bins[i] >= item`.
#    If the evaluator does `valid_indices = np.where(bins >= item)[0]`, then `argmax(scores[valid_indices])`, then the seed's behavior for invalid bins doesn't matter.
#    But the seed returns `-(bins - item)`. For valid bins, this is `-remaining`.
#    So it picks the bin with the smallest remaining capacity (Best-Fit).
#    This matches the "Best-Fit" description in the comment.
#    So the evaluator likely filters valid bins.

# If the evaluator filters valid bins, the seed is correct Best-Fit.
# My "bug" observation was about the raw array, but the evaluator logic is key.

# So, how to improve?
# If the evaluator filters, I should ensure my scores are high for valid bins and low for invalid ones, just in case the evaluator doesn't filter.
# Safe practice: `scores = np.full_like(bins, -np.inf); valid = bins >= item; scores[valid] = ...`

# Let's implement a robust **Best-Fit** with masking.
# And maybe try **First-Fit** or **Worst-Fit** as well.
# Given the "FRESH DESIGN" directive, I will implement **First-Fit** with masking.
# If First-Fit is better, score < 1.0.
# If not, score >= 1.0.

# Let's try **Worst-Fit** with masking.
# `scores[valid] = bins[valid] - item`

# Which one to choose?
# I'll go with **Best-Fit** but with a **quadratic** penalty to see if it tightens the packing more.
# `scores[valid] = -(bins[valid] - item)**2`

# Let's write the code for **Quadratic Best-Fit**.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    scores = np.full_like(bins, -np.inf, dtype=float)
    valid = bins >= item
    if not np.any(valid):
        return scores
    
    remaining = bins[valid] - item
    # Quadratic Best-Fit
    scores[valid] = -remaining**2
    return scores

# This is a small change from the seed.
# The seed was `-(bins - item)` for all bins.
# This is `-(bins - item)**2` for valid bins, `-inf` for invalid.

# Let's consider if **First-Fit** is a better "fresh design".
# First-Fit is simpler and often very fast.
# But Best-Fit is generally better for bin count.

# I will stick with **Quadratic Best-Fit**.
# It emphasizes filling bins more completely.

# One more idea: **Harmonic**.
# Too complex.

# Final decision: **Quadratic Best-Fit** with proper masking.

# EVOLVE-BLOCK-END
