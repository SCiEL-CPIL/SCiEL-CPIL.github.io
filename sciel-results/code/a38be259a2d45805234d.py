# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Strategy: Dynamic, stateful conflict-avoidance using a local "independence" heuristic.
# Instead of a static global balance score, we maintain the current set of accepted vectors.
# For each candidate vector, we estimate its "conflict potential" with the existing set.
# A vector v conflicts with a pair (u, w) in the set if u, v, w form a 3-term AP.
# In F_3^n, a 3-term AP is {a, a+d, a+2d}.
# So v is the middle term if v = (u+w)/2. Since we are in integers mod 3, 
# v_i = (u_i + w_i) / 2 mod 3.
# Or v is an endpoint: v = a or v = a+2d.
# 
# The greedy algorithm adds vectors in descending priority.
# The priority function is called for each vector *before* it is added.
# However, the set of *already added* vectors is known to the evaluator's internal state,
# but NOT to our `priority` function directly, unless we use global state.
# The evaluator calls `priority(v, n)` for each v in the full set {0,1,2}^n, sorts them,
# and then greedily adds them if they don't create an AP with the *currently accepted* set.
# 
# Key Insight: We can track the accepted set in a global variable inside `priority`.
# Every time `priority` is called, we don't know if the vector will be accepted yet.
# BUT, the evaluator sorts *all* vectors first based on their static priority scores,
# and THEN iterates through the sorted list, adding them if valid.
# Therefore, the `priority` function is called exactly once for each vector,
# likely in a batch or sequence to generate the sorted list.
# We CANNOT use the state of the accepted set *during* the priority calculation
# because the priority calculation happens *before* the greedy selection phase.
#
# WAIT. Let's re-read the contract: "vectors of {0,1,2}^n are added greedily in descending priority".
# This implies:
# 1. Calculate priority for ALL vectors.
# 2. Sort vectors by priority (descending).
# 3. Iterate through sorted list:
#    - If vector can be added to current set (no AP formed), add it.
#
# If this is the case, the `priority` function must be STATIC or depend only on the vector itself,
# because the "current set" is empty at the start of the sorting phase and evolves during the greedy phase,
# which happens *after* all priorities are computed.
#
# HOWEVER, some evaluators might call `priority` dynamically?
# "Interface: priority(vector: tuple, n: int) -> float; vectors of {0,1,2}^n are added greedily in descending priority"
# Usually, this means a static sort. If it were dynamic (like a priority queue where the priority of remaining items changes as items are added), the contract would likely specify that.
# Given the score of 1.2148 with a static heuristic, and inspiration 2 mentioning "dynamic, stateful" but only scoring 1.2305 (which is close), it's possible that dynamic state is NOT fully exploitable in the way I thought, OR the evaluator re-calls priority?
#
# Let's assume the standard FunSearch contract:
# 1. `priority` is called for all 3^n vectors to produce a list of (score, vector).
# 2. The list is sorted.
# 3. Greedy selection happens.
#
# If this is the case, we CANNOT know which vectors are already in the set when computing the priority of a new vector.
# Therefore, the priority MUST be a function of the vector alone (and n).
#
# So, how do we improve?
# The current heuristic is "balanced counts".
# Let's try a different structural feature: **Symmetry Breaking / Geometric Distribution**.
#
# In F_3^n, lines are affine subspaces of dimension 1.
# A vector with many 0s is "close" to the origin.
# A vector with many 2s is "close" to 2*origin = 0? No.
#
# Let's look at the "zero-heavy" hypothesis from the directive:
# "Model the priority as a weighted sum of characteristically biased coordinates by assigning higher weights to positions where the vector contains the digit 0... vectors with more zeros are structurally less likely to form the symmetric triples"
#
# Let's test the "Zero-Heavy" hypothesis.
# Prioritize vectors with a high count of 0s.
# If we prioritize 0s, we fill the {0,2}^n subspace first?
# {0,2}^n is a cap set of size 2^n.
# If we fill this first, we have a large base.
# Then we try to add vectors with 1s.
# Vectors with 1s conflict with pairs in {0,2}^n?
# Let u, w in {0,2}^n. Can v with 1s form an AP?
# v = (u+w)/2. If u,w have only 0,2, then u+w has even entries (0, 2, 4->1).
# (u+w)/2 mod 3:
# 0/2 = 0
# 2/2 = 1
# 4/2 = 2
# So if u,w are in {0,2}^n, the midpoint v can have 0s, 1s, 2s.
# Specifically, if u_i=0, w_i=0 -> v_i=0
# u_i=0, w_i=2 -> v_i=1
# u_i=2, w_i=0 -> v_i=1
# u_i=2, w_i=2 -> v_i=2
# So midpoints of pairs from {0,2}^n can have 1s.
#
# If we fill {0,2}^n completely, then any vector v that is the midpoint of two vectors in {0,2}^n CANNOT be added.
# Which vectors are midpoints of pairs in {0,2}^n?
# For any v in {0,1,2}^n, can we find u,w in {0,2}^n such that v = (u+w)/2?
# v_i = 0 -> u_i=0, w_i=0
# v_i = 1 -> (0,2) or (2,0)
# v_i = 2 -> u_i=2, w_i=2
# So EVERY vector in {0,1,2}^n is the midpoint of some pair in {0,2}^n?
# Let's check.
# If v has a 1, we need one 0 and one 2 in the corresponding coordinates of u and w.
# If v has a 0, we need 0,0.
# If v has a 2, we need 2,2.
# So yes, for ANY v, we can construct u and w in {0,2}^n such that v is the midpoint.
# Therefore, if the set contains ALL of {0,2}^n, NO other vector can be added!
# Because any other vector v would form an AP with u and w from {0,2}^n.
#
# Conclusion: Filling {0,2}^n first is a TERRIBLE strategy, as it blocks all other vectors.
# We need a strategy that picks a SUBSET of {0,2}^n that leaves room for vectors with 1s.
# Or, prioritize vectors with 1s early?
#
# Let's look at the structure of a good cap set.
# The known best cap sets often have a specific structure.
# For n=8, the best is 20.
#
# Let's try a heuristic that penalizes vectors that are "too close" to the origin (too many 0s) AND "too close" to the hyperplane (too many 1s)?
#
# Alternative Idea: **Randomized Hashing / Chaos**.
# The current heuristic is smooth. Maybe a discontinuous or pseudo-random ordering based on the vector's value in a finite field representation helps break symmetries that lead to early APs?
#
# Let's try: Map the vector to an integer in [0, 3^n).
# Use a hash function to create a priority.
# But we need to bias it.
#
# Let's try a "Layered" approach.
# Layer 0: Vectors with 0 ones. (Size 2^n)
# Layer 1: Vectors with 1 one. (Size n * 2^(n-1))
# Layer 2: Vectors with 2 ones.
# ...
#
# If we prioritize Layer 0, we block everything.
# If we prioritize Layer 1, we might get a good set?
#
# Let's try a score that peaks at exactly 1 or 2 ones.
# Score = - (c1 - 1.5)^2
#
# Let's implement a score that favors a specific "weight" (sum of digits).
# In F_3, weight = sum of digits mod 3?
#
# Let's try the "Geometric Series Tie-Breaker" more aggressively.
# The current program uses `tie_breaker / 10000`.
# What if the base score is very coarse (e.g., just count of 1s), and the tie-breaker is a strong hash?
#
# Let's try:
# Base Score: - (count_1)  # Favor fewer 1s, but not zero?
# No, let's try: Base Score = 0 for all.
# Tie-Breaker: A stable hash of the vector.
# This effectively picks a random permutation, biased by the hash.
# But random permutations usually score ~1.0.
#
# Let's go back to the "Zero-Heavy" directive but with a twist:
# "vectors with more zeros are structurally less likely to form the symmetric triples"
# This is false if we fill ALL zeros. But maybe a *sparse* set of zero-heavy vectors is good?
#
# Let's try a score that rewards **Variance** in the digits.
# A vector like (0,0,0,1,1,1,2,2) has high variance?
# A vector like (1,1,1,1,1,1,1,1) has low variance.
#
# Let's try: Score = Variance(vector)
# High variance means the digits are spread out.
#
# Implementation:
# 1. Compute mean of vector.
# 2. Compute variance.
# 3. Return variance.
#
# Let's test this logic mentally.
# (0,0,0,0,0,0,0,0) -> Var 0.
# (1,1,1,1,1,1,1,1) -> Var 0.
# (0,0,0,0,2,2,2,2) -> Mean 1, Var 1.
# (0,1,2,0,1,2,0,1) -> Mean 1, Var 1.
#
# Vectors with high variance have a mix of 0s, 1s, 2s.
# These are the "balanced" vectors.
# The current program uses "closeness to mean count", which is similar.
#
# Let's try a DIFFERENT structural change:
# **Local Interaction Penalty**.
# We can't see the set, but we can approximate "density".
#
# Actually, let's look at the inspiration 2 which scored 1.2305.
# It used "dynamic, stateful conflict-avoidance".
# If the evaluator allows state, we can track the set.
# How?
# The evaluator calls `priority(v, n)` for each v.
# If we store `v` in a global list every time `priority` is called, we have the full list of vectors.
# But we don't know the order of calls.
# However, if the evaluator calls `priority` for ALL vectors, sorts them, and then adds them, we can't update the set during the sorting phase.
#
# UNLESS... the evaluator calls `priority` inside the greedy loop?
# "vectors ... are added greedily in descending priority"
# This phrasing is ambiguous. It could mean:
# 1. Pre-compute all priorities, sort, then greedy add. (Static)
# 2. Use a priority queue, pop max, check if valid, if not, discard and pop next. (Dynamic)
#
# If it's (2), then `priority` is called, returns a score, the vector is popped, and then checked.
# If the check fails, the vector is discarded.
# In this case, the "current set" is NOT available to `priority` either, because `priority` is called *before* the check.
#
# So, in both cases, `priority` is effectively static with respect to the set.
#
# Therefore, we must find a better STATIC heuristic.
#
# Let's try a heuristic based on **Modular Arithmetic**.
# Sum of digits mod 3?
# Sum of (i * v_i) mod 3?
#
# Let's try: Score = (sum(v) % 3) * 10 + (sum(i*v_i) % 3)
# This creates 9 distinct classes.
#
# Let's try to maximize the "distance" from the subspace {0,2}^n.
# Distance = count of 1s.
# We want to avoid 0 (blocks everything) and avoid high (conflicts).
# Peak at 1 or 2.
#
# Let's try: Score = -abs(count_1 - 2)
#
# And a strong tie-breaker:
# Hash = sum(v_i * 31**i) % 1000000
# Score = base + hash / 1e6
#
# Let's implement this.

def priority(vector, n):
    # Count 1s
    c1 = sum(1 for v in vector if v == 1)
    
    # Base score: Peak at c1 = 2
    base_score = -abs(c1 - 2)
    
    # Tie-breaker: Stable hash
    # Use a simple polynomial hash
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v) % 1000003
        
    # Combine
    # Base score range: [-n, 0]
    # Hash range: [0, 1000003)
    # We want base score to dominate.
    # If base score difference is 1, hash difference should be < 1.
    # So scale hash by 1e-6.
    
    final_score = base_score + h * 1e-6
    
    return final_score

# EVOLVE-BLOCK-END
