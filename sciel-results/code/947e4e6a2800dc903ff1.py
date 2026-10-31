import time
import zlib

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# State for the simulated greedy set
# We maintain a list of vectors accepted into our simulated cap set.
# The key insight is that if we assign priorities such that the evaluator's
# greedy selection order matches our simulation's order, we can predict
# which vectors will be accepted and assign high priority to those that
# cause the least conflict with the *current* simulated set.
#
# However, the evaluator calls priority() for ALL vectors first to sort them,
# then iterates through the sorted list to build the set.
# This means during the priority() calls, we do NOT know the final order yet.
#
# Wait, if the evaluator sorts by priority, it calls priority(v) for all v.
# Then it iterates v in sorted order:
#   if is_cap_set(S U {v}): S.add(v)
#
# We cannot know S during the priority() call because S is being built *after*
# all priorities are assigned.
#
# UNLESS... the evaluator calls priority(v) *during* the greedy step?
# "vectors ... are added greedily in descending priority".
# Standard FunSearch contract usually implies:
# 1. Generate candidates.
# 2. Score candidates.
# 3. Sort.
# 4. Greedy select.
#
# If the contract is "score all, then sort", we cannot be stateful with respect to the *actual* set.
#
# Let's re-read the inspiration 1: "stateful, geometry-aware heuristic... uses math, time; defines priority".
# And the current program's comments: "I do NOT know the current set during the priority call."
#
# Is it possible the interface is different?
# "Interface: priority(vector: tuple, n: int) -> float"
#
# If I cannot see the set, how can I do "Conflict Minimization"?
# I can't.
#
# So, what is the "Stateful" part?
# Maybe the "Stateful" refers to maintaining a *predicted* set based on the assumption that the
# greedy algorithm will pick the highest priority vectors first?
#
# If I assume the greedy algorithm picks vectors in the order of my assigned priorities,
# I can simulate the set growth *as I assign priorities*?
# No, I assign priorities in some order (the order the evaluator calls me).
# The evaluator likely calls me for vectors in a fixed order (e.g., lexicographic).
# If I assign a high priority to v, it will be considered early.
#
# Strategy: **Online Greedy Simulation**
# 1. We maintain a list `sim_set` of vectors we *think* will be accepted.
# 2. We also maintain a list `all_vectors` or just process them as they come?
#    The evaluator calls `priority(v)` for each v. We don't know if v is the first or last.
#    But we can assume the calls are in a deterministic order (e.g., lexicographic).
#    
#    If we assume the calls are in lexicographic order, we can't simulate the *final* greedy
#    order because we don't know the priorities of the *other* vectors yet.
#
# Alternative Interpretation:
# Perhaps the "Stateful" inspiration implies that the evaluator *does* provide the set?
# No, the signature is fixed.
#
# Let's look at the "Multiplicative Complexity" directive.
# "Prioritize vectors by maximizing the product of their coordinate values..."
# This is a static heuristic.
#
# Let's look at the "Graph Coloring" hint again.
# "Assign priority based on the vector's degree in the induced subgraph of the greedy set".
# This strongly implies access to the set.
#
# Is it possible that the `priority` function is called *incrementally*?
# i.e., The evaluator picks a candidate, calls priority, adds it, then picks another?
# "vectors ... are added greedily in descending priority".
# This phrasing usually implies a global sort.
#
# However, if I implement a heuristic that *strongly* favors vectors that are "independent"
# in a multiplicative sense, it might align with the structure of large cap sets.
#
# Let's try the **Multiplicative Rank Heuristic** as directed, but combined with a
# **Stateful Conflict Prediction** if possible.
#
# Actually, there is a known technique for this specific FunSearch task:
# If the evaluator calls priority for all vectors, we can use the *call count* or
# *time* to estimate progress? No.
#
# Let's stick to the directive: **Multiplicative Complexity**.
# Why?
# In $\mathbb{F}_3^n$, a set is a cap set if it contains no 3-term AP.
# 3-term AP: $x, y, z$ such that $x+z = 2y$.
# If we treat coordinates independently, a "full rank" or "high product" structure
# avoids linear dependencies.
# Specifically, vectors with many non-zero entries (1 or 2) are "richer" algebraically.
# Vectors with many 0s are "sparse" and more likely to lie on lines with other sparse vectors.
#
# Heuristic:
# Score = $\log(\prod (1 + v_i))$ ? No, 0 is a problem.
# Directive says: "treating 0 as 1 for multiplicative stability".
# So, effective value $v'_i = 1 + v_i$.
# Product $P = \prod_{i=0}^{n-1} (1 + v_i)$.
# We want to maximize $P$.
# This favors vectors with more 2s (value 3) and 1s (value 2), and penalizes 0s (value 1).
# This biases the set towards the "corner" of the cube where values are high.
#
# But wait, cap sets are symmetric under coordinate permutation and value permutation (0<->2).
# If we favor high values, we might get stuck in a sub-cube?
# The $\{0,1\}^n$ hypercube is a cap set of size $2^n$.
# If we favor 1s and 2s over 0s, we might fill the $\{1,2\}^n$ hypercube?
# $\{1,2\}^n$ is also a cap set of size $2^n$.
# The maximum size is much larger than $2^n$ for $n \ge 7$ (approx $3^n / n$).
# So just filling a hypercube is not optimal.
#
# We need to break the hypercube trap.
# The current program uses "Balance" to avoid hypercubes (which are unbalanced in one coordinate direction? No, $\{0,1\}^n$ has 0s and 1s, no 2s. It is unbalanced in the symbol distribution).
# The "Multiplicative" heuristic favors 2s.
#
# Let's combine **Multiplicative Complexity** with **Symbol Balance**.
# 1. Multiplicative Score: $\sum \log(1+v_i)$.
# 2. Balance Score: As before, penalize deviation from $n/3$ per symbol.
#
# The directive says: "shifting the heuristic from statistical balance to multiplicative complexity".
# This suggests *replacing* balance with multiplicative, or making multiplicative dominant.
#
# Let's try:
# `score = alpha * MultiplicativeScore + beta * BalanceScore + gamma * HashScore`
#
# If we follow the directive strictly, we should prioritize the multiplicative term.
#
# Implementation:
# Precompute log values for 1, 2, 3 (corresponding to v=0,1,2).
# log(1)=0, log(2)=0.693, log(3)=1.098.
#
# We also need to handle the "Stateful" aspect if possible.
# If I can't be stateful, I will use a strong deterministic hash to break symmetries
# that the multiplicative score might create (e.g., all permutations of the same multiset
# of values will have the same multiplicative score).
#
# To make it "Materially Different", I will drop the FFT/Spectral component entirely
# and use a purely local, coordinate-wise multiplicative heuristic with strong hash perturbation.

import math
import random

# Precompute logs for multiplicative score
# v=0 -> 1+0=1 -> log(1)=0
# v=1 -> 1+1=2 -> log(2)
# v=2 -> 1+2=3 -> log(3)
_LOG_VALS = [0.0, math.log(2), math.log(3)]

# Precompute weights for positional symmetry breaking
# We use a fixed seed to ensure reproducibility but break the permutation symmetry
_rng = random.Random(12345)
_MAX_N = 100
_POS_WEIGHTS = [_rng.uniform(-0.1, 0.1) for _ in range(_MAX_N)]

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    mult_score = 0.0
    pos_score = 0.0
    
    # Counts for balance
    c0 = 0
    c1 = 0
    c2 = 0
    
    for i, val in enumerate(vector):
        # Multiplicative component
        mult_score += _LOG_VALS[val]
        
        # Positional symmetry breaking
        if i < _MAX_N:
            pos_score += val * _POS_WEIGHTS[i]
        else:
            pos_score += val * _POS_WEIGHTS[i % _MAX_N]
            
        # Balance counts
        if val == 0:
            c0 += 1
        elif val == 1:
            c1 += 1
        else:
            c2 += 1
            
    # Normalize multiplicative score
    # Max possible mult_score is n * log(3)
    max_mult = n * math.log(3)
    norm_mult = mult_score / max_mult if max_mult > 0 else 0.0
    
    # Balance Score
    ideal = n / 3.0
    var = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    max_var = 2 * n * n / 3.0
    if max_var > 0:
        norm_var = var / max_var
    else:
        norm_var = 0.0
    balance = 1.0 - norm_var
    
    # Hash for tie-breaking and further symmetry breaking
    h = 0
    for val in vector:
        h = (h * 31 + val) & 0xFFFFFFFF
    hash_score = (h / 4294967296.0)
    
    # Combine
    # Directive: Shift to multiplicative complexity.
    # Let's weight Multiplicative heavily, Balance moderately, Hash lightly.
    # If we weight Mult too high, we might get stuck in {1,2}^n.
    # Let's try: 0.5 * Mult + 0.4 * Balance + 0.1 * Hash
    # The positional score is added to the hash part to break symmetries locally.
    
    score = 0.5 * norm_mult + 0.4 * balance + 0.1 * hash_score + pos_score
    
    return float(score)
