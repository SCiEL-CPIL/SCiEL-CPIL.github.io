import time
import zlib

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# Stateful storage for the greedy set
# The evaluator adds vectors greedily. We assume the order of calls to priority()
# reflects the order in which vectors are considered.
# However, the interface is priority(vector, n) -> float.
# We do not know which vectors have been *accepted* into the cap set by the evaluator.
# Wait, the contract says: "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set".
# This implies the evaluator maintains the set.
# My priority function only sees the candidate vector. It does NOT see the current set.
# Therefore, I CANNOT compute exact conflicts with the current set unless I can infer the set.
#
# Re-reading the contract: "Interface: priority(vector: tuple, n: int) -> float".
# The function is called for a candidate. The evaluator decides if it's added.
# I do not have access to the current set S.
#
# If I cannot see S, I cannot do "stateful conflict minimization" in the strict sense of checking against S.
# However, I can maintain a *model* of S if I assume the greedy algorithm is deterministic based on my priorities?
# No, that's risky. If my priority function is non-deterministic or if the evaluator's tie-breaking differs, my model drifts.
#
# BUT, looking at the inspirations:
# "inspiration 1 ... uses itertools, math, numpy, time; defines _count_conflicts, _hamming_weight, priority, _accept_vector"
# This suggests that some solutions *do* try to track the set or use a different mechanism.
# Actually, if the evaluator calls priority() for ALL vectors in {0,1,2}^n to determine the order, then I can't know which ones were accepted until I see the result, which I don't.
#
# Let's re-read carefully: "vectors ... are added greedily in descending priority".
# This usually means:
# 1. Evaluator generates all vectors.
# 2. Evaluator calls priority(v) for each v.
# 3. Evaluator sorts v by priority.
# 4. Evaluator iterates through sorted list: if v keeps set a cap set, add v.
#
# In this mode, `priority` is called for every vector exactly once (or in batches).
# I do NOT know the current set during the `priority` call.
#
# So, "stateful conflict minimization" against the *actual* set is impossible with this interface unless the interface is different (e.g., `priority(vector, current_set)`).
#
# However, the inspiration 1 mentions `_accept_vector`. This might imply a different interface or a misunderstanding of the contract in the inspiration note, OR it implies that the priority function can have side effects that persist, but without knowing which were accepted, it's useless.
#
# Wait, look at the current program (score 1.29). It is stateless.
# Look at inspiration 3: "stateful, adaptive heuristic ... relative to the current set size".
# How can it know the current set size?
# Maybe the number of calls to `priority` correlates with the process?
# If the evaluator calls `priority` for all $3^n$ vectors, the total number of calls is fixed.
# But I don't know *which* ones are being called.
#
# Let's look at the "graph coloring" hint in the directive:
# "Model the cap set constraint as a graph coloring problem ... assign priority based on the vector's degree in the induced subgraph of the greedy set".
# This also implies knowing the set.
#
# Is it possible that the evaluator calls `priority` sequentially and I can assume that if I return a high score, it *might* be added? No, I don't get feedback.
#
# Alternative interpretation:
# Perhaps the "greedy" part is implemented *by me*?
# No, "The evaluator measures the raw value... The search score is value/seed".
# "vectors ... are added greedily ... (FunSearch contract)".
# This confirms the evaluator does the adding.
#
# If I cannot see the set, I must use a heuristic that *anticipates* good cap set structure.
# The current program uses FFT smoothness.
# The "graph coloring" hint might be metaphorical or referring to a different interface variant.
#
# However, there is a way to make a "stateful" heuristic if the calls are ordered in a way that reveals the set? No.
#
# Let's reconsider the "different strategy" requirement.
# The current strategy is "Spectral Smoothness".
# A different strategy: "Combinatorial Distance to Known Cap Set Structures".
# Or: "Randomized Hashing with Symmetry Breaking".
#
# Actually, let's look at the cell findings:
# "Replacing the discrete distinct-symbol step and L1 balance penalty with smooth quadratic surrogates ... gives the greedy ordering a smoother, more informative landscape ... escape the {0,1}^n sub-cube trap".
# This suggests that *balance* and *smoothness* are key.
#
# Let's try a strategy based on **Hamming Weight Distribution and Pairwise Independence**.
# Instead of FFT (which is global frequency), let's use **Local Pairwise Constraints**.
# Vectors that are "far" from each other in Hamming distance are less likely to form APs?
# No, APs depend on linear structure.
#
# Let's try the **Graph Degree Heuristic** but approximated.
# Since I can't see the set, I can't compute the degree.
#
# What if I use a **Deterministic Randomization** strategy that breaks symmetries more aggressively?
# The current program has a small hash tie-breaker.
# Cap sets are highly symmetric. The greedy algorithm often gets stuck in symmetric traps (like picking all 0s then all 1s).
# A strong, fixed random perturbation (seeded by the vector itself) might help escape these traps.
#
# Let's combine:
# 1. **Balance Score**: How close is the vector to having n/3 of each symbol? (Like current, but maybe weighted differently).
# 2. **Symmetry Breaking**: A strong, deterministic hash of the vector to break ties and disrupt symmetric patterns.
# 3. **Anti-Clustering**: Penalize vectors that are "too similar" to the "center" (all 1s) or "corners" (all 0s/2s).
#
# Wait, the current program *already* does Centrality (balance) and Hash.
# Score = -HFE + Centrality + Hash.
#
# What is materially different?
# **Change the objective from "Smoothness" to "Maximizing Minimum Distance to Existing Typical Points"**.
# Since I don't know existing points, I can't do that.
#
# Let's look at the "Graph Coloring" hint again.
# "Assign priority based on the vector's degree in the induced subgraph of the greedy set".
# If I assume the greedy set is "typical", what is the degree?
# A vector $v$ creates an AP if there exists $u$ in $S$ such that $2v-u \in S$ or $v + (v-u)/2 \in S$?
# In $\mathbb{Z}_3$, a 3-term AP is $\{a, a+d, a+2d\}$.
# $v$ creates an AP with $S$ if:
# 1. $v, v+d, v+2d \in S \cup \{v\}$? No, $v$ is new.
# 2. There exist $u_1, u_2 \in S$ such that $\{v, u_1, u_2\}$ is an AP.
# This requires checking pairs in $S$.
#
# If I truly cannot access $S$, I must rely on intrinsic properties of $v$.
#
# Let's try a **Polynomial Hashing** approach that is more complex than the current linear hash.
# The current hash is `h = (h * 37 + val)`. This is a simple rolling hash.
# A more robust hash might better distribute priorities and avoid systematic biases.
#
# However, the biggest lever in greedy cap set construction is often **breaking the symmetry of the coordinates**.
# The current program treats all coordinates symmetrically.
# What if I introduce a **Positional Bias**?
# For example, prioritize vectors that have specific values at specific indices?
# This breaks the permutation symmetry of the coordinates.
#
# Strategy: **Positional Symmetry Breaking with Balance**.
# 1. Compute Balance (Centrality) as before.
# 2. Compute a "Positional Hash" that gives different weights to different coordinate indices.
#    e.g., `score += sum(v[i] * weight[i] for i in range(n))`.
#    The weights `weight[i]` are fixed pseudo-random numbers.
# 3. This creates a unique "landscape" that is not invariant under coordinate permutation.
#    This might help the greedy algorithm avoid getting stuck in the same symmetric trap every time, or find a different branch of the search tree.
#
# Let's implement this.
# We need fixed weights. We can generate them using a simple PRNG seeded with a constant.
#
# Additionally, let's drop the FFT (which is computationally expensive and maybe not the right feature) and replace it with a **Hamming Weight Variance** term.
# Vectors with Hamming weight close to $n/3$ (for 0s) etc. are balanced.
#
# New Score Components:
# 1. **Balance**: $1 - \frac{\sum (c_i - n/3)^2}{2n^2/3}$. (Same as current Centrality).
# 2. **Positional Bias**: $\sum_{i=0}^{n-1} v_i \cdot W_i$, where $W_i$ are fixed random weights in $[-1, 1]$.
# 3. **Anti-Constant Penalty**: Penalize vectors that are constant or near-constant (low entropy).
#    Entropy can be approximated by $1 - \max(c_i)/n$.
#
# Let's refine the Positional Bias.
# If I use random weights, I might favor vectors with high values at high-weight indices.
# This effectively rotates the objective function in the vector space.
#
# Implementation details:
# - Precompute weights for max n (say 50).
# - `priority` computes balance and positional score.
# - Combine: `score = 2.0 * Balance + 1.0 * PositionalScore`.
# - Add a tiny random tie-breaker to ensure strict ordering.

import random

# Precompute positional weights
# We need to handle variable n. We'll generate a long list of weights.
_rng = random.Random(42)
_MAX_N = 100
_WEIGHTS = [_rng.uniform(-1.0, 1.0) for _ in range(_MAX_N)]

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # 1. Balance Score (Centrality)
    c0 = 0
    c1 = 0
    c2 = 0
    pos_score = 0.0
    
    # Iterate to count and compute positional score
    # We assume vector is a tuple of ints
    for i, val in enumerate(vector):
        if i < _MAX_N:
            pos_score += val * _WEIGHTS[i]
        else:
            # If n > _MAX_N, wrap around or ignore? 
            # n is likely small.
            pos_score += val * _WEIGHTS[i % _MAX_N]
            
        if val == 0:
            c0 += 1
        elif val == 1:
            c1 += 1
        elif val == 2:
            c2 += 1

    ideal = n / 3.0
    var = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    max_var = 2 * n * n / 3.0
    if max_var > 0:
        norm_var = var / max_var
    else:
        norm_var = 0.0
    
    balance = 1.0 - norm_var
    
    # 2. Positional Score
    # Normalize positional score to roughly [-1, 1] or similar?
    # The sum of val * weight can be up to 2 * sum(|weights|).
    # Let's just use it as is, scaled.
    # To make it comparable to balance [0,1], we can scale it.
    # Max possible pos_score is roughly 2 * n * 1 = 2n.
    # Let's normalize by n.
    normalized_pos = pos_score / n if n > 0 else 0.0
    # Map to [0, 1] range? 
    # pos_score is in [-n, n] roughly.
    # Let's map it to [0, 1] by adding 0.5 and scaling?
    # Or just use it as a tie-breaker / bias.
    # If we want it to significantly affect the order, we should weight it.
    # Let's try: score = balance + 0.5 * normalized_pos
    
    # 3. Tie-breaker
    # Use a strong hash to break remaining ties
    h = 0
    for val in vector:
        h = (h * 31 + val) & 0xFFFFFFFF
    hash_score = (h / 4294967296.0) * 0.01 # Very small
    
    # Combine
    # Balance is the primary driver (cap sets need balance).
    # Positional score breaks symmetry.
    score = balance + 0.5 * normalized_pos + hash_score
    
    return float(score)
