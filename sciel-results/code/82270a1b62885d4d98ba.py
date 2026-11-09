# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
import random

# Global state for the greedy cap set construction
_cap_set_state = {}

def _reset_state(n, all_vectors):
    """
    Reset the global state for a new dimension.
    Pre-compute all vectors as numpy arrays for speed.
    """
    if n in _cap_set_state:
        return
    
    _cap_set_state['n'] = n
    _cap_set_state['start_time'] = time.time()
    _cap_set_state['deadline'] = _cap_set_state['start_time'] + 320.0  # Safety margin
    
    # Generate all vectors in {0,1,2}^n
    # For n up to 10, 3^10 = 59049, which is manageable.
    # For n > 10, we might need to be careful, but let's assume n <= 10 for now.
    # If n is larger, we can generate on the fly or use a different strategy.
    # Let's check the range. If n > 10, 3^n grows fast. 
    # n=11 -> 177147, n=12 -> 531441.
    # The cap set size for n=11 is known to be small (around 120?).
    # But generating 500k vectors and sorting them might be slow?
    # Sorting 500k floats is fast in numpy/python.
    # The greedy check is the bottleneck.
    
    # Let's generate the list of tuples first, then sort by priority.
    # Then we process them one by one.
    
    # To save memory and time for large n, we might want to generate lazily?
    # But we need to sort by priority.
    # So we must generate all priorities.
    
    # Let's optimize the vector generation.
    # Using itertools.product is fine.
    
    # However, the priority function might depend on the state?
    # No, the priority function is static (as required by the interface: priority(vector, n)).
    # The state is only used by the evaluator's greedy loop?
    # Wait, the task says: "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set".
    # This implies the *evaluator* runs the greedy loop using my priority function.
    # OR, do I run the greedy loop inside priority?
    # "Interface: priority(vector: tuple, n: int) -> float"
    # "The evaluator measures the raw value (higher is better). The search score is value/seed..."
    # "Score: cap set size"
    
    # This implies I return a float.
    # If I return a priority score, the evaluator uses it to order vectors.
    # But the "raw value" is the cap set size?
    # "Score: cap set size"
    # "The evaluator measures the raw value (higher is better)."
    
    # This is ambiguous. Does the evaluator:
    # 1. Call priority(v, n) for all v, sort by priority, run greedy, and measure the size?
    #    In this case, my priority function just needs to be a good heuristic.
    #    The "raw value" is the size of the cap set produced by that heuristic.
    # 2. Or do I have to return the size of the cap set I constructed?
    #    But the signature is priority(vector, n) -> float.
    #    It doesn't take the current set as input.
    #    So I cannot compute a dynamic priority that depends on the current set unless I maintain global state.
    #    But if the evaluator calls priority(v, n) for all v first (to get a sorted list), then the state is empty.
    #    If the evaluator calls priority(v, n) one by one in a loop, I *could* maintain state, but the order of calls is determined by the priority?
    #    No, usually "greedy in descending priority" means:
    #       1. Compute priority for all v.
    #       2. Sort v by priority.
    #       3. Iterate through sorted v, adding to set if valid.
    #    
    #    If this is the case, my priority function MUST be static (independent of the current set).
    #    Because the sorting happens before the greedy addition.
    #    
    #    Let's re-read carefully: "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set (FunSearch contract)."
    #    This standard phrasing in FunSearch/GDM tasks usually implies:
    #       - The program provides a scoring function.
    #       - The harness uses this score to determine the order.
    #       - The harness performs the greedy selection.
    #    
    #    Therefore, my job is to design the BEST static priority function that leads to a large cap set when used for greedy selection.
    #    
    #    This confirms that the "dynamic" state approach (where I track which vectors have been added) is NOT possible if the harness sorts first.
    #    I must rely on a static heuristic that predicts which vectors are "good" to include in a cap set.
    
    #    However, looking at the current program, it returns a float based on the vector.
    #    So I just need to return a good static score.
    
    #    Let's refine the static heuristic.
    
    pass

# --- STATIC HEURISTIC DESIGN ---
# We want to prioritize vectors that are likely to be part of a large cap set.
# A cap set is a set with no 3-term AP.
# In {0,1,2}^n, a 3-term AP is (a, b, c) such that a+c = 2b (mod 3).
# This is equivalent to a, b, c being distinct and collinear in F_3^n.
#
# Key Insight:
# Vectors with more "1"s are often more "central" and conflict more.
# Vectors with only 0s and 2s (i.e., no 1s) form a subspace {0,2}^n.
# This subspace has size 2^n.
# Is {0,2}^n a cap set?
# Check: a, b, c in {0,2}^n.
# a_i, b_i, c_i in {0,2}.
# Condition: a_i + c_i = 2*b_i (mod 3).
# Possible values for a_i, c_i:
# If b_i = 0, 2*b_i = 0. a_i + c_i = 0 => (0,0) or (2,2) (since 2+2=4=1!=0). Wait.
# In mod 3:
# 0+0=0. 2+2=4=1.
# So if b_i=0, we need a_i+c_i=0. Only (0,0) works. So a_i=c_i=0.
# If b_i=2, 2*b_i=4=1. We need a_i+c_i=1.
# Pairs (0,1) no 1. (1,0) no 1. (2,2)->4=1. No, 2+2=1.
# So if b_i=2, a_i=2, c_i=2.
# So for any i, if b_i=0, then a_i=c_i=0. If b_i=2, then a_i=c_i=2.
# This implies a=c=b.
# So there are no *distinct* a, b, c in {0,2}^n forming an AP.
# Thus {0,2}^n is a cap set of size 2^n.
#
# Can we do better?
# Yes, for large n, the maximal cap set size is larger than 2^n.
# For n=3, 2^3=8. Max cap set is 9.
# For n=4, 2^4=16. Max cap set is 20.
#
# The vectors that are NOT in {0,2}^n contain at least one 1.
# Adding a vector with 1s might allow adding more vectors?
# Or does it force us to remove some {0,2} vectors?
#
# Heuristic Idea:
# 1. Prioritize vectors in {0,2}^n (no 1s). They are safe.
# 2. Among those, prioritize... ? All are equivalent in terms of AP-free-ness within the set.
#    But they conflict with vectors that have 1s.
# 3. Vectors with 1s:
#    We want vectors that are "far" from each other and from the {0,2} set?
#    Actually, we want vectors that don't form APs with each other or with the {0,2} set.
#    
#    Let's look at the structure of APs involving a vector v with 1s.
#    If v has a 1 at position i, then for any u, w such that u+w = 2v, we have an AP.
#    This is complex.
#    
#    Alternative Heuristic:
#    Randomization? No, must be deterministic.
#    
#    Let's try to prioritize vectors with an ODD number of 1s?
#    Or vectors with specific weight?
#    
#    Known result: The "balanced" heuristic (score 1.21) worked well.
#    It favored vectors with c0, c1, c2 close to n/3.
#    This means it deprioritizes pure {0,2} vectors (c1=0, c0+c2=n).
#    If n is large, c1=0 is far from n/3.
#    So the balanced heuristic actually *excludes* many {0,2} vectors from the top priority.
#    This suggests that a mix of {0,2} and {1}-containing vectors is better than just {0,2}.
#    
#    Let's refine the "balanced" heuristic.
#    Maybe we should weight the "1" count differently.
#    In F_3, 1 is the "middle" element.
#    0 and 2 are "opposites".
#    
#    Let's try a score based on the "energy" or "potential" for conflict.
#    A vector v conflicts with u if there exists w such that u, v, w is an AP.
#    This is hard to pre-compute.
#    
#    Let's go back to the balanced heuristic but tune it.
#    Score = - sum((count_i - n/3)^2)
#    
#    What if we add a penalty for having *any* 1s?
#    No, balanced already penalizes low 1s if n/3 is significant.
#    
#    What if we prioritize vectors that are "orthogonal" in some sense?
#    
#    Let's try a different structural change:
#    Instead of a single scalar score, can we use a vector score? No, interface is float.
#    
#    Let's try to incorporate the "position" of 1s.
#    Two vectors with the same counts but different positions of 1s might have different conflict potentials.
#    Vectors with 1s in "spread out" positions might be better than those with 1s clustered?
#    Or vice versa?
#    
#    Let's test a hypothesis:
#    Vectors with 1s in positions that are "far apart" (high Hamming distance between 1-positions) might be better.
#    Why? If 1s are close, they might form small local APs.
#    If 1s are spread, they are more independent.
#    
#    Let's compute the sum of pairwise distances between 1-positions.
#    Or simpler: the sum of the indices of 1s.
#    Or the variance of the indices of 1s.
#    
#    Let's try:
#    Primary: Balanced counts (as before).
#    Secondary: High variance of positions of 1s (spread out).
#    
#    Implementation:
#    1. Count c0, c1, c2.
#    2. Compute balanced_score = -sum((c_i - n/3)^2).
#    3. Compute positions of 1s: pos_1s = [i for i, v in enumerate(vector) if v == 1].
#    4. If len(pos_1s) > 1:
#       mean_pos = sum(pos_1s) / len(pos_1s)
#       variance = sum((p - mean_pos)^2 for p in pos_1s)
#    Else:
#       variance = 0
#    5. score = balanced_score + variance / (n^2)  # Normalize variance
#    
#    Let's think if this makes sense.
#    If 1s are spread out, variance is high. We want high score, so we add it.
#    If 1s are clustered, variance is low.
#    
#    Is this a structural change? Yes, it uses spatial information of the digits, not just counts.
#    
#    Let's also consider the interaction with 2s.
#    In an AP a, b, c, if b has a 1, then a and c must have 0s or 2s in a specific way.
#    Actually, if b_i = 1, then a_i + c_i = 2 (mod 3).
#    Possible (a_i, c_i): (0,2), (2,0).
#    So if b has a 1, a and c must have opposite 0/2 in that position.
#    This means a and c are "opposite" in the 0/2 sense at positions where b is 1.
#    
#    This suggests that vectors with many 1s constrain their neighbors to be "anti-correlated" in the 0/2 subspace.
#    This might make them harder to pack.
#    
#    So maybe we should penalize 1s?
#    But the balanced heuristic (which penalizes *low* 1s for large n) worked.
#    For n=8, n/3 = 2.66.
#    Pure {0,2} has 0 ones. Deviation = 2.66.
#    Vector with 3 ones. Deviation = 0.33.
#    So balanced prefers 3 ones.
#    
#    Let's stick with balanced as the primary, and add the spread of 1s as a secondary.
#    
#    Also, consider the spread of 2s?
#    By symmetry, 0 and 2 are similar.
#    Let's just use 1s spread.
    
def priority(vector, n):
    c0 = 0
    c1 = 0
    c2 = 0
    pos_1s = []
    
    for i, v in enumerate(vector):
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            pos_1s.append(i)
        else:
            c2 += 1
            
    target = n / 3.0
    dist_sq = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    base_score = -dist_sq
    
    # Secondary: Spread of 1s
    # We want 1s to be spread out to reduce local clustering conflicts.
    if len(pos_1s) > 1:
        mean_pos = sum(pos_1s) / len(pos_1s)
        variance = sum((p - mean_pos)**2 for p in pos_1s)
    else:
        variance = 0
        
    # Normalize variance: Max variance is when 1s are at 0 and n-1.
    # Variance of {0, n-1} is ((n-1)/2)^2.
    # So divide by (n/2)^2 to get a value in [0,1] roughly.
    spread_score = variance / (n * n) if n > 0 else 0
    
    # Combine
    # Base score range: ~ -n to 0.
    # Spread score range: 0 to 1.
    # We want spread to be a tie-breaker, so scale it down.
    final_score = base_score + spread_score * 0.1
    
    return final_score

# EVOLVE-BLOCK-END
