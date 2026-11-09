"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def priority(vector, n):
    # This function is kept for interface compatibility but the core logic
    # is implemented in the helper below to allow state-aware selection.
    # In a true dynamic greedy, 'priority' is not static.
    return 0.0

def solve_cap_set(n):
    """
    Solves the cap set problem for dimension n using a dynamic greedy strategy.
    Returns the size of the largest cap set found.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    # Generate all vectors in {0, 1, 2}^n
    # For efficiency, we work with integers 0, 1, 2.
    # Total vectors: 3^n
    # We use a list of tuples for the vectors.
    all_vectors = list(itertools.product([0, 1, 2], repeat=n))
    num_vectors = len(all_vectors)
    
    # Map vector to index for O(1) lookup
    vec_to_idx = {v: i for i, v in enumerate(all_vectors)}
    
    # State
    selected = set()
    remaining = set(range(num_vectors))
    
    # To check if adding a vector v creates an AP with existing set S:
    # An AP is (a, b, c) such that a + c = 2b.
    # If we add v, it creates an AP if:
    # 1. There exist a, c in S such that a + c = 2v (v is the middle term)
    # 2. There exists a in S such that v + a = 2c for some c in S (v is an end term)
    #    Note: If a+c=2v, then v+a=2c implies c=v, which is not in S yet.
    #    So we just need to check if v is the middle of an AP with two elements in S,
    #    OR if v is an end of an AP with one element in S and the middle also in S.
    
    # Actually, a simpler check:
    # Adding v to S creates a 3-term AP iff there exists u in S such that:
    # - (2v - u) is in S (v is middle)
    # - (v + u)/2 is in S and (v+u) is even (v is end, u is other end)
    
    # Let's precompute nothing, just check on the fly.
    # To optimize, we can maintain a count of how many APs each remaining vector is "involved" in
    # with the current selected set. But since the set grows, this is expensive.
    
    # Alternative efficient greedy:
    # 1. Start with empty set.
    # 2. For each step, find the vector v in remaining that minimizes the number of APs it forms with current S.
    # 3. If min count is 0, add v.
    # 4. If no v has count 0, we might be stuck or need to pick the one with least "future damage".
    #    However, standard greedy for cap sets usually just picks any valid one.
    #    If we pick randomly among valid ones, we might get stuck.
    #    Dynamic heuristic: Pick the one that leaves the most options open?
    #    Cost(v) = number of pairs (a, b) in S such that v, a, b form an AP.
    #    If Cost(v) > 0, v cannot be added.
    #    So we only consider v where Cost(v) == 0.
    #    Among those, which one to pick?
    #    Heuristic: Pick the one that "blocks" the fewest other remaining vectors.
    #    i.e., Minimize sum_{u in remaining, u != v} [Cost(u) increases if v is added]
    
    # This is O(N^2) per step if done naively.
    # N = 3^n. For n=8, N=6561. N^2 = 43M. Steps ~ N. Total ~ 280B ops. Too slow.
    
    # We need a faster heuristic or approximation.
    # Known result: Random greedy works well.
    # Let's try:
    # 1. Shuffle remaining vectors.
    # 2. Iterate. If adding v keeps it a cap set, add it.
    # 3. Repeat with different shuffles if time permits?
    # But the task requires a `priority` function? No, the task says "vectors are added greedily in descending priority".
    # The interface is `priority(vector, n) -> float`.
    # The evaluator calls `priority` to sort the vectors?
    # "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
    # This implies the evaluator sorts ALL vectors by `priority(v, n)` once, and then iterates through them in that order, adding if valid.
    # If this is the case, `priority` MUST be static (cannot depend on the current set state during the greedy loop, because the order is fixed before the loop starts).
    
    # RE-READING CONTRACT:
    # "Interface: priority(vector: tuple, n: int) -> float; vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
    # This strongly suggests a STATIC ordering. The evaluator generates the list, sorts by priority, and runs the greedy.
    # If the order is static, my "dynamic" idea is impossible via the `priority` interface alone unless I can modify the global state or the evaluator calls `priority` dynamically.
    # Standard FunSearch/SCiEL contracts usually imply the function defines the heuristic for the static sort.
    
    # HOWEVER, if I can't do dynamic, how do I improve on "balanced digits"?
    # Inspiration 1 says "Uniform priority...".
    # Inspiration 2 says "dynamic, state-aware...". But if the interface is static, how?
    # Maybe the `priority` function can use global variables to track state?
    # If the evaluator calls `priority` for each vector to determine the sort order, it calls it N times.
    # If the evaluator calls `priority` *during* the greedy loop to pick the next element, it would be dynamic.
    # "added greedily in descending priority" usually means:
    # 1. Compute priority for all.
    # 2. Sort.
    # 3. Iterate.
    # If it's step 1-3, it's static.
    
    # Let's assume the standard static interpretation first.
    # What static heuristic is better than "balanced digits"?
    # "Balanced digits" prefers vectors with equal counts of 0, 1, 2.
    # This avoids early APs.
    # Maybe we should prefer vectors with more 1s? Or fewer?
    # Actually, the known best static heuristic for cap sets is often related to the "linear" structure or specific algebraic properties.
    # But wait, if I can use global state, I can simulate a dynamic selection IF the evaluator allows it.
    # If the evaluator sorts once, global state won't help during the sort.
    
    # Let's look at the "Inspiration 2" again: "uses time; defines priority".
    # Maybe it uses `time.time()` to vary the priority? That would be random.
    # Random priority (shuffle) is a strong baseline.
    # "Balanced digits" got 1.223.
    # Random might get similar or worse.
    
    # Let's try a different static heuristic:
    # Prefer vectors that are "linearly independent" from previous high-priority ones?
    # No, static can't see previous.
    
    # How about: Priority = -sum(digit^2) ? (Prefers 1s)
    # Or Priority = -sum((digit-1)^2) ? (Prefers 1s, same as above)
    # Balanced digits minimizes variance.
    
    # Let's try: Priority based on the number of 1s.
    # Vectors with more 1s are "central".
    # In {0,1,2}^n, the vector (1,1,...,1) is the center.
    # APs involving the center are common.
    # Maybe we should AVOID the center?
    # Priority = sum((digit-1)^2). Higher priority = further from center?
    # Or lower priority = further from center?
    # If we add center first, it creates many APs.
    # So we should add center LAST.
    # So priority = -sum((digit-1)^2). (Center has lowest priority).
    # "Balanced digits" also puts center low priority (if balanced means equal counts, center is most balanced).
    # Wait, if balanced means counts are equal, then (1,1,1) is most balanced.
    # If we add most balanced first, we add center first. That's bad.
    # So maybe we should add LEAST balanced first?
    # i.e., Priority = sum((digit-1)^2). (Center has 0, corners have high).
    # Let's test this hypothesis: Adding "extreme" vectors first might be better.
    
    # Let's implement:
    # Priority = sum((d-1)^2 for d in vector)
    # This prioritizes vectors with many 0s or 2s (corners) over those with 1s.
    
    # Another idea:
    # Use a hash of the vector to break ties randomly?
    # No, deterministic is better for reproducibility, but random can escape local minima.
    # But we only have one shot (static sort).
    
    # Let's try the "Anti-Center" heuristic:
    # Prioritize vectors with high variance in digits.
    
    # Let's also consider:
    # The "balanced digits" strategy measured 1.223.
    # The seed (uniform) is 1.0.
    # So balanced digits is good.
    # What if we invert it?
    
    # Let's try: Priority = sum(digit * (n - index)) ? (Lexicographic variant)
    # No, lexicographic is basically uniform.
    
    # Let's try: Priority = sum( (digit-1)^2 * (index+1) ) ?
    
    # Actually, let's look at the structure of cap sets.
    # A cap set in {0,1,2}^n is equivalent to a set of vectors with no 3-term AP.
    # In F_3^n, this is a cap set.
    # The largest cap sets are known to be related to "linear" constructions for small n, but greedy is suboptimal.
    
    # If I can't do dynamic, I must find a better static order.
    # Let's try "Anti-Center" (High Variance First).
    
    def priority(vector, n):
        # Higher value = added earlier.
        # We want to add vectors that are "less likely" to form APs with each other early on?
        # Or vectors that "block" the most APs?
        
        # Hypothesis: Vectors with many 1s (center) form many APs.
        # So we should add them LAST.
        # So priority should be LOW for center.
        # Priority = sum((d-1)^2 for d in vector)
        # Center (1,1,1) -> 0.
        # Corner (0,0,0) -> 3.
        # So Corner has higher priority.
        
        return sum((d - 1) ** 2 for d in vector)

# EVOLVE-BLOCK-END
