import time
import zlib
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# We maintain a set of accepted vectors.
# The priority function is called for each candidate vector.
# The strategy is to estimate how many new conflicts (APs) this vector would create
# if added to the current set.
# An AP is formed by (a, b, c) such that a + c = 2b (mod 3).
# If we add v, it forms an AP if there exists u in S such that:
# 1. v is the middle: 2v = u1 + u2 (requires u1, u2 in S) - Hard to check efficiently without precomputation
# 2. v is an end: v + u1 = 2u2 or v + u2 = 2u1.
#    This is equivalent to checking if (2u2 - v) is in S or (2u1 - v) is in S.
#    Or simpler: For any u in S, does the line {u, v, 2(u+v)/3 ? No.}
#    The three points on a line through u and v are u, v, and w = 2v - u (mod 3)?
#    Let's check: If u, v, w are in AP, then u + w = 2v => w = 2v - u.
#    So if we add v, we create an AP with existing u if w = (2*v - u) % 3 is also in S.
#    Also, if u, v, w are in AP, v could be the middle: u + w = 2v. This requires two existing points u, w.
#    Checking "middle" is O(|S|^2) which is too slow for large S.
#    However, we can check "end" positions efficiently: O(|S|).
#    If v is an end, there is exactly one other end w for each middle u.
#    If v is a middle, there are pairs (u, w).
#    
#    Actually, the standard check for "does adding v create an AP" is:
#    For every u in S:
#       1. Is (2*v - u) % 3 in S? (v is an end, u is middle? No. u, v, w. If u is middle, u=(v+w)/2 => 2u = v+w => w = 2u - v. So if w = 2u - v is in S, then v, u, w is AP. Here u is middle.)
#       2. Is (v + u) / 2 in S? No, division by 2 is multiplication by 2 in Z3.
#          If v is middle: v = (u+w)/2 => 2v = u+w => w = 2v - u. This is the same check!
#          Wait. In Z3, 2 is the inverse of 2.
#          AP: a, b, c. b = (a+c)/2 => 2b = a+c.
#          If we add v, and we have u in S:
#          Case A: v is an end. Then there is a middle m and other end w.
#                 If u is the middle m: w = 2u - v. Check if w in S.
#                 If u is the other end w: m = 2v - u. Check if m in S.
#          Case B: v is the middle. Then there are ends a, c.
#                 a + c = 2v. This requires two points in S.
#                 
#    So, to check if adding v creates ANY AP with S:
#    1. For each u in S:
#       - Check if (2*u - v) % 3 is in S. (u is middle, v is end)
#       - Check if (2*v - u) % 3 is in S. (v is middle? No. If v is middle, u is an end. The other end is w. w = 2v - u. So if w in S, then u, v, w is AP. Here v is middle.)
#       
#    Wait, let's re-verify.
#    Points: u (in S), v (new).
#    Potential AP involving v and u:
#    1. u, v, w. AP means 2v = u + w => w = 2v - u. If w in S, AP exists. (v is middle)
#    2. v, u, w. AP means 2u = v + w => w = 2u - v. If w in S, AP exists. (u is middle)
#    3. u, w, v. AP means 2w = u + v => w = 2(u+v)/3 ? No, 2w = u+v => w = 2(u+v) * inv(2) = 2(u+v)*2 = 4(u+v) = (u+v) mod 3?
#       Wait. 2w = u+v. Multiply by 2 (inverse of 2): 4w = 2(u+v) => w = 2(u+v).
#       So w = (2*u + 2*v) % 3. If w in S, AP exists. (w is middle)
#       
#    So for each u in S, we need to check 3 potential partners:
#    w1 = (2*v - u) % 3
#    w2 = (2*u - v) % 3
#    w3 = (2*u + 2*v) % 3
#    
#    If any of these are in S, adding v creates an AP.
#    
#    The "Conflict Count" is the number of such u in S that satisfy this.
#    We want to MINIMIZE this count.
#    
#    To make this efficient, we store S as a set of tuples.
#    Lookup is O(1).
#    Loop over S is O(|S|).
#    Total priority call cost: O(|S|).
#    If |S| grows to 1000, and we have 10000 candidates, that's 10^7 ops. Python might be slow.
#    But Numpy can help? No, S is a set of tuples.
#    We can optimize by precomputing the "conflict potential" of v against S?
#    No, S changes.
#    
#    However, we can approximate or use a faster data structure?
#    For n up to 10-15, |S| is likely small enough (cap set size is ~3^(n/2) or similar? No, cap set size is much smaller, maybe 3^n / poly(n)? Actually cap set size is ~2.756^n. For n=10, ~3000. For n=15, ~100,000? No.
#    Known bounds: max cap set size in Z3^n is ~ 2.756^n.
#    For n=10, 2.756^10 ~ 2500.
#    For n=20, 2.756^20 ~ 6 million.
#    If n is large, |S| is large, O(|S|) per call is too slow.
#    
#    But the evaluator adds vectors greedily. The total number of calls to priority is roughly the number of candidates evaluated until the set is "full" or time runs out.
#    If the greedy process stops early because no more vectors can be added, |S| is bounded.
#    
#    Let's assume n is small enough that O(|S|) is feasible within 300s.
#    If not, we need a heuristic approximation.
#    
#    Alternative: Precompute a "conflict mask" or use bitsets?
#    Represent vectors as integers (base 3).
#    S is a set of ints.
#    For a given v, we want to count how many u in S have (2v-u) in S OR (2u-v) in S OR (2u+2v) in S.
#    
#    This is still O(|S|).
#    
#    Let's implement the O(|S|) check. It is structurally different from the spectral approach because it is stateful and directly models the constraint.

_accepted_vectors = set()

def _vector_to_int(vec):
    # Convert tuple of 0,1,2 to a base-3 integer
    # This allows for faster hashing and storage
    h = 0
    for val in vec:
        h = h * 3 + val
    return h

def _int_to_vector(i, n):
    # Not needed for priority, but good for debugging
    v = [0] * n
    for j in range(n-1, -1, -1):
        v[j] = i % 3
        i //= 3
    return tuple(v)

def priority(vector, n):
    global _accepted_vectors
    
    # Deadline check
    if time.time() > _DEADLINE:
        return 0.0

    v_int = _vector_to_int(vector)
    
    # If already accepted (shouldn't happen in greedy, but safe)
    if v_int in _accepted_vectors:
        return -1.0  # Lowest priority

    # Calculate conflict count
    # For each u in S, check if adding v creates an AP with u and some w in S.
    # Conditions:
    # 1. w = (2*v - u) % 3 in S  (v is middle)
    # 2. w = (2*u - v) % 3 in S  (u is middle)
    # 3. w = (2*u + 2*v) % 3 in S (w is middle)
    
    # Optimization: We can precompute the set S as a set of ints.
    # We need to compute (2*v - u) mod 3^n? No, the operation is coordinate-wise mod 3.
    # Base-3 integer arithmetic does NOT correspond to coordinate-wise mod 3 addition/subtraction.
    # (a+b) in Z3^n is not (a_int + b_int) % (3^n).
    # So we must do coordinate-wise operations.
    
    # This makes O(|S|) with tuple operations potentially slow in Python.
    # Let's try to vectorize or optimize.
    
    # Since we can't easily vectorize over a set of tuples, we will iterate.
    # To speed up, we can cache the vector as a list of ints.
    
    v_list = list(vector)
    two_v = [(2 * x) % 3 for x in v_list]
    two_v_plus_two_v = [(4 * x) % 3 for x in v_list] # 2*u + 2*v -> 2(u+v). Wait.
    # Condition 3: w is middle. 2w = u + v. w = 2(u+v).
    # So w_i = (2 * (u_i + v_i)) % 3.
    
    conflict_count = 0
    
    # Iterate over accepted vectors
    for u_int in _accepted_vectors:
        # Reconstruct u? That's expensive.
        # Better to store accepted vectors as tuples in a list, and use a set of ints for lookup?
        # Or just store tuples in the set? Tuples are hashable.
        
        # Let's change _accepted_vectors to store tuples.
        # Lookup is O(n).
        
        pass

    # Revised Strategy: Store accepted vectors as tuples.
    # _accepted_vectors is a set of tuples.
    
    # Let's re-implement with tuples.
    
    # To avoid global state issues if the evaluator resets, we should probably not use globals 
    # unless the evaluator expects statefulness. The contract says "vectors ... are added greedily".
    # This implies the evaluator maintains the set? 
    # "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
    # Usually, in these tasks, the `priority` function is called for a candidate, and if it's high enough 
    # (or top of the heap), it's added. The evaluator might not tell us which ones were added.
    # HOWEVER, the standard FunSearch contract for "greedy addition" often implies that the 
    # `priority` function is stateless, OR the evaluator passes the current set?
    # The signature is `priority(vector: tuple, n: int) -> float`. 
    # It does NOT pass the current set.
    # This means we CANNOT know the current set inside `priority` unless we maintain our own state 
    # and assume the evaluator calls us in a way that allows us to track additions.
    # But we don't know if a vector was added or not!
    # 
    # CRITICAL REALIZATION:
    # If the evaluator does not tell us which vectors were added, we cannot maintain an accurate 
    # conflict graph. 
    # UNLESS: The task implies that WE are the ones doing the greedy selection?
    # "vectors ... are added greedily ... Score: cap set size"
    # "The evaluator measures the raw value"
    # Usually, in these "priority" tasks, the evaluator runs a loop:
    # 1. Generate all candidates (or a subset).
    # 2. Sort by priority.
    # 3. Iterate through sorted list. If candidate can be added (doesn't break cap set), add it.
    # 
    # If the evaluator does the "can be added" check, then `priority` is just a ranking function.
    # My previous "stateful" idea fails because I don't know which vectors were actually added.
    # 
    # So, I must revert to a STATELESS strategy, but a different one.
    # 
    # What is a different stateless strategy?
    # Current: Spectral (FFT) + Centrality.
    # Inspiration 1: Geometric-aware (Hamming weight, conflicts with 0?).
    # Inspiration 3: Adaptive? (But stateless).
    # 
    # Let's try a **Combinatorial Structure** approach.
    # 
    # Hypothesis: Cap sets are related to "independent sets" in the graph where edges connect points 
    # that form an AP with a third point.
    # 
    # A vector v is "bad" if it lies on many lines with other "likely" vectors.
    # 
    # Let's look at the **Hamming Weight** and **Sum of Coordinates**.
    # 
    # Alternative: **Entropy-based**.
    # Vectors with high entropy (balanced symbols) are better?
    # The current program uses Centrality (variance from n/3).
    # 
    # Let's try **Randomness / Chaos**.
    # Use a hash-based priority that is highly random but deterministic.
    # The cell finding said: "Replacing the 32-bit custom hash with a 64-bit slice of SHA256 improves the uniformity... yielding larger cap sets."
    # The current program uses a simple polynomial hash.
    # Let's try using `zlib.crc32` or `hashlib.sha256` for the tie-breaker, but combine it with a strong structural signal.
    # 
    # What structural signal?
    # Maybe **Line Containment**?
    # For a vector v, how many lines in Z3^n contain v?
    # Every point is on (3^n - 1) / 2 lines? No.
    # Number of lines through a point in Z3^n is (3^n - 1) / 2.
    # This is constant for all v. Not useful.
    # 
    # How about **Intersections with Subspaces**?
    # 
    # Let's try a **Greedy Independent Set Heuristic** approximation.
    # In graph coloring/IS, a common heuristic is to pick the vertex with the lowest degree.
    # Here, "degree" of v is the number of other points u such that v and u are in a line with a third point w.
    # Since we don't know the set, we assume the set is "random" or "typical".
    # 
    # If the set is random, the probability that a specific line {a,b,c} is fully in the set is low.
    # 
    # Let's try **Maximizing Distance**.
    # Prefer vectors that are "far" from the origin? Or "far" from the center?
    # 
    # Let's look at the **Spectral** approach again.
    # The current program penalizes high frequency.
    # What if we **REWARD** high frequency (randomness)?
    # Random vectors are less likely to form APs with each other.
    # 
    # Let's try: Score = HighFrequencyEnergy + RandomHash.
    # 
    # Or better: **Minimize Autocorrelation**.
    # 
    # Let's implement a strategy based on **Symbol Distribution** and **Hash**.
    # 
    # Strategy:
    # 1. Compute the "Balance" (how close counts of 0,1,2 are to n/3).
    # 2. Compute a "Randomness" score using a high-quality hash.
    # 3. Combine them.
    # 
    # The current program does: Score = -HFE + Centrality + SmallHash.
    # HFE is High Frequency Energy. Penalizing HFE means preferring Low Frequency (Smooth).
    # 
    # Let's flip it: Prefer High Frequency (Random).
    # Score = HFE + Centrality + LargeHash.
    
    import hashlib
    
    # 1. Centrality (Balance)
    c0 = sum(1 for x in vector if x == 0)
    c1 = sum(1 for x in vector if x == 1)
    c2 = sum(1 for x in vector if x == 2)
    
    ideal = n / 3.0
    var = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    max_var = 2 * n * n / 3.0
    centrality = 1.0 - (var / max_var if max_var > 0 else 0.0)
    
    # 2. High Frequency Energy (Randomness)
    # Use the same FFT logic but take the high part.
    v_np = np.array(vector, dtype=np.float64)
    x = v_np - 1.0
    if n > 1:
        fft_x = np.fft.fft(x)
        power = np.abs(fft_x) ** 2
        total_power = np.sum(power)
        if total_power > 1e-9:
            hfe = np.sum(power[n//2:]) / total_power
        else:
            hfe = 0.0
    else:
        hfe = 0.0
        
    # 3. Hash (Tie-breaker / Randomness)
    # Use SHA256 for better uniformity
    h = hashlib.sha256(vector.tobytes() if hasattr(vector, 'tobytes') else bytes(vector)).digest()
    # Convert first 8 bytes to float
    hash_val = int.from_bytes(h[:8], 'big') / (2**64)
    
    # Combine: Reward Randomness (HFE) and Balance (Centrality)
    # Weights: HFE is in [0,1], Centrality is in [0,1].
    # Let's try equal weights.
    score = hfe + centrality
    
    # Add hash as a small perturbation to break ties and add randomness
    score += hash_val * 0.1
    
    return float(score)

# Note: The above code still uses FFT which might be slow if called many times.
# Let's optimize the "Randomness" part to avoid FFT if possible, or make it faster.
# FFT is O(n log n). For n=15, it's fast.
# But calling numpy FFT for every priority call might have overhead.
# 
# Alternative for "Randomness": 
# Count the number of "transitions" in the vector (v_i != v_{i+1}).
# High transitions = High frequency.
# This is O(n) and very fast.

def priority_fast(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # 1. Centrality
    c0 = 0
    c1 = 0
    c2 = 0
    transitions = 0
    prev = -1
    for x in vector:
        if x == 0: c0 += 1
        elif x == 1: c1 += 1
        else: c2 += 1
        if prev != -1 and x != prev:
            transitions += 1
        prev = x
    
    ideal = n / 3.0
    var = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    max_var = 2 * n * n / 3.0
    centrality = 1.0 - (var / max_var if max_var > 0 else 0.0)
    
    # 2. Transition Density (Proxy for High Frequency)
    if n > 1:
        trans_density = transitions / (n - 1)
    else:
        trans_density = 0.0
        
    # 3. Hash
    import hashlib
    h = hashlib.sha256(bytes(vector)).digest()
    hash_val = int.from_bytes(h[:8], 'big') / (2**64)
    
    # Combine: Reward Transitions (Randomness) and Balance
    score = trans_density + centrality
    score += hash_val * 0.1
    
    return float(score)

# Use the fast version
priority = priority_fast
