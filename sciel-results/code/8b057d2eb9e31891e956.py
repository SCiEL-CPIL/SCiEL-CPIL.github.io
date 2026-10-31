import time
import itertools
import math

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0  # 300s limit, keep margin

# Global cache for priorities
_PRIORITY_CACHE = {}
# Global flag to indicate if we finished precomputation
_PRECOMPUTED = False

def _forms_line(a, b, c, n):
    """
    Check if a, b, c form a line in F_3^n.
    A line is a set {a, b, c} such that a_i + b_i + c_i = 0 mod 3 for all i.
    """
    for i in range(n):
        if (a[i] + b[i] + c[i]) % 3 != 0:
            return False
    return True

def _is_cap_set_subset(vecs, n):
    """
    Check if a list of vectors is a cap set.
    vecs is a list of tuples.
    """
    m = len(vecs)
    # Optimized check: for each pair, check if their "midpoint" (in F_3) is in the set.
    # In F_3, the third point c on the line through a and b is c = -a - b (mod 3).
    # So we just need to check if (-a_i - b_i) mod 3 is in the set.
    
    # Convert to set of tuples for O(1) lookup
    vec_set = set(vecs)
    
    for i in range(m):
        a = vecs[i]
        for j in range(i + 1, m):
            b = vecs[j]
            # Calculate c = -a - b mod 3
            c = tuple((-a[k] - b[k]) % 3 for k in range(n))
            if c in vec_set:
                return False
    return True

def _precompute_priorities(n):
    """
    Simulate greedy growth to estimate "importance" of each vector.
    We prioritize vectors that are part of many potential cap sets or that
    allow the set to grow further.
    
    Strategy:
    1. Generate all vectors.
    2. Use a heuristic greedy builder to find a large cap set S_max.
    3. Vectors in S_max get high priority.
    4. Vectors not in S_max get lower priority, but we refine by checking
       how many vectors in S_max they conflict with.
       
    To make it more robust, we run multiple greedy builds with different initial seeds
    or randomizations (if time permits) and count how often each vector is included.
    """
    global _PRECOMPUTED
    global _PRIORITY_CACHE
    
    if _PRECOMPUTED:
        return

    # If n is large, full precomputation is impossible. 
    # For n <= 6, total vectors is 3^n <= 729. This is small.
    # For n > 6, we fall back to a simpler heuristic or partial simulation.
    
    if n > 8:
        # Fallback for large n: use a simple geometric heuristic
        # Vectors with more '1's and '2's might be better? 
        # Actually, in F_3^n, the "distance" or structure matters.
        # Let's just use a deterministic hash-based priority to ensure correctness,
        # but biased towards "balanced" vectors as a baseline.
        for vec in itertools.product([0, 1, 2], repeat=n):
            # Simple heuristic: prefer vectors with count of 1s close to n/3
            c1 = vec.count(1)
            c2 = vec.count(2)
            c0 = vec.count(0)
            # Deviation from ideal balance
            dev = abs(c0 - n/3.0) + abs(c1 - n/3.0) + abs(c2 - n/3.0)
            # Hash for tie breaking
            h = 0
            for v in vec:
                h = (h * 31 + v) % (10**9 + 7)
            _PRIORITY_CACHE[vec] = -dev * 1000.0 + h / (10**9 + 7)
        _PRECOMPUTED = True
        return

    # For n <= 8, we can do actual simulation
    all_vecs = list(itertools.product([0, 1, 2], repeat=n))
    num_vecs = len(all_vecs)
    
    # Count how many times each vector is included in a "good" greedy cap set
    # We run K greedy builds. 
    # To save time, we limit K based on remaining time.
    
    inclusion_count = {vec: 0 for vec in all_vecs}
    
    # We will run multiple greedy passes.
    # Pass 1: Standard greedy with a fixed "base" priority (e.g., hash order)
    # Pass 2+: If time permits, try different base orders.
    
    # Base priority for the simulation: 
    # We want to find a large cap set. 
    # A common heuristic is to prefer vectors that are "far" from existing ones.
    # But for precomputation, we can just run a few random-ish greedy builds.
    
    # Since we need deterministic output for the same input n, we use a fixed seed for "randomness"
    # or just multiple fixed permutations.
    
    # Let's use 10 different fixed permutations of the vector list to simulate different "views".
    # Permutation i is defined by sorting by (hash_i(vec), vec)
    
    # Actually, a better structural approach:
    # The "value" of a vector is how many maximal cap sets it belongs to.
    # Approximation: Run greedy algorithm with different tie-breaking rules.
    
    # Rule 1: Lexicographic
    # Rule 2: Sum of elements
    # Rule 3: Count of 1s
    # Rule 4: Count of 2s
    # Rule 5: Hash
    
    # We will run 5 such greedy builds.
    
    keys = [
        lambda v: v,
        lambda v: sum(v),
        lambda v: v.count(1),
        lambda v: v.count(2),
        lambda v: hash(v) # Python hash is randomized per run, so we need a stable hash
    ]
    
    # Replace hash with stable polynomial hash
    def stable_hash(v):
        h = 0
        for x in v:
            h = (h * 31 + x) % (10**9 + 7)
        return h
        
    keys[4] = stable_hash
    
    # Run greedy builds
    for k in range(5):
        # Check time
        if time.time() - _START_TIME > _DEADLINE - 10:
            break
            
        # Sort vectors by key k
        # We want to add vectors in an order that maximizes cap set size.
        # Greedy: Start with empty set. Iterate through sorted list.
        # If adding vec keeps it a cap set, add it.
        
        # To make it non-trivial, we can try to start with a specific vector?
        # No, standard greedy is fine for this heuristic.
        
        sorted_vecs = sorted(all_vecs, key=keys[k])
        
        current_set = []
        current_set_tuples = set()
        
        for vec in sorted_vecs:
            # Check if adding vec maintains cap set property
            # Optimized check: 
            # For each existing vector u in current_set,
            # let w = -u - vec (mod 3).
            # If w is in current_set, then {u, vec, w} is a line.
            
            is_valid = True
            for u in current_set:
                w = tuple((-u[i] - vec[i]) % 3 for i in range(n))
                if w in current_set_tuples:
                    is_valid = False
                    break
            
            if is_valid:
                current_set.append(vec)
                current_set_tuples.add(vec)
                inclusion_count[vec] += 1
        
        # We could also count "near misses" or conflicts, but inclusion is a strong signal.
        
    # Normalize counts to [0, 1]
    max_count = max(inclusion_count.values()) if inclusion_count else 1
    if max_count == 0:
        max_count = 1
        
    for vec in all_vecs:
        score = inclusion_count[vec] / max_count
        # Add small hash tie-breaker to ensure distinct priorities
        h = stable_hash(vec)
        _PRIORITY_CACHE[vec] = score + (h / (10**9 + 7)) * 0.001
        
    _PRECOMPUTED = True

def priority(vector, n):
    """
    Returns a priority score for the vector. Higher is better.
    """
    # Deadline check
    if time.time() - _START_TIME > _DEADLINE - 5:
        # Return a safe default if we are running out of time
        # This shouldn't happen if precomputation is fast enough for n<=8
        return 0.0

    # Convert to tuple for hashing
    vec_tuple = tuple(vector)
    
    if vec_tuple in _PRIORITY_CACHE:
        return _PRIORITY_CACHE[vec_tuple]
    
    # If not precomputed (e.g., first call for this n), trigger precomputation
    # Note: The evaluator might call priority with different n values?
    # The contract says "vectors of {0,1,2}^n". Usually n is fixed per evaluation run.
    # But to be safe, we precompute for the specific n requested.
    
    # Check if we have precomputed for this n.
    # We use a global flag, but we should store per-n if n changes.
    # Given the 300s limit, n is likely small and fixed.
    # Let's assume n is fixed for the duration of the process.
    
    if not _PRECOMPUTED:
        _precompute_priorities(n)
    
    # Retrieve or compute on the fly if precomputation failed/was skipped
    if vec_tuple in _PRIORITY_CACHE:
        return _PRIORITY_CACHE[vec_tuple]
    
    # Fallback: Simple heuristic
    c0 = vec_tuple.count(0)
    c1 = vec_tuple.count(1)
    c2 = vec_tuple.count(2)
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    h = 0
    for v in vec_tuple:
        h = (h * 31 + v) % (10**9 + 7)
    return -dev * 1000.0 + h / (10**9 + 7)
