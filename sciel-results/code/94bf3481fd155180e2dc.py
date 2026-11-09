import time
import sys
import hashlib
import struct
import itertools

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

def _is_cap_set(vecs, n):
    """
    Check if a set of vectors in {0,1,2}^n is a cap set.
    A cap set contains no three distinct elements a, b, c such that a+b+c = 0 mod 3.
    Equivalent to: no a, b, c in the set such that a, b, c are distinct and a+b+c = (0,...,0) mod 3.
    """
    # For efficiency, we can check pairs.
    # If we add a new vector v to a set S, we need to ensure no a, b in S such that a+b+v = 0 mod 3.
    # i.e., a+b = -v mod 3.
    # Since the set is built greedily, we can maintain the set and check against existing elements.
    pass

def priority(vector, n):
    """
    This function is called for each vector to determine its priority.
    The evaluator sorts vectors in descending order of priority and adds them greedily.
    We cannot access the current set state directly in this function signature,
    so we must return a static priority or use a stateful approach if allowed.
    
    However, the contract says: "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set".
    This implies the evaluator does:
    1. Generate all vectors.
    2. Call priority(v, n) for each v.
    3. Sort vectors by priority descending.
    4. Iterate through sorted vectors, adding to set if cap set property holds.
    
    Since we don't have access to the current set during the sorting phase, a static priority is the only option
    unless the evaluator calls priority dynamically (which is unlikely given the signature).
    
    BUT, looking at the current program, it returns a hash.
    To improve, we need a better heuristic for which vectors to prefer.
    
    Known good cap sets often have structure.
    For small n, we can precompute or use a better heuristic.
    
    Let's try a different strategy:
    The problem is that a simple hash might pick a bad order.
    A better static order might be based on the "weight" or some structural property.
    
    However, there's a powerful trick: if we can't control the order dynamically, we can't easily optimize for the greedy constraint.
    
    Wait, the task says: "Cap set task evaluated in Generative Modeling for Mathematical Discovery."
    And "Interface: priority(vector: tuple, n: int) -> float".
    
    If the evaluator is fixed and just sorts by this function, then we are stuck with a static ordering.
    Is it possible that the evaluator allows us to maintain state?
    No, the signature is simple.
    
    Let's look at the "Inspiration" which returned 0.0 (lexicographic).
    The current program uses a hash.
    
    Is there a way to make the priority depend on the vector in a way that favors larger cap sets?
    
    Actually, for a fixed n, the maximum cap set size is known (or bounded).
    For n=4, max is 9.
    For n=5, max is 16?
    
    If the evaluator tests multiple n, we need a general strategy.
    
    One idea: use a priority that favors vectors with a certain property, e.g., balanced coordinates (more 1s) or something else.
    
    However, a more effective approach for cap sets in greedy construction is often to use a random permutation, but a *specific* random permutation that is known to work well, or a deterministic one that avoids collisions.
    
    Let's try a different hash function that might distribute the vectors more evenly or favor specific structures.
    
    Alternatively, we can use a priority based on the value of the vector interpreted as a base-3 number, but permuted.
    
    Let's try to use a more complex hash that mixes the bits better.
    
    Also, note that the current program has a bug:
    `return -h`
    `return h`
    The second return is unreachable.
    
    Let's fix that and try a better hash.
    
    Another idea: Since we can't change the greedy algorithm, we can only change the order.
    The best order is one that avoids "conflicts" early.
    
    Let's try a priority that is based on the sum of the coordinates.
    Vectors with sum close to n/2 might be better?
    
    Let's experiment with a few static heuristics.
    
    Heuristic 1: Hash (current)
    Heuristic 2: Sum of coordinates
    Heuristic 3: Product of non-zero coordinates
    Heuristic 4: Lexicographic (baseline)
    
    Since we can't test, we have to guess.
    
    The hash approach in the current program is decent but might have collisions or bad ordering.
    
    Let's use a better hash:
    Use Python's built-in hash? No, it's salted per run.
    Use a cryptographic hash?
    
    Let's use a simple, strong hash like SHA-256 on the string representation of the vector.
    
    Also, we should consider that for small n, the number of vectors is small (3^n).
    For n=10, 3^10 = 59049.
    
    Let's try to combine the hash with a secondary key that might be useful.
    
    Actually, there is a known result that random ordering works well for cap sets.
    So a good hash is likely sufficient.
    
    Let's improve the hash to be more uniform.
    """
    # Convert vector to a string or bytes for hashing
    # vector is a tuple of ints
    s = ''.join(map(str, vector))
    h = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    return h

# Let's refine this.
# The current program's hash:
# h = 0
# for i, val in enumerate(vector):
#     h = (h * 31 + val * (i + 1)) % 1000000007
# return h

# This is a polynomial hash.
# Let's try a different one.

# We can also try to prioritize vectors that are "far" from each other in some sense.
# But we don't know the set.

# Let's stick to a good random-like priority.
# To ensure reproducibility and avoid Python's hash salting, we use a deterministic hash.

def priority(vector, n):
    # Use a strong deterministic hash
    # Convert to bytes
    b = struct.pack(f'{len(vector)}B', *vector)
    h = int(hashlib.md5(b).hexdigest(), 16)
    return h
