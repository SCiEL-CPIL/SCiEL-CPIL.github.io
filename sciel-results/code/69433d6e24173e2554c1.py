"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

# Global state for the greedy cap set construction
_greedy_state = {
    'active': False,
    'n': 0,
    'cap_set': set(),
    'sums': set(),
    'deadline': 0.0,
    'best_size': 0,
    'current_size': 0,
}

def _check_cap_set_addition(vector, n, cap_set, sums):
    """
    Check if adding vector to cap_set keeps it a cap set.
    A cap set is a set with no three distinct elements a, b, c such that a+b+c = 0 mod 3.
    Equivalently, no a, b in cap_set such that (2a - b) mod 3 is in cap_set or is the new vector.
    More efficiently: for each a in cap_set, check if (2*vector - a) mod 3 is in cap_set.
    Also check if vector itself creates a triple with two existing elements.
    
    Actually, the standard check: a set is a cap set iff for all a, b in set (a != b),
    (a + b + c) mod 3 != 0 for all c in set.
    
    When adding new vector v to existing cap set S:
    1. For each a in S: check if (2*v - a) mod 3 is in S. If yes, then a, v, (2v-a) form a triple.
    2. For each a in S: check if (a + v + c) mod 3 = 0 for some c in S, i.e., c = (2*(a+v) - ... )
       Actually: a + v + c = 0 mod 3 => c = -(a+v) mod 3 = (2*a + 2*v) mod 3.
       So for each a in S, check if (2*a + 2*v) mod 3 is in S.
    
    Wait, let me be more careful. A triple is {a, b, c} distinct with a+b+c = 0 mod 3.
    When adding v to S, we need to check:
    - For any a, b in S with a != b: if a + b + v = 0 mod 3, then {a, b, v} is a triple.
      This means v = -(a+b) mod 3 = (2a + 2b) mod 3.
    - For any a in S: if a + a + v = 0 mod 3, that's 2a + v = 0 mod 3, so v = -2a mod 3 = a mod 3.
      But a + a + v has only two distinct elements if a = v, but v is new so v != a.
      Actually a, a, v are not three distinct elements. So we only care about distinct triples.
    
    So the check is: for all pairs (a, b) in S with a != b, check if (2a + 2b) mod 3 == v.
    But that's O(|S|^2) per insertion.
    
    Better: precompute for each possible vector v, the set of pairs (a,b) that would conflict.
    Or: maintain a set of "forbidden" vectors. When S is a cap set, the forbidden vectors are those v
    such that there exist a, b in S with a != b and a + b + v = 0 mod 3, i.e., v = (2a + 2b) mod 3.
    
    So we can maintain a set of forbidden vectors. When we add v to S:
    - Check if v is in forbidden set. If yes, can't add.
    - If we add v, update forbidden set: for each a in S, add (2a + 2v) mod 3 to forbidden.
    
    This is O(|S|) per addition after the initial check.
    """
    # Check if v is forbidden
    # We'll maintain _forbidden set externally
    pass

def _greedy_cap_set(n, priority_func, time_limit=290.0):
    """
    Greedily add vectors in descending priority order, keeping the set a cap set.
    Returns the size of the cap set.
    """
    start_time = time.time()
    
    # Generate all vectors in {0,1,2}^n
    # For large n, this is 3^n which can be huge. We need to be smart.
    # For n=1: 3 vectors
    # For n=2: 9 vectors
    # For n=3: 27 vectors
    # For n=4: 81 vectors
    # For n=5: 243 vectors
    # For n=6: 729 vectors
    # For n=7: 2187 vectors
    # For n=8: 6561 vectors
    # For n=9: 19683 vectors
    # For n=10: 59049 vectors
    # For n=11: 177147 vectors
    # For n=12: 531441 vectors
    # For n=13: 1594323 vectors
    # For n=14: 4782969 vectors
    # For n=15: 14348907 vectors - this is getting large
    
    # We'll generate vectors on the fly, sorted by priority.
    # But sorting 3^n vectors is expensive for large n.
    # Alternative: use a max-heap or generate in priority order.
    
    # For now, let's generate all vectors, compute priorities, sort, and greedily add.
    # This works for n up to about 12-13.
    
    all_vectors = list(itertools.product(range(3), repeat=n))
    
    # Compute priority for each vector
    vector_priorities = []
    for v in all_vectors:
        p = priority_func(v, n)
        vector_priorities.append((p, v))
    
    # Sort by priority descending
    vector_priorities.sort(key=lambda x: -x[0])
    
    cap_set = set()
    forbidden = set()
    
    # Precompute: for each vector v, what vectors u would form a triple with some existing pair?
    # Actually, let's maintain forbidden as the set of vectors that would create a triple if added.
    # Initially, forbidden is empty.
    # When we add v to cap_set, for each a in cap_set (before adding v), 
    # the vector (2*a + 2*v) mod 3 would form a triple with a and v... 
    # Wait, no. If we have a and v in the set, then a + v + c = 0 => c = (2a + 2v) mod 3.
    # So c would be forbidden.
    # Also, if we have a and b in the set, then a + b + v = 0 => v = (2a + 2b) mod 3.
    # So when we add v, we need to check if v is in forbidden.
    # And when we add v, we add to forbidden all vectors c such that there exists a in cap_set
    # with a + v + c = 0, i.e., c = (2a + 2v) mod 3.
    
    size = 0
    for p, v in vector_priorities:
        if time.time() > time_limit:
            break
        
        # Check if v is in forbidden
        if v in forbidden:
            continue
        
        # Check if v is already in cap_set
        if v in cap_set:
            continue
        
        # Add v to cap_set
        cap_set.add(v)
        size += 1
        
        # Update forbidden set
        # For each a in cap_set (excluding v), add (2*a + 2*v) mod 3 to forbidden
        for a in cap_set:
            if a == v:
                continue
            # c = (2*a + 2*v) mod 3
            c = tuple((2*a[i] + 2*v[i]) % 3 for i in range(n))
            forbidden.add(c)
    
    return size

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Uses a parametric weight scheme based on coordinate counts and hash.
    """
    # If this is the first call, initialize the greedy state
    if not _greedy_state['active']:
        _greedy_state['active'] = True
        _greedy_state['n'] = n
        _greedy_state['deadline'] = time.time() + 280.0  # Safety margin
        _greedy_state['best_size'] = 0
        _greedy_state['current_size'] = 0
    
    # Check deadline
    if time.time() > _greedy_state['deadline']:
        return 0.0
    
    # Parametric priority: 
    # - Count of 0s, 1s, 2s in the vector
    # - Hash of the vector for tie-breaking
    # We want to maximize cap set size.
    # Known good strategies: favor vectors with balanced coordinates, or use structured patterns.
    
    # Let's use a simple parametric score:
    # score = w0 * count_0 + w1 * count_1 + w2 * count_2 + w_hash * hash(vector)
    
    # For now, use a fixed reasonable set of weights.
    # We can tune these, but let's start with something that favors balanced vectors.
    
    count_0 = vector.count(0)
    count_1 = vector.count(1)
    count_2 = vector.count(2)
    
    # Hash for tie-breaking
    h = hash(vector) % 1000000
    
    # Weights: favor vectors with more 1s (or balanced)
    # Let's try: prefer vectors with equal counts of 0, 1, 2
    balance = 1.0 / (1 + abs(count_0 - count_1) + abs(count_1 - count_2) + abs(count_0 - count_2))
    
    # Add a small hash component for deterministic tie-breaking
    score = balance * 1.0 + (h / 1000000.0) * 0.001
    
    return score

# EVOLVE-BLOCK-END
