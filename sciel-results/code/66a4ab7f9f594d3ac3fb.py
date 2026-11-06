"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    N = 180
    # Generate all forbidden triples: distinct a,b,c in [1,180] with 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c  =>  c = ab/(a+b), need a+b | ab
    # Equivalently: for distinct a,b,c, check if a*b == a*c + b*c, i.e., ab = c(a+b), i.e., c = ab/(a+b)
    
    # Precompute all forbidden triples
    forbidden = []  # list of (a,b,c) tuples where 1/a = 1/b + 1/c, all distinct
    # We need to check all unordered triples? No, the equation is not symmetric.
    # 1/a = 1/b + 1/c. For given distinct a,b,c, exactly one of the six orderings might work.
    # Actually: 1/a = 1/b + 1/c. WLOG assume a is the "sum" side.
    # For each pair (b,c) distinct, a = bc/(b+c). Check if a is integer and distinct from b,c.
    
    triples = set()  # frozensets of {a,b,c} that are forbidden
    for b in range(1, N+1):
        for c in range(b+1, N+1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = frozenset([a, b, c])
                    triples.add(triple)
    
    # Build adjacency: for each element, which triples it's in
    elem_to_triples = {i: [] for i in range(1, N+1)}
    for t in triples:
        for x in t:
            elem_to_triples[x].append(t)
    
    # Greedy: start with all elements, iteratively remove the element in the most remaining triples
    # Better: use local search / simulated annealing style
    
    # Start with all elements
    S = set(range(1, N+1))
    
    def count_violations(S):
        """Count number of forbidden triples fully contained in S."""
        count = 0
        for t in triples:
            if t <= S:
                count += 1
        return count
    
    # Greedy removal: while there are violations, remove the element in the most violated triples
    # But this might not be optimal. Let's try a different approach.
    
    # Approach: Maximize |S| subject to no forbidden triple fully in S.
    # This is a maximum independent set in a 3-uniform hypergraph.
    # Use greedy: start with all, remove elements to break triples.
    
    # Better approach: start with a good initial guess and improve.
    # The current best is [91,180] with 90 elements.
    
    # Let's try: for each starting threshold, take [k, 180] and see if it's valid.
    # Then try to add elements below k.
    
    # First, find the largest k such that [k, 180] is valid.
    def is_valid(S):
        for t in triples:
            if t <= S:
                return False
        return True
    
    # Check [k, 180] for various k
    best_k = N+1
    for k in range(1, N+1):
        S_test = set(range(k, N+1))
        if is_valid(S_test):
            best_k = k
            break  # We want the smallest k (largest set)
    
    # Actually, [k,180] is valid for larger k. We want the smallest k.
    # Let me recheck: as k increases, the set shrinks, so it's easier to be valid.
    # So we want the smallest k such that [k,180] is valid.
    
    valid_k = None
    for k in range(1, N+1):
        S_test = set(range(k, N+1))
        if is_valid(S_test):
            valid_k = k
            break
    
    if valid_k is not None:
        S = set(range(valid_k, N+1))
    else:
        S = set(range(N, N+1))  # fallback
    
    # Now try to add elements below valid_k
    # For each element i < valid_k, check if adding i creates a violation.
    # If not, add it. Repeat until no more can be added.
    
    improved = True
    while improved:
        improved = False
        for i in range(1, valid_k):
            if i not in S:
                S_test = S | {i}
                if is_valid(S_test):
                    S = S_test
                    improved = True
    
    # Now try local search: remove one element, add another
    for _ in range(1000):
        if time.time() > deadline:
            break
        # Try removing a random element and adding a different one
        import random
        if len(S) == 0:
            break
        remove_candidates = list(S)
        add_candidates = [i for i in range(1, N+1) if i not in S]
        if not remove_candidates or not add_candidates:
            break
        r = random.choice(remove_candidates)
        a = random.choice(add_candidates)
        S_test = (S - {r}) | {a}
        if len(S_test) > len(S) and is_valid(S_test):
            S = S_test
    
    # Try to add more elements
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(1, N+1):
            if i not in S:
                S_test = S | {i}
                if is_valid(S_test):
                    S = S_test
                    improved = True
    
    return sorted(S)
# EVOLVE-BLOCK-END
