"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import sys
    sys.setrecursionlimit(100000)

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_sum_free(S):
        """
        Find a maximum sum-free subset of S.
        Sum-free means no a, b, c in subset (a,b not necessarily distinct) with a+b=c.
        
        We model this as finding a maximum independent set in a graph where:
        - Vertices are elements of S
        - Edges represent conflicts: we cannot have both a and c if a+b=c for some b in S,
          or we cannot have a and b if a+b is in S (as c), etc.
        
        Actually, the constraint is: for any a, b, c in the subset, a+b != c.
        This is a 3-uniform hypergraph constraint. But we can reformulate:
        For each pair (a, b) with a <= b (to avoid symmetry, though a=b is allowed),
        if a+b is in S, then we cannot have all three {a, b, a+b} in the subset.
        This is a hyperedge of size 3.
        
        However, a simpler way to think about it: the subset T is sum-free iff
        for every c in T, there do NOT exist a, b in T such that a+b=c.
        Equivalently, T ∩ (T+T) = empty, where T+T = {a+b : a,b in T}.
        
        Since the instances are small (n up to 240), we can use a branch-and-bound
        approach. But 240 is too large for exact exponential search in general.
        
        Let's use a greedy + local search or a smarter heuristic.
        Actually, the problem says "Maximize the total size" and gives a lower bound of (n+2)/3.
        The seed program uses odd numbers, which gives roughly n/2.
        
        Let's try a different approach: 
        1. Start with all elements.
        2. While there's a violation (a+b=c with a,b,c in set), remove one of them.
        3. Use a greedy strategy to remove the element that is involved in the most violations.
        
        But this might not be optimal. Let's try a randomized local search:
        - Start with a random subset (e.g., all odd numbers, or a random half).
        - Try to add elements that don't create violations.
        - Try to swap elements to improve.
        
        Given the time constraint (120s total), and n up to 240, let's use a heuristic:
        1. Partition S into three sets based on a random hash or value range.
        2. Or use the fact that if we take numbers from a range [L, R] where 2L > R, 
           then no sum of two elements can be in the set (if all are positive). But we have negatives.
        
        Better heuristic: 
        - Sort S.
        - Use a greedy algorithm: process elements in some order, add if it doesn't violate.
        - Try multiple orderings (e.g., by absolute value, by value, random).
        
        Let's implement a greedy algorithm with multiple strategies and pick the best.
        """
        n = len(S)
        if n <= 1:
            return S[:]
        
        # Precompute which pairs sum to which element
        # For each c, find all pairs (a, b) with a+b=c
        s_set = set(S)
        # conflicts[c] = list of (a, b) such that a+b=c and a,b in S
        # But for efficiency, let's precompute for each pair (i,j) if S[i]+S[j] is in S
        # and what its index is
        
        # Let's use a different approach: 
        # The sum-free condition is violated if there exist a, b, c in T with a+b=c.
        # We can think of this as: for each c, the set of pairs (a, b) with a+b=c must not
        # have both a and b in T.
        
        # Greedy algorithm:
        # 1. Start with empty set T.
        # 2. For each element x in S (in some order), check if adding x to T would create a violation.
        #    A violation is created if there exist a, b in T such that a+b=x, or x+b=a for some a in T,
        #    or a+x=b for some a, b in T.
        #    More precisely, adding x to T creates a violation if:
        #    - There exist a, b in T such that a+b=x (then {a,b,x} is a violating triple)
        #    - There exists a in T such that x+a is in T (then {x,a,x+a} is a violating triple)
        #    - There exists a in T such that a+x is in T (same as above)
        #    Actually, the condition is: after adding x, is there any triple (a,b,c) in T∪{x} with a+b=c?
        #    The new triples involving x are: (x, a, c) with x+a=c, (a, x, c) with a+x=c, (a, b, x) with a+b=x.
        #    So we need to check:
        #    1. Is there a, b in T with a+b=x?
        #    2. Is there a in T such that x+a is in T? (i.e., x+a in T∪{x}, but x+a=x implies a=0, which is not in S)
        #       So x+a in T for some a in T.
        #    3. Similarly, a+x in T for some a in T. (Same as 2)
        
        # To check these efficiently, we maintain T as a set.
        
        def is_sum_free_with(T, x):
            """Check if T ∪ {x} is sum-free."""
            # Check if there exist a, b in T with a+b=x
            for a in T:
                b = x - a
                if b in T:
                    return False
            # Check if there exists a in T such that x+a is in T
            for a in T:
                c = x + a
                if c in T:
                    return False
            return True
        
        # This is O(|T|) per check, and we do it for each x, so O(n^2) total, which is fine for n=240.
        
        # Try multiple orderings:
        import random
        rng = random.Random(42)
        
        best_subset = []
        
        # Strategy 1: Sort by absolute value, ascending
        order1 = sorted(S, key=lambda x: abs(x))
        # Strategy 2: Sort by value, ascending
        order2 = sorted(S)
        # Strategy 3: Sort by value, descending
        order3 = sorted(S, reverse=True)
        # Strategy 4: Random order
        order4 = S[:]
        rng.shuffle(order4)
        # Strategy 5: Sort by value, but start from the middle
        mid = n // 2
        order5 = sorted(S[mid:]) + sorted(S[:mid])
        
        for order in [order1, order2, order3, order4, order5]:
            T = set()
            for x in order:
                if is_sum_free_with(T, x):
                    T.add(x)
            if len(T) > len(best_subset):
                best_subset = list(T)
        
        # Try to improve by swapping: remove an element and try to add others
        # This is a simple local search
        current = set(best_subset)
        improved = True
        while improved:
            improved = False
            for x in current:
                # Try removing x and see if we can add more elements
                candidate = current - {x}
                added = 0
                for y in S:
                    if y not in candidate:
                        if is_sum_free_with(candidate, y):
                            candidate.add(y)
                            added += 1
                if len(candidate) > len(current):
                    current = candidate
                    improved = True
                    break
        
        return list(current)
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        subset = solve_sum_free(S)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
