"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(1000000)

def construct():
    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_sum_free(S, time_limit=30.0):
        """
        Find a maximum sum-free subset of S using branch-and-bound with memoization.
        
        The constraint: no a, b, c in T (a,b not necessarily distinct) with a+b=c.
        This is equivalent to: for each c in T, there are no a, b in T with a+b=c.
        
        We model this as a maximum independent set problem on a 3-uniform hypergraph
        where hyperedges are triples {a, b, a+b} for all valid a, b with a+b in S.
        
        Branch-and-bound:
        - State: (remaining_elements, current_selection)
        - At each step, pick an element and either include it or exclude it
        - If we include x, we must exclude all y such that there exists z in current_selection
          with x+y=z or y+x=z or x+z=y (i.e., x conflicts with some pair in selection)
        - Actually, the conflict is: if we have a, b in T, then a+b cannot be in T.
          So the constraint is: T ∩ (T+T) = ∅.
        
        Better formulation for branch-and-bound:
        - Process elements in a fixed order
        - For each element, decide include or exclude
        - If we include x, we need to ensure no a, b in current T have a+b=x,
          and no a in current T has x+a in T (or a+x in T)
        
        Let's use a different approach: build T incrementally.
        - Sort S in some order
        - For each element x in order, decide to include or exclude
        - If we include x, check that adding x doesn't create a violation with existing T
        - Use upper bound: len(T) + len(remaining) to prune
        
        For memoization, we can use the set of remaining elements as the key,
        but that's too large. Instead, we'll use a simpler branch-and-bound without
        memoization but with good pruning.
        """
        n = len(S)
        if n <= 1:
            return S[:]
        
        s_set = set(S)
        
        # Precompute for each element x, which pairs (a, b) in S have a+b=x
        # and which elements conflict with x given a set T
        # For efficiency, precompute:
        # pairs_summing_to[x] = list of (a, b) such that a+b=x and a,b in S
        # But we need this dynamically based on what's in T.
        
        # Let's precompute a lookup: for each x in S, list of (a, b) with a+b=x
        pairs_to = {x: [] for x in S}
        for i, a in enumerate(S):
            for b in S:
                s = a + b
                if s in s_set:
                    pairs_to[s].append((a, b))
        
        # Sort elements by a heuristic: elements with more conflicts first
        # An element x has "conflict degree" = number of pairs (a,b) with a+b=x
        # plus number of pairs (a, x) or (x, a) where a+x is in S
        def conflict_degree(x):
            # Pairs (a, b) with a+b=x
            d = len(pairs_to[x])
            # Pairs (a, x) with a+x in S, or (x, a) with x+a in S
            for a in S:
                if a + x in s_set:
                    d += 1
            return d
        
        # Order by conflict degree descending (most constrained first)
        order = sorted(range(n), key=lambda i: -conflict_degree(S[i]))
        ordered_S = [S[i] for i in order]
        
        best_size = 0
        best_subset = []
        
        # Use iterative deepening or just branch-and-bound
        # State: index in ordered_S, current set T (as a frozenset for hashing)
        # But frozenset hashing is expensive for large sets.
        
        # Alternative: use a boolean array and pass the index
        # Let's use a recursive function with the current T as a set
        
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
        
        # Branch and bound
        def bnb(idx, T, current_size):
            nonlocal best_size, best_subset
            
            if time.time() - start_time > time_limit:
                return
            
            # Upper bound: current_size + (n - idx)
            if current_size + (n - idx) <= best_size:
                return
            
            if idx == n:
                if current_size > best_size:
                    best_size = current_size
                    best_subset = list(T)
                return
            
            x = ordered_S[idx]
            
            # Option 1: Exclude x
            bnb(idx + 1, T, current_size)
            
            # Option 2: Include x (if it doesn't violate)
            if is_sum_free_with(T, x):
                T.add(x)
                bnb(idx + 1, T, current_size + 1)
                T.remove(x)
        
        start_time = time.time()
        T = set()
        bnb(0, T, 0)
        
        # If branch-and-bound didn't find anything good, use greedy as fallback
        if len(best_subset) < (n + 2) // 3:
            # Greedy fallback
            for order_strategy in [ordered_S, sorted(S), sorted(S, reverse=True)]:
                T = set()
                for x in order_strategy:
                    if is_sum_free_with(T, x):
                        T.add(x)
                if len(T) > len(best_subset):
                    best_subset = list(T)
        
        return best_subset
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        subset = solve_sum_free(S)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
