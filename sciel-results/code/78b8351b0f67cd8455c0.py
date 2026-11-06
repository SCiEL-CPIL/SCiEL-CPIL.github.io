"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import sys
    sys.setrecursionlimit(1000000)
    import time
    from collections import defaultdict

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
        n = len(S)
        if n == 0:
            return []
        if n == 1:
            return S[:]
        
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(S)}
        
        # Precompute for each index i, the list of pairs (j, k) such that S[i] + S[j] == S[k] or S[i] + S[k] == S[j]
        # Actually, we need to know for any subset T, if it's sum-free.
        # A subset T is sum-free if for all a,b,c in T, a+b != c.
        # Equivalently, for each c in T, the set {a in T : c-a in T} must be empty.
        
        # Let's precompute conflicts.
        # conflict[i] = set of indices j such that S[i] + S[j] is in S.
        # If we include i and j, we cannot include S[i]+S[j].
        
        # Better: Precompute triples.
        # triple[i] = list of (j, k) such that S[i] + S[j] = S[k].
        # If i and j are in T, then k cannot be in T.
        
        # Let's use a bitset approach for efficiency.
        # n is at most 240. We can use Python integers as bitsets.
        
        # bitset[i] has bit j set if S[i] + S[j] is in S, and the index of that sum is k.
        # Wait, the relation is symmetric in i and j for the sum, but the result k is specific.
        # Let's define:
        # For each i, j:
        #   s = S[i] + S[j]
        #   if s in val_to_idx:
        #      k = val_to_idx[s]
        #      # Then if i and j are in T, k cannot be in T.
        #      # Also, if i and k are in T, j cannot be in T.
        #      # Also, if j and k are in T, i cannot be in T.
        
        # So we can build a graph of "conflicts" or rather a 3-uniform hypergraph.
        # But for B&B, we can maintain a set of "forbidden" indices.
        
        # Let's try a different angle.
        # We want to find a maximum subset T of {0..n-1} such that no triple (i,j,k) is fully contained in T.
        
        # Precompute all triples (i, j, k) with i < j <= k (to avoid duplicates? No, order matters for sum, but set membership doesn't).
        # Actually, a+b=c. i, j, k are distinct indices? Not necessarily, but S has distinct values.
        # If i=j, then 2*S[i] = S[k].
        
        # Let's just iterate all pairs (i, j) and check if S[i]+S[j] is in S.
        triples = []
        for i in range(n):
            for j in range(i, n):  # i <= j to handle symmetry of addition, but we need to be careful
                s = S[i] + S[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # We have a triple (i, j, k)
                    # To avoid duplicates in the list of triples, we can sort (i, j, k) or just store unique sets.
                    # Since i and j are interchangeable in the sum, but k is the result.
                    # Let's store the set of indices.
                    triples.append((i, j, k))
        
        # Note: The same triple might be found multiple times?
        # If i < j, we find (i, j, k).
        # If we iterate j from i to n, we cover all unordered pairs {i, j}.
        # So each unordered pair is considered once.
        # The triple is uniquely determined by the pair and the sum.
        
        # However, we need to be efficient in checking if a set T is sum-free.
        # A set T is sum-free if no triple (i, j, k) is a subset of T.
        
        # Let's use a Branch and Bound approach.
        # State: current set T (as a frozenset or bitset), remaining candidates R.
        # We want to maximize |T|.
        
        # To speed up, we can precompute for each index i, the list of triples involving i.
        # triples_by_idx[i] = list of (j, k) such that (i, j, k) is a triple.
        
        triples_by_idx = defaultdict(list)
        for i, j, k in triples:
            triples_by_idx[i].append((j, k))
            triples_by_idx[j].append((i, k))
            triples_by_idx[k].append((i, j))
        
        # B&B:
        # We maintain a set of "available" indices.
        # We try to add indices one by one.
        # When we add an index i, we must remove all indices j such that there exists a triple (i, j, k) where k is already in T?
        # No, that's not right.
        # If we add i to T, we must ensure that no triple (i, j, k) has j and k also in T.
        # So, if j is in T, we must remove k. If k is in T, we must remove j.
        # This suggests a propagation step.
        
        # Let's define a function `propagate(T, R)`:
        # Given a current set T and a set of remaining candidates R,
        # check if T is sum-free (it should be by construction).
        # For each i in T, for each (j, k) in triples_by_idx[i]:
        #   If j in T and k in R, then k must be removed from R (because if k is added, we get a triple).
        #   If k in T and j in R, then j must be removed from R.
        #   If j in R and k in R, we don't remove them yet, but we note a constraint.
        
        # This propagation is local.
        
        # Let's implement a simple B&B with propagation.
        
        best_size = 0
        best_subset = set()
        
        # Initial greedy to get a good lower bound
        # Greedy: sort by degree (number of triples), pick lowest degree first?
        # Or just pick arbitrarily.
        
        # Let's try a few greedy strategies to get an initial bound.
        
        def greedy_order(order):
            T = set()
            R = set(range(n))
            for i in order:
                if i in R:
                    # Check if adding i creates a conflict with T
                    # i can be added if no (i, j, k) has j, k in T.
                    ok = True
                    for j, k in triples_by_idx[i]:
                        if j in T and k in T:
                            ok = False
                            break
                    if ok:
                        T.add(i)
                        # Remove from R
                        R.remove(i)
                        # Propagate: remove from R any j or k that would form a triple with i and an element in T
                        # Actually, if we add i, we must remove from R any x such that there exists y in T with (i, y, x) or (i, x, y) or (x, y, i) in triples.
                        # Wait, the propagation logic:
                        # If i is in T, and y is in T, then any z such that (i, y, z) or (i, z, y) or (y, i, z) or (y, z, i) or (z, i, y) or (z, y, i) is a triple must be removed from R.
                        # This is getting complicated to implement efficiently on the fly.
                        # For the greedy phase, let's just do the simple check.
            return T
        
        # Simple greedy:
        # Order by value, or random.
        
        import random
        rng = random.Random(42)
        
        orders = [
            list(range(n)),
            list(range(n-1, -1, -1)),
            sorted(range(n), key=lambda i: abs(S[i])),
            sorted(range(n), key=lambda i: S[i]),
        ]
        for _ in range(5):
            o = list(range(n))
            rng.shuffle(o)
            orders.append(o)
        
        for o in orders:
            T = greedy_order(o)
            if len(T) > best_size:
                best_size = len(T)
                best_subset = T.copy()
        
        # Now B&B
        # We will use a recursive search.
        # To avoid deep recursion, we might need to be careful, but n=240 is small enough for a bitset.
        
        # Let's represent T and R as bitsets (integers).
        # bit i is set if i is in the set.
        
        def to_bitset(s):
            b = 0
            for i in s:
                b |= (1 << i)
            return b
        
        def from_bitset(b):
            return [i for i in range(n) if (b >> i) & 1]
        
        best_size = len(best_subset)
        best_bitset = to_bitset(best_subset)
        
        # Precompute for each i, the bitset of indices j such that (i, j, k) is a triple for some k.
        # Actually, we need: if i is in T, which indices cannot be added to T if some other index is already in T?
        
        # Let's define `conflict[i]` as a list of (j, k) pairs.
        # If i is added to T, and j is already in T, then k must be removed from R.
        
        # In B&B:
        # State: T_bit, R_bit.
        # We pick an index i from R.
        # Branch 1: Add i to T.
        #   New T = T | {i}
        #   New R = R - {i}
        #   Propagate: For each j in T (excluding i), for each (j, k) in triples_by_idx[i]:
        #       if k in R, remove k from R.
        #   Also, for each j in T, for each (k, j) in triples_by_idx[i] (i.e., i+k=j? No, i is the one being added).
        #   Let's be precise.
        #   Triples involving i: (i, j, k) where S[i]+S[j]=S[k] or S[i]+S[k]=S[j].
        #   If i is in T, and j is in T, then k cannot be in T.
        #   So if j is in T, we must remove k from R.
        #   If k is in T, we must remove j from R.
        
        #   So, when adding i:
        #   For each (j, k) in triples_by_idx[i]:
        #      if j in T: remove k from R
        #      if k in T: remove j from R
        
        # Branch 2: Do not add i (remove i from R).
        #   New R = R - {i}
        
        # Pruning:
        # Upper bound = |T| + |R|. If <= best_size, prune.
        
        # Let's implement this.
        
        # To make it faster, we can order the branching.
        # Pick i from R with highest "impact" (most conflicts).
        
        def solve_bnb(T_bit, R_bit):
            nonlocal best_size, best_bitset
            
            # Prune
            if bin(T_bit).count('1') + bin(R_bit).count('1') <= best_size:
                return
            
            if R_bit == 0:
                if bin(T_bit).count('1') > best_size:
                    best_size = bin(T_bit).count('1')
                    best_bitset = T_bit
                return
            
            # Propagation
            # We can propagate before branching.
            # Check if any i in T forces removals from R.
            # This is already done when we added to T.
            # But maybe we can propagate more?
            # If i, j in R and k in T, and (i, j, k) is a triple, then we can't have both i and j in T.
            # This is a constraint on R, not an immediate removal.
            
            # Pick an index to branch on.
            # Let's pick the one with the most triples involving it and elements in T or R.
            # For simplicity, pick the first one in R.
            # Or pick the one with highest degree in the induced subgraph on R?
            
            # Let's pick the index i in R that is in the most triples with other elements in R.
            # This might be expensive to compute.
            # Let's just pick the smallest index in R for now.
            
            # Find the lowest set bit in R_bit
            i = (R_bit & -R_bit).bit_length() - 1
            
            # Branch 1: Add i
            T_new = T_bit | (1 << i)
            R_new = R_bit & ~(1 << i)
            
            # Propagate
            for j, k in triples_by_idx[i]:
                if (T_new >> j) & 1:
                    R_new &= ~(1 << k)
                if (T_new >> k) & 1:
                    R_new &= ~(1 << j)
            
            solve_bnb(T_new, R_new)
            
            # Branch 2: Do not add i
            R_new = R_bit & ~(1 << i)
            solve_bnb(T_bit, R_new)
        
        # Start B&B
        # We start with T=0, R=full set.
        # But we have a good initial bound.
        # We can start B&B with T=0, R=full, but prune using best_size.
        
        full_bitset = (1 << n) - 1
        solve_bnb(0, full_bitset)
        
        return from_bitset(best_bitset)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        subset = solve_sum_free(S)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
