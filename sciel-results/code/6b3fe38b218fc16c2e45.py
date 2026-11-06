"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    deadline = time.time() + 140.0

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_one(S, seed):
        n = len(S)
        if n == 0:
            return []
        
        # 1. Build conflict graph
        # Conflict if a+b=c, a+c=b, or b+c=a
        # We will find Max Independent Set in G_conflict
        # which is Max Clique in G_complement (compatibility graph)
        
        # Map values to indices
        val_to_idx = {v: i for i, v in enumerate(S)}
        
        # adj_complement[i] is a bitmask of vertices compatible with i
        # (i.e., no sum-free violation)
        adj_complement = [0] * n
        
        for i in range(n):
            vi = S[i]
            mask = 0
            for j in range(n):
                if i == j:
                    continue
                vj = S[j]
                # Check compatibility: no x, y, z in {vi, vj} such that x+y=z
                # Pairs to check: (vi, vi), (vi, vj), (vj, vj)
                # Sums: 2*vi, vi+vj, 2*vj
                # If any sum is in S, and the resulting z is vi or vj, it's a conflict.
                # Wait, the condition is "no a,b,c in S such that a+b=c".
                # If we add vi and vj to a set T, we must ensure:
                # 1. No a, b in T such that a+b in T.
                # This is a global property.
                # However, for the graph approach:
                # If we select a set of vertices C (Clique in complement), it is Sum-Free.
                # This requires that for ANY a, b in C, a+b is NOT in C.
                # This is NOT a pairwise property!
                # Example: 1, 2, 3. 
                # 1+2=3 (Conflict between {1,2} and 3).
                # 1+1=2 (Conflict between {1} and 2).
                # 2+2=4 (No conflict in {1,2,3}).
                # If we select {1, 3}: 1+1=2 (not in set), 1+3=4 (not in set), 3+3=6 (not in set). OK.
                # If we select {2, 3}: 2+2=4 (not in set), 2+3=5 (not in set), 3+3=6 (not in set). OK.
                # If we select {1, 2}: 1+1=2 (in set!). Conflict.
                
                # So, is Sum-Free a pairwise property?
                # If a+b=c, then {a,b,c} is not sum-free.
                # This implies that if we have a set C, it is sum-free iff 
                # for all a,b in C, a+b is not in C.
                # This is indeed a property that depends on triples.
                # However, we can define a graph G where edges represent "compatible pairs".
                # Two elements x, y are compatible if adding them to an empty set doesn't create a sum-free violation *between them*.
                # i.e., 2x != y and 2y != x and x+y != x (impossible) and x+y != y (impossible).
                # Wait, if x+y = z, and z is also in the set, that's a violation.
                # So x and y are compatible with each other ONLY IF x+y is not in the set?
                # No, x and y can be in the set together even if x+y is in S, AS LONG AS x+y is NOT selected.
                
                # The MIS approach on a simple graph works if the property is pairwise.
                # Sum-free is NOT pairwise. 
                # Example: S = {1, 2, 3}.
                # Max sum-free subset is {1, 3} or {2, 3} (size 2).
                # {1, 2} is not sum-free (1+1=2).
                # {1, 3} is sum-free.
                # {2, 3} is sum-free.
                # {1, 2, 3} is not.
                
                # If we build a graph where edges exist if x+y != z for any z in S? No.
                # The condition "C is sum-free" is equivalent to:
                # For all a, b in C: a+b not in C.
                
                # This can be modeled as a 3-uniform hypergraph independent set, or we can use a different strategy.
                # Given n <= 240, we can use a different exact solver or a stronger heuristic.
                
                # Let's stick to a robust heuristic that is fast and effective, but "structurally different" from the slow greedy.
                # The prompt asks for a STRUCTURAL change. 
                # "Different optimization or search method".
                # The current one is Greedy + Local Search.
                # Let's try Simulated Annealing or a Genetic Algorithm? 
                # Or a better Local Search: Min-Conflicts?
                
                # Actually, for n=240, we can try a "Greedy with Randomization + Multiple Restarts" which is often effective for such constraints.
                # Or, we can use the fact that the problem is equivalent to finding a large independent set in a 3-uniform hypergraph.
                
                # Let's implement a fast Local Search:
                # 1. Start with a random valid set (or odd numbers).
                # 2. Iteratively:
                #    - Try to add a random element. If valid, keep it.
                #    - Try to remove a random element and add a random other element (1-for-1 swap) if it increases size or maintains size but reduces "potential conflicts" (hard to define).
                #    - Better: 2-opt moves. Remove 2, add 2.
                
                # Let's try a different structural approach: **Max Cut / Ising Model approximation**? No, too complex.
                
                # Let's go with **Iterated Local Search (ILS)** with a specific move set:
                # 1. Add moves: Try to add x if valid.
                # 2. Remove moves: Remove x if it allows adding y, z (net gain).
                # 3. Swap moves: Remove x, add y (if size doesn't decrease, but we want to increase size, so this is only useful if it unlocks future adds).
                
                # To make it "structurally different", I will implement a **Bitset-based Conflict Checker** which is O(1) per check after precomputation, allowing for thousands of iterations per second.
                
                # Precomputation:
                # For each x in S, precompute:
                # - conflicts_add[x]: set of y in S such that adding y to a set containing x is invalid?
                #   No, validity depends on the whole set.
                #   However, we can precompute:
                #   - sum_pairs[x]: set of z in S such that x + x = z or x + z = x (impossible) or z + x = x (impossible).
                #     Actually, if x is in set, then 2x cannot be in set.
                #     And if x, y in set, x+y cannot be in set.
                
                # Let's use the Bitset trick for the Local Search:
                # Represent the current set as a bitset `current_bits`.
                # To check if `x` can be added:
                # 1. Check if `2*x` is in `current_bits`.
                # 2. Check if there exists `a` in `current_bits` such that `a+x` is in `current_bits`.
                #    This is equivalent to: `(current_bits & (current_bits >> x))` is not empty? 
                #    No, indices are not values. Values are in [-10000, 10000].
                #    We can map values to bits in a large integer or numpy array.
                #    Let's use a boolean array of size 20001 (offset 10000).
                
                # This allows O(1) or O(N/word_size) checks.
        
        # Implementation of Fast Local Search
        
        # 1. Map values to indices 0..20000
        OFFSET = 10000
        SIZE = 20001
        
        # 2. Precompute for each element x:
        #    - double_idx[x]: index of 2x (if in range and in S)
        #    - sum_partners[x]: list of indices y such that x+y is in S.
        #      Actually, for fast checking:
        #      We maintain `in_set` boolean array.
        #      To check if x can be added:
        #      a) in_set[2*x] must be False.
        #      b) For all a in S: in_set[a+x] must be False.
        #         This is a dot product or bitwise AND if we use bitsets.
        
        # Let's use Python integers as bitsets (arbitrary precision).
        # Bit i corresponds to value i - OFFSET.
        
        def get_bitset(values):
            mask = 0
            for v in values:
                if 0 <= v + OFFSET < SIZE:
                    mask |= (1 << (v + OFFSET))
            return mask

        # Precompute for each x in S:
        # shift_mask[x]: bitset of all y such that x+y is in S? 
        # No, we need to check if (current_set + x) intersects current_set.
        # If current_set is a bitset C.
        # We want to know if (C shifted by x) & C is non-zero.
        # Shifting a Python int is fast.
        
        # Let's define the fast check:
        # Can we add x to set represented by bitset C?
        # 1. Check 2x: if C has bit (2x+OFFSET), fail.
        # 2. Check sums: 
        #    Let C_shifted = C << x  (if x > 0) or C >> -x (if x < 0).
        #    Wait, bit i represents value i-OFFSET.
        #    Value v is at bit v+OFFSET.
        #    Value v+x is at bit v+x+OFFSET = (v+OFFSET) + x.
        #    So C shifted by x bits corresponds to values in C shifted by x.
        #    If (C << x) & C != 0, then there exists a, b in C such that a+x = b.
        #    This covers a+b=c where b=a+x.
        #    We also need to cover x+x=c, which is covered by 2x check.
        #    And a+x=c where a is in C and c is in C. This is the same as b=c.
        
        # So, check:
        # 1. (C >> (2*x+OFFSET)) & 1 == 0  (2x not in C)
        # 2. If x >= 0: (C << x) & C == 0
        #    If x < 0: (C >> -x) & C == 0
        
        # This is O(1) bitwise operations on big integers!
        
        # Algorithm:
        # 1. Generate initial set (e.g., all odds, or random valid set).
        # 2. Local Search:
        #    - While time < deadline:
        #       - Try to add random x not in C. If valid, add.
        #       - If no adds possible, try to remove random x and add random y (swap) if it allows future adds? 
        #         Actually, just try to add. If stuck, remove a random element and try to add others.
        #         Better: "Kick" - remove k random elements, then greedily add as many as possible.
        
        rng = Random(seed + 1000)
        
        # Initial Set: Odds are a good start.
        init_vals = [x for x in S if x % 2 != 0]
        if not init_vals:
            init_vals = [S[0]]
            
        current_set = set(init_vals)
        current_bits = get_bitset(current_set)
        current_size = len(current_set)
        
        best_set = set(current_set)
        best_size = current_size
        
        # Precompute valid additions for current state? No, state changes.
        
        def can_add(x, bits):
            idx = x + OFFSET
            if idx < 0 or idx >= SIZE:
                return False
            if bits & (1 << idx):
                return False # Already in set
            
            # Check 2x
            idx_2x = 2*x + OFFSET
            if 0 <= idx_2x < SIZE:
                if bits & (1 << idx_2x):
                    return False
            
            # Check a+x = b
            if x > 0:
                if (bits << x) & bits:
                    return False
            elif x < 0:
                if (bits >> -x) & bits:
                    return False
            # x=0 is not in S (nonzero integers)
            
            return True

        def add(x, bits):
            return bits | (1 << (x + OFFSET))

        def remove(x, bits):
            return bits & ~(1 << (x + OFFSET))

        # Local Search Loop
        start_time = time.time()
        stall_count = 0
        
        while time.time() - start_time < 10: # 10s per instance
            improved = False
            
            # 1. Try to add elements
            # Shuffle candidates for diversity
            candidates = [x for x in S if x not in current_set]
            rng.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, current_bits):
                    current_set.add(x)
                    current_bits = add(x, current_bits)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(current_set)
            
            if improved:
                stall_count = 0
                continue
            
            # 2. If no adds, try to remove and re-add (Kick)
            # Remove a random element
            if current_size == 0:
                break
                
            # Try removing one element to see if it allows more adds
            # Pick a random element to remove
            to_remove = rng.choice(list(current_set))
            current_set.discard(to_remove)
            current_bits = remove(to_remove, current_bits)
            current_size -= 1
            
            # Now try to add as many as possible
            # We might add back to_remove or others
            candidates = [x for x in S if x not in current_set]
            rng.shuffle(candidates)
            
            added_any = False
            for x in candidates:
                if can_add(x, current_bits):
                    current_set.add(x)
                    current_bits = add(x, current_bits)
                    current_size += 1
                    added_any = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(current_set)
            
            if not added_any:
                # Restore removed element if nothing was added
                current_set.add(to_remove)
                current_bits = add(to_remove, current_bits)
                current_size += 1
                stall_count += 1
            else:
                stall_count = 0
            
            if stall_count > 100:
                # Restart with a different initial set?
                # Or just break and try a new randomization
                break
        
        return list(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        results.append(solve_one(S, seed))
    
    return results
# EVOLVE-BLOCK-END
