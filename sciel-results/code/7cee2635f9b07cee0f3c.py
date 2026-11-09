import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin for 90s limit
    N = 128
    
    # 1. Precompute GCDs
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]
            
    # 2. Identify "bad" pairs (a, b)
    # A pair (a, b) is "bad" (must not both be in the solution) if there exists a c 
    # such that gcd(a,b) == gcd(a,c) == gcd(b,c).
    # If such a c exists, then {a,b,c} is a bad triple.
    # We want to find a subset S such that NO bad triple is fully contained in S.
    # This is equivalent to finding a Maximum Independent Set in a graph where 
    # edges connect pairs that are "incompatible".
    # Two numbers a and b are incompatible if they form a bad triple with ANY c.
    # Wait, if a and b are in S, and there exists c such that (a,b,c) is bad, 
    # does that mean S is invalid?
    # Yes, if a, b, c are ALL in S.
    # So, simply removing all pairs (a,b) that participate in ANY bad triple is 
    # sufficient but might be too restrictive (it forces us to exclude one of a,b 
    # even if c is not in S).
    # 
    # Let's re-read the condition: "no three distinct elements whose three pairwise 
    # greatest common divisors are equal".
    # If S contains a and b, it is valid as long as for ALL c in S, the triple (a,b,c) 
    # is NOT bad.
    # This is a 3-uniform hypergraph independence problem.
    # However, we can map this to a graph MIS if we define the graph differently?
    # No, standard MIS is for 2-uniform.
    # But note: if a and b share a gcd g > 1, and we include a and b, we must 
    # exclude ALL c such that gcd(a,c)=g and gcd(b,c)=g.
    # 
    # Let's look at the structure of "bad" triples.
    # gcd(a,b) = gcd(a,c) = gcd(b,c) = g.
    # This implies a, b, c are all multiples of g.
    # Let a = g*a', b = g*b', c = g*c'.
    # gcd(a', b') = 1, gcd(a', c') = 1, gcd(b', c') = 1.
    # So a', b', c' are pairwise coprime.
    # 
    # Strategy:
    # The "bad" triples are formed by picking a common divisor g, and then picking 
    # three numbers from the set of multiples of g that are pairwise coprime 
    # (after dividing by g).
    # 
    # Let's try a different structural approach:
    # Group numbers by their "radical" or prime factors?
    # 
    # Let's stick to the Branch and Bound on the Hypergraph, or use a specific 
    # Graph MIS approximation that works well for this structure.
    # 
    # Actually, there is a known result for this type of problem.
    # If we pick all numbers that are "prime powers" or have a unique large prime factor?
    # 
    # Let's implement a greedy + local search on the Hypergraph directly, 
    # but using a smarter initialization:
    # 
    # 1. Identify all bad triples.
    # 2. Use a greedy algorithm to build S:
    #    - Start with empty S.
    #    - Sort candidates by "conflict degree" (number of bad triples they are in).
    #    - Add element if it doesn't complete any bad triple with existing S.
    #    - This is O(N * T) per step.
    # 
    # Better: 
    # Let's use the "Max Clique in Compatibility Graph" idea from the prompt inspiration?
    # The prompt suggests: "edge exists between two numbers if their GCD is 1".
    # If we pick a set S such that ALL pairs in S are coprime (GCD=1), then 
    # for any triple (a,b,c) in S, gcd(a,b)=1, gcd(a,c)=1, gcd(b,c)=1. 
    # They are equal! So they ARE a bad triple.
    # So we CANNOT pick a set of pairwise coprime numbers larger than 2?
    # No, if gcd(a,b)=1, gcd(a,c)=1, gcd(b,c)=1, then the triple is bad.
    # So S cannot contain 3 pairwise coprime numbers.
    # This is a very strong constraint.
    # 
    # Therefore, the maximum size of a set of pairwise coprime numbers is 2.
    # But we can include numbers that are NOT pairwise coprime, as long as they 
    # don't form the specific "equal gcd" triple.
    # 
    # Example: 2, 4, 6.
    # gcd(2,4)=2, gcd(2,6)=2, gcd(4,6)=2. Equal! Bad.
    # 
    # Example: 2, 3, 6.
    # gcd(2,3)=1, gcd(2,6)=2, gcd(3,6)=3. Not equal. Good.
    # 
    # So, we can have numbers with shared factors, as long as the gcds are distinct.
    # 
    # Let's try a Branch and Bound on the original set [1..128] directly.
    # State: (current_set, candidate_set, current_size)
    # 
    # To make it fast, we need a good upper bound.
    # Upper bound: current_size + len(candidate_set) is trivial.
    # 
    # Let's use a Greedy Coloring bound on the "conflict graph"?
    # Define conflict graph H where an edge (u,v) exists if u and v are in a bad triple 
    # with SOME w? No, that's too strong.
    # 
    # Let's use the property: 
    # If we fix the set of "small" numbers (1..K), we can determine which "large" numbers 
    # are safe.
    # 
    # Alternative: Use the "Seed" approach but much faster.
    # The current seed got 1.75.
    # 
    # Let's try a specific constructive heuristic:
    # 1. Include all numbers that are powers of 2? 
    #    {1, 2, 4, 8, 16, 32, 64, 128}
    #    Check triples: (1,2,4): gcd(1,2)=1, gcd(1,4)=1, gcd(2,4)=2. Not equal.
    #    (2,4,8): gcd(2,4)=2, gcd(2,8)=2, gcd(4,8)=4. Not equal.
    #    It seems powers of 2 are safe. Size 8.
    # 
    # 2. Add numbers with a large prime factor > 64?
    #    If p > 64, then only multiples of p are p, 2p, ... but 2p > 128 if p>64.
    #    So only p itself is in [1,128].
    #    Primes in (64, 128]: 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113.
    #    These are coprime to each other and to most small numbers.
    #    Let's check (67, 2, 4). gcd(67,2)=1, gcd(67,4)=1, gcd(2,4)=2. Safe.
    #    Let's check (67, 1, 2). gcd(67,1)=1, gcd(67,2)=1, gcd(1,2)=1. BAD.
    #    So we cannot have 1, 2, and a prime p>64.
    #    If we keep 1 and 2, we must exclude all primes > 64?
    #    Or exclude 1 or 2?
    #    1 is "bad" because gcd(1,x)=1 for all x.
    #    If S contains 1, then for any a,b in S\{1}, if gcd(a,b)=1, then (1,a,b) is bad.
    #    So if 1 is in S, S\{1} must be a set where NO two elements are coprime.
    #    This means S\{1} must be a clique in the "non-coprime" graph.
    #    Max size of a set of pairwise non-coprime numbers in [1,128]?
    #    All even numbers: 64 numbers.
    #    Check (2,4,6): gcds 2,2,2. Bad.
    #    So even numbers alone are not safe.
    #    
    #    This is getting complex. Let's use a robust Local Search on the Hypergraph.
    
    # Precompute bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = G[a][b]
            for c in range(b + 1, N + 1):
                if G[a][c] == g_ab and G[b][c] == g_ab:
                    bad_triples.append((a, b, c))
    
    T = len(bad_triples)
    
    # Map element to list of triple indices
    triples_by_elem = [[] for _ in range(N + 1)]
    for i, t in enumerate(bad_triples):
        for x in t:
            triples_by_elem[x].append(i)
    
    # Helper to check if adding x to S is valid
    # S is a set of integers.
    def is_valid_addition(S, x):
        # Check all triples containing x
        for t_idx in triples_by_elem[x]:
            a, b, c = bad_triples[t_idx]
            # Check if the other two are in S
            # We need to find the other two elements in the triple
            others = [y for y in (a, b, c) if y != x]
            if others[0] in S and others[1] in S:
                return False
        return True

    # Greedy Construction
    def greedy_construct():
        S = set()
        # Sort by degree (number of conflicts) descending? 
        # Or ascending to keep flexibility?
        # Usually, picking "safe" elements first is good.
        # Degree = number of bad triples the element is in.
        degrees = [len(triples_by_elem[i]) for i in range(1, N + 1)]
        candidates = list(range(1, N + 1))
        # Sort by degree ascending (least constrained first)
        candidates.sort(key=lambda x: degrees[x])
        
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if is_valid_addition(S, x):
                S.add(x)
        return S

    # Local Search: Swap based
    # State: S (set)
    # Objective: Maximize |S|
    # Move: Remove one element from S, Add one element not in S.
    # Or: Remove one, Add one.
    # To ensure validity, we can use a "repair" step or only accept valid states.
    # Let's do a "kick" strategy:
    # 1. Pick random element in S to remove.
    # 2. Pick random element not in S to add.
    # 3. If valid, accept. If not, reject.
    # This is very slow if acceptance rate is low.
    
    # Better: 
    # Maintain S.
    # For each x in S, check if it can be removed.
    # For each y not in S, check if it can be added.
    # If we can add y, we do it.
    # If we can remove x, we do it (to free up space for others?).
    # Actually, we want to MAXIMIZE size.
    # So we should always add if possible.
    # The problem is getting stuck in a local optimum where no single addition is possible, 
    # but a swap (remove x, add y) is possible.
    
    def local_search(S):
        S = set(S)
        no_improve_count = 0
        max_no_improve = 1000
        current_size = len(S)
        
        while time.time() < deadline - 2.0:
            if no_improve_count > max_no_improve:
                break
            
            improved = False
            
            # 1. Try to add any element
            # Sort candidates by "potential" (e.g., number of conflicts)
            # Just iterate through all not in S
            for y in range(1, N + 1):
                if y not in S:
                    if is_valid_addition(S, y):
                        S.add(y)
                        current_size += 1
                        improved = True
                        no_improve_count = 0
                        break # Restart scan after addition
            
            if improved:
                continue
                
            # 2. Try swaps
            # Remove x, Add y
            # To be efficient, only try x that are "critical" or random.
            # And y that are "compatible" with S\{x}.
            
            # Precompute for each x in S, which y's become valid if x is removed?
            # This is expensive.
            
            # Randomized Swap:
            if S:
                x = random.choice(list(S))
                # Find a y not in S that is valid with S \ {x}
                # We can check is_valid_addition(S - {x}, y)
                
                # To speed up, we can precompute for each x, the set of y's that 
                # are ONLY blocked by x?
                # If y is blocked by x and z, removing x doesn't help.
                
                # Let's just try a few random y's.
                # Or iterate through y's that are in conflict with x?
                
                # Let's try all y not in S, but stop early if found.
                # To avoid O(N^2) per step, we can limit tries.
                
                found_swap = False
                # Shuffle y's not in S for randomness
                candidates_y = [y for y in range(1, N + 1) if y not in S]
                random.shuffle(candidates_y)
                
                for y in candidates_y:
                    if is_valid_addition(S - {x}, y):
                        S.remove(x)
                        S.add(y)
                        # Size unchanged, but we might enable future additions
                        # We consider this an improvement if it leads to future adds?
                        # For now, accept to escape local optima.
                        improved = True
                        found_swap = True
                        no_improve_count = 0
                        break
                
                if found_swap:
                    continue
            
            no_improve_count += 1
            
        return S

    # Initialize
    best_S = greedy_construct()
    best_S = local_search(best_S)
    
    # Multiple restarts with different greedy orderings?
    # Given time constraints, one good run is better than many bad ones.
    # But we have ~80s.
    # Let's try a few restarts with different random seeds for the swap phase.
    
    global_best = best_S
    global_best_size = len(best_S)
    
    # Run a few more iterations of local search from the current best
    # Or restart greedy with different sorting?
    
    # Let's try sorting by "degree" descending (most constrained first)
    def greedy_construct_desc():
        S = set()
        degrees = [len(triples_by_elem[i]) for i in range(1, N + 1)]
        candidates = list(range(1, N + 1))
        candidates.sort(key=lambda x: degrees[x], reverse=True)
        
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if is_valid_addition(S, x):
                S.add(x)
        return S

    S2 = greedy_construct_desc()
    S2 = local_search(S2)
    if len(S2) > global_best_size:
        global_best = S2
        global_best_size = len(S2)
        
    # Randomized Greedy
    def greedy_construct_random():
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if is_valid_addition(S, x):
                S.add(x)
        return S

    # Run a few random restarts
    import random
    for i in range(5):
        if time.time() > deadline - 10.0:
            break
        S3 = greedy_construct_random()
        S3 = local_search(S3)
        if len(S3) > global_best_size:
            global_best = S3
            global_best_size = len(S3)

    # Final verification
    # Ensure no bad triple is in global_best
    for t in bad_triples:
        if t[0] in global_best and t[1] in global_best and t[2] in global_best:
            # Invalid! Repair by removing one element
            # This shouldn't happen if is_valid_addition is correct
            global_best.remove(t[0])
            
    return list(global_best)

# EVOLVE-BLOCK-START
# (Note: The code above is the implementation. The EVOLVE-BLOCK markers are for the evolution system.)
# EVOLVE-BLOCK-END
