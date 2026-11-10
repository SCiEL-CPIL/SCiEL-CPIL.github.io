import sys
import time

def construct():
    # Moduli from 2 to 128
    moduli = list(range(2, 129))
    max_mod = 128
    
    # Create a list of all possible (modulus, residue) pairs
    # Index mapping:
    # We need to map (m, r) to a unique index for bitsets
    # Total pairs = sum(m for m in 2..128) = 8255
    pairs = []
    index_map = {}
    idx = 0
    for m in moduli:
        for r in range(m):
            pairs.append((m, r))
            index_map[(m, r)] = idx
            idx += 1
            
    N = len(pairs)
    
    # Precompute GCDs? Not needed for every pair if we compute on the fly or precompute compatibility.
    # Let's precompute the compatibility bitsets.
    # compat[i] is a bitset of all j such that pairs[i] and pairs[j] are disjoint.
    
    # Using Python integers as bitsets is efficient for N < 10000.
    # 8255 bits is about 1 KB per integer. 8255 integers is about 8 MB.
    
    compat = [0] * N
    
    # Optimization: Group by modulus?
    # For each pair i, check all j? O(N^2) = 8255^2 ~ 68 million operations.
    # In Python, 68 million operations might take ~10-30 seconds.
    # We need to be careful.
    
    # Let's try to optimize the graph construction.
    # Two pairs (m1, r1) and (m2, r2) are compatible if r1 != r2 mod gcd(m1, m2).
    
    # We can iterate i, and for each i, iterate j > i?
    # Or just build the full bitset.
    
    # Let's try to speed up the inner loop.
    # Precompute gcd for all pairs of moduli?
    # moduli are 2..128. 127 moduli.
    # gcd_matrix[m1_idx][m2_idx]
    
    m_indices = {m: i for i, m in enumerate(moduli)}
    num_mods = len(moduli)
    gcd_mat = [[0]*num_mods for _ in range(num_mods)]
    for i in range(num_mods):
        for j in range(num_mods):
            gcd_mat[i][j] = __import__('math').gcd(moduli[i], moduli[j])
            
    # Now build compat bitsets
    # For each i, set bit j if compatible.
    
    # To make it faster, we can group nodes by modulus.
    # Nodes for modulus m are at indices [start_m, end_m)
    
    start_idx = [0] * (max_mod + 1)
    for m in moduli:
        start_idx[m] = idx
        idx += m
    # Adjust start_idx to be actual start
    # Let's just use the index_map and pairs list.
    
    # Re-indexing for clarity
    # Let's create a list of lists: nodes_by_mod[m] = [indices]
    nodes_by_mod = {m: [] for m in moduli}
    for i, (m, r) in enumerate(pairs):
        nodes_by_mod[m].append(i)
        
    # Build compat
    # This is O(N^2) which is risky in Python.
    # 8255 * 8255 / 2 = 34 million checks.
    # Each check involves gcd lookup and modulo.
    # Might be too slow (10-20s).
    
    # Alternative: 
    # We only need to find the Max Clique.
    # We can use a greedy approach to get a good initial clique, 
    # and then branch and bound.
    # But we need the graph.
    
    # Let's try to construct the graph faster.
    # For a fixed i (m1, r1), a node j (m2, r2) is INCOMPATIBLE if r2 % gcd(m1, m2) == r1 % gcd(m1, m2).
    # So j is COMPATIBLE if NOT that.
    
    # Let's precompute for each modulus m2, and each possible residue class mod g (where g divides m1 and m2?), 
    # it's complicated.
    
    # Let's just do the O(N^2) but with some pruning or vectorization? 
    # No vectorization for logic like this easily.
    
    # Maybe we can reduce N?
    # The maximum clique size is at most 127.
    # We can use a heuristic: 
    # Pick moduli that are pairwise coprime? 
    # If moduli are pairwise coprime, any residues are disjoint? 
    # No, if gcd=1, they are always disjoint? 
    # Yes, if gcd(m1, m2)=1, then r1 != r2 mod 1 is always true (0!=0 is false? wait).
    # Condition: r1 != r2 mod g. If g=1, r1 mod 1 = 0, r2 mod 1 = 0. 
    # 0 != 0 is False. 
    # So if gcd=1, they are NEVER disjoint? 
    # Let's re-read the criterion.
    # "disjoint if r1 not congruent r2 mod gcd(m1, m2)"
    # If gcd=1, r1 mod 1 = 0, r2 mod 1 = 0. 0 is congruent to 0. 
    # So they are NOT disjoint. 
    # This means pairwise coprime moduli CANNOT be used together?
    # Wait. 
    # If m1=2, m2=3. gcd=1. 
    # Classes: x=0 mod 2, x=0 mod 3. Intersection: x=0 mod 6. Not disjoint.
    # Classes: x=0 mod 2, x=1 mod 3. Intersection: x=4 mod 6. Not disjoint.
    # Any two arithmetic progressions with coprime moduli intersect infinitely often.
    # So, if gcd(m1, m2) = 1, they are NEVER disjoint.
    # Therefore, we CANNOT have two moduli with gcd 1 in our set.
    
    # This is a huge insight.
    # The moduli we choose MUST have gcd > 1 for every pair?
    # No, if gcd > 1, we CAN choose residues to be disjoint.
    # If gcd = 1, we CANNOT make them disjoint.
    # So, the set of moduli must be such that gcd(mi, mj) > 1 for all i != j.
    # This is a "pairwise non-coprime" set of moduli.
    
    # Let's verify.
    # If gcd(m1, m2) = g > 1.
    # We need r1 != r2 mod g.
    # This is possible if we choose r1 and r2 appropriately.
    
    # So the problem reduces to:
    # 1. Select a subset of moduli M from {2..128} such that gcd(mi, mj) > 1 for all distinct mi, mj in M.
    # 2. For each selected modulus, select a residue such that for all pairs, the disjointness condition holds.
    
    # Step 1: Maximize the size of a set of integers from [2, 128] that are pairwise non-coprime.
    # Step 2: Check if residues can be assigned.
    
    # Let's focus on Step 1 first.
    # This is a Maximum Clique problem in the graph where nodes are moduli 2..128 and edges connect non-coprime pairs.
    # This graph is much smaller (127 nodes).
    
    # Let's find the Max Clique in the "non-coprime" graph of moduli.
    
    import math
    
    moduli = list(range(2, 129))
    n_mods = len(moduli)
    
    # Build adjacency for moduli
    adj_mod = [0] * n_mods
    for i in range(n_mods):
        for j in range(i+1, n_mods):
            if math.gcd(moduli[i], moduli[j]) > 1:
                adj_mod[i] |= (1 << j)
                adj_mod[j] |= (1 << i)
                
    # Find Max Clique in this graph
    # N=127. Bitset size 2 words. Very fast.
    
    def max_clique_bitset(adj, n):
        # Tomita's algorithm
        best = 0
        best_set = []
        
        def expand(R, P):
            nonlocal best, best_set
            if not P:
                if len(R) > best:
                    best = len(R)
                    best_set = R.copy()
                return
            
            # Heuristic: pick vertex with max degree in P
            # For small n, we can just iterate
            # Or use coloring bound
            
            # Simple branch and bound with coloring
            # Order vertices in P by degree in P
            # This implementation is a simplified version
            
            # Let's use a greedy coloring to bound
            # We need to color the graph induced by P
            # Colors give an upper bound on clique size in P
            
            # For n=127, we can just use a simple recursive search with pruning
            
            # To speed up, we can use the fact that we want the MAXIMUM size.
            
            # Let's try a simple DFS with bitsets
            
            # Choose a vertex v from P
            # To improve pruning, we should order vertices.
            
            # This might be slow if done naively in Python for 127 nodes.
            # But 127 is small.
            
            # Let's use a standard Max Clique algorithm
            pass

    # Actually, we can just find the largest set of pairwise non-coprime numbers.
    # A well-known large set is all even numbers.
    # Evens in 2..128: 2, 4, ..., 128. Count = 64.
    # Can we add any odd number?
    # An odd number must be non-coprime with ALL evens.
    # gcd(odd, even) > 1 implies odd shares a factor with even.
    # If we pick odd number 3, it must be non-coprime with 2? gcd(3,2)=1. No.
    # So if 2 is in the set, no odd number can be in the set?
    # gcd(odd, 2) = 1 always.
    # So if 2 is selected, NO odd number can be selected.
    # So the set of evens (size 64) is a candidate.
    
    # What if we remove 2?
    # Then we can include odds that are non-coprime with all other selected numbers.
    # If we select all multiples of 2 except 2? i.e., 4, 6, 8...
    # And add 3?
    # 3 must be non-coprime with 4 (gcd=1). No.
    # So 3 cannot be with 4.
    
    # It seems the set of all multiples of a prime p is a clique.
    # Multiples of 2: 64 numbers.
    # Multiples of 3: 42 numbers (3, 6, ..., 126).
    # Multiples of 5: 25 numbers.
    
    # Can we mix?
    # If we take multiples of 2, we cannot take any number coprime to 2 (i.e., odd).
    # So all multiples of 2 is a maximal clique of size 64.
    
    # Is there a larger one?
    # What if we take multiples of 2 AND multiples of 3?
    # We need gcd(a, b) > 1 for all pairs.
    # If a is multiple of 2, b is multiple of 3.
    # gcd(a, b) must be > 1.
    # If a=2, b=3, gcd=1. Fail.
    # If a=4, b=3, gcd=1. Fail.
    # If a=6, b=3, gcd=3. OK.
    # So if we include 6, we can include 3.
    # But if we include 2, we cannot include 3.
    
    # So we can construct a clique by taking all numbers divisible by 2 OR 3, provided that any number divisible by 2 only (not 3) is non-coprime with any number divisible by 3 only (not 2).
    # Let A = {x | 2|x, 3!|x}
    # Let B = {x | 3|x, 2!|x}
    # Let C = {x | 6|x}
    # For a in A, b in B: gcd(a, b) must be > 1.
    # a is even, not div by 3. b is div by 3, odd.
    # gcd(a, b) must have a factor. Since a is not div by 3, the factor must be in a.
    # So a must share a factor with b.
    # If we pick a=2, b=3, gcd=1.
    # So we cannot have 2 and 3.
    # If we remove 2 from A, and 3 from B?
    # A' = {4, 8, 10, 14, ...}
    # B' = {9, 15, 21, ...}
    # gcd(4, 9)=1. Fail.
    # So it's hard to mix primes.
    
    # The largest clique is likely the set of all multiples of 2 (size 64) or multiples of 3 (size 42) or 5 (25).
    # 64 is the largest prime factor set.
    
    # So Max Moduli Set Size is likely 64.
    
    # Now, can we assign residues to these 64 moduli such that they are pairwise disjoint?
    # Moduli: 2, 4, 6, 8, ..., 128.
    # For any pair (2i, 2j), gcd is at least 2.
    # We need r_i != r_j mod gcd(2i, 2j).
    
    # This is a CSP.
    # We can use backtracking or greedy.
    # Given the density, it's likely solvable.
    
    # Let's try to construct the solution for the set of even moduli.
    # If we find a valid assignment for the 64 even moduli, the score is 64.
    # The current seed score is 1.0, which corresponds to 7 classes.
    # 64 is a huge improvement.
    
    # Let's verify if 64 is possible.
    # We need to assign r_m for m in {2, 4, ..., 128} such that r_m != r_n mod gcd(m, n).
    
    # Greedy strategy:
    # Process moduli in increasing order.
    # For each m, pick a residue r that is valid for all previous moduli.
    
    # Let's implement this.
    
    def solve_residues(mods):
        n = len(mods)
        residues = [None] * n
        
        for i in range(n):
            m = mods[i]
            found = False
            for r in range(m):
                valid = True
                for j in range(i):
                    mj = mods[j]
                    rj = residues[j]
                    g = math.gcd(m, mj)
                    if r % g == rj % g:
                        valid = False
                        break
                if valid:
                    residues[i] = r
                    found = True
                    break
            if not found:
                return None
        return residues

    # Try even moduli
    even_mods = [m for m in range(2, 129, 2)]
    res = solve_residues(even_mods)
    
    if res:
        return [[res[i], even_mods[i]] for i in range(len(even_mods))]
    else:
        # Fallback: try to find max clique of moduli and solve
        # For now, return the seed solution or a smaller valid set
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
