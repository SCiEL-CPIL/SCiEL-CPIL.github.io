import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # We want 12 distinct positive integers <= 10000.
    # Minimize the number of distinct primes dividing sums of distinct pairs.
    
    # Strategy:
    # 1. Identify a small set of primes P = {p1, p2, ..., pk}.
    # 2. Find 12 numbers such that all pairwise sums are P-smooth (only have prime factors in P).
    # 3. Start with small k and increase if no solution found.
    
    # Candidate primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    # Let's try increasing sets of primes.
    
    # Precompute smooth numbers? Or check on the fly?
    # Given the constraint N=12, pairwise sums are 66 numbers.
    # If we pick a set of primes P, we need 12 numbers x1..x12 such that xi+xj is P-smooth for all i!=j.
    
    # This is equivalent to finding a clique of size 12 in the graph G_P where vertices are 1..10000 and edge (u,v) exists if u+v is P-smooth.
    
    # Since we want to minimize k (number of primes), we iterate k=1, 2, 3, ...
    # k=1: primes={2}. Sums must be powers of 2.
    #    Can we find 12 distinct numbers <=10000 such that all pairwise sums are powers of 2?
    #    Let's analyze. If x+y=2^a, x+z=2^b.
    #    This structure is very restrictive.
    #    Example: {1, 3, 5, 11, ...}?
    #    1+3=4=2^2
    #    1+5=6 (not power of 2) -> No.
    #    Actually, if x+y=2^a and x+z=2^b, then y-z = 2^a - 2^b.
    #    Let's just search.
    
    # k=2: primes={2,3}. Sums must be of form 2^a * 3^b.
    #    These are "3-smooth" numbers (or Hamming numbers).
    #    Density is higher.
    
    # k=3: primes={2,3,5}.
    # k=4: primes={2,3,5,7}.
    
    # We will generate the set of allowed sums for a given set of primes.
    # Then we build the graph and look for a clique.
    
    # Optimization:
    # Instead of building the full graph of 10000 nodes (which is heavy),
    # we can use a heuristic search or backtracking.
    
    # However, 10000 nodes is small enough for bitsets.
    # Bitset size: 10000 bits = 1250 bytes.
    # Adjacency matrix: 10000 * 1250 bytes = 12.5 MB. This is fine.
    
    # Let's implement a function to check if a number is P-smooth.
    # And a function to find a clique of size 12.
    
    # We'll try prime sets in order of size, then lexicographically.
    # Small primes are more likely to produce smooth sums.
    
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We need to be careful with time.
    # Let's start with k=3: {2,3,5}
    # If that fails, k=4: {2,3,5,7}
    
    # To speed up clique finding, we can use a greedy heuristic or a simple backtracking with pruning.
    # Given the high density of smooth numbers (especially for 2,3,5,7), a clique of size 12 should exist.
    
    # Let's try to construct the set directly using a greedy approach first, then verify.
    # Or better, use a randomized local search / simulated annealing?
    # No, let's stick to the clique search on the compatibility graph.
    
    # Step 1: Define smoothness check for a given set of primes.
    def is_smooth(num, primes):
        if num <= 0:
            return False
        for p in primes:
            while num % p == 0:
                num //= p
        return num == 1
    
    # Step 2: For a given set of primes, build the adjacency bitsets.
    # We only care about numbers 1..10000.
    # But we can restrict the search space to numbers that are "good".
    # A number x is good if it has at least 11 other numbers y such that x+y is smooth.
    # Let's just build the full graph for the first few prime sets.
    
    MAX_VAL = 10000
    N = 12
    
    # We'll iterate over prime set sizes k from 3 upwards.
    # For each k, we try combinations of the first k primes?
    # Or just the first k primes? The smallest primes give the smoothest numbers.
    # It's likely that {2,3,5} or {2,3,5,7} will work.
    
    # Let's try k=3: [2, 3, 5]
    # Let's try k=4: [2, 3, 5, 7]
    
    # We will use a simple backtracking algorithm to find a clique of size N.
    # To make it fast, we order vertices by degree (descending).
    
    def find_clique(primes):
        # Precompute smooth sums
        # smooth_sums[s] = True if s is smooth
        # Max sum is 20000.
        max_sum = 2 * MAX_VAL
        smooth = [False] * (max_sum + 1)
        for s in range(2, max_sum + 1):
            if is_smooth(s, primes):
                smooth[s] = True
        
        # Build adjacency lists
        # adj[i] is a set of j such that i+j is smooth
        # We use 1-based indexing for values 1..MAX_VAL
        # But we only need to consider values that have enough connections.
        
        # Let's first count degrees.
        # To save time, we can restrict the candidate pool.
        # Not all numbers 1..10000 are useful.
        # Let's identify "active" nodes.
        
        # Optimization: Instead of full 10000x10000, let's pick a subset of candidates.
        # Which numbers are most likely to be in a large clique?
        # Numbers that are smooth themselves? Not necessarily.
        # Numbers that are "close" to smooth numbers?
        
        # Let's just try the full graph but with early termination.
        # If we find a clique, we return it.
        
        # Build adjacency bitsets using Python integers (bitsets)
        # adj_bits[i] is an integer where bit j is set if i and j are connected.
        adj_bits = [0] * (MAX_VAL + 1)
        degrees = [0] * (MAX_VAL + 1)
        
        # This loop is O(MAX_VAL^2) which is 10^8 operations. Too slow in Python.
        # We need a faster way.
        
        # Alternative:
        # For each pair (i, j), check if i+j is smooth.
        # We can iterate i from 1 to MAX_VAL, and for each i, iterate j from i+1 to MAX_VAL.
        # That's still 50 million pairs.
        # Checking is_smooth is fast, but 50M * cost(is_smooth) might be too slow.
        
        # Let's optimize is_smooth.
        # For fixed primes, we can precompute a table of smooth numbers up to 20000.
        # Then the check is just a lookup.
        
        # Precompute smooth numbers set
        smooth_set = set()
        for s in range(2, max_sum + 1):
            if is_smooth(s, primes):
                smooth_set.add(s)
        
        # Now build adjacency.
        # We can iterate i, and for each possible sum s in smooth_set,
        # j = s - i.
        # If 1 <= j <= MAX_VAL and j != i, then i and j are connected.
        
        # This is O(MAX_VAL * |smooth_set|).
        # |smooth_set| for {2,3,5} up to 20000:
        # 2^a * 3^b * 5^c <= 20000.
        # a <= 14 (2^14=16384)
        # b <= 9 (3^9=19683)
        # c <= 5 (5^5=3125)
        # Roughly 15*10*6 = 900 numbers.
        # So 10000 * 900 = 9,000,000 operations. This is feasible.
        
        adj_bits = [0] * (MAX_VAL + 1)
        degrees = [0] * (MAX_VAL + 1)
        
        for i in range(1, MAX_VAL + 1):
            for s in smooth_set:
                j = s - i
                if 1 <= j <= MAX_VAL and j != i:
                    # Set bit j in adj_bits[i]
                    adj_bits[i] |= (1 << j)
                    degrees[i] += 1
        
        # Now we have the graph.
        # We want to find a clique of size N=12.
        # Use backtracking.
        # Order vertices by degree descending.
        candidates = list(range(1, MAX_VAL + 1))
        candidates.sort(key=lambda x: -degrees[x])
        
        # We can prune candidates with degree < N-1.
        candidates = [x for x in candidates if degrees[x] >= N - 1]
        
        if not candidates:
            return None
        
        # Backtracking search
        # We maintain a list of selected vertices.
        # And a set of common neighbors.
        
        # To speed up, we can use a recursive function.
        # But Python recursion limit might be an issue? Depth is 12, so it's fine.
        
        # Let's use a helper function.
        # common_neighbors is a bitset (integer) of vertices that are connected to all selected vertices.
        
        def backtrack(selected, common_bits):
            if len(selected) == N:
                return selected[:]
            
            # Get candidates from common_bits that are also in our candidate list
            # We need to iterate through the bits in common_bits.
            
            # Extract candidates from bitset
            # We only care about vertices in `candidates` list.
            # To make this faster, we can map vertex value to index in candidates list?
            # Or just iterate through candidates and check if bit is set.
            
            # Since candidates is sorted by degree, we try high degree first.
            
            for v in candidates:
                if len(selected) > 0 and not (common_bits & (1 << v)):
                    continue
                
                # If we reach here, v is a valid candidate.
                
                # Check if v is connected to all selected vertices.
                # This is already handled by common_bits if we update it correctly.
                # Initially common_bits is all 1s (for the first vertex).
                # When we add a vertex u, new_common = common_bits & adj_bits[u].
                
                # But wait, the loop above checks `if not (common_bits & (1 << v))`.
                # This means v must be in the current common neighborhood.
                
                # Add v to selected
                new_selected = selected + [v]
                
                # Update common bits
                new_common = common_bits & adj_bits[v]
                
                # Recurse
                result = backtrack(new_selected, new_common)
                if result:
                    return result
                
                # Backtrack
                # No need to undo selected as we passed by value (list copy)
            
            return None
        
        # Initial common bits: all bits set for vertices in candidates?
        # Actually, for the first vertex, any vertex is a candidate.
        # So common_bits should have bits set for all vertices in `candidates`.
        
        init_common = 0
        for v in candidates:
            init_common |= (1 << v)
        
        return backtrack([], init_common)
    
    # Try prime sets
    # We start with small sets of small primes.
    
    # k=3: [2, 3, 5]
    # k=4: [2, 3, 5, 7]
    # k=5: [2, 3, 5, 7, 11]
    
    # Let's try them in order.
    
    prime_sets_to_try = [
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
    ]
    
    # Also consider permutations? No, the set of primes is what matters.
    # But maybe [2, 3, 7] is better than [2, 3, 5]? Unlikely, 5 is smaller.
    # So we try the first k primes.
    
    for primes in prime_sets_to_try:
        if time.time() > deadline:
            break
            
        # We need to be careful: find_clique might take a long time if it fails.
        # We should add a timeout or step limit to the backtracking.
        
        # Let's modify backtrack to have a step limit.
        
        steps = [0]
        step_limit = 100000  # Adjust this based on performance
        
        def backtrack_limited(selected, common_bits):
            if time.time() > deadline:
                return None
            if steps[0] > step_limit:
                return None
            steps[0] += 1
            
            if len(selected) == N:
                return selected[:]
            
            for v in candidates:
                if len(selected) > 0 and not (common_bits & (1 << v)):
                    continue
                
                new_selected = selected + [v]
                new_common = common_bits & adj_bits[v]
                
                result = backtrack_limited(new_selected, new_common)
                if result:
                    return result
            
            return None
        
        # Re-run the clique finding with limits
        # We need to rebuild the graph inside the loop or pass it.
        # Let's refactor find_clique to accept a timeout.
        
        # Actually, the code above builds the graph inside find_clique.
        # Let's just call a modified version.
        
        # To avoid duplicating code, let's just run the search and hope it's fast.
        # If it's too slow, we break and try next prime set.
        
        # Let's wrap the whole thing in a try/except or time check.
        
        # Re-implementing the search part here to use the time limit.
        
        # 1. Build smooth set
        max_sum = 2 * MAX_VAL
        smooth_set = set()
        for s in range(2, max_sum + 1):
            if is_smooth(s, primes):
                smooth_set.add(s)
        
        # 2. Build adjacency
        adj_bits = [0] * (MAX_VAL + 1)
        degrees = [0] * (MAX_VAL + 1)
        
        for i in range(1, MAX_VAL + 1):
            for s in smooth_set:
                j = s - i
                if 1 <= j <= MAX_VAL and j != i:
                    adj_bits[i] |= (1 << j)
                    degrees[i] += 1
        
        # 3. Filter candidates
        candidates = list(range(1, MAX_VAL + 1))
        candidates.sort(key=lambda x: -degrees[x])
        candidates = [x for x in candidates if degrees[x] >= N - 1]
        
        if not candidates:
            continue
        
        init_common = 0
        for v in candidates:
            init_common |= (1 << v)
        
        steps = [0]
        
        def backtrack_limited(selected, common_bits):
            if time.time() > deadline:
                return None
            if steps[0] > 200000:
                return None
            steps[0] += 1
            
            if len(selected) == N:
                return selected[:]
            
            for v in candidates:
                if len(selected) > 0 and not (common_bits & (1 << v)):
                    continue
                
                new_selected = selected + [v]
                new_common = common_bits & adj_bits[v]
                
                result = backtrack_limited(new_selected, new_common)
                if result:
                    return result
            
            return None
        
        result = backtrack_limited([], init_common)
        if result:
            return result
    
    # If no solution found with these prime sets, return a fallback.
    # The fallback should be valid (distinct, <=10000).
    # We can return the first 12 integers.
    return list(range(1, 13))
# EVOLVE-BLOCK-END
