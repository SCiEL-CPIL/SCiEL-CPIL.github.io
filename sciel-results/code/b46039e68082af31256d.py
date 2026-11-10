import time
import itertools

# Global cache for prime factors
# We only need to check smoothness, so we can just divide.

def is_smooth(n, primes):
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Candidate prime sets to try, ordered by size and "smallness"
    # We want to minimize the count of distinct primes.
    # Start with smallest sets.
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_set = None
    best_prime_count = 100
    
    # Try sets of primes of increasing size
    # Size 2: {2, 3}, {2, 5}, {2, 7}...
    # Size 3: {2, 3, 5}, ...
    
    # It's highly likely that 2 is involved in the sums.
    # If we scale by 2^k, 2 is always a factor.
    # So we focus on the other primes.
    
    # Strategy:
    # 1. Try P = {2, 3}
    # 2. Try P = {2, 3, 5}
    # 3. Try P = {2, 3, 5, 7}
    # ...
    
    # To find the set of integers, we search for a clique in the graph G_P
    # where vertices are integers 1..L and edges connect u,v if u+v is P-smooth.
    # We want a clique of size 12.
    
    L = 200  # Search space for x_i before scaling
    
    # Precompute smoothness for all pairs (u,v) in 1..L for a given P?
    # That's L^2 checks. 200^2 = 40000. Fast enough.
    
    def find_clique(P):
        # P is a list of primes
        # Build adjacency list or matrix
        # We just need to find 12 distinct nodes that are mutually connected.
        
        # Greedy heuristic:
        # Sort nodes by degree (number of smooth sums)
        # Pick highest degree, then recurse.
        
        # Simpler: Backtracking with ordering.
        
        # Let's precompute the list of valid partners for each i
        valid_partners = [[] for _ in range(L+1)]
        for i in range(1, L+1):
            for j in range(i+1, L+1):
                if is_smooth(i+j, P):
                    valid_partners[i].append(j)
                    valid_partners[j].append(i)
        
        # Sort nodes by degree descending to guide search
        nodes = list(range(1, L+1))
        nodes.sort(key=lambda x: len(valid_partners[x]), reverse=True)
        
        clique = []
        
        def backtrack(start_idx, current_clique):
            if len(current_clique) == 12:
                return current_clique[:]
            
            # Pruning: if not enough nodes left, return
            if len(current_clique) + (len(nodes) - start_idx) < 12:
                return None
            
            for k in range(start_idx, len(nodes)):
                node = nodes[k]
                
                # Check if node is connected to all in current_clique
                connected = True
                for c in current_clique:
                    if node not in valid_partners[c]:
                        connected = False
                        break
                
                if connected:
                    res = backtrack(k+1, current_clique + [node])
                    if res:
                        return res
            return None
        
        return backtrack(0, [])

    # Try prime sets
    # We prioritize sets containing 2, as scaling makes 2 "free".
    # But the set of primes in the sum is what matters.
    # If we use P={2,3}, the sum has primes {2,3}. Count=2.
    
    candidate_sets = []
    
    # Size 2
    for p in base_primes[1:]:
        candidate_sets.append([2, p])
        
    # Size 3
    for p1 in base_primes[1:]:
        for p2 in base_primes[1:]:
            if p1 < p2:
                candidate_sets.append([2, p1, p2])
                
    # Size 4
    for p1 in base_primes[1:]:
        for p2 in base_primes[1:]:
            if p1 >= p2: continue
            for p3 in base_primes[1:]:
                if p2 >= p3: continue
                candidate_sets.append([2, p1, p2, p3])
                
    # Sort by size, then by sum of primes (smaller primes usually allow more solutions)
    candidate_sets.sort(key=lambda x: (len(x), sum(x)))
    
    for P in candidate_sets:
        if time.time() > deadline:
            break
            
        # Find clique
        clique = find_clique(P)
        
        if clique:
            # We found 12 distinct x_i
            # Scale them to fit <= 10000
            max_x = max(clique)
            # a_i = x_i * 2^k
            # We want max_x * 2^k <= 10000
            # Also a_i must be distinct (guaranteed if x_i distinct)
            # And a_i >= 1
            
            # Choose k such that max_x * 2^k <= 10000
            # To be safe and maybe "nice", we can just pick the largest k possible?
            # Or any k. The prime factors of sums are {2} U primes(x_i+x_j).
            # Since x_i+x_j is P-smooth, primes(x_i+x_j) subset P.
            # So total primes = {2} U P.
            # Note: if 2 is in P, it's just P.
            
            k = 0
            while max_x * (2 ** (k+1)) <= 10000:
                k += 1
                
            scale = 2 ** k
            a_set = [x * scale for x in clique]
            
            # Verify distinctness and bounds
            if len(set(a_set)) == 12 and all(1 <= a <= 10000 for a in a_set):
                # Calculate actual prime count to be sure
                # (Should be len(P) if 2 in P, else len(P)+1)
                # But let's just return it.
                # The score is the number of distinct primes.
                # We found a solution for P.
                # The number of primes is len(set(P) | {2})?
                # Wait, if P={3,5}, and we scale by 2, sums are 2*(3-smooth).
                # Primes are {2, 3, 5}.
                # So we should always include 2 in our consideration if we scale by 2.
                # My candidate sets all include 2.
                
                best_set = a_set
                best_prime_count = len(P)
                
                # Since we sorted by size, the first valid set has the minimum size.
                # We can break if we want the absolute minimum.
                # The task says "minimize".
                # So we should return the first one found in the sorted order.
                break

    if best_set:
        return best_set
    
    # Fallback: Random or simple construction
    # If no small prime set worked, use a heuristic random search
    # or return a simple set like range(1,13) which has many primes.
    
    # Simple fallback: Powers of 2?
    # 1, 2, 4, 8... sums have many primes.
    # Let's just return range(1,13) to avoid None.
    return list(range(1, 13))

# EVOLVE-BLOCK-END
