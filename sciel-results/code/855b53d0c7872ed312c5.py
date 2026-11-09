import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    max_val = 10000
    target_size = 12
    
    def get_primes(n):
        """Return the set of prime factors of n."""
        if n == 1:
            return set()
        primes = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                primes.add(d)
                n //= d
            d += 1
        if n > 1:
            primes.add(n)
        return primes

    def evaluate(set_a):
        """Calculate the union of prime factors of all pairwise sums."""
        all_primes = set()
        n = len(set_a)
        for i in range(n):
            for j in range(i + 1, n):
                s = set_a[i] + set_a[j]
                all_primes.update(get_primes(s))
        return all_primes

    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

    def find_clique_bitset(primes, time_limit=10.0):
        """
        Finds a clique of size target_size in the graph where nodes are 1..max_val
        and edges exist if sum is smooth w.r.t 'primes'.
        Uses bitsets for fast intersection.
        """
        search_start = time.time()
        
        # 1. Identify smooth sums
        smooth_sums = set()
        for s in range(2, 2 * max_val + 1):
            if is_smooth(s, primes):
                smooth_sums.add(s)
        
        if len(smooth_sums) < target_size * (target_size - 1) / 2:
            # Not enough edges potentially, though this bound is loose
            pass

        # 2. Build neighbor bitsets
        # neighbors[i] is a bitmask where bit j is 1 if i and j are connected
        # We only need to consider j > i to avoid symmetry issues in the search, 
        # but for the bitset intersection in Bron-Kerbosch, having the full neighborhood 
        # is easier. We will enforce ordering in the search.
        
        neighbors = [0] * (max_val + 1)
        
        # Efficient construction:
        # For each s in smooth_sums:
        #   i ranges from max(1, s-max_val) to min(max_val, s//2)
        #   j = s - i
        #   Set bit j in neighbors[i] and bit i in neighbors[j]
        
        for s in smooth_sums:
            min_i = max(1, s - max_val)
            max_i = min(max_val, s // 2)
            for i in range(min_i, max_i + 1):
                j = s - i
                if i != j:
                    neighbors[i] |= (1 << j)
                    neighbors[j] |= (1 << i)
        
        # 3. Initial candidates: all nodes 1..max_val
        # Bitmask with bits 1 to max_val set
        all_nodes_mask = (1 << (max_val + 1)) - 1
        # Clear bit 0
        all_nodes_mask &= ~(1 << 0)
        
        # We want to find a clique of size 12.
        # To prune, we can order nodes by degree.
        # However, bitset operations are fast. Let's try a simple recursive search.
        
        # Optimization: Only consider nodes that have degree >= target_size - 1
        # Actually, for a clique of size K, every node must have at least K-1 neighbors 
        # within the final clique, so global degree must be >= K-1.
        
        valid_nodes_mask = 0
        for i in range(1, max_val + 1):
            if bin(neighbors[i]).count('1') >= target_size - 1:
                valid_nodes_mask |= (1 << i)
        
        if valid_nodes_mask == 0:
            return None
            
        def backtrack(clique, candidates):
            """
            clique: list of integers
            candidates: bitmask of integers that can be added to clique
            """
            if time.time() - search_start > time_limit:
                return None
                
            if len(clique) == target_size:
                return clique[:]
                
            # If not enough candidates left
            if bin(candidates).count('1') < target_size - len(clique):
                return None
                
            # Extract a node from candidates
            # To improve heuristic, pick a node with high degree within candidates?
            # Simple approach: pick the lowest set bit, or random?
            # Let's pick a node that maximizes the potential. 
            # For now, just iterate through set bits.
            
            # To avoid trying all permutations, we rely on the fact that we remove 
            # nodes from candidates as we go.
            
            # Get list of nodes in candidates
            # This conversion is O(N) but N=10000 is small.
            # However, doing this at every recursion step might be slow.
            # Alternative: use a while loop on the bitmask.
            
            c = candidates
            while c:
                if time.time() - search_start > time_limit:
                    return None
                
                # Get lowest set bit
                lsb = c & -c
                v = lsb.bit_length() - 1
                
                # Remove v from candidates for subsequent iterations
                c ^= lsb
                
                # New candidates must be in current candidates AND neighbors of v
                new_candidates = candidates & neighbors[v]
                
                # Note: We should probably exclude nodes <= v to maintain order?
                # The standard Bron-Kerbosch with pivot doesn't strictly require ordering 
                # if we just remove v from the candidate pool. 
                # However, to avoid finding the same clique in different orders, 
                # we typically enforce an ordering. 
                # If we don't enforce ordering, we might find {1,2,3} and {3,2,1}.
                # Since we return the first one found, it doesn't matter for correctness, 
                # but it might waste time.
                # Let's enforce v > max(clique) if clique is not empty?
                # Actually, the bitset `candidates` already contains only nodes not yet in clique.
                # If we just pick any node, we cover all combinations.
                
                res = backtrack(clique + [v], new_candidates)
                if res:
                    return res
                    
            return None

        # Start search
        # To speed up, we can try starting from high-degree nodes.
        # But `backtrack` handles the branching.
        # Let's just call it.
        
        res = backtrack([], valid_nodes_mask)
        
        if res:
            # Verify distinctness and range
            if len(set(res)) == target_size and all(1 <= x <= max_val for x in res):
                return res
        return None

    best_set = None
    best_prime_count = float('inf')
    
    # Fallback: Generate a random set to ensure we always have a valid answer
    random.seed(42)
    fallback_set = random.sample(range(1, max_val + 1), target_size)
    fallback_primes = evaluate(fallback_set)
    best_set = fallback_set
    best_prime_count = len(fallback_primes)
    
    # Try prime sets in order of increasing size
    # Based on inspiration, {2,3,5,7} should work.
    prime_sets_to_try = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13]
    ]
    
    for primes in prime_sets_to_try:
        if time.time() > deadline - 5:
            break
            
        # Try to find a clique for this prime set
        # We use the bitset approach which is much faster than the current program's greedy.
        # The current program uses a greedy approach with random restarts, which can be slow and get stuck.
        # The bitset approach uses Bron-Kerbosch with bitset intersection, which is efficient for finding cliques.
        
        # We need to be careful about the time limit for the clique search.
        # If the prime set is small, the graph is dense, and finding a clique should be fast.
        # If the prime set is large, the graph is sparse, and finding a clique might be harder.
        
        # Let's allocate more time to smaller prime sets.
        time_alloc = 15.0 if len(primes) <= 3 else 10.0
        
        clique = find_clique_bitset(primes, time_alloc)
        
        if clique:
            # The clique consists of numbers k_i such that k_i + k_j is smooth w.r.t primes.
            # We can construct a_i = k_i * M, where M is the product of primes.
            # This ensures that a_i + a_j = M * (k_i + k_j), so the prime factors of a_i + a_j
            # are a subset of primes ∪ {prime factors of k_i + k_j}.
            # Since k_i + k_j is smooth w.r.t primes, the prime factors of a_i + a_j are a subset of primes.
            
            M = 1
            for p in primes:
                M *= p
                
            # Check if M * max(k_i) <= max_val
            if M * max(clique) > max_val:
                # Scale down? No, we can't scale down without breaking the smoothness.
                # We can try a different set of k_i.
                # Or we can try a smaller prime set.
                # For now, skip this prime set.
                continue
                
            a_set = [k * M for k in clique]
            
            # Verify distinctness and bounds
            if len(set(a_set)) == target_size and all(x <= max_val for x in a_set):
                primes_found = evaluate(a_set)
                if len(primes_found) < best_prime_count:
                    best_prime_count = len(primes_found)
                    best_set = a_set
                    # If we found a set with the same number of primes as the prime set, we can't do better for this prime set.
                    # But we should still try smaller prime sets to see if we can do even better.
                    # However, the loop is ordered by increasing size, so we are already trying smaller sets first.
                    # So if we found a set for [2,3], we can break.
                    if len(primes_found) == len(primes):
                        # We found a set with exactly the primes in the prime set.
                        # We can't do better than this for this prime set.
                        # But we should check if there's a smaller prime set that works.
                        # Since we are iterating in order, if this is the first prime set, we can break.
                        # If this is a later prime set, we have already tried smaller ones and they didn't work.
                        # So we can break.
                        break
    
    return best_set

# Test
if __name__ == "__main__":
    result = construct()
    print("Result:", result)
    print("Primes:", evaluate(result))
    print("Count:", len(evaluate(result)))
