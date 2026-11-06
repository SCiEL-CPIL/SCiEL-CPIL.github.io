import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute Smallest Prime Factor (SPF) up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    def get_prime_factors(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def evaluate_set(nums):
        """Calculate the number of distinct prime factors of all pairwise sums."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(get_prime_factors(nums[i] + nums[j]))
        return len(all_primes)

    # Strategy 1: Exact Clique Search for Prime Set {2, 3}
    # We look for 12 numbers x_i such that x_i + x_j is only divisible by 2 and 3.
    # This implies x_i + x_j = 2^a * 3^b.
    # If we set x_i = 6*k_i + 3, then x_i + x_j = 6(k_i + k_j) + 6 = 6(k_i + k_j + 1).
    # We need k_i + k_j + 1 to be of the form 2^a * 3^b.
    
    max_k = 10000 // 6
    # Generate all 2^a * 3^b <= 2*max_k + 1
    smooth_sums = set()
    for a in range(15):
        pow2 = 1 << a
        if pow2 > 2 * max_k + 1:
            break
        for b in range(15):
            val = pow2 * (3 ** b)
            if val > 2 * max_k + 1:
                break
            smooth_sums.add(val)
            
    # k_i must be such that x_i <= 10000 => 6*k_i + 3 <= 10000 => k_i <= 1666
    # Also k_i >= 0.
    valid_ks = list(range(0, 1667))
    
    # Build graph: edge between k_i, k_j if k_i + k_j + 1 in smooth_sums
    # We want a clique of size 12.
    # This is a large graph, so use a heuristic clique finder or backtracking with pruning.
    
    # Optimization: Sort valid_ks by "degree" in the potential graph?
    # Or just try to find a clique in the subgraph of small k's first, as sums are smaller.
    
    # Let's try a randomized backtracking search for the clique.
    # To make it fast, we only consider k in [0, 500] first, then expand if needed.
    # Actually, the density of 2-3-smooth numbers is low.
    # Number of 2-3-smooth numbers up to 3333 is roughly (log 3333)^2 / (2 log 2 log 3) ~ 100.
    # So the graph is sparse.
    
    # Let's generate candidate k's that have high potential.
    # A k is good if k + 1 + k' is smooth for many k'.
    
    # Simpler: Just use the local search. The clique search might fail or be slow.
    # But if we find a clique, score is 2.0 (optimal).
    
    found_clique = False
    best_clique = None
    
    # Heuristic Clique Search
    # Pick a random k, try to extend.
    for _ in range(100):
        if time.time() > deadline:
            break
        # Start with a random k
        start_k = random.randint(0, 500)
        clique = [start_k]
        # Try to add 11 more
        for _ in range(11):
            # Find a k' not in clique such that k' + k + 1 is smooth for all k in clique
            candidates = []
            for k_prime in range(0, 1667):
                if k_prime in clique:
                    continue
                ok = True
                for k in clique:
                    if (k + k_prime + 1) not in smooth_sums:
                        ok = False
                        break
                if ok:
                    candidates.append(k_prime)
            
            if not candidates:
                break
            # Pick random from candidates to diversify
            next_k = random.choice(candidates)
            clique.append(next_k)
            
        if len(clique) == 12:
            found_clique = True
            # Convert to x
            best_clique = [6 * k + 3 for k in clique]
            break

    if found_clique:
        # Verify
        if len(set(best_clique)) == 12 and all(1 <= x <= 10000 for x in best_clique):
            if evaluate_set(best_clique) == 2:
                return best_clique

    # Strategy 2: Local Search to minimize prime count
    # Start with a good initial guess
    # Guess 1: Multiples of 6
    init1 = [6 * i for i in range(1, 13)]
    # Guess 2: 6k+3
    init2 = [6 * i + 3 for i in range(0, 12)]
    # Guess 3: 6k+1
    init3 = [6 * i + 1 for i in range(0, 12)]
    # Guess 4: Random
    init4 = random.sample(range(1, 10001), 12)
    
    best_set = None
    best_score = float('inf')
    
    for init in [init1, init2, init3, init4]:
        curr_set = init.copy()
        curr_score = evaluate_set(curr_set)
        
        if curr_score < best_score:
            best_score = curr_score
            best_set = curr_set.copy()
            if best_score <= 2:
                return best_set

        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Pick a random index to change
            idx = random.randint(0, 11)
            old_val = curr_set[idx]
            
            # Generate candidates for new value
            # 1. Neighbors
            # 2. Random
            # 3. Smooth-based
            candidates = set()
            candidates.add(old_val + random.randint(-10, 10))
            candidates.add(random.randint(1, 10000))
            
            # Bias towards numbers that make sums smooth with existing neighbors
            # For each neighbor j, we want x_i + x_j to have few primes.
            # This is hard to target directly without knowing the target prime set.
            # Just random and local steps usually suffice.
            
            # Add some "structured" candidates
            # Try numbers of form 6k+r
            for r in [0, 1, 2, 3, 4, 5]:
                candidates.add(random.randint(1, 1666) * 6 + r)
                
            for new_val in candidates:
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in curr_set:
                    continue
                
                # Calculate new score efficiently?
                # For n=12, full evaluation is 66 sums.
                # Factoring 66 numbers up to 20000 is very fast with SPF.
                # 66 * ~10 ops = 660 ops. Very fast.
                # So we can just do full evaluation.
                
                new_set = curr_set.copy()
                new_set[idx] = new_val
                new_score = evaluate_set(new_set)
                
                if new_score < curr_score:
                    curr_score = new_score
                    curr_set = new_set
                    improved = True
                    if curr_score < best_score:
                        best_score = curr_score
                        best_set = curr_set.copy()
                        if best_score <= 2:
                            return best_set
                    # Break early if we found a very good one? No, keep improving.
                    # But we should restart the "improved" loop to pick a new index
                    break # Restart the while loop to pick a new index
    
    return best_set
# EVOLVE-BLOCK-END
