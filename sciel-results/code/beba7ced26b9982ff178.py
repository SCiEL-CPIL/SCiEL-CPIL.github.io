"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # 90s limit, use 45s for safety
    deadline = start_time + 45.0
    
    # Helper to get prime factors
    def prime_factors(n):
        factors = set()
        if n <= 1:
            return factors
        if n % 2 == 0:
            factors.add(2)
            while n % 2 == 0:
                n //= 2
        p = 3
        while p * p <= n:
            if n % p == 0:
                factors.add(p)
                while n % p == 0:
                    n //= p
            p += 2
        if n > 1:
            factors.add(n)
        return factors

    def evaluate_set(nums):
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(prime_factors(nums[i] + nums[j]))
        return all_primes

    def score_of(nums):
        return len(evaluate_set(nums))

    # Strategy 1: Try to find 12 numbers where all pairwise sums have primes in a small set
    # Try prime sets: {2}, {2,3}, {2,3,5}, {2,3,5,7}, etc.
    # For {2}: need all sums to be powers of 2. x_i + x_j = 2^k
    # This means all x_i must be of form 2^a - 2^b? Actually if x_i + x_j is always power of 2,
    # then x_i must all be equal (since x_1+x_2 = x_1+x_3 => x_2=x_3). So {2} alone is impossible for distinct numbers.
    
    # For {2,3}: x_i + x_j must be of form 2^a * 3^b
    # Let x_i = 6 * k_i. Then x_i + x_j = 6(k_i + k_j), so k_i + k_j must be of form 2^a * 3^b.
    # Find 12 distinct k_i in [1, 10000//6] such that all pairwise sums are of form 2^a * 3^b.
    
    max_x = 10000
    max_k = max_x // 6
    
    # Generate all numbers of form 2^a * 3^b <= 2*max_k (since k_i + k_j <= 2*max_k)
    valid_sums = set()
    for a in range(0, 20):
        val = 1 << a
        if val > 2 * max_k:
            break
        for b in range(0, 15):
            v = val * (3 ** b)
            if v > 2 * max_k:
                break
            valid_sums.add(v)
    
    # Find k values such that k is of form 2^a * 3^b and k + k' is also of form 2^a * 3^b
    # Actually k itself doesn't need to be of form 2^a*3^b, only k_i + k_j needs to be.
    # But we need to find 12 k_i such that all pairwise sums are in valid_sums.
    
    # Let's try a different approach: search for k values directly
    # k_i must be in [1, max_k]
    # We want k_i + k_j in valid_sums for all i != j
    
    # Try small k values first
    k_candidates = list(range(1, min(max_k, 2000) + 1))
    
    # Build adjacency: i and j are compatible if k_i + k_j in valid_sums
    n_k = len(k_candidates)
    adj = [set() for _ in range(n_k)]
    for i in range(n_k):
        ki = k_candidates[i]
        for j in range(i + 1, n_k):
            kj = k_candidates[j]
            if ki + kj in valid_sums:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find clique of size 12
    clique = []
    found = False
    
    def backtrack_clique(cands):
        nonlocal found, clique
        if time.time() > deadline:
            return
        if len(clique) == 12:
            found = True
            return
        if len(clique) + len(cands) < 12:
            return
        if not cands:
            return
        
        for idx, v in enumerate(cands):
            clique.append(v)
            new_cands = [u for u in cands[idx+1:] if u in adj[v]]
            backtrack_clique(new_cands)
            if found:
                return
            clique.pop()
    
    # Filter: only nodes with degree >= 11
    init_cands = [i for i in range(n_k) if len(adj[i]) >= 11]
    if init_cands:
        backtrack_clique(init_cands)
    
    if found:
        result = [6 * k_candidates[k] for k in clique]
        if len(set(result)) == 12 and all(1 <= x <= max_x for x in result):
            return result
    
    # Strategy 2: Try prime set {2, 3, 5}
    # x_i + x_j must be of form 2^a * 3^b * 5^c
    valid_sums_235 = set()
    for a in range(0, 20):
        val_a = 1 << a
        if val_a > 2 * max_x:
            break
        for b in range(0, 12):
            val_ab = val_a * (3 ** b)
            if val_ab > 2 * max_x:
                break
            for c in range(0, 10):
                v = val_ab * (5 ** c)
                if v > 2 * max_x:
                    break
                valid_sums_235.add(v)
    
    # Find 12 numbers x_i in [1, max_x] such that all pairwise sums are in valid_sums_235
    # Try x_i = 30 * k_i (multiple of 2,3,5)
    max_k30 = max_x // 30
    k_candidates30 = list(range(1, min(max_k30, 500) + 1))
    n_k30 = len(k_candidates30)
    adj30 = [set() for _ in range(n_k30)]
    for i in range(n_k30):
        ki = k_candidates30[i]
        for j in range(i + 1, n_k30):
            kj = k_candidates30[j]
            # x_i + x_j = 30*(ki+kj), need 30*(ki+kj) in valid_sums_235
            # 30 = 2*3*5, so 30*(ki+kj) has primes {2,3,5} iff ki+kj has primes subset of {2,3,5}
            # But we already check if 30*(ki+kj) is in valid_sums_235
            if 30 * (ki + kj) in valid_sums_235:
                adj30[i].add(j)
                adj30[j].add(i)
    
    clique30 = []
    found30 = False
    
    def backtrack_clique30(cands):
        nonlocal found30, clique30
        if time.time() > deadline:
            return
        if len(clique30) == 12:
            found30 = True
            return
        if len(clique30) + len(cands) < 12:
            return
        if not cands:
            return
        
        for idx, v in enumerate(cands):
            clique30.append(v)
            new_cands = [u for u in cands[idx+1:] if u in adj30[v]]
            backtrack_clique30(new_cands)
            if found30:
                return
            clique30.pop()
    
    init_cands30 = [i for i in range(n_k30) if len(adj30[i]) >= 11]
    if init_cands30:
        backtrack_clique30(init_cands30)
    
    if found30:
        result = [30 * k_candidates30[k] for k in clique30]
        if len(set(result)) == 12 and all(1 <= x <= max_x for x in result):
            return result
    
    # Strategy 3: Local search to minimize prime count
    best_score = float('inf')
    best_set = list(range(1, 13))
    best_score = score_of(best_set)
    
    def try_local_search(start_set, max_iters=5000):
        nonlocal best_score, best_set
        curr_set = start_set.copy()
        curr_score = score_of(curr_set)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if curr_score <= 3:
                return curr_set
            
            # Pick random index
            idx = random.randint(0, 11)
            old_val = curr_set[idx]
            
            # Generate candidates
            cands = []
            # Random in range
            cands.append(random.randint(1, max_x))
            # Near old value
            for delta in random.sample(range(-50, 51), min(10, 101)):
                v = old_val + delta
                if 1 <= v <= max_x:
                    cands.append(v)
            # Multiples of small numbers
            for base in [2, 3, 5, 6, 10, 15, 30]:
                for _ in range(3):
                    cands.append(base * random.randint(1, max_x // base))
            
            improved = False
            for new_val in cands:
                if new_val < 1 or new_val > max_x:
                    continue
                if new_val in curr_set:
                    continue
                
                new_set = curr_set.copy()
                new_set[idx] = new_val
                new_score = score_of(new_set)
                
                if new_score < curr_score:
                    curr_score = new_score
                    curr_set = new_set
                    improved = True
                    if curr_score < best_score:
                        best_score = curr_score
                        best_set = curr_set.copy()
                    if curr_score <= 3:
                        return curr_set
                    break
            
            if not improved:
                # Try a different index
                pass
        
        if curr_score < best_score:
            best_score = curr_score
            best_set = curr_set.copy()
        return None
    
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if random.random() < 0.3:
            start = random.sample(range(1, 100), 12)
        elif random.random() < 0.6:
            start = random.sample(range(1, 500), 12)
        else:
            start = random.sample(range(1, max_x + 1), 12)
        
        result = try_local_search(start, max_iters=3000)
        if result is not None:
            return result
    
    return best_set
# EVOLVE-BLOCK-END
