import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 192
    MAX_SIZE = 48
    all_candidates = list(range(1, N + 1))
    
    # Pre-compute combinations of indices? No, S changes.
    # But we can pre-compute triples of values? Too many.
    
    def is_valid_full(S):
        """Full validity check for a set S. Used for final verification."""
        n = len(S)
        S_list = list(S)
        for i in range(n):
            x = S_list[i]
            # x must not divide sum of any 3 distinct OTHER elements
            # Iterate over combinations of 3 from S excluding x
            others = [v for v in S_list if v != x]
            for a, b, c in combinations(others, 3):
                if (a + b + c) % x == 0:
                    return False
        return True

    def can_add(x, S):
        """
        Check if x can be added to set S (list of distinct ints).
        Returns True if valid.
        """
        n = len(S)
        
        # 1. Check if x divides sum of any 3 distinct others in S
        if n >= 3:
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. Check if any a in S divides x + u + v for distinct u, v in S \ {a}
        # This is the expensive part.
        for a in S:
            # Find two largest elements in S excluding a to bound the sum
            # If x + max1 + max2 < a, then x + u + v < a for all u,v, so (x+u+v)%a != 0 (since sum > 0)
            # Actually, if x+u+v < a, then (x+u+v) % a = x+u+v != 0. So it's safe.
            # We only need to check if x+u+v is a multiple of a.
            
            # Get two largest elements in S excluding a
            # To be safe and fast, we can just iterate pairs, but with pruning.
            others = [v for v in S if v != a]
            if len(others) < 2:
                continue
            
            # Sort others descending to find max2 quickly? 
            # Or just find max1, max2.
            m1 = 0
            m2 = 0
            for v in others:
                if v > m1:
                    m2 = m1
                    m1 = v
                elif v > m2:
                    m2 = v
            
            # Max possible sum with x is x + m1 + m2
            if x + m1 + m2 < a:
                continue
            
            # Check all pairs u, v in others
            # Optimization: If (x + u + v) % a == 0, return False
            # We can iterate u, then v
            # Note: u and v must be distinct
            for i in range(len(others)):
                u = others[i]
                for j in range(i + 1, len(others)):
                    v = others[j]
                    if (x + u + v) % a == 0:
                        return False
        return True

    def run_greedy(order):
        S = []
        for x in order:
            if can_add(x, S):
                S.append(x)
        return S

    def local_search(S, rng, time_limit=2.0):
        """
        Improve set S using local search (additions, swaps).
        """
        S = list(S)
        Sset = set(S)
        best_S = S
        best_size = len(S)
        start_ls = time.time()
        
        improved = True
        while improved and (time.time() - start_ls) < time_limit:
            improved = False
            
            # 1. Try to add any element
            for x in all_candidates:
                if x in Sset:
                    continue
                if can_add(x, S):
                    S.append(x)
                    Sset.add(x)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try swaps: remove a, add x
            # To avoid getting stuck in a specific local optimum, try a few random swaps
            # or systematic ones.
            # We try to find a swap that results in a set that can potentially grow more,
            # but simply swapping to a different valid set of same size can help escape.
            
            # Let's try swapping elements that are "small" or "causing conflicts"
            # Heuristic: Try removing each element a in S, and see if we can add a larger x?
            # Or just random swaps.
            
            # Given time constraints, random swaps with a bias towards larger numbers might be good.
            # Or: Try to replace a with x where x > a.
            
            candidates_to_remove = list(S)
            rng.shuffle(candidates_to_remove)
            
            for a in candidates_to_remove:
                if (time.time() - start_ls) > time_limit:
                    break
                
                S_temp = [v for v in S if v != a]
                S_temp_set = set(S_temp)
                
                # Find a valid x to add
                # Prefer larger x? Or just any x?
                # Let's try adding x from all_candidates not in S_temp
                # Sort candidates to try larger ones first?
                # Actually, just iterating all_candidates is fine.
                found = False
                for x in all_candidates:
                    if x in S_temp_set:
                        continue
                    if can_add(x, S_temp):
                        # Accept swap
                        S = S_temp + [x]
                        Sset = S_temp_set | {x}
                        improved = True
                        found = True
                        break
                if found:
                    break
        
        return S

    best_set = []
    best_size = 0
    
    # 1. Deterministic Greedy Strategies
    orders = [
        sorted(all_candidates, reverse=True),
        sorted(all_candidates),
        [x for x in all_candidates if x > N//2] + [x for x in all_candidates if x <= N//2],
        [x for x in all_candidates if x % 2 == 1] + [x for x in all_candidates if x % 2 == 0],
        [x for x in all_candidates if x % 2 == 0] + [x for x in all_candidates if x % 2 == 1],
    ]
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        S = run_greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # 2. Randomized Greedy + Local Search
    rng = random.Random(42)
    
    num_iterations = 500 # Upper bound, controlled by time
    
    for i in range(num_iterations):
        if time.time() > deadline - 5:
            break
            
        # Create a random order, biased towards larger numbers or mixed
        # Strategy A: Random shuffle
        # Strategy B: Random shuffle but keep top 50% in random order, bottom 50% in random order?
        # Let's just do random shuffle with some bias.
        
        # Bias: Larger numbers are "safer" (fewer divisors). 
        # But we want to maximize count.
        # Let's try a few different randomization styles.
        
        style = rng.choice(['random', 'biased_large', 'biased_small', 'prime_heavy'])
        
        cands = all_candidates.copy()
        
        if style == 'random':
            rng.shuffle(cands)
        elif style == 'biased_large':
            # Mix of sorted desc and random
            rng.shuffle(cands)
            # Sort by random key + large bias
            cands.sort(key=lambda x: x + rng.uniform(0, 100), reverse=True)
        elif style == 'biased_small':
            cands.sort(key=lambda x: x + rng.uniform(0, 100))
        elif style == 'prime_heavy':
            # Primes are good candidates as they have few divisors
            # Simple primality check for 1..192
            def is_prime(n):
                if n < 2: return False
                for p in range(2, int(n**0.5)+1):
                    if n % p == 0: return False
                return True
            
            primes = [x for x in cands if is_prime(x)]
            non_primes = [x for x in cands if not is_prime(x)]
            rng.shuffle(primes)
            rng.shuffle(non_primes)
            # Interleave or just concatenate?
            # Let's try putting primes first, then non-primes
            cands = primes + non_primes

        S = run_greedy(cands)
        
        # Local Search
        # Time budget for local search per iteration: ~1-2 seconds?
        # Total time 85s. If 50 iterations, 1.7s each.
        # Let's be more conservative: 100 iterations, 0.5s each?
        # Or fewer iterations, more time per local search.
        # Local search can be slow.
        
        # Let's estimate cost of can_add.
        # If |S| ~ 40.
        # Check 1: C(40,3) ~ 9880 mods.
        # Check 2: 40 * C(39,2) ~ 40 * 741 ~ 29640 mods.
        # Total ~ 40k ops.
        # Python can do ~10M ops/sec? So 40k is ~4ms.
        # Greedy: 192 * 4ms = 0.8s.
        # Local search: 100 steps * (192 checks * 4ms) = 100 * 0.8s = 80s. Too slow.
        
        # We need to optimize can_add or reduce local search steps.
        
        # Optimization for can_add:
        # The second check is the bottleneck.
        # Can we precompute sums?
        # For a fixed a, we need to check if x + u + v is div by a.
        # This is equivalent to x + u + v = k*a.
        
        # Let's stick to a smaller time budget for local search, e.g., 0.2s.
        # And increase number of restarts.
        
        S = local_search(S, rng, time_limit=0.3)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # 3. Final Polish on Best Set
    S = list(best_set)
    Sset = set(S)
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for x in all_candidates:
            if x in Sset:
                continue
            if can_add(x, S):
                S.append(x)
                Sset.add(x)
                changed = True
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Verify result
    if not is_valid_full(best_set):
        # Fallback to a known valid set if something went wrong
        # Simple greedy descending is usually valid
        S = run_greedy(sorted(all_candidates, reverse=True))
        return S

    return best_set

# EVOLVE-BLOCK-END
