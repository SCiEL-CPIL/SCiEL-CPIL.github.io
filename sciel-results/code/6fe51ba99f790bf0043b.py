import math
import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s limit

    N = 256

    # Precompute prime factors for each number 1..N
    prime_factors = [set() for _ in range(N + 1)]
    prime_factors[1] = set()
    for i in range(2, N + 1):
        x = i
        d = 2
        while d * d <= x:
            if x % d == 0:
                prime_factors[i].add(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            prime_factors[i].add(x)

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    all_numbers = list(range(1, N + 1))
    num_numbers = N

    # Build bipartite graph: left = primes (num_primes nodes), right = integers 1..256 (num_numbers nodes)
    # Edge weight: if prime p divides integer n, weight = n (we want to maximize sum)
    # If p doesn't divide n, weight = 0 (or -infinity to discourage)
    # We also need to handle 1: it's coprime to everything, so we can always include it
    # Strategy: assign each integer to at most one prime, and each prime to at most one integer
    # The sum of assigned integers is our objective

    # For Hungarian algorithm, we need a cost matrix. We'll maximize, so negate weights.
    # Create cost matrix: rows = primes, cols = integers
    # cost[i][j] = -n if primes[i] divides n, else a large positive number (to discourage)
    
    # Actually, let's think differently. We want to select a set of integers such that:
    # - Each integer's prime factors are "covered" by distinct primes
    # - We maximize the sum of selected integers
    
    # Alternative: Use the fact that if we assign integer n to prime p (where p|n), 
    # then n can only be selected if all its prime factors are assigned to it or to other selected numbers.
    
    # Simpler approach: Build a graph where we try to match primes to their multiples,
    # and the weight of matching p to n is n. Then find max weight matching.
    
    # But this doesn't fully capture the constraint. Let me use a different formulation.
    
    # Key insight: Two numbers are coprime iff they share no prime factors.
    # So we can think of this as: each prime can be "used" by at most one selected number.
    # If number n is selected, it "uses" all primes in its factorization.
    
    # This is a weighted set packing problem, which is NP-hard in general.
    # But with 54 primes and 256 numbers, we can use integer programming or clever heuristics.
    
    # Let's try a different approach: Use scipy's linear_sum_assignment on a modified problem.
    # We'll create a bipartite graph where:
    # - Left nodes: primes (54 nodes)
    # - Right nodes: integers 2..256 (255 nodes) + dummy nodes
    # - Edge from prime p to integer n exists if p|n, with weight = n
    # - Find max weight matching
    
    # But this still doesn't enforce that if we select n, ALL its prime factors must be matched to it.
    
    # Let me try a completely different approach: Since the problem is small enough,
    # let's use a greedy + local search but with smarter initial solution from matching.
    
    # First, compute a good initial solution using a matching-based heuristic:
    # For each prime, find its largest multiple <= 256 that doesn't conflict with already-chosen numbers.
    
    # Actually, let's just improve the local search with better moves and more restarts.
    # The current approach is decent; let's add:
    # 1. Better initial solution using iterative improvement
    # 2. More aggressive local search with 2-for-2 swaps
    # 3. Simulated annealing-like acceptance
    
    # Precompute divisibility: divisors[n] = list of primes dividing n
    divisors = [set() for _ in range(N + 1)]
    for n in range(2, N + 1):
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                divisors[n].add(p)
                while n % p == 0:
                    n //= p
        if n > 1:
            divisors[n].add(n)
    
    # Rebuild prime_factors correctly
    prime_factors = [set() for _ in range(N + 1)]
    prime_factors[1] = set()
    for i in range(2, N + 1):
        x = i
        d = 2
        while d * d <= x:
            if x % d == 0:
                prime_factors[i].add(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            prime_factors[i].add(x)
    
    def is_coprime_to_set(x, S):
        if x == 1:
            return True
        pf_x = prime_factors[x]
        for y in S:
            if y == 1:
                continue
            if pf_x & prime_factors[y]:
                return False
        return True

    # Build initial solution using greedy with lookahead
    # For each prime, we can either use it alone (as a prime power) or merge it with other primes
    # into a composite number.
    
    # Key idea: For small primes, it's often better to merge them into composites with large primes.
    # E.g., 2*31=62, but 2*127=254 is better. Or 3*83=249, etc.
    
    # Let's use a different strategy: Start with all prime powers, then try to improve
    # by replacing pairs of prime powers with their product (if <= 256) or other composites.
    
    initial_set = set()
    for p in primes:
        x = p
        while x * p <= N:
            x *= p
        initial_set.add(x)
        
    current_set = set(initial_set)
    current_sum = sum(current_set)
    
    def local_search(S, s_sum, max_time):
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            
            # Phase 1: Add
            for x in all_numbers:
                if x not in S and is_coprime_to_set(x, S):
                    S.add(x)
                    s_sum += x
                    improved = True
                    break
            
            # Phase 2: Swap 1 for 1
            if not improved:
                s_list = list(S)
                candidates_in = [y for y in all_numbers if y not in S]
                candidates_in.sort(reverse=True)
                for a in s_list:
                    temp_set = S - {a}
                    for b in candidates_in:
                        if b <= a:
                            break
                        if is_coprime_to_set(b, temp_set):
                            S.discard(a)
                            S.add(b)
                            s_sum += b - a
                            improved = True
                            break
                    if improved:
                        break
                        
            # Phase 3: Swap 2 for 1
            if not improved:
                s_list = list(S)
                for i in range(len(s_list)):
                    if time.time() - start_time > max_time - 1:
                        break
                    for j in range(i + 1, len(s_list)):
                        a, c = s_list[i], s_list[j]
                        if a + c >= N:
                            continue
                        temp_set = S - {a, c}
                        for b in all_numbers:
                            if b in S or b <= a + c:
                                continue
                            if is_coprime_to_set(b, temp_set):
                                S.discard(a)
                                S.discard(c)
                                S.add(b)
                                s_sum += b - a - c
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
            # Phase 4: Swap 1 for 2
            if not improved:
                s_list = list(S)
                for a in s_list:
                    if time.time() - start_time > max_time - 1:
                        break
                    temp_set = S - {a}
                    candidates = [y for y in all_numbers if y not in S and is_coprime_to_set(y, temp_set)]
                    candidates.sort(reverse=True)
                    found = False
                    for i in range(len(candidates)):
                        if time.time() - start_time > max_time - 1:
                            break
                        b = candidates[i]
                        if b <= a:
                            break
                        for j in range(i + 1, len(candidates)):
                            d = candidates[j]
                            if b + d <= a:
                                break
                            if is_coprime_to_set(d, temp_set | {b}):
                                S.discard(a)
                                S.add(b)
                                S.add(d)
                                s_sum += b + d - a
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if improved:
                        break
                        
            # Phase 5: Swap 2 for 2
            if not improved:
                s_list = list(S)
                n_s = len(s_list)
                for i in range(n_s):
                    if time.time() - start_time > max_time - 1:
                        break
                    for j in range(i + 1, n_s):
                        a, c = s_list[i], s_list[j]
                        temp_set = S - {a, c}
                        a_c_sum = a + c
                        # Find two numbers b, d not in S, coprime to temp_set and each other, sum > a+c
                        candidates = [y for y in all_numbers if y not in S and is_coprime_to_set(y, temp_set)]
                        candidates.sort(reverse=True)
                        found = False
                        for ii in range(len(candidates)):
                            if time.time() - start_time > max_time - 1:
                                break
                            b = candidates[ii]
                            if b <= a_c_sum:
                                break
                            for jj in range(ii + 1, len(candidates)):
                                d = candidates[jj]
                                if b + d <= a_c_sum:
                                    break
                                if is_coprime_to_set(d, temp_set | {b}):
                                    S.discard(a)
                                    S.discard(c)
                                    S.add(b)
                                    S.add(d)
                                    s_sum += b + d - a - c
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if improved:
                        break
                        
        return S, s_sum

    best_sum = current_sum
    best_set = set(current_set)
    
    current_set, current_sum = local_search(current_set, current_sum, time_limit - 10)
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = set(current_set)
        
    # Random Restarts
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        num_restarts = int(remaining_time / 3)
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 5:
                break
            # Perturb
            random_set = set(best_set)
            # Remove 1-2 random elements
            for _ in range(random.randint(1, 2)):
                if random_set:
                    random_set.remove(random.choice(list(random_set)))
            # Greedy fill
            random_sum = sum(random_set)
            for x in all_numbers:
                if x not in random_set and is_coprime_to_set(x, random_set):
                    random_set.add(x)
                    random_sum += x
            
            random_set, random_sum = local_search(random_set, random_sum, time_limit - 5)
            if random_sum > best_sum:
                best_sum = random_sum
                best_set = set(random_set)

    result = list(best_set)
    result = [int(x) for x in result]
    return result
