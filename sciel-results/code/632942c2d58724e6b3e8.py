import time
import random
from itertools import combinations

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s kill (evaluator says 90s kill, but task says 60s)
    
    # Candidate numbers 2..256
    candidates = list(range(2, P['n'] + 1))
    n_cand = len(candidates)
    
    # Precompute prime factorization for all candidates
    # factorize[n] = list of prime factors with multiplicity
    factorize = [None] * (P['n'] + 1)
    for i in range(2, P['n'] + 1):
        if factorize[i] is None:
            # i is prime
            factorize[i] = [i]
        for j in range(i * 2, P['n'] + 1, i):
            if factorize[j] is None:
                factorize[j] = []
            factorize[j].append(i)
    
    # Build a divisibility-check helper using factorization
    # a divides b*c iff for every prime p, v_p(a) <= v_p(b) + v_p(c)
    # Precompute prime power exponents
    # exps[n][p] = exponent of prime p in n
    # But primes up to 256... let's use a dict or list
    # Actually, let's precompute for each number a list of (prime, exponent)
    
    def prime_factorize(n):
        if n < 2:
            return {}
        result = {}
        x = n
        p = 2
        while p * p <= x:
            while x % p == 0:
                result[p] = result.get(p, 0) + 1
                x //= p
            p += 1
        if x > 1:
            result[x] = result.get(x, 0) + 1
        return result
    
    # Precompute factorizations
    factor_dict = [None] * (P['n'] + 1)
    for i in range(2, P['n'] + 1):
        factor_dict[i] = prime_factorize(i)
    
    def divides_product(a, b, c):
        """Check if a divides b*c"""
        # For each prime p in factorization of a, check v_p(a) <= v_p(b) + v_p(c)
        fa = factor_dict[a]
        fb = factor_dict[b]
        fc = factor_dict[c]
        for p, ea in fa.items():
            eb = fb.get(p, 0)
            ec = fc.get(p, 0)
            if ea > eb + ec:
                return False
        return True
    
    # Precompute conflict hyperedges: for each triple (a,b,c) distinct, check if a|b*c
    # A set S is valid iff for all distinct a,b,c in S, NOT (a | b*c)
    # So a "conflict" is a triple (a,b,c) where a | b*c
    # We want a maximum subset with no conflicting triple.
    
    # This is a 3-uniform hypergraph independent set problem.
    # Let's build the conflict structure.
    
    # For each ordered pair (b,c) with b != c, find all a such that a | b*c
    # Then for a triple {a,b,c} to be invalid, we need a | b*c (with a distinct from b,c)
    
    # Actually, the condition is: for every THREE DISTINCT members a,b,c, a does not divide b*c.
    # So for a set S to be valid, for all distinct a,b,c in S: a ∤ b*c.
    
    # Let's precompute: for each pair (b,c), b!=c, the set of a's that divide b*c.
    # Then a conflict hyperedge exists for {a,b,c} if a | b*c.
    
    # To check if adding x to S creates a conflict:
    # 1. x | b*c for some distinct b,c in S
    # 2. a | x*c for some distinct a,c in S (a,c != x, which is true since x not in S)
    # 3. a | b*x for some distinct a,b in S (same as 2 by symmetry of b,c)
    
    # Let's use branch and bound with a good heuristic.
    
    # First, let's build a fast validity check for a set represented as a bitmask or set.
    
    # Given the small size (255 candidates), we can use a set of ints.
    
    def can_add(S, x):
        """Check if x can be added to set S without creating a conflict."""
        S_list = list(S)
        n = len(S_list)
        
        # Check 1: x | b*c for any distinct b,c in S
        for i in range(n):
            for j in range(i+1, n):
                b = S_list[i]
                c = S_list[j]
                if divides_product(x, b, c):
                    return False
                    
        # Check 2: a | x*c for any distinct a,c in S
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                c = S_list[j]
                if i == j:
                    continue
                if divides_product(a, x, c):
                    return False
                    
        return True
    
    # Initial solution: all primes
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    best_set = set(primes)
    best_size = len(primes)
    
    # Local search / greedy improvement
    # Try random restarts with greedy + local search
    
    # Conflict density: for each candidate x, count how many pairs (b,c) have x | b*c
    # This helps order the greedy addition.
    
    # Precompute for each x, the number of pairs (b,c) in candidates with x | b*c
    conflict_count = [0] * (P['n'] + 1)
    for x in candidates:
        count = 0
        for b in candidates:
            for c in candidates:
                if b != c:
                    if divides_product(x, b, c):
                        count += 1
        conflict_count[x] = count
    
    # Greedy: add candidates in order of increasing conflict_count (least restrictive first)
    # Or try multiple orderings.
    
    def greedy_add(S, order):
        """Try to add candidates in given order to S."""
        S = set(S)
        for x in order:
            if x not in S:
                if can_add(S, x):
                    S.add(x)
        return S
    
    # Local search: try to swap elements
    def local_search(S):
        """Try 1-for-1 swaps to improve the set."""
        S = set(S)
        improved = True
        while improved:
            improved = False
            S_list = list(S)
            for a in S_list:
                S_without_a = S - {a}
                for b in candidates:
                    if b in S:
                        continue
                    if can_add(S_without_a, b):
                        new_S = S_without_a | {b}
                        # This swap doesn't change size, so we need a different criterion
                        # Actually, for max independent set, we want to increase size.
                        # So local search should try to add, not swap.
                        pass
        return S
    
    # Better local search: try to find an element to remove and two to add
    # Or just do greedy with different orderings and keep the best.
    
    # Strategy: 
    # 1. Start with primes
    # 2. Try to greedily add composites in various orders
    # 3. Use random restarts
    
    composites = [x for x in candidates if not is_prime(x)]
    
    # Order composites by conflict_count (ascending = easier to add)
    comp_by_conflict = sorted(composites, key=lambda x: conflict_count[x])
    comp_by_conflict_desc = sorted(composites, key=lambda x: -conflict_count[x])
    
    # Try several strategies
    strategies = [
        comp_by_conflict,
        comp_by_conflict_desc,
        list(reversed(comp_by_conflict)),
    ]
    
    # Add random orderings
    for _ in range(20):
        random.shuffle(composites)
        strategies.append(list(composites))
    
    for strat in strategies:
        if time.time() > deadline - 5:
            break
        S = greedy_add(set(primes), strat)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Now try local search on the best set found
    # Try removing one element and adding up to two others
    if time.time() < deadline - 10:
        S = set(best_set)
        S_list = list(S)
        
        # Try 1-for-2 swaps: remove a, add b and c
        for a in S_list:
            if time.time() > deadline - 5:
                break
            S_without_a = S - {a}
            # Try to add pairs
            for b in candidates:
                if b in S_without_a:
                    continue
                if can_add(S_without_a, b):
                    S_with_b = S_without_a | {b}
                    for c in candidates:
                        if c in S_with_b:
                            continue
                        if can_add(S_with_b, c):
                            new_S = S_with_b | {c}
                            if len(new_S) > len(S):
                                S = new_S
                                S_list = list(S)
                                break
                    if len(S) > len(S_without_a) + 1:
                        break
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    # Try more aggressive local search: hill climbing with random perturbations
    if time.time() < deadline - 5:
        current = set(best_set)
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            # Random perturbation: remove 1-2 random elements
            perturbed = set(current)
            for _ in range(random.randint(1, 2)):
                if perturbed:
                    elem = random.choice(list(perturbed))
                    perturbed.remove(elem)
            
            # Greedy add
            remaining = [x for x in candidates if x not in perturbed]
            random.shuffle(remaining)
            for x in remaining:
                if time.time() > deadline - 2:
                    break
                if x not in perturbed:
                    if can_add(perturbed, x):
                        perturbed.add(x)
            
            if len(perturbed) > len(current):
                current = perturbed
                if len(current) > best_size:
                    best_size = len(size)
                    best_set = current
    
    # Final polish: try to add any remaining candidate
    final = set(best_set)
    for x in candidates:
        if x not in final:
            if can_add(final, x):
                final.add(x)
    if len(final) > best_size:
        best_set = final
        best_size = len(final)
    
    # Convert to list of Python ints
    result = list(best_set)
    # Ensure all are Python ints
    result = [int(x) for x in result]
    
    return result
