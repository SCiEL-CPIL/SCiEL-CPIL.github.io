import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    N = len(numbers)
    idx = {x: i for i, x in enumerate(numbers)}
    
    # Helper: LCM calculation
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Precompute bad triples
    # A triple (a,b,c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    triple_idx = []  # For each variable, list of triple indices containing it
    
    for i in range(N):
        a = numbers[i]
        for j in range(i + 1, N):
            b = numbers[j]
            lab = lcm(a, b)
            for k in range(j + 1, N):
                c = numbers[k]
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    bad_triples.append((i, j, k))
    
    M = len(bad_triples)
    triple_idx = [[] for _ in range(N)]
    for t, (a, b, c) in enumerate(bad_triples):
        triple_idx[a].append(t)
        triple_idx[b].append(t)
        triple_idx[c].append(t)
    
    # Count how many triples each variable is in
    var_degree = [len(triple_idx[i]) for i in range(N)]
    
    # Initial solution: start with all variables set to 1 (included)
    # Then use local search to minimize unsatisfied clauses
    # Unatisfied clause = triple where all 3 variables are set to 1
    
    def count_unsatisfied(sol):
        """Count number of bad triples where all 3 variables are 1."""
        count = 0
        for t in range(M):
            a, b, c = bad_triples[t]
            if sol[a] and sol[b] and sol[c]:
                count += 1
        return count
    
    # Better: incremental count using neighbor lists
    def initial_solution():
        """Start with a good initial solution using greedy addition."""
        sol = [0] * N
        
        # Use prime powers as core
        def get_primes(limit):
            sieve = [True] * (limit + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(limit**0.5) + 1):
                if sieve[i]:
                    for j in range(i*i, limit + 1, i):
                        sieve[j] = False
            return [i for i, is_prime in enumerate(sieve) if is_prime]
        
        primes = get_primes(n)
        for p in primes:
            power = p
            while power <= n:
                sol[idx[power]] = 1
                power *= p
        sol[idx[1]] = 1
        
        # Try to add remaining variables greedily
        remaining = [i for i in range(N) if sol[i] == 0]
        remaining.sort(key=lambda i: (var_degree[i], numbers[i]))
        
        for i in remaining:
            if time.time() > deadline - 30:
                break
            # Check if adding i creates any unsatisfied triple
            can_add = True
            for t in triple_idx[i]:
                a, b, c = bad_triples[t]
                others = [x for x in (a, b, c) if x != i]
                if sol[others[0]] and sol[others[1]]:
                    can_add = False
                    break
            if can_add:
                sol[i] = 1
        
        return sol
    
    # WalkSAT-style local search
    def walk_sat_local_search(sol, time_budget=60.0):
        """Local search to minimize unsatisfied clauses."""
        current_sol = sol[:]
        current_unsat = count_unsatisfied(current_sol)
        best_sol = current_sol[:]
        best_unsat = current_unsat
        
        # Precompute for fast evaluation: for each variable, which triples it's in
        # and for each triple, which variables are set
        
        # Incremental update: when we flip variable v, we only need to check
        # triples containing v
        def flip_and_evaluate(v, current_sol, current_unsat):
            """Flip variable v and return new unsat count."""
            old_val = current_sol[v]
            new_val = 1 - old_val
            current_sol[v] = new_val
            
            delta = 0
            for t in triple_idx[v]:
                a, b, c = bad_triples[t]
                # Check if this triple was unsatisfied before and after
                others = [x for x in (a, b, c) if x != v]
                was_unsat = current_sol[others[0]] and current_sol[others[1]] and (old_val == 1)
                is_unsat = current_sol[others[0]] and current_sol[others[1]] and (new_val == 1)
                
                if was_unsat and not is_unsat:
                    delta -= 1
                elif not was_unsat and is_unsat:
                    delta += 1
            
            return delta
        
        # Main loop
        p_flip = 0.25  # Probability of random flip
        max_iter = 100000
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            if current_unsat == 0:
                break
            
            # Find all variables that, if flipped, would decrease unsat
            improving_vars = []
            for v in range(N):
                delta = flip_and_evaluate(v, current_sol, current_unsat)
                # Revert
                current_sol[v] = 1 - current_sol[v]
                if delta < 0:
                    improving_vars.append((v, delta))
            
            if improving_vars:
                # Pick the best improving variable (most negative delta)
                improving_vars.sort(key=lambda x: x[1])
                v = improving_vars[0][0]
                delta = improving_vars[0][1]
                current_sol[v] = 1 - current_sol[v]
                current_unsat += delta
            else:
                # No improving move; do a random flip (WalkSAT)
                if random.random() < p_flip:
                    v = random.randint(0, N - 1)
                else:
                    # Pick variable in a random unsatisfied triple
                    # Find an unsatisfied triple
                    unsat_triples = []
                    for t in range(M):
                        a, b, c = bad_triples[t]
                        if current_sol[a] and current_sol[b] and current_sol[c]:
                            unsat_triples.append(t)
                            if len(unsat_triples) >= 10:
                                break
                    if unsat_triples:
                        t = random.choice(unsat_triples)
                        a, b, c = bad_triples[t]
                        v = random.choice([a, b, c])
                    else:
                        break
                
                delta = flip_and_evaluate(v, current_sol, current_unsat)
                current_sol[v] = 1 - current_sol[v]
                current_unsat += delta
            
            # Update best
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_sol = current_sol[:]
        
        return best_sol, best_unsat
    
    # Phase 1: Initial solution
    sol = initial_solution()
    unsat = count_unsatisfied(sol)
    
    # Phase 2: WalkSAT local search
    if time.time() < deadline - 10:
        sol, unsat = walk_sat_local_search(sol)
    
    # Phase 3: Deterministic refinement - try to add back any variable that doesn't create conflict
    for v in range(N):
        if time.time() > deadline - 3:
            break
        if sol[v] == 0:
            can_add = True
            for t in triple_idx[v]:
                a, b, c = bad_triples[t]
                others = [x for x in (a, b, c) if x != v]
                if sol[others[0]] and sol[others[1]]:
                    can_add = False
                    break
            if can_add:
                sol[v] = 1
    
    # Convert to list of numbers
    result = [numbers[i] for i in range(N) if sol[i] == 1]
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result
