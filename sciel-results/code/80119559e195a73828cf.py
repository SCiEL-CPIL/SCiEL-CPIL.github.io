"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

# Fixed seed for reproducibility within the run, but we can vary if needed
# However, the task says "construct()", so we should probably be deterministic 
# or use a fixed seed if randomness is used to ensure consistent evaluation.
# The prompt implies a single evaluation.
random.seed(42)

P = {'n': 256}

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True

def _prime_factors(n):
    """Return a dictionary of prime factors and their exponents."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divides(a, b):
    """Check if a divides b."""
    return b % a == 0

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Precompute prime factorizations for all candidates
    # This is crucial for fast violation checks
    pf_cache = {}
    for x in candidates:
        pf_cache[x] = _prime_factors(x)
    
    # A helper to check if a | b*c efficiently using prime factors
    # a | b*c iff for all p^k || a, the sum of exponents of p in b and c is >= k
    def _a_divides_bc(a, b, c):
        fa = pf_cache[a]
        fb = pf_cache[b]
        fc = pf_cache[c]
        for p, exp_a in fa.items():
            exp_bc = fb.get(p, 0) + fc.get(p, 0)
            if exp_bc < exp_a:
                return False
        return True

    # Initial state: Start with a random subset or a greedy heuristic subset
    # Let's start with all primes, which is a valid set, and try to add composites.
    # Or start with a large random set and remove violators.
    # Given the constraint is tight, starting with primes is a good "safe" base.
    primes = [p for p in candidates if _is_prime(p)]
    current_set = set(primes)
    
    # To maximize cardinality, we want to add as many composites as possible.
    # Let's use Simulated Annealing on the set of *excluded* numbers or the set itself.
    # It's easier to work with the set of included numbers.
    
    # Initial temperature
    T = 10.0
    T_min = 0.001
    cooling_rate = 0.999
    
    # Pre-compute list of all composites to try adding
    composites = [x for x in candidates if not _is_prime(x)]
    
    # Function to count violations in a set
    def count_violations(s):
        lst = list(s)
        violations = 0
        # Optimization: only check triples where a is the smallest? No, a can be any.
        # But if a > b*c, a cannot divide b*c (since a,b,c >= 2, b*c >= 4, but a can be large).
        # Actually if a > b*c, it's impossible. 
        # So we can skip checks if a > b*c.
        for i in range(len(lst)):
            a = lst[i]
            for j in range(i+1, len(lst)):
                b = lst[j]
                for k in range(j+1, len(lst)):
                    c = lst[k]
                    if a <= b * c: # Necessary condition
                        if _a_divides_bc(a, b, c):
                            violations += 1
        return violations

    # Initial violations
    current_violations = count_violations(current_set)
    
    # We want to MINIMIZE violations (target 0) and MAXIMIZE size.
    # Objective: Maximize Size - Penalty * Violations
    # Or just minimize violations first, then maximize size?
    # Since we start with valid set (primes), violations = 0.
    # We want to ADD numbers.
    
    # Strategy: 
    # 1. Start with all primes (Valid, Size ~54)
    # 2. Try to add composites one by one.
    # 3. If adding a composite creates violations, we might need to remove some existing numbers to fix it.
    # This is a hard combinatorial problem.
    
    # Let's use a Greedy + Local Search approach which is more robust than SA for small N.
    # The "different strategy" is using a **Conflict Graph** and **Max Independent Set** heuristic.
    
    # Build Conflict Graph
    # Node: number x
    # Edge (x, y) if x and y cannot coexist in a valid set?
    # No, validity is a 3-wise constraint. It's a 3-uniform hypergraph.
    # A set S is valid if it contains NO violating triple.
    # This is the problem of finding a Maximum 3-Independent Set.
    
    # Heuristic for Max 3-Independent Set:
    # 1. Start with empty set.
    # 2. Add nodes one by one.
    # 3. If adding a node creates a violation with existing pairs, we have a choice:
    #    - Don't add it.
    #    - Remove one of the existing nodes involved in the conflict.
    
    # Let's implement a Greedy Algorithm with Backtracking/Local Search.
    
    def is_valid_with_new(s, new_node):
        """Check if s U {new_node} is valid."""
        # Check all pairs (b, c) in s
        # If new_node divides b*c -> invalid
        # If b divides new_node*c -> invalid
        # If c divides new_node*b -> invalid
        lst = list(s)
        for i in range(len(lst)):
            b = lst[i]
            # Check if new_node | b * c for any c in s (c != b)
            # We can optimize by precomputing products?
            # For now, direct check.
            for j in range(len(lst)):
                if i == j: continue
                c = lst[j]
                # Triple (new_node, b, c)
                if _a_divides_bc(new_node, b, c):
                    return False
                # Triple (b, new_node, c) -> b | new_node * c
                if _a_divides_bc(b, new_node, c):
                    return False
                # Triple (c, new_node, b) -> c | new_node * b
                # Note: _a_divides_bc(c, new_node, b) is same as c | new_node * b
                if _a_divides_bc(c, new_node, b):
                    return False
        return True

    # Greedy Construction
    # Sort candidates by "potential" (e.g., primes first, then composites with few factors)
    # Primes are "safe" because they only divide b*c if they divide b or c.
    # Since we start with primes, adding a prime p is safe if p doesn't divide any b*c?
    # If p is prime, p | b*c implies p|b or p|c.
    # If we only have primes in the set, p|b is false (distinct primes).
    # So any set of primes is valid.
    
    # The challenge is adding composites.
    # A composite x is dangerous if its prime factors are "shared" heavily.
    
    # Let's try a different ordering: Add numbers that are "least likely" to cause conflicts.
    # Numbers with large prime factors are safer?
    # Or numbers that are prime powers?
    
    # Let's stick to the SA idea but applied to the SET of INCLUDED numbers.
    # State: A set S.
    # Move: Add a number not in S, or Remove a number in S, or Swap.
    # Energy: Number of violations.
    # We want Energy = 0 and Size = Max.
    
    # Since we start with valid set, Energy = 0.
    # We try to add numbers. If adding causes Energy > 0, we accept with probability exp(-DeltaE/T).
    # If we accept, we have a violation. We must try to fix it by removing other numbers.
    
    # This is effectively a Local Search on the space of subsets.
    
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Current state
    s = set(current_set)
    E = 0 # Violations
    
    # Precompute all potential conflicts? No, too many.
    
    # Optimization: 
    # Instead of checking all triples for E, we can maintain a list of violations.
    # But for simplicity and given N=256, let's see if we can afford full checks.
    # Size ~ 100. Triples ~ 100^3 / 6 ~ 160,000.
    # 160,000 checks per step. 10,000 steps = 1.6 billion checks. Too slow in Python.
    
    # We need a faster violation check.
    # Incremental update:
    # When adding x to S, new violations are those triples involving x.
    # Count = sum_{b,c in S, b!=c} [x|bc or b|xc or c|xb]
    
    def calc_new_violations(s, x):
        """Count violations introduced by adding x to s."""
        count = 0
        lst = list(s)
        for i in range(len(lst)):
            b = lst[i]
            for j in range(len(lst)):
                if i == j: continue
                c = lst[j]
                if _a_divides_bc(x, b, c):
                    count += 1
                if _a_divides_bc(b, x, c):
                    count += 1
                if _a_divides_bc(c, x, b):
                    count += 1
        return count

    # This is O(N^2) per addition. N~100. 10,000 ops.
    # 10,000 steps * 10,000 ops = 100 million ops. Feasible in 85s?
    # Python is slow. 100M simple ops might take 5-10s.
    # We can run ~20,000-50,000 steps.
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Candidates to add: all composites
    # Candidates to remove: all elements in s
    
    # Shuffle composites for random access
    # We'll just pick random candidates from the full pool not in s
    
    while time.time() < deadline - 2: # Leave 2s for final cleanup
        if T < T_min:
            break
            
        # Choose a move
        # 50% chance add, 50% chance remove?
        # Or just add? If we only add, we never fix errors.
        # If we accept a bad add, we MUST remove something to get back to 0.
        
        # Let's use a "Fix-up" strategy:
        # 1. Try to add a random candidate not in s.
        # 2. Calculate DeltaE.
        # 3. If DeltaE == 0, accept.
        # 4. If DeltaE > 0, accept with prob exp(-DeltaE/T).
        # 5. If accepted (now invalid), perform local search to remove violations:
        #    - Find a violating triple.
        #    - Remove one element from it (randomly or the one with most conflicts).
        #    - Repeat until valid.
        
        if len(s) < n:
            # Pick a candidate to add
            # Prefer composites not in s
            not_in_s = [x for x in candidates if x not in s]
            if not_in_s:
                x = random.choice(not_in_s)
                delta_e = calc_new_violations(s, x)
                
                if delta_e == 0:
                    s.add(x)
                    E = 0
                else:
                    # Metropolis acceptance
                    if random.random() < math.exp(-delta_e / T):
                        s.add(x)
                        E = delta_e
                        # Now we are in an invalid state. Try to fix it.
                        # Local search to remove violations
                        fix_attempts = 0
                        while E > 0 and fix_attempts < 50 and time.time() < deadline - 2:
                            # Find a violating triple
                            # We need to find one quickly.
                            # Since we know E > 0, we can scan.
                            # To speed up, we can just pick a random element involved in a conflict?
                            # Let's find one explicitly.
                            found = False
                            lst = list(s)
                            # Randomize order to avoid bias
                            random.shuffle(lst)
                            for i in range(len(lst)):
                                a = lst[i]
                                for j in range(i+1, len(lst)):
                                    b = lst[j]
                                    for k in range(j+1, len(lst)):
                                        c = lst[k]
                                        if _a_divides_bc(a, b, c):
                                            # Found violation (a,b,c)
                                            # Remove one of them
                                            # Heuristic: remove the one with highest "degree" in conflict graph?
                                            # Or just random?
                                            # Let's remove 'a' (the divisor) as it's often the "cause"
                                            # Or remove the composite one?
                                            # Let's pick randomly from a,b,c
                                            victim = random.choice([a, b, c])
                                            s.remove(victim)
                                            # Recalculate E?
                                            # This is expensive.
                                            # Let's just recalculate E from scratch after removal.
                                            E = count_violations(s)
                                            found = True
                                            break
                                    if found: break
                                if found: break
                            if not found:
                                # Should not happen if E>0, but just in case
                                E = 0
                            fix_attempts += 1
                
        T *= cooling_rate
        
        # Track best
        if E == 0 and len(s) > best_size:
            best_set = set(s)
            best_size = len(best_set)
            
    # Final cleanup: ensure best_set is valid
    # It should be, but let's verify
    if count_violations(best_set) != 0:
        # Fallback to primes
        best_set = set(primes)
        
    return [int(x) for x in sorted(best_set)]
# EVOLVE-BLOCK-END
