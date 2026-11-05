import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # We want to maximize L such that 1..L are all covered
    # Strategy: greedy set cover with budget constraint
    # Process primes in order of decreasing 1/p (smallest primes first)
    # For each prime, find the residue that covers the most uncovered integers in 1..L_max
    
    # We'll try to cover up to some max range
    MAX_L = 100000
    
    # Track which integers are covered (use a set or boolean array)
    # For efficiency, use a bytearray for coverage of 1..MAX_L
    covered = bytearray(MAX_L + 1)  # covered[i] = 1 if i is covered
    
    # Budget tracking with exact rationals
    budget = Fraction(1, 1)
    used_primes = set()
    classes = []
    
    # Sort primes by 1/p descending (smallest primes first)
    sorted_primes = sorted(primes, key=lambda p: -1/p)
    
    # For greedy: at each step, try each available prime and find best residue
    # But that's O(n_primes * p) per step which could be slow
    # Alternative: process primes in order of 1/p, pick best residue for each
    
    # Actually, let's do a smarter greedy:
    # 1. Start with no classes
    # 2. Repeatedly find the (p, r) that gives the best "marginal coverage per unit budget"
    #    among all unused primes
    # 3. Add it if it improves things
    # 4. Stop when no improvement or budget exhausted
    
    # For efficiency, let's first do a simpler approach:
    # Process primes in order of 1/p descending, for each find the best residue
    # that covers the most uncovered integers in 1..current_max+1
    
    # Track the current maximum consecutive coverage from 1
    def get_max_consecutive(covered, max_check):
        """Find the largest L such that 1..L are all covered."""
        L = 0
        for i in range(1, min(max_check, MAX_L) + 1):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    # Simple greedy: process primes in order, pick best residue for each
    # But we need to be smart about which residues to try
    
    # For a prime p, residue r covers numbers: r, r+p, r+2p, ... in 1..MAX_L
    # We want to maximize coverage of uncovered numbers
    
    # Let's do iterative improvement:
    # Start with empty set, repeatedly add the best (p,r) pair
    
    # To make this tractable, let's limit the search range initially
    # and expand as we find better solutions
    
    # Actually, let's use a different approach:
    # 1. Start with a reasonable initial solution
    # 2. Try local search improvements
    
    # Initial solution: use the seed approach but optimized
    # The seed uses primes 13,17,19,... mapping i->(i+1)%p
    # Let's try a greedy from scratch
    
    # Greedy approach:
    # At each step, for each unused prime p, compute the best residue r
    # that covers the most uncovered integers in 1..L_target
    # where L_target is the current max consecutive + some lookahead
    
    # For efficiency, let's cap the number of iterations
    
    best_classes = []
    best_L = 0
    
    # Try greedy construction
    covered = bytearray(MAX_L + 1)
    budget = Fraction(1, 1)
    used_primes = set()
    classes = []
    
    # We'll do up to 200 iterations of greedy selection
    for iteration in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Find the best (p, r) pair among unused primes
        best_gain = 0
        best_p = None
        best_r = None
        best_budget_cost = None
        
        for p in sorted_primes:
            if p in used_primes:
                continue
            
            # Check if we can afford this prime
            cost = Fraction(1, p)
            if budget - cost < 0:
                continue
            
            # Find the best residue r for this prime
            # We want to maximize the number of uncovered integers covered
            # in the range 1..some reasonable max
            
            # For efficiency, only check residues that might help
            # The best residue should cover many uncovered numbers
            # Try all residues 0..p-1 but stop early if possible
            
            best_r_for_p = -1
            best_count_for_p = 0
            
            # For large p, this is expensive. Let's limit.
            # If p > 500, there are at most 200 numbers in 1..100000
            # covered by any single residue class
            # So the gain is bounded
            
            # Try each residue
            for r in range(p):
                # Count how many uncovered integers in 1..MAX_L are covered by (p, r)
                # Numbers: r, r+p, r+2p, ... <= MAX_L
                # But we only care about those that are currently uncovered
                # And we want to focus on extending the consecutive prefix
                
                count = 0
                # Optimize: only count if it helps extend the prefix
                # For now, count all uncovered covered
                n = r
                while n <= MAX_L:
                    if n >= 1 and not covered[n]:
                        count += 1
                    n += p
                
                if count > best_count_for_p:
                    best_count_for_p = count
                    best_r_for_p = r
            
            if best_r_for_p >= 0 and best_count_for_p > best_gain:
                best_gain = best_count_for_p
                best_p = p
                best_r = best_r_for_p
                best_budget_cost = cost
        
        if best_p is None or best_gain == 0:
            break
        
        # Add the best class
        p = best_p
        r = best_r
        used_primes.add(p)
        classes.append([p, r])
        budget -= best_budget_cost
        
        # Update covered array
        n = r
        while n <= MAX_L:
            if n >= 1:
                covered[n] = 1
            n += p
        
        # Check current max consecutive
        current_L = get_max_consecutive(covered, MAX_L)
        if current_L > best_L:
            best_L = current_L
            best_classes = list(classes)
    
    # Verify the best solution
    if best_classes:
        return best_classes
    else:
        # Fallback to seed
        primes = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]
