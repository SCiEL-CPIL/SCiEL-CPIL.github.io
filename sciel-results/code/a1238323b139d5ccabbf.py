"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate all primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    primes = _sieve(1000)
    
    # Greedy: select primes to maximize coverage.
    # Strategy: try to build a set of primes where sum(1/p) <= 1,
    # then assign residues greedily to maximize consecutive coverage from 1.
    
    # First, find a good set of primes within budget.
    # We'll try different strategies and pick the best.
    
    def compute_coverage(classes, max_check=200000):
        """Compute the largest L such that 1..L are all covered."""
        if not classes:
            return 0
        covered = set()
        for p, r in classes:
            # Add all numbers n in 1..max_check where n ≡ r (mod p)
            # Start from the smallest n >= 1 with n % p == r
            if r == 0:
                start = p  # p, 2p, 3p, ...
            else:
                start = r  # r, r+p, r+2p, ...
            if start < 1:
                start += p * ((1 - start + p - 1) // p)
            for n in range(start, max_check + 1, p):
                covered.add(n)
        
        # Find the largest L such that 1..L all in covered
        L = 0
        for i in range(1, max_check + 1):
            if i in covered:
                L = i
            else:
                break
        return L
    
    def greedy_residues(prime_list, max_check=200000):
        """Assign residues greedily to maximize consecutive coverage from 1."""
        if not prime_list:
            return []
        
        # Sort primes: we'll process in order of increasing p (smaller primes cover denser)
        # Actually, let's try processing in the order given and pick best residue for each.
        
        covered = set()
        classes = []
        
        for p in prime_list:
            # Try all residues and pick the one that maximizes new coverage in 1..max_check
            best_r = 0
            best_new = -1
            best_covered = set()
            
            for r in range(p):
                # Compute which numbers in 1..max_check are covered by (p, r)
                new_covered = set()
                if r == 0:
                    start = p
                else:
                    start = r
                if start < 1:
                    start += p
                for n in range(start, max_check + 1, p):
                    if n not in covered:
                        new_covered.add(n)
                
                # We want to maximize coverage of consecutive integers from 1
                # A simple heuristic: maximize the count of new numbers covered
                # that are in the range 1..current_max+some_buffer
                current_max = 0
                for i in range(1, max_check + 1):
                    if i in covered:
                        current_max = i
                    else:
                        break
                
                # Score: prioritize covering numbers near current_max+1
                score = 0
                for n in new_covered:
                    if n == current_max + 1:
                        score += 10000
                    elif n <= current_max + 100:
                        score += max(0, 100 - (n - current_max))
                    else:
                        score += 1
                
                if score > best_new or (score == best_new and len(new_covered) > len(best_covered)):
                    best_new = score
                    best_r = r
                    best_covered = new_covered
            
            classes.append([p, best_r])
            covered |= best_covered
            
            if time.time() > deadline:
                break
        
        return classes
    
    # Strategy: select primes greedily by trying to maximize coverage
    # We'll use a priority-based approach: try adding primes one by one
    
    # First, let's find which primes we can afford
    # Try different subsets and evaluate
    
    # Greedy prime selection: start with empty, add prime that gives best coverage improvement
    # This is expensive, so let's use a simpler approach.
    
    # Approach: select primes in order of increasing p (smallest first gives densest coverage)
    # but constrained by budget.
    
    # Let's try: sort primes by p (ascending), greedily add if budget allows
    # But also try: sort by p descending (cheapest first)
    
    best_classes = []
    best_L = 0
    
    # Try strategy 1: smallest primes first (densest)
    primes_asc = sorted(primes)
    budget = Fraction(1)
    selected = []
    for p in primes_asc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    
    if selected:
        classes = greedy_residues(selected, max_check=200000)
        L = compute_coverage(classes, max_check=200000)
        if L > best_L:
            best_L = L
            best_classes = classes
    
    # Try strategy 2: largest primes first (cheapest, most primes)
    primes_desc = sorted(primes, reverse=True)
    budget = Fraction(1)
    selected = []
    for p in primes_desc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    
    if selected:
        classes = greedy_residues(selected, max_check=200000)
        L = compute_coverage(classes, max_check=200000)
        if L > best_L:
            best_L = L
            best_classes = classes
    
    # Try strategy 3: mixed - use small primes for density, fill with large
    # Start with a few small primes, then add large ones
    if time.time() < deadline - 5:
        for num_small in range(1, 6):
            small_primes = primes_asc[:num_small]
            budget = Fraction(1)
            for p in small_primes:
                budget -= Fraction(1, p)
            if budget < 0:
                break
            
            selected = list(small_primes)
            for p in primes_desc:
                if p in small_primes:
                    continue
                cost = Fraction(1, p)
                if budget >= cost:
                    budget -= cost
                    selected.append(p)
            
            if selected:
                classes = greedy_residues(selected, max_check=200000)
                L = compute_coverage(classes, max_check=200000)
                if L > best_L:
                    best_L = L
                    best_classes = classes
            
            if time.time() > deadline:
                break
    
    return best_classes if best_classes else [[2, 1]]

# EVOLVE-BLOCK-END
