"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _canonical_pairwise_disjoint(residues, moduli):
    """Check if all pairs are disjoint using the exact gcd criterion."""
    n = len(moduli)
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            if residues[i] % g != residues[j] % g:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 90s
    
    # We want to maximize the number of disjoint residue classes
    # with distinct moduli in 2..128.
    #
    # The seed uses powers of 2: 2, 4, 8, 16, 32, 64, 128 (7 classes).
    # Residues: 0, 1, 3, 7, 15, 31, 63 (i.e., 2^k - 1 for k=0..6)
    #
    # Key insight: Two residue classes (r1, m1) and (r2, m2) are disjoint
    # iff r1 not congruent r2 mod gcd(m1, m2).
    #
    # To maximize count, we should use moduli that share large gcds,
    # because then the constraint is weaker (fewer residues are excluded).
    # Powers of 2 have nested gcd structure: gcd(2^a, 2^b) = 2^min(a,b).
    #
    # Let's try to build a larger set by mixing in other prime powers
    # or by using a different strategy.
    #
    # Strategy: Greedy with backtracking, trying to add as many as possible.
    # Order moduli by increasing value (smaller moduli constrain more).
    
    max_mod = 128
    all_moduli = list(range(2, max_mod + 1))
    
    # Try different orderings and strategies
    # First, let's verify the seed solution
    seed_moduli = [2**k for k in range(1, 8)]  # 2,4,8,16,32,64,128
    seed_residues = [2**(k-1) - 1 for k in range(1, 8)]  # 0,1,3,7,15,31,63
    assert _canonical_pairwise_disjoint(seed_residues, seed_moduli), "Seed should be valid"
    
    # Try to find a larger set using greedy + local search
    best_count = 7
    best_solution = [(seed_residues[i], seed_moduli[i]) for i in range(7)]
    
    # Greedy approach: try adding moduli in various orders
    # Sort by prime factorization complexity or by gcd overlap
    
    def try_greedy(moduli_order):
        """Try to build a set greedily following moduli_order."""
        used_moduli = set()
        residues = []
        moduli = []
        
        for m in moduli_order:
            if m in used_moduli:
                continue
            # Try all residues 0..m-1 and see if any works
            found = False
            for r in range(m):
                test_r = residues + [r]
                test_m = moduli + [m]
                if _canonical_pairwise_disjoint(test_r, test_m):
                    residues.append(r)
                    moduli.append(m)
                    used_moduli.add(m)
                    found = True
                    break
            if not found:
                continue
        return list(zip(residues, moduli))
    
    # Try ordering by decreasing modulus (larger moduli first, more residue choices)
    desc_order = sorted(all_moduli, reverse=True)
    sol = try_greedy(desc_order)
    if len(sol) > best_count:
        best_count = len(sol)
        best_solution = sol
    
    # Try ordering by increasing modulus
    asc_order = sorted(all_moduli)
    sol = try_greedy(asc_order)
    if len(sol) > best_count:
        best_count = len(sol)
        best_solution = sol
    
    # Try ordering by prime factors count (fewer prime factors first)
    def prime_factor_count(n):
        count = 0
        temp = n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                count += 1
                temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    pf_order = sorted(all_moduli, key=lambda x: (prime_factor_count(x), x))
    sol = try_greedy(pf_order)
    if len(sol) > best_count:
        best_count = len(sol)
        best_solution = sol
    
    # Try ordering by decreasing prime factor count
    pf_order_desc = sorted(all_moduli, key=lambda x: (-prime_factor_count(x), x))
    sol = try_greedy(pf_order_desc)
    if len(sol) > best_count:
        best_count = len(sol)
        best_solution = sol
    
    # Try a more sophisticated approach: only use prime powers
    # Prime powers up to 128
    prime_powers = []
    for base in range(2, 129):
        # Check if base is prime
        is_prime = True
        for d in range(2, int(base**0.5) + 1):
            if base % d == 0:
                is_prime = False
                break
        if not is_prime:
            continue
        p = base
        while p <= 128:
            prime_powers.append(p)
            p *= base
    
    # Also add prime powers where base is composite but power is prime power
    # e.g., 4=2^2, 8=2^3, 9=3^2, etc.
    # Actually, let's just enumerate all prime powers
    prime_powers = set()
    for base in range(2, 129):
        # Check if base is prime
        is_prime = True
        for d in range(2, int(base**0.5) + 1):
            if base % d == 0:
                is_prime = False
                break
        if not is_prime:
            continue
        p = base
        while p <= 128:
            prime_powers.add(p)
            p *= base
    
    prime_powers_list = sorted(prime_powers)
    
    # Try greedy on prime powers
    sol = try_greedy(prime_powers_list)
    if len(sol) > best_count:
        best_count = len(sol)
        best_solution = sol
    
    sol = try_greedy(sorted(prime_powers_list, reverse=True))
    if len(sol) > best_count:
        best_count = len(sol)
        best_solution = sol
    
    # Check deadline
    if time.time() > deadline:
        return best_solution
    
    # Try a local improvement: start from seed, try to add more
    # by replacing some elements or trying different residues
    
    # For each pair of consecutive powers of 2, see if we can insert
    # another modulus between them
    
    # Actually, let's try a complete search for small cases
    # The maximum possible is limited. Let's try to find if 8 is achievable.
    
    # Key constraint: if we use moduli m1, m2, ..., mk, for each pair
    # gcd(mi, mj) must divide (ri - rj).
    
    # Let's try a randomized approach with many restarts
    import random
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random ordering
        random_order = all_moduli[:]
        random.shuffle(random_order)
        sol = try_greedy(random_order)
        if len(sol) > best_count:
            best_count = len(sol)
            best_solution = sol
    
    return best_solution

# EVOLVE-BLOCK-END
