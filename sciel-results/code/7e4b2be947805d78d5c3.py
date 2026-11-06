import math
import random
import time

MAX_MOD = 128
DEADLINE_MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _greedy_construct(moduli, deadline):
    """
    Greedy construction:
    1. Sort moduli by number of prime factors (ascending) to prioritize "constrained" moduli?
       Actually, moduli with more prime factors have more potential conflicts.
       Let's try sorting by "difficulty" - maybe just ascending value or by number of divisors.
       For even moduli, they all share factor 2.
       The constraint is that for any pair (m_i, m_j), we need r_i not congruent to r_j mod gcd(m_i, m_j).
       
       A simple strategy:
       Assign residues one by one.
       For each modulus m, choose a residue r that is compatible with all previously assigned.
       
       To maximize success rate, we can use a randomized greedy with restarts.
    """
    n = len(moduli)
    if n == 0:
        return []
    
    # We will try to assign residues to the given moduli.
    # We return the list of (r, m) pairs.
    
    # Heuristic: Sort moduli. 
    # Moduli that are multiples of more small primes might be more restrictive?
    # Or maybe we should just process them in increasing order of modulus?
    # Let's try sorting by the number of distinct prime factors, descending.
    # More prime factors -> more gcds > 1 with other numbers -> more constraints.
    
    def num_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    # Sort by number of prime factors, descending.
    # This puts highly composite numbers first.
    sorted_mods = sorted(moduli, key=lambda m: (-num_prime_factors(m), m))
    
    residues = [None] * n
    moduli_list = sorted_mods[:]
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
            
        if idx == n:
            return True
            
        m = moduli_list[idx]
        # Try residues in random order
        residues_to_try = list(range(m))
        random.shuffle(residues_to_try)
        
        for r in residues_to_try:
            valid = True
            for j in range(idx):
                if not _disjoint(r, m, residues[j], moduli_list[j]):
                    valid = False
                    break
            if valid:
                residues[idx] = r
                if backtrack(idx + 1):
                    return True
        residues[idx] = None
        return False

    if backtrack(0):
        return [(residues[i], moduli_list[i]) for i in range(n)]
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    # Candidate sets of moduli, ordered by expected size (larger is better)
    # 1. All even moduli (64 items)
    # 2. All odd moduli (63 items) - but odd moduli include coprime pairs, so they are likely invalid.
    #    Wait, the requirement is that the CHOSEN set must be pairwise disjoint.
    #    If we pick odd moduli, we must ensure no two are coprime? No, we just need to find residues.
    #    But it's much harder to find disjoint residues for coprime moduli (always possible actually, since gcd=1 means any residues are disjoint? No.
    #    If gcd(m1, m2)=1, then (r2-r1) % 1 == 0 is always true. So _disjoint returns False.
    #    Wait, _disjoint returns (r2-r1)%g != 0. If g=1, (r2-r1)%1 is 0. So it returns False.
    #    This means if two moduli are coprime, they CAN NEVER be disjoint!
    #    Let's re-verify the math.
    #    Classes are {x | x ≡ r1 (m1)} and {y | y ≡ r2 (m2)}.
    #    They are disjoint iff there is no x such that x ≡ r1 (m1) and x ≡ r2 (m2).
    #    By CRT, a solution exists iff r1 ≡ r2 (mod gcd(m1, m2)).
    #    So they are disjoint iff r1 NOT ≡ r2 (mod gcd(m1, m2)).
    #    If gcd(m1, m2) = 1, then r1 ≡ r2 (mod 1) is always true.
    #    So a solution ALWAYS exists.
    #    Therefore, the classes are NEVER disjoint if gcd(m1, m2) = 1.
    #    So we CANNOT include any pair of coprime moduli in our set.
    #    This confirms that we must pick a set of moduli where NO two are coprime.
    #    i.e., the set must be "pairwise non-coprime".
    
    # What is the largest subset of [2, 128] such that no two elements are coprime?
    # This is a known problem.
    # The set of all even numbers is pairwise non-coprime (gcd >= 2). Size 64.
    # Can we add any odd number to the evens?
    # If we add an odd number m, it must share a factor with every even number in the set.
    # Since the set contains 2 (or 4, 6, etc.), m must share a factor with 2? No, m is odd, so gcd(m, 2)=1.
    # So m cannot be coprime to ANY even number.
    # But gcd(m, 2) = 1 always for odd m.
    # So we cannot add any odd number to a set containing 2.
    # What if we remove 2?
    # Set: {4, 6, 8, ...} (even numbers >= 4).
    # Can we add an odd number m?
    # m must share a factor with 4 -> m must be even? No, gcd(m,4)=1 if m is odd.
    # So m cannot be added if 4 is in the set.
    # Generally, if the set contains any even number E, any odd number O added must have gcd(O, E) > 1.
    # But if the set contains 2, gcd(O, 2)=1, so O cannot be added.
    # If the set contains 4 but not 2, gcd(O, 4)=1 for odd O. So O cannot be added.
    # If the set contains 6 but not 2 or 4?
    # We need a set of moduli such that every pair shares a factor.
    # The largest such set in [2, N] is typically the set of even numbers.
    # Proof sketch:
    # Let S be a subset of [2, N] with pairwise non-coprime elements.
    # If S contains an odd number o, then every other element in S must share a factor with o.
    # Let p be a prime factor of o. Then every other element in S must be divisible by p OR share another factor with o.
    # This gets complex.
    # However, it is a known result that the maximum size of a pairwise non-coprime subset of [2, N] is floor(N/2) (the evens).
    # Let's check N=128. Evens are 2, 4, ..., 128. Count = 64.
    # Can we get more?
    # If we pick all multiples of 2, we get 64.
    # If we pick all multiples of 3, we get floor(128/3) = 42.
    # If we pick all multiples of 2 and some odds?
    # As shown, if 2 is in the set, no odd can be added.
    # If we remove 2, can we add odds?
    # Suppose we have {4, 6, 8, ...}.
    # Add 3? gcd(3,4)=1. Fail.
    # Add 9? gcd(9,4)=1. Fail.
    # It seems very hard to beat the evens.
    
    # So the target is 64.
    # The current program already targets 64.
    # Why did it score 1.7143?
    # 1.7143 * 7 (seed) = 12? No, seed score is relative.
    # Score = value / seed_value.
    # If seed value was 7 (just guessing), 1.7143 * 7 = 12.
    # But the inspiration says "Including all 63 even moduli... Measured: 1 -> 1.857".
    # Wait, the inspiration text says:
    # "Including all 63 even moduli (vs seed's 7) ... Measured: 1 -> 1.857."
    # This implies the seed had 7? Or the baseline was 1?
    # The task says: "search score is value/seed".
    # If the current program scores 1.7143, and it returns 64 classes, then seed_value = 64 / 1.7143 = 37.3?
    # Or maybe the seed program returned a small number.
    
    # Let's look at the inspiration again.
    # "inspiration 1 (score 2.2857): ... defines _solve_csp ..."
    # The diff shows a change from _construct_residues to _solve_csp.
    # The child p37 scored 2.2857.
    # The parent p33 scored 1.7143.
    # The change was in the backtracking strategy.
    # p33 used random shuffling of moduli in each attempt?
    # p37 sorted moduli by "density" of co-prime factors?
    
    # The current program (p33) shuffles `even_mods` randomly in each attempt.
    # p37 (the better one) sorted them.
    # Let's adopt the sorting strategy from p37.
    # p37 sorted by "decreasing density of co-..." (cut off).
    # Likely: sort by number of prime factors, or by value.
    
    # Let's try sorting by value (ascending) or by number of divisors.
    # Actually, for CSP, "fail-first" heuristic is good.
    # Variables with smallest domain first? Domain size is m. So small m first.
    # Constraints with high degree first?
    
    # Let's try sorting moduli in ASCENDING order.
    # Small moduli have fewer residues to try, so they fail faster if no solution.
    
    # Also, the current program restarts with `random.seed(attempts)`.
    # This is fine.
    
    # Let's refine the backtracking:
    # 1. Sort moduli ascending.
    # 2. In backtrack, try residues in a specific order?
    #    Maybe residues that are "compatible" with the most previous assignments?
    #    Or just random.
    
    # Let's also consider that we might not need to restart many times if the order is good.
    # Ascending order should be very effective.
    
    # One more thing: The current program has a fallback greedy.
    # If backtracking fails, it uses greedy.
    # But backtracking should succeed for 64 evens.
    
    # Let's implement the sorted backtracking.
    
    even_mods = [m for m in all_mods if m % 2 == 0]
    # Sort ascending
    even_mods.sort()
    
    best_pairs = []
    
    # Try to solve with sorted order
    # We might still need random restarts if the first attempt fails, but with sorted order it should be fast.
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Use a fixed order for moduli (ascending)
        # But randomize the residue trial order?
        # Or use a deterministic residue order?
        # Random residue order is usually safer to avoid getting stuck in a local bad choice.
        
        residues = [None] * len(even_mods)
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(even_mods):
                return True
            
            m = even_mods[idx]
            # Try residues
            # Optimization: We can filter residues based on previous constraints?
            # For now, just shuffle.
            
            # To add variety between attempts, we can seed random with attempt
            # But we are inside the loop.
            
            residues_to_try = list(range(m))
            random.shuffle(residues_to_try)
            
            for r in residues_to_try:
                valid = True
                for j in range(idx):
                    if not _disjoint(r, m, residues[j], even_mods[j]):
                        valid = False
                        break
                if valid:
                    residues[idx] = r
                    if backtrack(idx + 1):
                        return True
            residues[idx] = None
            return False
        
        # Set seed for reproducibility of the attempt's randomness
        random.seed(42 + attempt)
        
        if backtrack(0):
            best_pairs = [(residues[i], even_mods[i]) for i in range(len(even_mods))]
            break
        
    # If we failed to find a solution for all 64, try greedy fallback
    if not best_pairs:
        # Greedy
        pairs = []
        for m in even_mods:
            if time.time() > deadline:
                break
            found = False
            for r in range(m):
                ok = True
                for (r2, m2) in pairs:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    pairs.append((r, m))
                    found = True
                    break
        best_pairs = pairs
        
    if not best_pairs:
        best_pairs = [(0, 2)]
        
    return [[r, m] for r, m in best_pairs]
