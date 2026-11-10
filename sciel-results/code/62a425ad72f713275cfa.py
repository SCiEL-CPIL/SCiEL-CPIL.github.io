import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin for 90s limit
    
    MAX_MOD = 128
    # Moduli from 2 to 128
    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)
    
    # Two residue classes (r1, m1) and (r2, m2) are disjoint iff
    # r1 % gcd(m1, m2) != r2 % gcd(m1, m2)
    # We need to select a subset of moduli (all distinct by construction) and assign
    # residues such that for every pair, the disjointness condition holds.
    
    # Key insight: if we select moduli that are pairwise coprime, we can always
    # assign residues (e.g., all 0s won't work if gcd=1 means 0%1=0=0%1... wait)
    # Actually if gcd(m1,m2)=1, then r1%1 = 0 and r2%1 = 0, so they're always "equal" mod 1.
    # So pairwise coprime moduli with all residues 0 would NOT be disjoint!
    
    # Let me re-read: "Every two infinite residue classes must be disjoint"
    # Class A = {x : x ≡ r1 (mod m1)}, Class B = {x : x ≡ r2 (mod m2)}
    # They are disjoint iff there's no x such that x≡r1 mod m1 AND x≡r2 mod m2.
    # By CRT, a solution exists iff r1 ≡ r2 (mod gcd(m1,m2)).
    # So they are disjoint iff r1 ≠ r2 (mod gcd(m1,m2)).
    
    # If gcd(m1,m2) = 1, then r1 mod 1 = 0 = r2 mod 1, so they are NEVER disjoint!
    # This means we CANNOT use pairwise coprime moduli!
    
    # We need gcd(m1,m2) >= 2 for every pair, and then choose residues carefully.
    # Actually we need: for every pair (i,j), r_i mod gcd(m_i,m_j) ≠ r_j mod gcd(m_i,m_j).
    
    # Strategy: select moduli that share a common factor. E.g., all even numbers.
    # If all moduli are even, gcd(m_i, m_j) >= 2 for all pairs.
    
    # For all-even moduli: gcd(m_i, m_j) is at least 2.
    # We need r_i mod gcd(m_i,m_j) ≠ r_j mod gcd(m_i,m_j) for all pairs.
    
    # Simpler approach: use moduli that are all multiples of a large power of 2.
    # E.g., moduli that are multiples of 64: 64, 128. Only 2 moduli.
    # Or multiples of 32: 32, 64, 96, 128. 4 moduli.
    # Or all even: 2,4,6,...,128. 64 moduli.
    
    # For all even moduli, let's try to assign residues.
    # If m_i is even, gcd(m_i, m_j) >= 2.
    # We need r_i mod gcd(m_i,m_j) ≠ r_j mod gcd(m_i,m_j).
    
    # A simple scheme: set r_i = m_i - 1 (i.e., r_i ≡ -1 mod m_i).
    # Then r_i mod gcd(m_i,m_j) = (m_i-1) mod gcd(m_i,m_j) = -1 mod gcd(m_i,m_j) = gcd(m_i,m_j) - 1.
    # Similarly r_j mod gcd = gcd - 1.
    # So they'd be EQUAL, not disjoint! Bad.
    
    # Try r_i = 0 for all. Then r_i mod gcd = 0 = r_j mod gcd. Equal. Bad.
    
    # Try r_i = 1 for all. Same problem: 1 mod gcd is same for both.
    
    # We need a scheme where r_i mod gcd(m_i,m_j) differs for each pair.
    
    # Key: if we pick moduli m_i = 2 * p_i where p_i are distinct primes,
    # then gcd(m_i, m_j) = 2 for i≠j (assuming p_i ≠ p_j).
    # Then we need r_i mod 2 ≠ r_j mod 2 for all i≠j.
    # But r mod 2 can only be 0 or 1. So at most 2 classes!
    
    # Better: pick moduli with larger shared factors.
    # If m_i = 2^k * p_i, and we ensure gcd(m_i, m_j) = 2^k for all pairs,
    # then we need r_i mod 2^k to be all distinct. Max 2^k classes.
    
    # Maximize 2^k * (number of distinct primes p_i such that 2^k * p_i <= 128)
    # Subject to 2^k * p_i <= 128.
    
    # For k=1: 2*p <= 128, p <= 64. Primes <= 64: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61. 
    #   But p=2 gives m=4, gcd(4, 2*3)=gcd(4,6)=2. OK.
    #   Actually we need distinct moduli, and moduli must be distinct.
    #   Moduli: 2*2=4, 2*3=6, 2*5=10, ..., 2*61=122. That's 18 moduli.
    #   But we can also use m=2 itself? 2 = 2*1, but 1 is not prime. 
    #   Let's include m=2: gcd(2, 2p) = 2 for any prime p. So m=2 works too.
    #   Moduli: 2, 4, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126.
    #   Wait, 2*p for primes p <= 64: p in {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61}
    #   2*2=4, 2*3=6, 2*5=10, 2*7=14, 2*11=22, 2*13=26, 2*17=34, 2*19=38, 2*23=46, 2*29=58, 2*31=62, 2*37=74, 2*41=82, 2*43=86, 2*47=94, 2*53=110, 2*59=118, 2*61=122.
    #   Plus m=2. Total: 19 moduli.
    #   gcd of any two: gcd(2, 2p) = 2, gcd(2p, 2q) = 2*gcd(p,q) = 2 (for distinct primes p,q).
    #   So gcd is always 2. We need r_i mod 2 all distinct → at most 2!
    #   This doesn't work for large sets.
    
    # Better approach: use powers of 2 times small numbers to get larger gcds.
    # Use moduli of the form 2^k where k varies: 2,4,8,16,32,64,128. 7 moduli.
    # gcd(2^a, 2^b) = 2^min(a,b).
    # We need r_i mod 2^min(a_i,a_j) ≠ r_j mod 2^min(a_i,a_j).
    # This is the original seed problem!
    
    # To do better, mix in other moduli.
    # Let's use a greedy/backtracking approach.
    
    # Generate all candidate (residue, modulus) pairs.
    # For each modulus m in 2..128, residues 0..m-1.
    # Total candidates: sum_{m=2}^{128} m = (2+128)*127/2 = 127*65 = 8255.
    
    # Build conflict graph: two candidates conflict if they're NOT disjoint.
    # Candidates (r1,m1) and (r2,m2) conflict iff r1 % gcd(m1,m2) == r2 % gcd(m1,m2).
    # Also, we can only pick one candidate per modulus.
    
    # This is a maximum independent set in a graph with the additional constraint
    # of at most one per modulus. This is hard in general.
    
    # Greedy approach: try to build a large set.
    # Sort candidates by some heuristic and greedily add if compatible.
    
    # Let's try: for each modulus, pick a residue. Use backtracking.
    
    # Simpler: use the fact that if we pick moduli that are all multiples of L,
    # and assign residues such that they're all distinct mod L, we get disjoint classes.
    # Maximize the count.
    
    # For L=2: multiples of 2 in 2..128: 64 numbers. Residues mod 2: 0 or 1. Max 2.
    # For L=4: multiples of 4: 32 numbers. Residues mod 4: 0,1,2,3. Max 4.
    # For L=8: 16 numbers. Max 8.
    # For L=16: 8 numbers. Max 8.
    # For L=32: 4 numbers. Max 32, but only 4 moduli available.
    # For L=64: 2 numbers.
    # For L=128: 1 number.
    
    # So using multiples of 8: moduli 8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128. 16 moduli.
    # Assign residues 0,1,2,...,7,0,1,2,...,7,0,1,2 (each used at most 8 times, but we need all distinct mod 8).
    # 16 moduli, residues mod 8: we can assign 0,1,...,7,0,1,...,7 → but 0 appears twice!
    # We need all 16 residues distinct mod 8, but there are only 8 residue classes mod 8.
    # So max 8 classes with L=8!
    
    # Hmm, this approach is limited. Let me think differently.
    
    # Use a different structure: moduli with varying gcds.
    # m1=2, m2=4, m3=8, m4=16, m5=32, m6=64, m7=128 (powers of 2, 7 moduli)
    # Plus: m8=6 (gcd with 2 is 2, with 4 is 2, with 8 is 2, etc.)
    # m9=10 (gcd with 2 is 2, with others >=2)
    
    # This is getting complex. Let me just do a greedy search.
    
    # Greedy: try to add (r, m) pairs one by one.
    # Order moduli by some criterion, try residues for each.
    
    # Better: use a branch and bound or just greedy with multiple orderings.
    
    best_result = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]  # seed fallback
    best_len = len(best_result)
    
    def is_compatible(existing, r_new, m_new):
        for (r_old, m_old) in existing:
            g = math.gcd(m_old, m_new)
            if r_new % g == r_old % g:
                return False
        return True
    
    def try_greedy(order):
        result = []
        for m in order:
            best_r = None
            for r in range(m):
                if is_compatible(result, r, m):
                    best_r = r
                    break
            if best_r is not None:
                result.append((best_r, m))
        return result
    
    # Try various orderings
    orders = [
        list(range(2, 129)),
        list(range(128, 1, -1)),
        sorted(range(2, 129), key=lambda m: -m.bit_length()),
        sorted(range(2, 129), key=lambda m: m.bit_length()),
        list(range(2, 129, 2)),  # evens
        list(range(128, 1, -2)),
    ]
    
    for order in orders:
        res = try_greedy(order)
        if len(res) > best_len:
            best_len = len(res)
            best_result = [[r, m] for (r, m) in res]
        if time.time() > deadline:
            break
    
    # Try a smarter approach: use moduli that are all multiples of a base, 
    # but with varying structures.
    
    # Actually, let's try to use all even moduli with a careful residue assignment.
    # For even moduli m, we need r_i mod gcd(m_i, m_j) != r_j mod gcd(m_i, m_j).
    
    # If we set r_i = m_i // 2 for each even modulus m_i, then:
    # r_i mod gcd(m_i, m_j) = (m_i/2) mod gcd(m_i, m_j)
    # This is complex. Let me just do a more thorough greedy.
    
    # Try: for each modulus, try ALL residues and pick the one that maximizes future options.
    # Or just try multiple random orderings.
    
    import random
    random.seed(42)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(2, 129))
        random.shuffle(order)
        res = try_greedy(order)
        if len(res) > best_len:
            best_len = len(res)
            best_result = [[r, m] for (r, m) in res]
    
    # Try a specific structure: powers of 2 plus carefully chosen others
    # Powers of 2: 2,4,8,16,32,64,128 with residues 1,3,7,15,31,63,127
    # These are pairwise disjoint (verified by seed).
    # Can we add more?
    
    base = [(1, 2), (3, 4), (7, 8), (15, 16), (31, 32), (63, 64), (127, 128)]
    
    # Try adding other moduli
    for m in range(2, 129):
        if m in [b[1] for b in base]:
            continue
        for r in range(m):
            if is_compatible(base, r, m):
                # Check if adding this breaks anything (it doesn't, since is_compatible checks all)
                # But we can only add one per modulus, and we want to maximize count
                # Let's try to build a maximal set
                pass
    
    # Let's do a proper greedy: start with empty, add candidates in order of (modulus, residue)
    # But try to be smart about residue choice.
    
    # For each modulus, the "best" residue is one that's incompatible with the fewest 
    # already-chosen classes. Since we process in order, this is just the first compatible one.
    
    # Let's try processing in order of decreasing modulus (harder constraints first).
    result = []
    for m in range(128, 1, -1):
        if time.time() > deadline:
            break
        found = False
        for r in range(m):
            if is_compatible(result, r, m):
                result.append((r, m))
                found = True
                break
        if not found:
            continue
    
    if len(result) > best_len:
        best_len = len(result)
        best_result = [[r, m] for (r, m) in result]
    
    # Try another strategy: process moduli in order of increasing number of prime factors
    # (simpler moduli first, leaving more flexibility)
    
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    result2 = []
    for m in sorted(range(2, 129), key=num_prime_factors):
        if time.time() > deadline:
            break
        for r in range(m):
            if is_compatible(result2, r, m):
                result2.append((r, m))
                break
    
    if len(result2) > best_len:
        best_len = len(result2)
        best_result = [[r, m] for (r, m) in result2]
    
    # Convert to required format
    if time.time() > deadline - 2:
        # Return what we have
        return best_result
    
    return best_result
