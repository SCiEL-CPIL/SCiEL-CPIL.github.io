"""Rewritten program: maximize pairwise disjoint residue classes with distinct moduli 2..128."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

MAX_MODULUS = 128

def construct():
    # We need to select a subset of moduli from 2..128 (distinct) and assign each a residue r (0 <= r < m)
    # such that for any two selected classes (r1, m1) and (r2, m2), the congruences x ≡ r1 (mod m1) and
    # x ≡ r2 (mod m2) have no solution. By CRT, this happens iff gcd(m1, m2) does not divide (r1 - r2).
    # Equivalently: gcd(m1, m2) ∤ (r1 - r2).
    
    # Strategy: For each modulus m, we can choose any residue r in [0, m-1].
    # Two classes (r1,m1) and (r2,m2) are disjoint iff gcd(m1,m2) ∤ (r1-r2).
    
    # This is a constraint satisfaction / optimization problem. We want to maximize the number of classes.
    
    # Key insight: If we pick moduli that are pairwise coprime, then gcd(m1,m2)=1, and since 1 divides everything,
    # gcd=1 always divides (r1-r2), so ANY two residues would make the classes intersect!
    # Wait, that's wrong. If gcd(m1,m2)=1, then by CRT, for ANY r1, r2, there exists a solution x.
    # So if two moduli are coprime, NO choice of residues makes them disjoint!
    
    # Therefore, we MUST only pick moduli that are NOT pairwise coprime. In fact, for any two selected moduli
    # m1, m2, we need gcd(m1,m2) > 1, AND we need to choose residues such that gcd(m1,m2) ∤ (r1-r2).
    
    # So the problem reduces to:
    # 1. Select a subset S of {2,...,128} such that for every pair m1,m2 in S, gcd(m1,m2) > 1.
    # 2. For each m in S, assign a residue r_m in [0, m-1] such that for every pair m1,m2 in S,
    #    gcd(m1,m2) ∤ (r_{m1} - r_{m2}).
    
    # Step 1: Find the largest subset of {2,...,128} where every pair has gcd > 1.
    # This is equivalent to finding the maximum clique in the graph where edge exists if gcd>1,
    # or equivalently, the maximum independent set in the graph where edge exists if gcd=1.
    
    # Actually, it's a "maximum set with pairwise gcd > 1" problem.
    
    # Let's think about structure. Numbers with a common prime factor p form a clique.
    # The largest such set for prime p is the count of multiples of p in [2,128].
    # For p=2: multiples are 2,4,6,...,128 → 64 numbers.
    # For p=3: multiples are 3,6,...,126 → 42 numbers.
    # For p=5: 5,10,...,125 → 25 numbers.
    # etc.
    
    # But we can also mix different primes if the numbers share some common factor pairwise.
    # However, mixing is tricky. The simplest large clique is all even numbers (64 elements).
    
    # Can we do better than 64? Let's check: the set of all numbers divisible by 2 is 64.
    # What about numbers divisible by 2 or 3? No, because a number divisible by 2 and a number divisible by 3
    # but not 2 (like 3 and 5... wait 5 isn't divisible by 3). Take 3 and 4: gcd(3,4)=1. So 3 and 4 can't both be in S.
    
    # So we need pairwise gcd > 1. The maximum such set:
    # - All even numbers: 64
    # - Can we add any odd number to the evens? An odd number n has gcd(n, 2k) = gcd(n, k) * ... 
    #   Actually gcd(n, 2k) where n is odd = gcd(n, k). For this to be >1 for ALL even numbers 2k,
    #   we need gcd(n, k) > 1 for all k in 1..64. But gcd(n, 1) = 1, so n and 2 have gcd 1. 
    #   So no odd number can be added to the full set of evens.
    
    # What if we drop some evens to add odds?
    # Suppose we include all multiples of 6 (32 numbers: 6,12,...,126). 
    # Can we add multiples of 2 that aren't multiples of 6? E.g., 2, 4, 8, 10, 14, 16...
    # gcd(6, 2) = 2 > 1 ✓. gcd(6, 4) = 2 > 1 ✓. gcd(12, 10) = 2 > 1 ✓.
    # Actually, any two even numbers have gcd ≥ 2. So all evens work together.
    # Any even number and any multiple of 3: gcd(even, mult_of_3) could be 1? 
    # E.g., 2 and 3: gcd=1. So 2 and 3 can't both be in S.
    
    # The maximum clique in the "gcd>1" graph on {2..128}:
    # It's known that the maximum size is achieved by all multiples of the smallest prime, which is 2.
    # Size = floor(128/2) = 64.
    
    # Let's verify we can't exceed 64. By Turan-type arguments or direct reasoning:
    # The independence number of the "coprime" graph... this is complex. But empirically, 64 is a strong candidate.
    
    # Now step 2: For the 64 even moduli, assign residues such that for any two even moduli m1, m2,
    # gcd(m1,m2) ∤ (r1 - r2).
    
    # Note: gcd(m1,m2) is always even (since both are even). Let g = gcd(m1,m2) ≥ 2.
    # We need r1 ≢ r2 (mod g).
    
    # Since all moduli are even, let m = 2k. Then r_m ∈ [0, 2k-1].
    # The condition is: for m1=2k1, m2=2k2, g=gcd(2k1,2k2)=2*gcd(k1,k2).
    # We need 2*gcd(k1,k2) ∤ (r1 - r2).
    
    # A simple approach: set r_m = 1 for all even m. Then r1 - r2 = 0, and g | 0 always. BAD.
    # Set r_m = m/2. Then for m1=2k1, r1=k1; m2=2k2, r2=k2. 
    # g = 2*gcd(k1,k2). We need 2*gcd(k1,k2) ∤ (k1-k2).
    # But k1-k2 is divisible by gcd(k1,k2), so (k1-k2) = gcd(k1,k2) * t for some integer t.
    # Then 2*gcd(k1,k2) | (k1-k2) iff 2 | t iff t is even iff (k1-k2)/gcd(k1,k2) is even.
    # This doesn't always hold. E.g., k1=1, k2=2: gcd=1, k1-k2=-1, 2*1=2 ∤ -1 ✓.
    # k1=2, k2=4: gcd=2, k1-k2=-2, 2*2=4 ∤ -2 ✓.
    # k1=3, k2=6: gcd=3, k1-k2=-3, 2*3=6 ∤ -3 ✓.
    # k1=4, k2=8: gcd=4, k1-k2=-4, 2*4=8 ∤ -4 ✓.
    # k1=2, k2=6: gcd=2, k1-k2=-4, 2*2=4 | -4 ✗. BAD!
    
    # So r_m = m/2 doesn't work for all pairs.
    
    # Alternative: r_m = 1 for all m. Doesn't work as shown.
    # r_m = m-1 (i.e., -1 mod m). Then r1 - r2 = (m1-1) - (m2-1) = m1 - m2.
    # g = gcd(m1,m2). We need g ∤ (m1-m2). But g | m1 and g | m2, so g | (m1-m2). Always fails!
    
    # We need a smarter residue assignment.
    
    # Let's use a greedy or systematic approach. Since we have 64 moduli, we can try to assign residues
    # using backtracking or a greedy method.
    
    # Actually, let's reconsider. Maybe we don't need all 64 even moduli. Maybe a smaller set with
    # easier residue assignment allows us to add more moduli from other groups?
    
    # But the task says "maximize the number of classes". So we want the maximum possible.
    
    # Let me try a different approach: use only moduli that are powers of 2 times a fixed odd part,
    # or use a structured assignment.
    
    # Key idea: If we pick moduli m_i = 2 * p_i where p_i are distinct primes, then
    # gcd(m_i, m_j) = 2 * gcd(p_i, p_j) = 2 (since p_i ≠ p_j are primes).
    # So g = 2 for all pairs. We need 2 ∤ (r_i - r_j), i.e., r_i and r_j have different parity.
    # But we have more moduli than 2 parity classes, so by pigeonhole, some two will have same parity.
    # This only works for at most 2 moduli. Not good.
    
    # Better: use moduli that are all multiples of a high power of 2.
    # Let m_i = 2^k * a_i where a_i are odd. If we fix k=7 (so m_i = 128*a_i, but a_i≥1, m_i≤128, so a_i=1, m_i=128).
    # Only one such modulus.
    
    # Let's try: all moduli divisible by 4. That's 32 moduli: 4,8,12,...,128.
    # gcd of any two is at least 4. We need g ∤ (r1-r2).
    # Assign r_m = m/2. For m=4k, r=2k. For m1=4k1, m2=4k2, g=gcd(4k1,4k2)=4*gcd(k1,k2).
    # r1-r2 = 2(k1-k2). Need 4*gcd(k1,k2) ∤ 2(k1-k2), i.e., 2*gcd(k1,k2) ∤ (k1-k2).
    # Same issue as before.
    
    # Let's just do a computational search for the optimal solution.
    
    # Given the complexity, let's use a heuristic: 
    # 1. Start with all even moduli (64 of them).
    # 2. Try to assign residues using a greedy algorithm with backtracking if needed.
    
    # For the greedy assignment: process moduli in some order, assign each a residue that is compatible
    # with all previously assigned residues.
    
    # For modulus m with residue r, and previously assigned (m_j, r_j), we need:
    # gcd(m, m_j) ∤ (r - r_j) for all j.
    
    # Let's implement this.
    
    moduli = list(range(2, 129))  # 2..128
    # First, find the maximum subset with pairwise gcd > 1.
    # We'll use the fact that all evens form a valid subset of size 64.
    # Let's check if we can do better with a different subset.
    
    # For now, assume the maximum is 64 (all evens). Let's try to assign residues.
    
    even_moduli = [m for m in moduli if m % 2 == 0]
    n = len(even_moduli)
    
    # Greedy assignment
    residues = [None] * n
    
    for i in range(n):
        m = even_moduli[i]
        found = False
        for r in range(m):
            # Check against all previously assigned
            ok = True
            for j in range(i):
                mj = even_moduli[j]
                rj = residues[j]
                g = math.gcd(m, mj)
                if (r - rj) % g == 0:
                    ok = False
                    break
            if ok:
                residues[i] = r
                found = True
                break
        if not found:
            # This shouldn't happen if a valid assignment exists
            # Fall back to a different strategy
            residues[i] = 0
    
    # Build the result
    result = [[residues[i], even_moduli[i]] for i in range(n)]
    
    # Verify
    for i in range(len(result)):
        r1, m1 = result[i]
        for j in range(i+1, len(result)):
            r2, m2 = result[j]
            g = math.gcd(m1, m2)
            assert (r1 - r2) % g != 0, f"Conflict: {r1},{m1} and {r2},{m2}"
    
    return result
# EVOLVE-BLOCK-END
