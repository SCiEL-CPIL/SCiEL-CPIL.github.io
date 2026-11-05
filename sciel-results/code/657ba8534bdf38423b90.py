import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit for 90s total
    
    max_modulus = 128
    all_moduli = list(range(2, max_modulus + 1))
    
    # Precompute gcd for all pairs of moduli
    gcd_table = {}
    for i, m1 in enumerate(all_moduli):
        for m2 in all_moduli[i:]:
            d = gcd(m1, m2)
            gcd_table[(m1, m2)] = d
            if m1 != m2:
                gcd_table[(m2, m1)] = d
    
    def is_compatible(r, m, selected):
        """Check if (r, m) is disjoint from all selected classes."""
        for r_i, m_i in selected:
            d = gcd_table[(m, m_i)]
            if r % d == r_i % d:
                return False
        return True
    
    # Strategy: We want to maximize the number of classes.
    # This is equivalent to finding a large independent set in a graph where
    # nodes are (r, m) pairs, but with the constraint that we can pick at most one
    # r per m. Actually, it's a packing problem.
    
    # A key insight: If we pick residues randomly, the probability of conflict is
    # roughly 1/gcd(m, m_i). For coprime moduli, gcd=1, so any residue works?
    # No. If gcd(m, m_i) = 1, then r % 1 == r_i % 1 is always 0 == 0.
    # Wait. The condition for disjointness is:
    # There is NO x such that x ≡ r (mod m) and x ≡ r_i (mod m_i).
    # By CRT, a solution exists iff r ≡ r_i (mod gcd(m, m_i)).
    # So they are DISJOINT iff r NOT congruent r_i (mod gcd(m, m_i)).
    
    # If gcd(m, m_i) = 1, then r % 1 is always 0, and r_i % 1 is always 0.
    # So r % 1 == r_i % 1 is ALWAYS true.
    # This means if gcd(m, m_i) = 1, the classes ALWAYS intersect!
    # Wait, let me re-read the task description.
    # "Every two infinite residue classes must be disjoint"
    # "verifier uses the exact gcd criterion"
    
    # Let's check the math again.
    # System:
    # x = r + k*m
    # x = r_i + j*m_i
    # r + k*m = r_i + j*m_i
    # k*m - j*m_i = r_i - r
    # This linear Diophantine equation has a solution iff gcd(m, m_i) divides (r_i - r).
    # So, if gcd(m, m_i) | (r_i - r), then they intersect.
    # They are disjoint iff gcd(m, m_i) does NOT divide (r_i - r).
    # i.e., (r_i - r) % gcd(m, m_i) != 0
    # i.e., r % gcd(m, m_i) != r_i % gcd(m, m_i)
    
    # So my previous logic was correct.
    # BUT, if gcd(m, m_i) = 1, then (r_i - r) % 1 is always 0.
    # So gcd(m, m_i) ALWAYS divides (r_i - r).
    # This implies that if two moduli are coprime, their residue classes ALWAYS intersect.
    # Therefore, we can NEVER pick two coprime moduli!
    
    # This is a huge constraint. We need to pick a set of moduli such that NO two are coprime.
    # This is a problem of finding a large subset of {2..128} with no coprime pairs.
    # The largest such set is the set of even numbers?
    # If we pick all even numbers, any two have gcd >= 2.
    # Are there larger sets?
    # The set of multiples of 2 is size 63 (2, 4, ..., 128).
    # What if we include some odd numbers?
    # If we include an odd number m, it must share a factor with every other selected number.
    # If we have a set of even numbers, and we add an odd number m, m must share a factor with all of them.
    # Since the even numbers include 2, 4, 6, 8..., if we include 2, then m must be even (contradiction) or share a factor with 2? No, gcd(m, 2) must be > 1. Since m is odd, gcd(m, 2) is always 1.
    # So if 2 is in the set, NO odd number can be in the set.
    # So the best strategy involving 2 is just the evens.
    
    # What if we exclude 2?
    # Then we can potentially include odd numbers.
    # However, the set of evens is quite large (63).
    # Can we do better than 63?
    # The maximum size of a subset of {1..N} with no coprime pairs is known to be roughly N/2 + something small.
    # Actually, the largest subset of {2..128} with no coprime pairs is likely the set of even numbers, or the set of multiples of 2 plus some adjustments?
    # Wait, if we take all multiples of 2, that's 63 numbers.
    # If we take all multiples of 3? 2, 3, 4, 5, 6...
    # Multiples of 3 in 2..128: 3, 6, 9, ..., 126. Count = 42. Smaller.
    # What about a mix?
    # It is a known result that the largest subset of {1, ..., N} with no two coprime elements has size floor(N/2).
    # For N=128, floor(128/2) = 64.
    # The set {2, 4, ..., 128} has size 64?
    # 2, 4, ..., 128. 128/2 = 64. Yes.
    # So the maximum number of moduli we can pick is 64.
    
    # So the problem reduces to:
    # 1. Pick a set of 64 moduli from {2..128} such that no two are coprime.
    #    The set of even numbers {2, 4, ..., 128} is a valid candidate.
    # 2. For each modulus m in this set, pick a residue r such that for all other m' in the set,
    #    r % gcd(m, m') != r' % gcd(m, m').
    
    # Since we are picking all even numbers, let m = 2k.
    # gcd(2k, 2j) = 2 * gcd(k, j).
    # The condition is r % (2*gcd(k,j)) != r' % (2*gcd(k,j)).
    
    # If we simply pick r = 0 for all of them?
    # Then 0 % d == 0 % d is always true. So they all intersect.
    # We need distinct residues modulo the gcds.
    
    # This looks like we can assign residues greedily.
    # Let's try to construct a solution for the set of even numbers.
    # Sort moduli: 2, 4, 6, 8, ...
    # For m=2, pick r=0. (0 mod 2)
    # For m=4, gcd(4,2)=2. r % 2 != 0 % 2 => r % 2 != 0. So r must be odd. Pick r=1.
    # For m=6, gcd(6,2)=2. r % 2 != 0. r odd.
    #             gcd(6,4)=2. r % 2 != 1 % 2 => r % 2 != 1.
    #             Contradiction! r must be odd AND r must be even.
    # So we cannot just pick r=0, 1, ... arbitrarily.
    
    # We need a more careful assignment.
    # This is a constraint satisfaction problem.
    
    # Let's use a backtracking or greedy approach with the set of even moduli.
    # Or better, let's use the randomized greedy approach but restricted to a good set of moduli.
    # The set of even moduli is the best candidate for size.
    
    # Let's try to find residues for the set of even moduli.
    # If we fail, we might need to drop some moduli.
    
    # Algorithm:
    # 1. Define candidate moduli: all even numbers from 2 to 128.
    # 2. Shuffle them? Or sort by some heuristic?
    #    Maybe sort by number of divisors? Or just random.
    # 3. Greedily assign residues.
    #    For each m, try all r in 0..m-1.
    #    If a valid r is found, add (r, m) to solution.
    #    If no valid r is found, skip m.
    # 4. Repeat with different orderings to maximize count.
    
    best_solution = []
    
    # We know the max possible count is 64 (or close to it).
    # Let's try to get as close to 64 as possible.
    
    # Optimization:
    # Instead of trying all r for each m, we can be smarter.
    # But m <= 128, so trying all r is fast.
    # 64 moduli * 128 residues * 64 checks = ~500k ops.
    # We can do many restarts.
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
            
        moduli = list(range(2, max_modulus + 1, 2))  # Only even moduli
        import random
        random.shuffle(moduli)
        
        current_solution = []
        
        for m in moduli:
            found = False
            # Try residues in a randomized order? Or 0..m-1?
            # Random order might help escape local minima?
            # Actually, for a fixed set of previous selections, the valid residues are fixed.
            # So order of trying r doesn't matter for feasibility, only for which r we pick.
            # Picking a "good" r might help future steps?
            # Let's try r=0 first, then 1, etc.
            
            for r in range(m):
                if is_compatible(r, m, current_solution):
                    current_solution.append([r, m])
                    found = True
                    break
            
        if len(current_solution) > len(best_solution):
            best_solution = current_solution
            
        # If we found 64, we can't do better (assuming max is 64).
        if len(best_solution) == 64:
            break
            
    # What if we can do better than 64 by including some odd numbers?
    # As discussed, if we include 2, we can't include odds.
    # If we exclude 2, we might include odds.
    # But the set of evens without 2 is size 63.
    # Can we add odds to {4, 6, ..., 128}?
    # The smallest even is 4.
    # If we add an odd m, it must share a factor with 4, 6, 8, ...
    # gcd(m, 4) > 1 => m is even (impossible) or... wait.
    # If m is odd, gcd(m, 4) is 1 or... no, 4=2^2. Odd m is not divisible by 2.
    # So gcd(m, 4) = 1.
    # So if 4 is in the set, NO odd number can be in the set.
    # So if we have any power of 2 in the set, we can't have odds.
    # The set of evens contains 4.
    # So we can't mix evens and odds if we keep 4.
    
    # What if we remove 4?
    # Set: {2, 6, 8, 10, ...} (evens except 4).
    # Now can we add odds?
    # If we add odd m, it must share factor with 2? No, gcd(m,2)=1.
    # So if 2 is in the set, no odds.
    
    # It seems the set of all evens is indeed the best structural choice.
    
    # One more check: Is it possible to have a set of size > 64?
    # The problem is equivalent to finding a clique in the graph where edges connect non-coprime numbers?
    # No, we want an independent set in the "coprime" graph.
    # Or a clique in the "non-coprime" graph.
    # The largest clique in the non-coprime graph of {2..128}.
    # The set of even numbers is a clique (all pairwise non-coprime).
    # Size 64.
    # Can we add any other number?
    # If we add an odd number m, it must be non-coprime with all evens.
    # But it must be non-coprime with 2.
    # gcd(m, 2) must be > 1.
    # Since m is odd, gcd(m, 2) = 1.
    # So no odd number can be added to the set of evens.
    # Can we replace some evens with odds to get a larger set?
    # Suppose we remove 2.
    # Set: {4, 6, 8, ..., 128}. Size 63.
    # Can we add odds?
    # Must be non-coprime with 4.
    # gcd(m, 4) > 1.
    # Since m is odd, gcd(m, 4) = 1.
    # So no odds can be added if 4 is present.
    
    # What if we remove 4 too?
    # Set: {6, 8, 10, ..., 128}. (Evens >= 6).
    # Size: 128/2 - 2 = 62.
    # Can we add odds?
    # Must be non-coprime with 6.
    # gcd(m, 6) > 1.
    # m can be multiple of 3 (odd) or multiple of 2 (even, already in set).
    # So odd multiples of 3: 3, 9, 15, 21, ...
    # But we must also be non-coprime with 8.
    # gcd(m, 8) > 1.
    # m odd => gcd(m, 8) = 1.
    # So no odds can be added if 8 is present.
    
    # It seems that as long as we have a power of 2 in the set, we can't have odds.
    # To have odds, we must remove ALL powers of 2.
    # Powers of 2 in 2..128: 2, 4, 8, 16, 32, 64, 128. (7 numbers).
    # If we remove them, we have 64 - 7 = 57 evens left.
    # Can we add odds?
    # The remaining evens are {6, 10, 12, 14, ...}.
    # Let's check if we can add odds to this set.
    # This is getting complicated. The set of evens (size 64) is a very strong candidate.
    # Given the time limit, sticking to evens and optimizing residue assignment is the safest bet.
    
    return best_solution
