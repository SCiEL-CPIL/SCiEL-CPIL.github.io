import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Strategy: Greedy set packing / local search on the compatibility graph.
    # We want to maximize sum(1/m) for a set of pairwise disjoint classes.
    # Disjointness condition: for classes (m1, r1) and (m2, r2), r1 % gcd(m1, m2) != r2 % gcd(m1, m2).
    # This is equivalent to finding a maximum weight independent set in a conflict graph,
    # where vertices are potential classes (m, r) and edges connect conflicting classes.
    # However, the number of potential classes is large (181 moduli * avg ~100 residues = ~18k vertices).
    # A full MIP or exhaustive search is too slow.
    # Instead, we use a greedy approach with randomization and local improvement.
    
    # 1. Generate candidate moduli and residues
    # We prioritize smaller moduli because they have higher weight 1/m.
    # For each modulus m, we can choose any residue r in [0, m).
    # To keep the search space manageable, we can limit the number of residues considered per modulus
    # or use a greedy selection that picks the "best" residue for a given set of moduli.
    
    # A more effective heuristic:
    # Start with an empty set.
    # Iteratively add the class (m, r) that gives the highest marginal gain in score
    # while maintaining validity.
    # Since validity depends on interactions, we can try to add classes in order of decreasing weight 1/m.
    # For each candidate modulus m (sorted by 1/m descending), we try to find a residue r
    # that is compatible with all currently selected classes.
    
    # Let's refine this:
    # 1. Sort all moduli in [20, 200] by weight 1/m descending (i.e., m ascending).
    # 2. Maintain a list of selected classes.
    # 3. For each modulus m in sorted order:
    #    - Try to find a residue r in [0, m) such that (m, r) is disjoint from all selected classes.
    #    - If found, add (m, r) to the selection.
    #    - Which residue to pick? Any valid one works for the score, but some might block future additions more than others.
    #    - To be safe, we can pick the residue that minimizes future conflicts. A simple heuristic is to pick r=0 if valid, or the first valid r.
    #    - However, this greedy approach might get stuck in local optima.
    
    # Let's try a randomized greedy with multiple restarts.
    # In each restart:
    #   - Shuffle the order of moduli slightly (or use a random priority).
    #   - Greedily add classes.
    #   - Use local search to swap residues if it allows adding more classes or higher weight classes? 
    #     Actually, the set of moduli is fixed by the greedy selection. The residues just need to be valid.
    #     If a set of moduli is "packable" (i.e., there exist residues making them disjoint), then the score is fixed by the moduli.
    #     So the problem reduces to: Find the subset of moduli S ⊆ [20, 200] that is "packable" and maximizes sum(1/m for m in S).
    
    # How to check if a set of moduli is packable?
    # This is a constraint satisfaction problem.
    # For a given set of moduli M = {m1, ..., mk}, we need to assign residues r_i such that
    # r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j) for all i != j.
    # This is equivalent to checking if the graph of constraints is satisfiable.
    # Since the constraints are only pairwise, and the modulus of the constraint is gcd(m_i, m_j),
    # this is a system of difference constraints modulo varying moduli.
    
    # A simpler sufficient condition for packability:
    # If all moduli are pairwise coprime, we can always assign residues (e.g., all 0) and they are disjoint?
    # No, if m1, m2 are coprime, gcd=1, r1%1=0, r2%1=0, so 0!=0 is false. They are NOT disjoint if residues are 0.
    # Wait, the condition is r1 != r2 mod gcd(m1, m2).
    # If gcd=1, then r1 % 1 == 0 and r2 % 1 == 0. So 0 == 0, which means they ARE disjoint?
    # No, the condition for disjointness is that the sets {x | x ≡ r1 (mod m1)} and {x | x ≡ r2 (mod m2)} are disjoint.
    # By CRT, if gcd(m1, m2) = g, the intersection is non-empty iff r1 ≡ r2 (mod g).
    # So they are disjoint iff r1 NOT ≡ r2 (mod g).
    # If g=1, r1 ≡ 0 (mod 1) and r2 ≡ 0 (mod 1) is always true. So r1 ≡ r2 (mod 1) is always true.
    # Thus, if gcd(m1, m2)=1, the sets are NEVER disjoint?
    # Let's check: m1=2, m2=3. r1=0, r2=0. Sets: {0, 2, 4...} and {0, 3, 6...}. Intersection is {0, 6, ...}. Not disjoint.
    # r1=0, r2=1. Sets: {0, 2, 4...} and {1, 4, 7...}. Intersection is {4, ...}. Not disjoint.
    # Actually, any two arithmetic progressions with coprime moduli intersect infinitely often.
    # So, if gcd(m1, m2) = 1, they CANNOT be disjoint.
    # Therefore, we must only pick moduli that are NOT pairwise coprime?
    # No, that's not right. The condition is r1 != r2 mod g.
    # If g=1, r1 mod 1 is always 0, r2 mod 1 is always 0. So r1 == r2 mod 1.
    # The condition for disjointness is r1 != r2 mod g.
    # If g=1, this condition is 0 != 0, which is False.
    # So yes, if gcd(m1, m2)=1, the classes are NEVER disjoint.
    
    # This is a crucial insight!
    # We can only pick moduli that share a common factor > 1 with every other modulus in the set?
    # Or at least, for every pair, gcd(m_i, m_j) > 1 AND we can choose residues such that r_i != r_j mod gcd(m_i, m_j).
    
    # If we pick all moduli to be multiples of a common K > 1, then gcd(m_i, m_j) is at least K.
    # Then we can assign residues r_i = i * (K / something)?
    # If we pick moduli m_i = K * k_i.
    # gcd(m_i, m_j) = K * gcd(k_i, k_j).
    # We need r_i != r_j mod (K * gcd(k_i, k_j)).
    # If we simply set r_i = i (assuming i < m_i), does this work?
    # Not necessarily.
    
    # However, if we pick moduli that are all multiples of K, and we pick at most K moduli,
    # and we assign residues 0, 1, ..., K-1 to them, then:
    # For any pair i, j, r_i = i, r_j = j.
    # gcd(m_i, m_j) is a multiple of K.
    # So r_i mod gcd = i mod (multiple of K).
    # r_j mod gcd = j mod (multiple of K).
    # If i != j and i, j < K, then i mod K != j mod K.
    # Since gcd is a multiple of K, i mod gcd = i (if i < gcd) and j mod gcd = j.
    # So i != j implies disjointness.
    
    # This is the strategy used in the current program!
    # It picks multiples of K, and assigns residues 0..K-1.
    # The current program iterates over K and picks the smallest K multiples of K.
    
    # Can we do better?
    # The current program restricts to moduli that are ALL multiples of a single K.
    # What if we use a mix of moduli that don't share a single common factor, but pairwise have gcd > 1?
    # Example: m1=6, m2=10, m3=15.
    # gcd(6,10)=2, gcd(6,15)=3, gcd(10,15)=5.
    # Can we assign residues?
    # r1 mod 2 != r2 mod 2
    # r1 mod 3 != r3 mod 3
    # r2 mod 5 != r3 mod 5
    # Try r1=0, r2=1, r3=0.
    # 0 mod 2 = 0, 1 mod 2 = 1. OK.
    # 0 mod 3 = 0, 0 mod 3 = 0. Fail.
    # Try r1=0, r2=1, r3=1.
    # 0 mod 2 = 0, 1 mod 2 = 1. OK.
    # 0 mod 3 = 0, 1 mod 3 = 1. OK.
    # 1 mod 5 = 1, 1 mod 5 = 1. Fail.
    # Try r1=0, r2=1, r3=2.
    # 0 mod 2 = 0, 1 mod 2 = 1. OK.
    # 0 mod 3 = 0, 2 mod 3 = 2. OK.
    # 1 mod 5 = 1, 2 mod 5 = 2. OK.
    # So {6, 10, 15} is packable with residues {0, 1, 2}.
    # Score: 1/6 + 1/10 + 1/15 = 5/30 + 3/30 + 2/30 = 10/30 = 1/3 ≈ 0.333.
    # Compare with K=2 strategy: Multiples of 2 in [20, 200]. Smallest 2 are 20, 22.
    # Score: 1/20 + 1/22 = 0.05 + 0.045 = 0.095.
    # Compare with K=3 strategy: Multiples of 3 in [20, 200]. Smallest 3 are 21, 24, 27.
    # Score: 1/21 + 1/24 + 1/27 ≈ 0.047 + 0.041 + 0.037 = 0.125.
    # The mixed strategy {6, 10, 15} uses smaller moduli (6, 10, 15) but they are < 20.
    # The constraint is 20 <= m <= 200.
    # So we can't use 6, 10, 15.
    # Let's try {30, 42, 35}? No, 35 is 5*7.
    # Let's try to find a set of moduli in [20, 200] that are "mixed" but packable.
    
    # The current program's best score is 1.379.
    # Let's see what K gives that.
    # For K=10, multiples: 20, 30, 40, 50, 60, 70, 80, 90, 100, 110.
    # Score: 1/20+1/30+...+1/110 ≈ 0.05+0.033+0.025+0.02+0.016+0.014+0.0125+0.011+0.01+0.009 = 0.2005.
    # Wait, the score is 1.379? That's much higher.
    # Ah, the current program sums 1/m.
    # 1/20 = 0.05.
    # If it picks many small moduli, the sum can be large.
    # For K=1, multiples are all integers. But K=1 means gcd(m_i, m_j) can be 1.
    # If K=1, the strategy "assign residues 0..K-1" means only residue 0 is assigned.
    # Then r_i=0 for all.
    # Condition: 0 != 0 mod gcd(m_i, m_j). This requires gcd(m_i, m_j) to be such that 0 != 0 mod g is false.
    # So K=1 strategy only works if gcd(m_i, m_j) > 1 for all pairs? No, if gcd=1, 0!=0 is false, so invalid.
    # So for K=1, we can only pick moduli that are pairwise non-coprime?
    # But the program picks the smallest K multiples of K. For K=1, it picks 20, 21, ..., 29.
    # Are these pairwise disjoint with residue 0?
    # gcd(20, 21)=1. 0!=0 mod 1 is False. Invalid.
    # So the program's logic "assign residues 0..k_pick-1" and "verify validity" would reject K=1 if it picks coprime pairs.
    # Let's check the code:
    # `for K in range(1, 101):`
    # `multiples = [m for m in range(20, 201) if m % K == 0]`
    # `k_pick = min(K, len(multiples))`
    # `selected_moduli = sorted(multiples)[:k_pick]`
    # `classes = [[m, i] for i, m in enumerate(selected_moduli)]`
    # For K=1, selected_moduli = [20, 21, ..., 29] (10 items).
    # classes: [20,0], [21,1], ..., [29,9].
    # Check [20,0] and [21,1]. gcd=1. 0%1=0, 1%1=0. 0==0. Invalid.
    # So K=1 yields no valid classes?
    # The loop continues.
    
    # What K gives 1.379?
    # Let's estimate. Sum of 1/m for m in [20, 200] is approx ln(200/20) = ln(10) ≈ 2.3.
    # If we could pick all, it would be 2.3.
    # 1.379 is about 60% of the total sum.
    # This suggests a large number of moduli are picked.
    
    # Let's try a different strategy:
    # Local Search / Hill Climbing on the set of moduli.
    # Start with a random valid set.
    # Try to add a modulus m not in the set.
    # Check if it can be packed with the existing set (by finding a valid residue).
    # If yes, add it.
    # Try to remove a modulus m from the set.
    # Check if removing it allows adding two or more smaller moduli? (Probably not, since 1/m decreases).
    # Actually, we just want to maximize the sum.
    # So we should try to add any valid modulus.
    # The order of addition matters.
    
    # Algorithm:
    # 1. Initialize empty set.
    # 2. Create a list of all moduli 20..200.
    # 3. Shuffle the list.
    # 4. For each m in shuffled list:
    #    a. Try to find a residue r for m such that (m, r) is disjoint from all current classes.
    #    b. If found, add (m, r) to the set.
    # 5. Repeat with different shuffles (random restarts).
    # 6. Keep the best set found.
    
    # How to find a valid residue r for m?
    # We have a set of existing classes C = {(m_i, r_i)}.
    # We need r such that for all (m_i, r_i) in C:
    # r % gcd(m, m_i) != r_i % gcd(m, m_i).
    # This is a set of forbidden residues modulo g_i = gcd(m, m_i).
    # The number of forbidden residues for m_i is 1 (specifically r_i mod g_i).
    # So we need to find r in [0, m) that avoids these forbidden values modulo their respective g_i.
    # Since m is up to 200, we can just iterate r from 0 to m-1 and check.
    # This is fast.
    
    # Complexity:
    # 181 moduli.
    # For each modulus, check up to 200 residues.
    # For each residue, check against up to 181 existing classes.
    # Total ops: 181 * 200 * 181 ≈ 6.5 million.
    # This is very fast in Python (should take < 1 second).
    # So we can do many random restarts.
    
    best_score = 0.0
    best_classes = []
    
    num_restarts = 1000
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle moduli
        moduli = list(range(20, 201))
        import random
        random.shuffle(moduli)
        
        current_classes = []
        for m in moduli:
            # Try to find a valid residue
            valid_residue = -1
            for r in range(m):
                is_valid = True
                for m_i, r_i in current_classes:
                    g = math.gcd(m, m_i)
                    if r % g == r_i % g:
                        is_valid = False
                        break
                if is_valid:
                    valid_residue = r
                    break
            
            if valid_residue != -1:
                current_classes.append([m, valid_residue])
        
        score = sum(1.0 / m for m, _ in current_classes)
        if score > best_score:
            best_score = score
            best_classes = current_classes
    
    if not best_classes:
        best_classes = [[20 * j, j - 1] for j in range(1, 11)]
        
    return best_classes
