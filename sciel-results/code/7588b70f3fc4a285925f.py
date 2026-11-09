import math

def construct():
    # We want to maximize sum of 1/m over a set of moduli m in [20, 200]
    # with distinct moduli and pairwise disjoint residue classes.
    #
    # Key insight: If we pick all moduli that are multiples of a single prime p,
    # then gcd(m_i, m_j) is at least p for all pairs. This makes the disjointness
    # constraint much easier to satisfy.
    #
    # For a fixed prime p, the moduli are p, 2p, 3p, ... up to 200 (but >= 20).
    # Let m = k*p. Then gcd(m, m') = p * gcd(k, k').
    # The constraint is r != r' mod gcd(m, m').
    #
    # A simple strategy: for each modulus m = k*p, assign residue r = 0.
    # Then we need 0 != 0 mod gcd(m, m'), i.e., gcd(m, m') != 1, which is true
    # since all are multiples of p. But wait, the constraint is r_i != r_j mod gcd(m_i, m_j).
    # If r_i = r_j = 0, then 0 != 0 mod g is FALSE. So all residues 0 don't work.
    #
    # Let's think differently. If we pick moduli that are all multiples of p,
    # and assign residues r_m = 0 for all m, that fails.
    #
    # What if we assign r_m = m mod (2p)? No, that's not right.
    #
    # Actually, a cleaner approach: pick a set of moduli that are all multiples of p.
    # For m = k*p, let's assign r = k mod (some base). 
    #
    # Let's reconsider. The constraint is: for each pair (m_i, m_j), 
    # r_i != r_j mod gcd(m_i, m_j).
    #
    # If we take all even numbers in [20, 200], that's 90 moduli.
    # gcd(2a, 2b) = 2*gcd(a, b).
    # If we set r_{2a} = a mod 2 (i.e., 0 if a even, 1 if a odd), then:
    # r_i != r_j mod gcd(2a, 2b) = 2*gcd(a,b).
    # Since gcd(a,b) >= 1, the modulus is at least 2.
    # r_i and r_j are in {0, 1}. They are different mod 2 iff they are different.
    # So we need a mod 2 != b mod 2 whenever gcd(a,b) >= 1? No, that's not right.
    # We need r_i != r_j mod (2*gcd(a,b)).
    # If a and b have the same parity, r_i = r_j, so r_i - r_j = 0, which is 0 mod anything.
    # So that fails if a and b have the same parity.
    #
    # Let's try a different residue assignment.
    # For m = k*p, assign r = k. 
    # Then r_i - r_j = k_i - k_j.
    # We need k_i - k_j not divisible by gcd(k_i*p, k_j*p) = p*gcd(k_i, k_j).
    # So we need k_i - k_j not divisible by p*gcd(k_i, k_j).
    # This is hard to guarantee for all pairs.
    #
    # Alternative: use a greedy approach with backtracking, but guided by the prime structure.
    # 
    # Actually, the simplest strong approach: 
    # 1. For each prime p in {2, 3, 5, 7, 11, 13, 17, 19}, consider all multiples of p in [20, 200].
    # 2. Sort these multiples in increasing order of 1/m (i.e., decreasing m).
    # 3. Greedily add each modulus if a valid residue exists given the current set.
    # 4. For residue assignment, try all possible residues (0 to m-1) and pick the first valid one.
    # 5. Keep the best result across all primes.
    #
    # But this might be slow. Let's optimize.
    #
    # Actually, a known result: the optimal set is often all multiples of a small prime.
    # Let's try p=2: all even numbers from 20 to 200.
    # Can we assign residues to all 90 even numbers?
    #
    # Let's use a greedy algorithm with a good heuristic:
    # - Start with the smallest moduli (largest 1/m).
    # - For each modulus, try to find a valid residue.
    # - Use a deterministic residue choice: the smallest valid residue.
    #
    # To make it faster, we can precompute conflicts.
    #
    # Let's implement a greedy algorithm that tries to include as many small moduli as possible.
    # We'll try different orderings: by m ascending, by m descending, by 1/m descending.

    best_classes = []
    best_score = 0.0

    # Try different primes as the "core" prime
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
        # Get all multiples of p in [20, 200]
        moduli = [m for m in range(20, 201) if m % p == 0]
        
        # Sort by 1/m descending (i.e., m ascending)
        moduli.sort()
        
        classes = []
        current_score = 0.0
        
        for m in moduli:
            # Find a valid residue for m
            valid_res = None
            for r in range(m):
                ok = True
                for m_j, r_j in classes:
                    g = math.gcd(m, m_j)
                    if r % g == r_j % g:
                        ok = False
                        break
                if ok:
                    valid_res = r
                    break
            
            if valid_res is not None:
                classes.append([m, valid_res])
                current_score += 1.0 / m
        
        if current_score > best_score:
            best_score = current_score
            best_classes = classes

    # Also try a general greedy approach without prime restriction
    # Sort all moduli by 1/m descending
    all_moduli = list(range(20, 201))
    all_moduli.sort()
    
    classes = []
    current_score = 0.0
    
    for m in all_moduli:
        valid_res = None
        for r in range(m):
            ok = True
            for m_j, r_j in classes:
                g = math.gcd(m, m_j)
                if r % g == r_j % g:
                    ok = False
                    break
            if ok:
                valid_res = r
                break
        
        if valid_res is not None:
            classes.append([m, valid_res])
            current_score += 1.0 / m
    
    if current_score > best_score:
        best_score = current_score
        best_classes = classes

    return best_classes
