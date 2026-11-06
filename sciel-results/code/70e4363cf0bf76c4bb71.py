"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a maximum set of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    
    Two residue classes (r1, m1) and (r2, m2) are disjoint iff
    r1 != r2 (mod gcd(m1, m2)).
    
    Strategy: Use a greedy approach with backtracking.
    We want to maximize the number of classes.
    """
    MAX_MOD = 128
    START = time.time()
    DEADLINE = START + 85  # Leave 5s safety margin for 90s limit
    
    # Precompute gcd for all pairs of moduli
    all_moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(all_moduli)
    
    # gcd_table[i][j] = gcd(all_moduli[i], all_moduli[j])
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(n_mod):
            gcd_table[i][j] = math.gcd(all_moduli[i], all_moduli[j])
    
    # We'll use a backtracking search to find the maximum number of classes.
    # Order moduli in a way that helps: start with "harder" moduli first.
    # A modulus is "harder" if it has more divisors or is more composite,
    # because it will have more constraints with other moduli.
    
    # Actually, let's think differently. The key constraint is:
    # For selected moduli m1, m2 with residues r1, r2, we need r1 != r2 mod gcd(m1,m2).
    # 
    # If we pick all moduli that are multiples of a prime p, and assign distinct
    # residues mod p, then all pairs are disjoint. This gives p classes.
    # 
    # Can we do better by mixing? Let's try a CSP/backtracking approach.
    
    # First, let's try the simple strategy: find the best single-prime family.
    # For prime p, we can use moduli p, 2p, 3p, ..., kp where kp <= 128.
    # Number of such moduli = floor(128/p).
    # We need distinct residues mod p, so we can have at most p classes.
    # So the limit is min(p, floor(128/p)).
    
    best_single = 0
    best_single_p = 0
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if p > 128:
            break
        count = min(p, 128 // p)
        if count > best_single:
            best_single = count
            best_single_p = p
    
    # best_single_p = 11 gives 11 classes.
    # Can we do better than 11?
    
    # Let's try a more sophisticated approach: select a set of moduli and
    # assign residues via backtracking.
    
    # Key insight: if gcd(m1, m2) = 1, then ANY residues work (they're always disjoint).
    # If gcd(m1, m2) = g > 1, then residues must differ mod g.
    
    # Let's try to build a large set. Start with the 11 moduli from prime 11:
    # 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121
    # Residues: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 mod 11
    
    # Can we add more moduli? The remaining moduli are those not divisible by 11.
    # For a new modulus m not divisible by 11, gcd(m, 11k) = gcd(m, k) * gcd(m, 11) = gcd(m, k) * 1 = gcd(m, k).
    # Wait, that's not right. gcd(m, 11k) where 11 doesn't divide m:
    # gcd(m, 11k) = gcd(m, k) if 11 doesn't divide m... no.
    # gcd(m, 11k) = gcd(m, k) * something... Let me just compute.
    
    # Let's try a direct backtracking approach to see if we can exceed 11.
    
    def try_add_to_solution(solution, available_moduli):
        """Try to add more moduli to the existing solution."""
        if time.time() > DEADLINE:
            return solution
        
        # For each available modulus, check if we can assign a residue
        for idx in range(len(available_moduli)):
            m = available_moduli[idx]
            
            # Check if we can find a valid residue for m
            # A residue r is valid if for all (r_i, m_i) in solution:
            # r != r_i mod gcd(m, m_i)
            
            # Build constraints: for each existing class, r must avoid certain residues mod g
            # Collect forbidden residues
            forbidden = set()
            valid = True
            for (r_i, m_i) in solution:
                g = math.gcd(m, m_i)
                # r must not be congruent to r_i mod g
                # This means r mod g != r_i mod g
                # So r mod g is forbidden for value r_i mod g
                # But different constraints may conflict
                # We need: for all i, (r - r_i) % g != 0
                # i.e., r % g != r_i % g for all i
                
                # Let's check if there's any residue in [0, m) that satisfies all
                # This is a system of congruence inequalities, which is complex.
                # Instead, let's just try all residues from 0 to m-1.
                pass
            
            # Try all residues
            for r in range(m):
                ok = True
                for (r_i, m_i) in solution:
                    g = math.gcd(m, m_i)
                    if r % g == r_i % g:
                        ok = False
                        break
                if ok:
                    new_solution = solution + [(r, m)]
                    remaining = available_moduli[:idx] + available_moduli[idx+1:]
                    return try_add_to_solution(new_solution, remaining)
        
        return solution
    
    # Start with the 11-class solution
    base_solution = [(i, 11 * (i + 1)) for i in range(11)]
    used_moduli = set(m for _, m in base_solution)
    available = [m for m in all_moduli if m not in used_moduli]
    
    # Try to extend
    extended = try_add_to_solution(base_solution, available)
    
    if len(extended) > len(base_solution):
        result = [[r, m] for r, m in extended]
        return result
    
    # If we couldn't extend, return the base solution
    return [[r, m] for r, m in base_solution]

# EVOLVE-BLOCK-END
