"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    
    Two residue classes (a, m) and (b, n) are disjoint iff gcd(a-b, m*n) != 1
    (i.e., the system a*x + ... has no solution; equivalently, the classes
    {a + k*m} and {b + l*n} are disjoint iff gcd(m,n) doesn't divide (a-b)
    in the right way... actually the criterion is: the classes are disjoint
    iff the congruence a ≡ b (mod gcd(m,n)) is NOT satisfied, i.e., 
    a mod gcd(m,n) != b mod gcd(m,n)).
    
    Wait, let me reconsider. Two arithmetic progressions {a + k*m : k∈Z} and 
    {b + l*n : l∈Z} intersect iff the system x ≡ a (mod m), x ≡ b (mod n) 
    has a solution. By CRT, this has a solution iff a ≡ b (mod gcd(m,n)).
    So they are DISJOINT iff a mod gcd(m,n) != b mod gcd(m,n).
    
    So for each pair of chosen classes (a_i, m_i) and (a_j, m_j), we need:
    a_i mod gcd(m_i, m_j) != a_j mod gcd(m_i, m_j)
    
    We want to maximize the number of such classes with distinct moduli from 2..128.
    """
    deadline = time.time() + 85  # Safety margin
    
    max_mod = 128
    available_moduli = list(range(2, max_mod + 1))
    
    # For a given modulus m, we can choose any residue r in [0, m-1].
    # We need to pick a subset of moduli and assign residues such that
    # for all pairs (i,j): r_i mod gcd(m_i, m_j) != r_j mod gcd(m_i, m_j)
    
    # Strategy: Greedy with backtracking / local search.
    # First, let's try a greedy approach: process moduli in some order,
    # and for each, pick a residue that is compatible with all previously chosen ones.
    
    # Key insight: For moduli that are pairwise coprime, gcd(m_i, m_j) = 1, 
    # so a mod 1 = b mod 1 = 0 always, meaning they'd ALWAYS intersect!
    # Wait, that's wrong. If gcd(m,n)=1, then the condition for intersection is
    # a ≡ b (mod 1), which is always true. So pairwise coprime moduli ALWAYS
    # have intersecting residue classes!
    
    # This means we CANNOT use pairwise coprime moduli. We need moduli that share
    # common factors.
    
    # Let me re-examine: the classes are disjoint iff a mod gcd(m,n) != b mod gcd(m,n).
    # If gcd(m,n) = 1, then both a mod 1 and b mod 1 are 0, so they're always equal,
    # meaning the classes always intersect. So we need gcd(m,n) >= 2 for every pair.
    
    # This means all chosen moduli must share a common factor of at least 2? No,
    # not necessarily a common factor across ALL, but every PAIR must have gcd >= 2.
    
    # Actually, we just need for every pair (m_i, m_j), gcd(m_i, m_j) >= 2.
    # This means no two chosen moduli can be coprime.
    
    # Let's think about which moduli in [2, 128] can form a large set where
    # every pair has gcd >= 2.
    
    # Moduli that are even: all pairs have gcd >= 2. There are 63 even numbers in [2,128].
    # But we also need to assign residues so that r_i mod gcd(m_i, m_j) != r_j mod gcd(m_i, m_j).
    
    # For all-even moduli, gcd(m_i, m_j) >= 2. Let's see if we can assign residues.
    # If we pick all residues to be 0, then r_i mod gcd = 0 = r_j mod gcd for all pairs,
    # which violates the condition.
    
    # We need r_i mod g_{ij} != r_j mod g_{ij} for all pairs.
    
    # This is a constraint satisfaction problem. Let's try a greedy approach.
    
    # Let's try: use only even moduli. For each even modulus m, we can choose r in [0, m-1].
    # The constraint between m_i and m_j (both even) is: r_i mod gcd(m_i,m_j) != r_j mod gcd(m_i,m_j).
    
    # Since gcd(m_i, m_j) is even (at least 2), we have at least 2 residue classes mod gcd.
    
    # Let's try a simple assignment: r_i = 1 for all i. Then r_i mod gcd = 1 mod gcd.
    # For this to work, we need 1 mod g != 1 mod g, which is false. So that doesn't work.
    
    # Try r_i = i (or some distinct values). Let's just do a proper search.
    
    # Given the complexity, let's use a greedy + local search approach.
    # Start with a subset of moduli, try to assign residues, and maximize the count.
    
    # Actually, let me think more carefully. The problem is:
    # - Choose a subset S of {2,...,128}
    # - For each m in S, choose r_m in [0, m-1]
    # - For all m, n in S, m != n: r_m mod gcd(m,n) != r_n mod gcd(m,n)
    # - Maximize |S|
    
    # Key observation: if all moduli are even, gcd(m,n) >= 2 for all pairs.
    # Let's focus on even moduli first.
    
    # For even moduli, let's write m = 2*k. Then gcd(m,n) = 2*gcd(k, l) where n=2*l.
    # The constraint is r_m mod (2*gcd(k,l)) != r_n mod (2*gcd(k,l)).
    
    # This is still complex. Let's just do a computational search.
    
    # Approach: 
    # 1. Consider candidate moduli sets (e.g., all even numbers, or numbers divisible by small primes)
    # 2. For a given set, try to assign residues via backtracking or greedy
    # 3. Optimize the set and assignment
    
    # Let's start with a greedy approach over even moduli:
    # Process moduli in order of increasing size (or some heuristic), and for each,
    # try all possible residues and pick one that's compatible with all previously assigned.
    
    def is_compatible(residue_new, modulus_new, assigned):
        """Check if (residue_new, modulus_new) is compatible with all assigned pairs."""
        for (r, m) in assigned:
            g = math.gcd(modulus_new, m)
            if residue_new % g == r % g:
                return False
        return True
    
    # Greedy: try to add as many moduli as possible
    # Order moduli by some heuristic - maybe smaller moduli first (more constraints)
    # or larger moduli first (more residue choices)
    
    best_result = []
    
    # Try different orderings
    orderings = [
        sorted(available_moduli),  # ascending
        sorted(available_moduli, reverse=True),  # descending
        # Custom: prioritize moduli with many shared factors
    ]
    
    for ordering in orderings:
        assigned = []
        for m in ordering:
            if time.time() > deadline:
                break
            # Try all residues for this modulus
            found = False
            for r in range(m):
                if is_compatible(r, m, assigned):
                    assigned.append((r, m))
                    found = True
                    break
            if not found:
                continue
        if len(assigned) > len(best_result):
            best_result = assigned[:]
    
    # Now try to improve by swapping/adding
    # Local search: try to add moduli that were skipped, or swap residues
    
    # Convert to list of [residue, modulus]
    result = [[r, m] for (r, m) in best_result]
    
    if not result:
        # Fallback: at least one class
        result = [[1, 2]]
    
    return result
# EVOLVE-BLOCK-END
