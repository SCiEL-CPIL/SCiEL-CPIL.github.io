"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _csp_solve(moduli, deadline):
    """Try to find residues for the given list of moduli such that all pairs are disjoint.
    Returns dict m->r or None. Uses backtracking with forward checking."""
    n = len(moduli)
    if n == 0:
        return {}
    # Order moduli: put those with more constraints first (higher degree / smaller gcd with others)
    # Compute conflict "tightness": for each pair, the number of forbidden residues = m_i / gcd
    # We'll just use a simple heuristic: sort by modulus descending (more residues to choose)
    # Actually, better: sort by number of "restrictions" = sum over others of 1/gcd... 
    # Keep it simple: try a few orderings.
    
    def backtrack(assignment, remaining, forbidden):
        """assignment: dict m->r; remaining: list of moduli; forbidden: dict m->set of forbidden residues"""
        if time.time() > deadline:
            return None
        if not remaining:
            return assignment
        # MRV: pick modulus with fewest available residues
        best_idx = -1
        best_count = None
        for i, m in enumerate(remaining):
            avail = m - len(forbidden.get(m, set()))
            if best_count is None or avail < best_count:
                best_count = avail
                best_idx = i
            if best_count <= 0:
                return None
        m = remaining[best_idx]
        new_remaining = remaining[:best_idx] + remaining[best_idx+1:]
        new_forbidden = {k: set(v) for k, v in forbidden.items()}
        # Try each available residue
        for r in range(m):
            if r in new_forbidden.get(m, set()):
                continue
            # Check feasibility: for each other modulus m2 in new_remaining, 
            # r must not be congruent to any already-assigned... wait, others not assigned yet.
            # Forward checking: for each m2 in new_remaining, the constraint gcd(m,m2) | (r - r2) 
            # means r2 cannot be in {r + k*gcd : ...}. We add forbidden residues for m2.
            feasible = True
            new_forbidden2 = {k: set(v) for k, v in new_forbidden.items()}
            for m2 in new_remaining:
                g = _gcd(m, m2)
                # r2 must satisfy r2 not ≡ r (mod g), i.e., r2 % g != r % g
                # Forbidden residues for m2: those with r2 % g == r % g
                # Count: m2 // g
                # Check if all residues of m2 are forbidden
                if m2 // g >= m2:
                    # all forbidden
                    feasible = False
                    break
                # We don't store the full set, just track count
                # Actually let's store the forbidden set explicitly for small m
                if m2 <= 256:
                    if m2 not in new_forbidden2:
                        new_forbidden2[m2] = set()
                    for r2 in range(m2):
                        if r2 % g == r % g:
                            new_forbidden2[m2].add(r2)
                    if len(new_forbidden2[m2]) >= m2:
                        feasible = False
                        break
            if feasible:
                assignment2 = dict(assignment)
                assignment2[m] = r
                result = backtrack(assignment2, new_remaining, new_forbidden2)
                if result is not None:
                    return result
        return None
    
    # Initial forbidden sets
    init_forbidden = {m: set() for m in moduli}
    return backtrack({}, list(moduli), init_forbidden)

def _try_grow(base_moduli, all_moduli, deadline):
    """Try to add more moduli to base_moduli via CSP."""
    current = list(base_moduli)
    # Try adding one at a time
    for m in all_moduli:
        if m in current:
            continue
        if time.time() > deadline:
            break
        trial = current + [m]
        # Check if CSP is solvable
        sol = _csp_solve(trial, deadline)
        if sol is not None:
            current = trial
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Baseline: one class per prime power up to 128
    # Prime powers: 2,4,8,16,32,64,128 ; 3,9,27,81 ; 5,25 ; 7,49 ; 11,13,17,19,23,29,31
    # But we can only pick ONE per prime (since same prime => gcd > 1)
    # Actually we can pick at most one modulus divisible by each prime.
    # So the prime-power skeleton picks one power per prime.
    # Which power to pick? The largest? Smallest? 
    # For disjointness between different primes, it doesn't matter.
    # But to allow adding composites later, smaller moduli might be better (more residues available).
    # Let's pick the smallest power of each prime.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    # Primes <= 128
    primes = [p for p in primes if p <= 128]
    
    # For each prime, pick the smallest power (just the prime itself)
    base_moduli = list(primes)
    # Residues: for coprime moduli, any residues work. Let's use r=0 for all.
    # But wait, we need to verify: gcd(p, q) = 1 for p != q, so 1 | (0-0) = 0, yes.
    # So [[0, p] for p in primes] is valid.
    
    # Now try to grow by adding composite moduli
    all_moduli = list(range(2, 129))
    
    # Try to add composites that are coprime to all current moduli
    # A composite m can be added if gcd(m, p) = 1 for all p in current set
    # But that's impossible since every composite has a prime factor, and that prime is in our set.
    # So we CANNOT add any composite if we have all primes.
    # 
    # Alternative: REMOVE some primes to add composites that cover multiple "slots"?
    # No, each modulus uses one "prime slot" per prime dividing it.
    # 
    # So the maximum is at most the number of distinct primes <= 128, which is 31.
    # But we can only pick one modulus per prime. So max is 31? 
    # Wait, can we pick two moduli that share a prime? 
    # If m1 and m2 both divisible by p, then gcd(m1,m2) >= p > 1.
    # We need gcd(m1,m2) ∤ (r1-r2). 
    # If gcd = g, we need r1 ≠ r2 (mod g). 
    # So it IS possible to have two moduli sharing a prime, as long as their residues differ mod gcd.
    # 
    # Example: m1=6, m2=10, gcd=2. Need r1 ≠ r2 (mod 2). So r1=0, r2=1 works.
    # 
    # So the problem is NOT simply one-per-prime. We can have multiple moduli per prime!
    # 
    # This changes everything. Let me reconsider.
    # 
    # The constraint is: for each pair (m1, m2), r1 ≠ r2 (mod gcd(m1,m2)).
    # 
    # This is a CSP. The maximum number of classes could be much larger than 31.
    
    # Let's use a greedy + local search approach.
    # Start with a good initial solution, then try to add more.
    
    # Greedy: add moduli one by one, choosing the one that is easiest to fit.
    random.seed(42)
    
    def greedy_construct(deadline):
        chosen = []  # list of (r, m)
        chosen_moduli = set()
        remaining = list(range(2, 129))
        random.shuffle(remaining)
        
        while remaining and time.time() < deadline:
            best_m = None
            best_r = None
            for m in remaining:
                # Try to find a residue r such that for all (r2, m2) in chosen:
                # gcd(m, m2) ∤ (r - r2)
                # i.e., r % gcd(m, m2) != r2 % gcd(m, m2)
                # For each chosen (r2, m2), the forbidden residues are those with r ≡ r2 (mod g)
                # where g = gcd(m, m2).
                # So forbidden = union over chosen of {r : r % g == r2 % g}
                # We need to find r in [0, m-1] not in forbidden.
                
                forbidden = [False] * m
                for r2, m2 in chosen:
                    g = _gcd(m, m2)
                    r2_mod = r2 % g
                    for r in range(m):
                        if r % g == r2_mod:
                            forbidden[r] = True
                
                # Find an available r
                available = [r for r in range(m) if not forbidden[r]]
                if available:
                    # Pick the one that causes least future conflict? For now, pick first.
                    best_m = m
                    best_r = available[0]
                    break
            
            if best_m is not None:
                chosen.append((best_r, best_m))
                chosen_moduli.add(best_m)
                remaining.remove(best_m)
        
        return chosen
    
    # Run greedy
    result = greedy_construct(deadline)
    
    # Try to improve with local search: try to add more moduli by reassigning residues
    # This is complex. For now, return what we have.
    
    # Ensure canonical form and distinct moduli
    seen_moduli = set()
    final = []
    for r, m in result:
        if m not in seen_moduli:
            seen_moduli.add(m)
            r = r % m
            final.append([r, m])
    
    if not final:
        # Fallback: prime skeleton
        final = [[0, p] for p in primes[:11]]
    
    return final
# EVOLVE-BLOCK-END

