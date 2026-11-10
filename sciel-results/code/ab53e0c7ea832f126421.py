import time
import random
import math
import sys
from functools import lru_cache

N = 5000
MAX_TIME = 290.0  # Safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def _construct_from_assignments(assignments, rng):
    """
    Given a partial assignment of forbidden residues for the first k primes,
    complete the assignment for subsequent primes using a greedy heuristic
    to maximize the size of the resulting admissible set.
    
    assignments: dict {prime: forbidden_residue}
    Returns: list of integers in [0, N] that are admissible.
    """
    # Start with the set of numbers that survive the initial assignments
    # We represent the set implicitly by checking membership, but for greedy steps,
    # we need to know counts.
    
    # To make this efficient, we maintain a boolean array or a set.
    # Given N=5000, a set of ints is fast enough.
    S = set(range(N + 1))
    
    # Apply initial assignments
    for p, r in assignments.items():
        S = {x for x in S if x % p != r}
        
    # If S is empty, return empty
    if not S:
        return []
        
    # Now we need to ensure admissibility for primes p <= len(S).
    # We might have violated admissibility for primes NOT in assignments.
    # Or, if assignments covered all primes p <= len(S), we are done?
    # No, len(S) changes as we remove more.
    
    # Strategy: 
    # 1. Check if current S is admissible.
    # 2. If not, find a violating prime p.
    # 3. If p is already in assignments, we can't change it (constraint fixed). 
    #    Wait, the task says "selecting a removed residue class for each prime".
    #    This implies we *can* choose the residue for *any* prime.
    #    The "assignments" in my strategy are just a *starting point* or *bias*.
    #    If a prime is in assignments but causes a violation, we must either:
    #    a) Accept that this branch is bad (if we treat assignments as hard constraints).
    #    b) Or, more likely, the "assignment" is just the *choice* we make.
    #    
    #    Let's refine: The "Backtracking" idea is that we *fix* the choices for small primes 
    #    to break symmetry/local optima, then greedily fix the rest.
    #    If the fixed choices lead to a non-admissible set that *cannot* be fixed by 
    #    changing other primes (because the violation is on a fixed prime), then this 
    #    branch is invalid?
    #    
    #    Actually, if p is fixed to residue r, and S contains all residues mod p, 
    #    then S is NOT admissible w.r.t p. We *must* remove one of the other residues?
    #    No, the definition is: "for every prime p <= len(tuple) some residue class mod p is missed".
    #    If we *chose* to remove residue r mod p, then r is missed.
    #    If S still has all residues, it means our "removal" didn't work? 
    #    No, S is defined as numbers NOT in any removed residue class.
    #    So if we removed r mod p, then no x in S has x % p == r.
    #    Therefore, r is ALWAYS missed in S.
    #    
    #    So, if we have a set of "removed residue classes" R = {(p_i, r_i)},
    #    then for every p_i in R, the residue r_i is missed.
    #    Admissibility requires: For EVERY prime p <= len(S), SOME residue is missed.
    #    
    #    If p is NOT in R, we haven't explicitly removed a residue mod p.
    #    However, it might be that by removing residues of OTHER primes, we accidentally
    #    removed all numbers of some residue mod p?
    #    Or, it might be that all residues mod p are still present in S.
    #    In that case, we MUST remove a residue mod p.
    #    
    #    So the algorithm is:
    #    1. Start with a set of "committed" removals (the assignments).
    #    2. Compute S = {x in [0,N] | x not in any committed removal}.
    #    3. Check admissibility.
    #    4. If violated for prime p:
    #       - If p is already committed, this is a CONTRADICTION? 
    #         No, if p is committed to r, then r is missed. So p cannot be violated.
    #         Wait, "violated" means "all residues present". If r is missed, not all are present.
    #         So committed primes are ALWAYS satisfied.
    #       - Therefore, any violation MUST be for a prime p NOT in committed set.
    #    5. So we just need to add a new removal for p.
    #    6. Choose which residue r of p to remove.
    #       Heuristic: Remove the one with min count in S.
    #    7. Repeat until admissible.
    
    committed = dict(assignments)
    
    # We need to be careful: len(S) shrinks, so the set of "relevant" primes (p <= len(S)) shrinks.
    # But we only add removals, so S only shrinks.
    
    while True:
        len_S = len(S)
        if len_S == 0:
            break
            
        # Find a violating prime
        viol_p = None
        for p in PRIMES:
            if p > len_S:
                break
            # If p is committed, it's satisfied (its assigned residue is missing).
            if p in committed:
                continue
            # Check if all residues mod p are present in S
            # Optimization: if p is small, this is fast.
            residues = set(x % p for x in S)
            if len(residues) == p:
                viol_p = p
                break
        
        if viol_p is None:
            break
            
        p = viol_p
        # Choose residue to remove
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
            
        # Randomized tie-breaking
        min_count = min(counts)
        candidates = [r for r in range(p) if counts[r] == min_count]
        r_to_remove = rng.choice(candidates)
        
        # Commit this removal
        committed[p] = r_to_remove
        S = {x for x in S if x % p != r_to_remove}
        
    return list(S)

def _construct():
    start_time = time.time()
    best_S = []
    best_size = 0
    
    # Strategy:
    # 1. Try empty assignments (pure greedy).
    # 2. Try fixing the first prime (2) to 0 or 1.
    # 3. Try fixing first two primes (2,3) to all combinations (4 combos).
    # 4. Try fixing first three primes (2,3,5) to all combinations (12 combos).
    # 5. If time permits, fix more.
    
    rng = random.Random(42)
    
    # Phase 1: No fixed assignments
    S = _construct_from_assignments({}, rng)
    if len(S) > best_size:
        best_S = S
        best_size = len(S)
        
    # Phase 2: Fix prime 2
    for r2 in [0, 1]:
        if time.time() - start_time > MAX_TIME - 50:
            break
        S = _construct_from_assignments({2: r2}, rng)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
    # Phase 3: Fix primes 2 and 3
    for r2 in [0, 1]:
        for r3 in [0, 1, 2]:
            if time.time() - start_time > MAX_TIME - 40:
                break
            S = _construct_from_assignments({2: r2, 3: r3}, rng)
            if len(S) > best_size:
                best_S = S
                best_size = len(S)
                
    # Phase 4: Fix primes 2, 3, 5
    # 2*3*5 = 30 combinations.
    # This is the "branching on first 5 primes" insight, but we start with 3.
    for r2 in [0, 1]:
        for r3 in [0, 1, 2]:
            for r5 in [0, 1, 2, 3, 4]:
                if time.time() - start_time > MAX_TIME - 30:
                    break
                S = _construct_from_assignments({2: r2, 3: r3, 5: r5}, rng)
                if len(S) > best_size:
                    best_S = S
                    best_size = len(S)
                    
    # Phase 5: Fix primes 2, 3, 5, 7
    # 2*3*5*7 = 210 combinations.
    # This might be too slow if each construction takes ~0.1s. 210 * 0.1 = 21s. Feasible.
    if time.time() - start_time < MAX_TIME - 60:
        for r2 in [0, 1]:
            for r3 in [0, 1, 2]:
                for r5 in [0, 1, 2, 3, 4]:
                    for r7 in [0, 1, 2, 3, 4, 5, 6]:
                        if time.time() - start_time > MAX_TIME - 20:
                            break
                        S = _construct_from_assignments({2: r2, 3: r3, 5: r5, 7: r7}, rng)
                        if len(S) > best_size:
                            best_S = S
                            best_size = len(S)
                            
    # Phase 6: Randomized restarts with random initial assignments for small primes
    # If we have time left, do random searches.
    time_left = MAX_TIME - (time.time() - start_time)
    if time_left > 10:
        # We can try random assignments for primes 2,3,5,7,11
        # and let the greedy finisher do the rest.
        small_primes = [2, 3, 5, 7, 11]
        while time.time() - start_time < MAX_TIME - 5:
            # Generate random assignments for a subset of small primes
            k = rng.randint(1, 5)
            chosen_primes = rng.sample(small_primes, k)
            assignments = {}
            for p in chosen_primes:
                assignments[p] = rng.randint(0, p - 1)
                
            S = _construct_from_assignments(assignments, rng)
            if len(S) > best_size:
                best_S = S
                best_size = len(S)
                
    return best_S

def construct():
    return _construct()

if __name__ == "__main__":
    result = construct()
    print(f"Best size: {len(result)}")
