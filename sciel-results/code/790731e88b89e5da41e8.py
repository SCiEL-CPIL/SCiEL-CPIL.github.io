import time
import math
import random

N = 5000
MAX_TIME = 295.0

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

# Precompute indices for each prime's residue classes
# indices[p][r] = list of x in [0, N] such that x % p == r
# This is memory heavy if stored as lists, so we use boolean masks or just iterate.
# Given N=5000, iterating is fast. But we can precompute counts.

def _count_admissible(S_removed):
    """
    S_removed is a dict {p: r} for primes p that have a removed residue.
    Returns the size of the tuple (count of x in [0, N] not removed).
    """
    count = 0
    for x in range(N + 1):
        removed = False
        for p, r in S_removed.items():
            if x % p == r:
                removed = True
                break
        if not removed:
            count += 1
    return count

def _is_admissible_tuple_size(size, S_removed):
    """
    Check if a tuple of the given size (determined by S_removed) is admissible.
    Admissibility: for every prime p <= size, some residue class mod p is missed.
    This is equivalent to: for every prime p <= size, there exists r in [0, p-1] 
    such that no x in the tuple has x % p == r.
    
    Wait, the definition of admissible tuple S is:
    For every prime p <= |S|, S does not cover all residues mod p.
    i.e., there exists r in [0, p-1] such that {x in S | x % p == r} is empty.
    
    In our formulation, S is the set of x NOT removed by S_removed.
    So for a prime p <= |S|, we need to check if there is a residue r such that
    NO x in S has x % p == r.
    This is equivalent to: for every residue r in [0, p-1], is it the case that
    ALL x in [0, N] with x % p == r are removed by S_removed?
    
    Actually, simpler:
    Let S be the tuple.
    For a fixed p <= |S|, S covers all residues mod p iff for every r in [0, p-1],
    there exists x in S such that x % p == r.
    
    We want S to NOT cover all residues. So we want there to exist r such that
    no x in S has x % p == r.
    This means: for some r, all x in [0, N] with x % p == r are NOT in S.
    Since S = [0, N] \ Removed, this means all x in [0, N] with x % p == r ARE in Removed.
    
    So, for a prime p <= |S|, the condition is:
    There exists r in [0, p-1] such that for all x in [0, N] with x % p == r,
    x is removed by S_removed.
    
    Note: x is removed by S_removed if there exists (q, r_q) in S_removed such that x % q == r_q.
    This is a complex check.
    
    Alternative view:
    The set of removed elements R_set = { x in [0, N] | exists (q, r_q) in S_removed, x % q == r_q }.
    The tuple S = [0, N] \ R_set.
    
    For a prime p <= |S|, S covers all residues mod p iff
    for every r in [0, p-1], there is x in S with x % p == r.
    This is false iff there is r in [0, p-1] such that for all x with x % p == r, x in R_set.
    
    So, to check admissibility for a given S_removed and size |S|:
    1. Calculate |S|.
    2. For each prime p <= |S|:
       a. Check if there exists r in [0, p-1] such that all x in [0, N] with x % p == r are in R_set.
       b. If no such r exists, the tuple is NOT admissible.
    """
    size = size
    for p in PRIMES:
        if p > size:
            break
        # Check if there is a residue r mod p that is completely covered by R_set
        found_missed_residue = False
        for r in range(p):
            # Check if all x in [0, N] with x % p == r are in R_set
            all_removed = True
            # Iterate through x in [0, N] with x % p == r
            # Start from r, step by p
            for x in range(r, N + 1, p):
                # Check if x is in R_set
                # x is in R_set if exists (q, rq) in S_removed such that x % q == rq
                is_removed = False
                for q, rq in S_removed.items():
                    if x % q == rq:
                        is_removed = True
                        break
                if not is_removed:
                    all_removed = False
                    break
            if all_removed:
                found_missed_residue = True
                break
        if not found_missed_residue:
            return False
    return True

def construct():
    start_time = time.time()
    
    # The primes that matter for defining the removal pattern are small.
    # Large primes p > N/2 have at most 2 numbers in [0, N] for any residue.
    # Actually, for p > N, there are no x in [0, N] with x % p == r for r > 0.
    # For p in (N/2, N], there are at most 2 x for each residue.
    
    # We will search over subsets of small primes.
    # Let's assume the optimal solution uses removals for primes up to some limit P_limit.
    # If a prime p is not in S_removed, it doesn't force any removals directly, 
    # but it might affect admissibility.
    # However, if p is not in S_removed, we are not removing any x based on p.
    # So R_set is determined only by the primes in S_removed.
    
    # Strategy: Randomized Local Search on the space of assignments for small primes.
    # Let P_small be the first K primes.
    # We assign a residue r_p in [0, p-1] to each p in P_small.
    # We can also choose to NOT remove for some p (r_p = -1).
    
    # Let's try K=10 (primes up to 29).
    P_search = PRIMES[:10]
    
    best_size = 0
    best_assignment = {}
    
    # Initial random assignment
    def random_assignment():
        assign = {}
        for p in P_search:
            if random.random() < 0.8:
                assign[p] = random.randint(0, p - 1)
            else:
                assign[p] = -1
        return assign
    
    def evaluate(assign):
        # Build S_removed dict
        S_removed = {p: r for p, r in assign.items() if r != -1}
        
        # Calculate size
        # Optimization: Use a boolean array for R_set
        removed = [False] * (N + 1)
        for p, r in S_removed.items():
            for x in range(r, N + 1, p):
                removed[x] = True
        
        size = 0
        for x in range(N + 1):
            if not removed[x]:
                size += 1
        
        # Check admissibility
        if not _is_admissible_tuple_size(size, S_removed):
            return -1
        
        return size

    # Run local search
    current_assign = random_assignment()
    current_size = evaluate(current_assign)
    if current_size > best_size:
        best_size = current_size
        best_assignment = dict(current_assign)
    
    for i in range(50000):
        if time.time() - start_time > MAX_TIME:
            break
            
        # Perturb
        new_assign = dict(current_assign)
        p_to_change = random.choice(P_search)
        if random.random() < 0.2:
            # Toggle on/off
            if new_assign[p_to_change] == -1:
                new_assign[p_to_change] = random.randint(0, p_to_change - 1)
            else:
                new_assign[p_to_change] = -1
        else:
            # Change residue
            new_assign[p_to_change] = random.randint(0, p_to_change - 1)
            
        new_size = evaluate(new_assign)
        
        if new_size > current_size:
            current_assign = new_assign
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_assignment = dict(current_assign)
        elif new_size == -1 and current_size > 0:
            # If we broke admissibility, maybe step back or random restart
            pass
        else:
            # Accept with some probability (simulated annealing style)
            # Or just random walk
            if random.random() < 0.05:
                current_assign = new_assign
                current_size = new_size
    
    # Reconstruct tuple from best_assignment
    S_removed = {p: r for p, r in best_assignment.items() if r != -1}
    removed = [False] * (N + 1)
    for p, r in S_removed.items():
        for x in range(r, N + 1, p):
            removed[x] = True
            
    result = [x for x in range(N + 1) if not removed[x]]
    
    # Final sanity check
    if _is_admissible_tuple_size(len(result), S_removed):
        return result
    else:
        # Fallback to empty set or a known valid set
        # An empty set is admissible (no primes p <= 0)
        # But we want the largest.
        # If best is invalid, return empty
        return []
