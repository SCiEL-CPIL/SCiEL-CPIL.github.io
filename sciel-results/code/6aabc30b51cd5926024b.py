"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

DEADLINE = 85.0  # safety margin before 90s kill

def _num_distinct_prime_factors(n):
    """Count distinct prime factors of n."""
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def _can_add_fast(S_set, S_list, candidate, n):
    """
    Fast check: can candidate be added to S_list (with S_set as a set for O(1) lookup)?
    
    Conditions to check:
    1. For all distinct b,c in S_list: candidate ∤ b*c
    2. For all distinct a,b in S_list: a ∤ b*candidate  (equivalently a ∤ candidate*b)
    
    We can optimize by precomputing for each a in S_list, which elements b in S_list
    would make a | b*candidate.
    """
    # Case 1: candidate ∤ b*c for all distinct b,c in S_list
    # This is equivalent to: for all distinct b,c in S_list, (b*c) % candidate != 0
    # We can check: does there exist b,c in S_list such that candidate | b*c?
    # candidate | b*c means that for each prime power p^k || candidate,
    # the total exponent of p in b*c is >= k.
    
    # Simpler approach: just iterate but with early exit
    # For case 1, we need to check all pairs (b,c) in S_list
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if (b * c) % candidate == 0:
                return False
    
    # Case 2: for each a in S_list, and each b in S_list (b != a), check a ∤ b*candidate
    # a | b*candidate iff (b*candidate) % a == 0
    # We can rewrite: let g = gcd(a, candidate), then a/g | b
    # So we need: for all a in S_list, for all b in S_list with b != a, (a/gcd(a,candidate)) ∤ b
    for i in range(n):
        a = S_list[i]
        g = gcd(a, candidate)
        a_reduced = a // g
        if a_reduced == 1:
            continue  # a divides candidate, so a always divides b*candidate
        # Check if there exists b in S_list, b != a, such that a_reduced | b
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            if b % a_reduced == 0:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute number of distinct prime factors for all numbers 2..256
    npf = {}
    for x in range(2, 257):
        npf[x] = _num_distinct_prime_factors(x)
    
    # Candidates sorted by (num_distinct_prime_factors, value) - prefer fewer prime factors
    candidates_by_npfs = sorted(range(2, 257), key=lambda x: (npf[x], x))
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy add in order of increasing prime factor count
    # Numbers with fewer distinct prime factors are "safer" to add
    
    # Try multiple random restarts with different orderings
    num_restarts = 200
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a candidate list: mostly ordered by npf, but with some randomization
        if attempt % 5 == 0:
            # Purely sorted by npf
            cand_list = candidates_by_npfs[:]
        elif attempt % 5 == 1:
            # Reverse sorted
            cand_list = candidates_by_npfs[::-1]
        elif attempt % 5 == 2:
            # Random shuffle
            cand_list = list(range(2, 257))
            random.shuffle(cand_list)
        elif attempt % 5 == 3:
            # Sorted by npf with random tie-breaking
            cand_list = list(range(2, 257))
            random.shuffle(cand_list)
            cand_list.sort(key=lambda x: npf[x])
        else:
            # Group by npf, shuffle within groups, shuffle group order
            groups = {}
            for x in range(2, 257):
                k = npf[x]
                if k not in groups:
                    groups[k] = []
                groups[k].append(x)
            group_keys = sorted(groups.keys())
            random.shuffle(group_keys)
            cand_list = []
            for k in group_keys:
                g = groups[k]
                random.shuffle(g)
                cand_list.extend(g)
        
        S = []
        S_set = set()
        for cand in cand_list:
            if time.time() > deadline:
                break
            if _can_add_fast(S_set, S, cand, len(S)):
                S.append(cand)
                S_set.add(cand)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Local search: try to add any remaining element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for cand in range(2, 257):
                if cand in S_set:
                    continue
                if _can_add_fast(S_set, S, cand, len(S)):
                    S.append(cand)
                    S_set.add(cand)
                    improved = True
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Final local search on the best set found
    if best_set and time.time() < deadline:
        S = best_set[:]
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for cand in range(2, 257):
                if cand in S_set:
                    continue
                if _can_add_fast(S_set, S, cand, len(S)):
                    S.append(cand)
                    S_set.add(cand)
                    improved = True
                    break
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    return best_set
# EVOLVE-BLOCK-END
