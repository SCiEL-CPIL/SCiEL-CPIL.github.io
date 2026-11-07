"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

DEADLINE = 85.0  # safety margin before 90s kill

def _is_valid_set(S):
    """Check if set S satisfies: for all distinct a,b,c in S, a does not divide b*c."""
    S = list(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(n):
            if j == i:
                continue
            b = S[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S[k]
                if (b * c) % a == 0:
                    return False
    return True

def _is_valid_with(S, candidate):
    """Check if adding candidate to S keeps the set valid."""
    S = list(S)
    n = len(S)
    # Check: for all distinct a,b,c in S ∪ {candidate}, a ∤ b*c
    # Cases:
    # 1. a = candidate: candidate ∤ b*c for all distinct b,c in S
    # 2. a in S, b = candidate: a ∤ candidate*c for all c in S, c ≠ a (and c distinct from a, candidate)
    # 3. a in S, c = candidate: a ∤ b*candidate for all b in S, b ≠ a (and b distinct from a, candidate)
    
    # Case 1: candidate ∤ b*c for all distinct b,c in S
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (S[i] * S[j]) % candidate == 0:
                return False
    
    # Case 2 & 3: for a in S, check a ∤ b*c where one of b,c is candidate
    for i in range(n):
        a = S[i]
        for j in range(n):
            if j == i:
                continue
            # b = S[j], c = candidate: a ∤ S[j] * candidate
            if (S[j] * candidate) % a == 0:
                return False
            # b = candidate, c = S[j]: same as above (symmetric in b,c for divisibility)
            # Actually b*c = c*b, so (S[j]*candidate)%a == (candidate*S[j])%a, same check
    
    return True

def _remove_and_check(S, remove_idx):
    """Check if removing element at remove_idx from S results in a valid set."""
    S_list = list(S)
    del S_list[remove_idx]
    return _is_valid_set(S_list)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with all primes in 2..256 as initial candidate set
    # Actually, let's start greedy: try to add elements in some order
    
    # Strategy: Start with a large set, then remove violations
    # Or: Greedy add
    
    # Let's try: start with all numbers 2..256, then iteratively remove elements that cause violations
    # But that's O(n^4) which is too slow for 255 elements
    
    # Better: Greedy construction
    # Order elements by some heuristic (e.g., primes first, then composites)
    
    # Let's try a different approach: start with primes > 128 (safe), then try to add more
    
    # Actually, let me try greedy with random restarts
    best_set = []
    best_size = 0
    
    # Precompute: for each pair (b,c), which a's divide b*c
    # This helps in checking
    
    # Let's try a simpler greedy: sort candidates, try to add each if valid
    candidates = list(range(2, 257))
    
    # Try multiple random orderings
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        S = []
        for cand in candidates:
            if time.time() > deadline:
                break
            if _is_valid_with(S, cand):
                S.append(cand)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Try local search: try to add elements not in S
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for cand in range(2, 257):
                if cand in S:
                    continue
                if _is_valid_with(S, cand):
                    S.append(cand)
                    improved = True
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    return best_set
# EVOLVE-BLOCK-END
