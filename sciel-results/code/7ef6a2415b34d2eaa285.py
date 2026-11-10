"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    N = 512
    # We want to maximize sum 1/(512 - a) for pairwise coprime a in 1..511
    # 1/(512-a) is largest when a is close to 511
    # So we want to include as many large values as possible, subject to pairwise coprime
    
    # Strategy: Greedy with backtracking / local search
    # First, let's get a good initial solution via greedy
    # Then improve via local search (swap operations)
    
    def score_of(chosen):
        s = 0
        for a in chosen:
            s += 1.0 / (512 - a)
        return s
    
    def is_coprime_with_all(a, chosen_set, chosen_list):
        for b in chosen_list:
            if math.gcd(a, b) != 1:
                return False
        return True
    
    # Greedy: sort by 1/(512-a) descending (i.e., a descending), pick if coprime
    def greedy_construct():
        chosen = []
        chosen_set = set()
        for a in range(511, 0, -1):
            if all(math.gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
                chosen_set.add(a)
        return chosen
    
    chosen = greedy_construct()
    best_score = score_of(chosen)
    
    # Local search: try swapping one element for another to improve score
    # Or try adding/removing elements
    # Better: try replacing an element a with a' where 1/(512-a') > 1/(512-a)
    # and a' is coprime to all other chosen elements
    
    chosen_set = set(chosen)
    
    # Precompute gcd matrix for speed? 511x511 is small
    # Actually let's just do direct checks
    
    # Local search: for each element in chosen, try to replace it with a larger-value element
    # that is coprime to all other elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, a in enumerate(list(chosen)):
            # Try to replace a with some a' > a (higher score) that is coprime to all others
            others = [b for j, b in enumerate(chosen) if j != i]
            # a' should have higher score, i.e., 1/(512-a') > 1/(512-a), i.e., a' > a
            # Also a' should not be in chosen (except a itself)
            for a_prime in range(min(511, a + 1), 511):
                if a_prime in chosen_set:
                    continue
                if all(math.gcd(a_prime, b) == 1 for b in others):
                    # Replace a with a_prime
                    chosen[i] = a_prime
                    chosen_set.discard(a)
                    chosen_set.add(a_prime)
                    new_score = score_of(chosen)
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                    break  # Only try the first valid replacement for this i
    
    # Also try adding elements: if there's a small a that can be added
    # Actually, we can't add if it's not coprime, but maybe we missed some
    # Let's do a final pass to try adding any element that fits
    for a in range(1, 512):
        if a in chosen_set:
            continue
        if all(math.gcd(a, b) == 1 for b in chosen):
            # Check if adding improves (it always does since 1/(512-a) > 0)
            chosen.append(a)
            chosen_set.add(a)
            best_score = score_of(chosen)
            improved = True
    
    # Another round of local search after additions
    while improved and time.time() < deadline:
        improved = False
        for i, a in enumerate(list(chosen)):
            others = [b for j, b in enumerate(chosen) if j != i]
            for a_prime in range(min(511, a + 1), 511):
                if a_prime in chosen_set:
                    continue
                if all(math.gcd(a_prime, b) == 1 for b in others):
                    chosen[i] = a_prime
                    chosen_set.discard(a)
                    chosen_set.add(a_prime)
                    new_score = score_of(chosen)
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                    break
    
    return chosen
# EVOLVE-BLOCK-END
