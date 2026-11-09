"""
Combined approach: Fast bitmask-based greedy + hill-climb with many restarts.
Replaces the slow 2-swap local search from the current program with a faster
multi-restart strategy inspired by the efficiency of inspiration 1.
"""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    # Safety margin to ensure we return within 120s (evaluator limit)
    deadline = start_time + 110.0

    N = 512

    # 1. Precomputation: Primes and Factor Masks
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # factor_masks[a] is a bitmask where bit i is 1 if primes[i] divides a
    # This allows O(1) coprime check: (mask_a & mask_b) == 0
    factor_masks = [0] * N
    for a in range(1, N):
        mask = 0
        tmp = a
        for i, p in enumerate(primes):
            if p * p > tmp:
                break
            if tmp % p == 0:
                mask |= (1 << i)
                while tmp % p == 0:
                    tmp //= p
        if tmp > 1:
            # tmp is a prime factor > sqrt(a), so it must be in our prime list (since a < 512)
            mask |= (1 << prime_to_idx[tmp])
        factor_masks[a] = mask

    # Precompute scores: 1 / (512 - a)
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)

    # Precompute number of prime factors for heuristic ordering
    def count_prime_factors(mask):
        return bin(mask).count('1')
    
    npf = [0] * N
    for a in range(1, N):
        npf[a] = count_prime_factors(factor_masks[a])

    # 2. Helper Functions
    
    def greedy_construct(ordering):
        """Greedy select in given order using bitmask coprimality."""
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        for a in ordering:
            if factor_masks[a] & chosen_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_set, chosen_mask

    def fast_hill_climb(chosen, chosen_set, chosen_mask, current_score, max_rounds=100):
        """
        Best-improvement local search with 1-add and 1-swap moves.
        Optimized for speed using bitmasks.
        """
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            best_gain = 0.0
            best_move = None  # ('add', b) or ('swap', a, b)
            
            # Try 1-add: add one element if possible
            # We iterate in descending score order to find the best add quickly
            # But since we want the GLOBAL best move, we must check all valid ones
            # However, scores are sorted by a (larger a -> smaller denominator -> larger score).
            # So iterating 511 down to 1 checks highest potential gains first.
            # We can break early if we find a gain that is maximal possible? No, because 
            # swapping might yield a larger net gain than adding.
            # But adding a high score item is always better than adding a low score item.
            
            for b in range(511, 0, -1):
                if b in chosen_set:
                    continue
                if factor_masks[b] & chosen_mask:
                    continue
                gain = scores[b]
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_move = ('add', b)
                    # Optimization: Since scores are descending, the first valid 'add' 
                    # is the best possible 'add'. We still need to check swaps though.
                    # We can't break here because a swap might be better.
            
            # Try 1-swap: remove one, add one
            # To optimize, we only consider removing 'a' if it has a low score, 
            # or if the best available 'b' is significantly higher.
            # A simple optimization: Only try swapping 'a' if scores[b_max] > scores[a].
            # Since we iterate b in descending order, the first valid b for a given a 
            # is the best b for that a.
            
            # To make this faster, we can limit which 'a's we try to remove.
            # If scores[a] is very high, it's unlikely swapping it for anything else improves score.
            # Specifically, if scores[a] >= max_possible_score_of_b, no swap involving removing a is useful.
            # max_possible_score is scores[511].
            
            # Let's iterate 'a' in ascending order of score (i.e., small 'a' first, which have low scores? No, small 'a' have LARGE scores? No.
            # score = 1/(512-a). 
            # a=1 -> 1/511 ~ 0.0019
            # a=511 -> 1/1 = 1.0
            # So large 'a' have high scores. We want to KEEP high scores.
            # We want to REMOVE low scores (small 'a').
            # So we should iterate 'a' from 1 upwards (lowest scores first) for removal candidates.
            
            for a in range(1, 512):
                if a not in chosen_set:
                    continue
                
                # If the best possible score (1.0) is not greater than current score of a, 
                # and we already have a good add move, maybe skip? 
                # Actually, if scores[a] is 1.0, we can't swap it for anything better.
                # If scores[a] is 0.5, we can swap for 1.0 (gain 0.5).
                
                temp_mask = chosen_mask & ~factor_masks[a]
                
                # Find best b to add if we remove a
                # Iterate b in descending score order
                for b in range(511, 0, -1):
                    if b in chosen_set:
                        continue
                    if factor_masks[b] & temp_mask:
                        continue
                    
                    gain = scores[b] - scores[a]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_move = ('swap', a, b)
                        # We found a valid b for this a. Since b is descending, this is the best b for this a.
                        # Can we break? No, a different 'a' might allow an even better 'b' or better net gain.
                        # But we can break the inner loop for this 'a'.
                        break 
                        # Note: We break the inner loop over b, but continue outer loop over a.
            
            if best_move is None:
                break
            
            mt = best_move[0]
            
            if mt == 'add':
                b = best_move[1]
                chosen.append(b)
                chosen_set.add(b)
                chosen_mask |= factor_masks[b]
                current_score += best_gain
            elif mt == 'swap':
                a, b = best_move[1], best_move[2]
                chosen.remove(a)
                chosen.append(b)
                chosen_set.remove(a)
                chosen_set.add(b)
                chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[b]
                current_score += best_gain
        
        return chosen, chosen_set, chosen_mask, current_score

    # 3. Main Search Loop
    best_chosen = []
    best_score = -1.0
    best_chosen_set = set()
    best_chosen_mask = 0

    # Strategy: Multiple Greedy Initializations + Hill Climb
    # We use a fixed number of restarts but dynamically adjust if time permits.
    # Given the speed of fast_hill_climb, we can do many restarts.
    
    rng = random.Random(42)
    
    # Define different ordering strategies
    # 1. Pure Score Descending (largest a first)
    order_score_desc = list(range(511, 0, -1))
    
    # 2. Score / NumPrimeFactors (prefer numbers that don't block many others)
    # This is a "density" heuristic.
    order_density = sorted(range(1, 512), key=lambda x: scores[x] / max(1, npf[x]), reverse=True)
    
    # 3. Randomized perturbations of the above
    
    restart_count = 0
    max_restarts = 500 # Upper bound, will stop earlier if time runs out
    
    while restart_count < max_restarts:
        if time.time() > deadline - 2.0: # Leave 2s margin for final check
            break
        
        restart_count += 1
        
        # Select ordering
        if restart_count == 1:
            ordering = order_score_desc
        elif restart_count == 2:
            ordering = order_density
        else:
            # Mix of random shuffle and density-based sorting
            # 50% chance: shuffle density order
            # 50% chance: weighted random selection
            if rng.random() < 0.5:
                ordering = order_density[:]
                # Randomly swap a few elements to perturb
                for _ in range(rng.randint(1, 20)):
                    i, j = rng.randint(0, 510), rng.randint(0, 510)
                    ordering[i], ordering[j] = ordering[j], ordering[i]
            else:
                # Weighted random ordering: higher score/density = higher probability of being early
                # Simple approach: shuffle, then sort by random key + weight
                ordering = list(range(1, 512))
                weights = [(rng.random() * 0.1 + scores[x] / max(1, npf[x]), x) for x in ordering]
                weights.sort(reverse=True)
                ordering = [x for _, x in weights]
        
        # Greedy Construct
        chosen, chosen_set, chosen_mask = greedy_construct(ordering)
        
        # Calculate initial score
        current_score = sum(scores[a] for a in chosen)
        
        # Hill Climb
        chosen, chosen_set, chosen_mask, current_score = fast_hill_climb(
            chosen, chosen_set, chosen_mask, current_score
        )
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_chosen = chosen[:]
            best_chosen_set = chosen_set.copy()
            best_chosen_mask = chosen_mask
            
            # Optional: If we found a very good solution, we could do a deeper search,
            # but fast_hill_climb is already quite good.
            
    # 4. Final Polish
    # Run hill climb one more time on the best found solution to ensure no obvious improvements left
    # This is redundant if the loop above converged, but safe.
    if best_chosen:
        best_chosen, best_chosen_set, best_chosen_mask, best_score = fast_hill_climb(
            best_chosen, best_chosen_set, best_chosen_mask, best_score, max_rounds=200
        )

    # Verification (Optional, for safety in development, remove for speed if needed)
    # The problem statement implies the evaluator checks validity.
    # We trust the bitmask logic.
    
    return best_chosen

# EVOLVE-BLOCK-END
