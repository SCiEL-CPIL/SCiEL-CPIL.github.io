"""Rewritten program: beam search over coprime sets to maximize sum 1/(512-a)."""
import time
from math import gcd
from fractions import Fraction
import heapq
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    N = 512
    # Candidate values: we want large 1/(512-a), so a close to 511 is best.
    # But we need pairwise coprime.
    # Greedy seed: pick from 511 down to 1, keeping coprime.
    
    def score_set_exact(chosen):
        s = Fraction(0, 1)
        for a in chosen:
            s += Fraction(1, N - a)
        return s
    
    def score_set_float(chosen):
        return sum(1.0 / (N - a) for a in chosen)
    
    def greedy_seed():
        chosen = []
        for a in range(511, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    best_set = greedy_seed()
    best_score_float = score_set_float(best_set)
    
    # Precompute float weights
    weights_float = [1.0 / (N - a) for a in range(1, 512)]
    
    # Precompute gcd compatibility matrix for speed
    # compat[a][b] = gcd(a,b) == 1
    # We only need this for the local search polish, not for beam search
    # (beam search checks gcd on the fly which is fine for tuples)
    
    # Beam search with float scores for speed
    beam_width = 500
    # Each beam entry: (float_score, chosen_tuple)
    
    # Initial beam: one empty set
    beam = [(0.0, ())]
    
    # Process candidates in order of decreasing a (511, 510, ..., 1)
    for a in range(511, 0, -1):
        if time.time() > deadline - 30.0:  # reserve 30s for polish
            break
        
        new_beam = []
        w = weights_float[a - 1]  # 1/(512-a)
        
        for score, chosen in beam:
            # Option 1: Skip a
            new_beam.append((score, chosen))
            
            # Option 2: Include a (if coprime with all in chosen)
            if all(gcd(a, b) == 1 for b in chosen):
                new_score = score + w
                new_chosen = chosen + (a,)
                new_beam.append((new_score, new_chosen))
        
        # Keep top beam_width entries by score
        if len(new_beam) > beam_width:
            new_beam = heapq.nlargest(beam_width, new_beam, key=lambda x: x[0])
        
        beam = new_beam
    
    # Find the best complete solution in the final beam
    for score, chosen in beam:
        if score > best_score_float:
            best_score_float = score
            best_set = list(chosen)
    
    # Local search polish: try swaps and insertions
    def local_search_polish(chosen, time_limit):
        """Improve a set via 1-opt swaps and insertions."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        current_score = score_set_float(chosen)
        
        # Build candidate pool: all numbers 1..511 not in chosen
        # For each candidate, check if it's coprime with all in chosen
        # For swaps: remove one element, add another
        
        # Precompute for each a in 1..511: is it coprime with all in chosen?
        # This changes when we modify chosen, so we'll recompute as needed
        
        def can_add(a, current_set):
            return all(gcd(a, b) == 1 for b in current_set)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try insertions: add a number not in chosen that's coprime with all
            # Sort by weight descending (largest a first)
            for a in range(511, 0, -1):
                if time.time() > time_limit:
                    return chosen, current_score
                if a not in chosen_set:
                    if can_add(a, chosen_set):
                        new_score = current_score + weights_float[a - 1]
                        if new_score > current_score + 1e-15:
                            chosen.append(a)
                            chosen_set.add(a)
                            current_score = new_score
                            improved = True
            
            # Try swaps: remove b, add a (where a is coprime with all except b)
            # For each b in chosen, for each a not in chosen:
            #   if a is coprime with (chosen - {b}), and weight(a) > weight(b), try swap
            for b in list(chosen):
                if time.time() > time_limit:
                    return chosen, current_score
                for a in range(511, 0, -1):
                    if a == b or a in chosen_set:
                        continue
                    if weights_float[a - 1] <= weights_float[b - 1]:
                        continue  # no gain possible
                    # Check if a is coprime with all in chosen except b
                    ok = True
                    for c in chosen:
                        if c != b and gcd(a, c) != 1:
                            ok = False
                            break
                    if ok:
                        new_score = current_score - weights_float[b - 1] + weights_float[a - 1]
                        if new_score > current_score + 1e-15:
                            # Perform swap
                            chosen.remove(b)
                            chosen_set.remove(b)
                            chosen.append(a)
                            chosen_set.add(a)
                            current_score = new_score
                            improved = True
        
        return chosen, current_score
    
    # Polish the best set
    if time.time() < deadline - 5.0:
        polished_set, polished_score = local_search_polish(best_set, deadline - 5.0)
        if polished_score > best_score_float:
            best_set = polished_set
            best_score_float = polished_score
    
    # Return the best set
    return list(best_set)

# Let's test with a quick run
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Size: {len(result)}")
    s = sum(Fraction(1, 512-a) for a in result)
    print(f"Score: {float(s)}")
