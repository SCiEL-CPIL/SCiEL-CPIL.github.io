import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Initial set: the known valid set of size 12
    S = set(range(13, 25))
    best_S = set(S)
    best_size = len(S)
    
    def is_valid(S):
        if not S:
            return True
        recip_frac = {x: Fraction(1, x) for x in S}
        for x in S:
            target = recip_frac[x]
            others = [recip_frac[y] for y in S if y != x]
            # Generate all subset sums of others
            sums = {Fraction(0)}
            for r in others:
                new_sums = set()
                for s in sums:
                    new_sums.add(s + r)
                sums.update(new_sums)
                if target in sums:
                    return False
        return True

    all_elements = set(range(1, 25))
    
    # To speed up, we can pre-check if adding u is likely to work?
    # No, just try.
    
    # Random restarts if stuck?
    # Let's run a loop of local search steps.
    
    no_improve_count = 0
    last_best_size = best_size
    
    while time.time() < deadline:
        # 1. Try to add a random element
        candidates = list(all_elements - S)
        if candidates:
            u = random.choice(candidates)
            S_new = S | {u}
            if is_valid(S_new):
                S = S_new
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    no_improve_count = 0
                    last_best_size = best_size
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # 2. Try to swap
        if S:
            v = random.choice(list(S))
            S_temp = S - {v}
            candidates = list(all_elements - S_temp)
            if candidates:
                u = random.choice(candidates)
                S_new = S_temp | {u}
                if is_valid(S_new):
                    if len(S_new) > best_size:
                        S = S_new
                        best_size = len(S_new)
                        best_S = set(S_new)
                        no_improve_count = 0
                        last_best_size = best_size
                    else:
                        no_improve_count += 1
                        # Accept same size swap with low prob to explore
                        if random.random() < 0.05:
                            S = S_new
        
        # 3. If stuck for a long time, try a random restart from a different seed
        if no_improve_count > 100:
            # Try a different starting point
            # Random set of size 12
            S = set(random.sample(list(all_elements), 12))
            if is_valid(S):
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
            no_improve_count = 0
            last_best_size = best_size

    return list(best_S)
