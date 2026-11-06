"""Improved program using Simulated Annealing with incremental updates for sum-free subsets."""

import random
import time
import math

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def max_sum_free_sa(vals, time_limit=10.0):
        n = len(vals)
        if n == 0:
            return []
        
        # Initial solution: Greedy
        rng = random.Random(123)
        cand = vals[:]
        rng.shuffle(cand)
        S = set()
        sums = set()
        diffs = set()
        
        for x in cand:
            if x in sums or x in diffs or (2*x in S):
                continue
            S.add(x)
            # Incremental add
            sums.add(2*x)
            for y in S:
                if y == x:
                    continue
                sums.add(x+y)
                diffs.add(x-y)
                diffs.add(y-x)
        
        best_S = set(S)
        best_size = len(S)
        
        start_time = time.time()
        T = 10.0
        cooling_rate = 0.999 # Slower cooling for more exploration
        
        while time.time() - start_time < time_limit:
            if T < 0.01:
                break
                
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try to add
                candidates_to_add = [x for x in vals if x not in S]
                if not candidates_to_add:
                    continue
                x = rng.choice(candidates_to_add)
                if x in sums or x in diffs or (2*x in S):
                    continue
                
                # Perform Add
                S.add(x)
                sums.add(2*x)
                for y in S:
                    if y == x:
                        continue
                    sums.add(x+y)
                    diffs.add(x-y)
                    diffs.add(y-x)
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)

            elif move_type < 0.8:
                # Try to remove
                if not S:
                    continue
                x = rng.choice(list(S))
                
                # Perform Remove
                S.remove(x)
                sums.discard(2*x)
                for y in S:
                    sums.discard(x+y)
                    diffs.discard(x-y)
                    diffs.discard(y-x)
                
                # Accept with prob exp(-1/T)
                if rng.random() < math.exp(-1/T):
                    pass # Keep removal
                else:
                    # Revert Remove (Add x back)
                    S.add(x)
                    sums.add(2*x)
                    for y in S:
                        if y == x:
                            continue
                        sums.add(x+y)
                        diffs.add(x-y)
                        diffs.add(y-x)

            else:
                # Try to swap
                if not S:
                    continue
                x = rng.choice(list(S))
                candidates_to_add = [y for y in vals if y not in S]
                if not candidates_to_add:
                    continue
                y = rng.choice(candidates_to_add)
                
                # Check if swap is valid
                # Temporarily remove x
                S.remove(x)
                # Save state to revert if invalid? 
                # Better: Check validity of y in S\{x} without modifying S permanently yet?
                # We can check:
                # 1. y not in sums(S\{x})
                # 2. y not in diffs(S\{x})
                # 3. 2y not in S\{x}
                
                # To check this, we need sums/diffs of S\{x}.
                # We already removed x from S, sums, diffs above.
                # So current sums/diffs are for S\{x}.
                
                is_valid_swap = True
                if y in sums or y in diffs or (2*y in S):
                    is_valid_swap = False
                
                if is_valid_swap:
                    # Perform Add y
                    S.add(y)
                    sums.add(2*y)
                    for z in S:
                        if z == y:
                            continue
                        sums.add(y+z)
                        diffs.add(y-z)
                        diffs.add(z-y)
                    
                    # Accept same-size move?
                    # Let's accept with prob 0.5 to allow exploration
                    if rng.random() < 0.5:
                        pass # Keep swap
                    else:
                        # Revert Swap (Remove y, Add x)
                        S.remove(y)
                        sums.discard(2*y)
                        for z in S:
                            sums.discard(y+z)
                            diffs.discard(y-z)
                            diffs.discard(z-y)
                        
                        S.add(x)
                        sums.add(2*x)
                        for z in S:
                            if z == x:
                                continue
                            sums.add(x+z)
                            diffs.add(x-z)
                            diffs.add(z-x)
                else:
                    # Revert Remove x (Add x back)
                    S.add(x)
                    sums.add(2*x)
                    for z in S:
                        if z == x:
                            continue
                        sums.add(x+z)
                        diffs.add(x-z)
                        diffs.add(z-x)

            T *= cooling_rate

        return list(best_S)

    results = []
    time_alloc = {60: 15.0, 120: 25.0, 240: 70.0}
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        t_limit = time_alloc.get(size, 10.0)
        subset = max_sum_free_sa(vals, t_limit)
        results.append(subset)
        
    return results
