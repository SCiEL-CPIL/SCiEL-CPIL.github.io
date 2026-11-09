import time
import random
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_instance(values):
        n = len(values)
        all_vals = list(values)
        
        # Greedy Initialization with multiple strategies
        best_init_S = set()
        best_init_size = 0
        
        strategies = [
            sorted(all_vals, reverse=True),
            sorted(all_vals),
            sorted(all_vals, key=abs, reverse=True),
            sorted(all_vals, key=abs)
        ]
        # Add a random shuffle
        rng_init = Random(42)
        shuffled = all_vals[:]
        rng_init.shuffle(shuffled)
        strategies.append(shuffled)
        
        for order in strategies:
            S = set()
            for v in order:
                valid = True
                for a in S:
                    if (v - a) in S or (a - v) in S:
                        valid = False
                        break
                if valid:
                    S.add(v)
            if len(S) > best_init_size:
                best_init_size = len(S)
                best_init_S = set(S)
        
        current_S = best_init_S
        current_size = best_init_size
        best_S = set(current_S)
        best_size = current_size
        
        # Maintain unselected list
        unselected = [v for v in all_vals if v not in current_S]
        
        rng = Random(12345)
        
        while time.time() < deadline:
            # Try to add from unselected
            added = False
            if unselected:
                # Try a few random candidates from unselected
                for _ in range(5):
                    idx = rng.randrange(len(unselected))
                    v = unselected[idx]
                    
                    valid = True
                    for a in current_S:
                        if (v - a) in current_S or (a - v) in current_S:
                            valid = False
                            break
                    
                    if valid:
                        current_S.add(v)
                        current_size += 1
                        unselected.pop(idx)
                        added = True
                        if current_size > best_size:
                            best_size = current_size
                            best_S = set(current_S)
                        break
            
            if added:
                continue
                
            # Try Swap: Remove one from S, Add one from unselected
            if current_S and unselected:
                # Remove random from S
                remove_idx = rng.randrange(len(current_S))
                # Convert to list for random access? 
                # It's faster to keep current_S as a set, but we need to pick random element.
                # list(current_S) is O(N). N is small.
                s_list = list(current_S)
                to_remove = s_list[remove_idx]
                
                current_S.remove(to_remove)
                current_size -= 1
                unselected.append(to_remove)
                
                # Try to add from unselected
                swap_success = False
                for _ in range(5):
                    idx = rng.randrange(len(unselected))
                    v = unselected[idx]
                    
                    # v might be to_remove itself
                    if v == to_remove:
                        continue
                        
                    valid = True
                    for a in current_S:
                        if (v - a) in current_S or (a - v) in current_S:
                            valid = False
                            break
                    
                    if valid:
                        current_S.add(v)
                        current_size += 1
                        unselected.pop(idx)
                        swap_success = True
                        if current_size > best_size:
                            best_size = current_size
                            best_S = set(current_S)
                        break
                        
                if not swap_success:
                    # If we couldn't add anything, put to_remove back
                    # to maintain size and not degrade
                    if to_remove not in current_S:
                        current_S.add(to_remove)
                        current_size += 1
                        # Remove to_remove from unselected
                        # Find index
                        for i, u in enumerate(unselected):
                            if u == to_remove:
                                unselected.pop(i)
                                break
        
        return list(best_S)

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        if time.time() > deadline - 5:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        
        vals = instance(size, seed)
        res = solve_instance(vals)
        results.append(res)
        
    return results
