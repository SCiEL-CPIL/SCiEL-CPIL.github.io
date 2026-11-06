import time
import random
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # Precompute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    weights = {s: L // s for s in range(1, 23)}
    
    # Map element to weight for faster access
    w_list = [0] * 23
    for s in range(1, 23):
        w_list[s] = weights[s]

    def check_add(sums, w):
        """Check if adding weight w to existing sums is dissociated."""
        for s in sums:
            if s + w in sums:
                return False
        return True

    def get_sums(S):
        sums = {0}
        for x in S:
            w = w_list[x]
            new_sums = {s + w for s in sums}
            sums.update(new_sums)
        return sums

    def greedy_with_order(order):
        """Greedy construction with given element order."""
        current_set = []
        current_sums = {0}
        for x in order:
            w = w_list[x]
            if check_add(current_sums, w):
                current_sums.update({s + w for s in current_sums})
                current_set.append(x)
        return current_set, current_sums

    best_set = []
    best_size = 0
    best_sums = {0}

    # Strategy 1: Multiple greedy passes with different orderings
    # Use both random and structured orderings
    all_elements = list(range(1, 23))
    
    # Deterministic orderings first
    deterministic_orders = [
        all_elements,
        sorted(all_elements, key=lambda x: -w_list[x]),  # descending weight
        sorted(all_elements, key=lambda x: w_list[x]),   # ascending weight
        sorted(all_elements, key=lambda x: x % 7),       # modular
        sorted(all_elements, key=lambda x: (x * 3) % 11),# another modular
    ]
    
    for order in deterministic_orders:
        if time.time() > deadline - 25:
            break
        s, sums = greedy_with_order(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = list(s)
            best_sums = sums

    # Random orderings
    for _ in range(150):
        if time.time() > deadline - 25:
            break
        order = list(all_elements)
        random.shuffle(order)
        s, sums = greedy_with_order(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = list(s)
            best_sums = sums

    # Strategy 2: Local Search with efficient swap handling
    current_best = list(best_set)
    current_sums = get_sums(current_best)
    
    # Precompute weight lookup for faster access
    in_set = set(current_best)
    
    for iteration in range(1000):
        if time.time() > deadline - 2:
            break
            
        improved = False
        missing = [x for x in range(1, 23) if x not in in_set]
        
        # 1. Try to add any missing element
        if missing:
            for x in missing:
                w = w_list[x]
                if check_add(current_sums, w):
                    current_sums.update({s + w for s in current_sums})
                    current_best.append(x)
                    in_set.add(x)
                    improved = True
                    if len(current_best) > best_size:
                        best_size = len(current_best)
                        best_set = list(current_best)
                        best_sums = get_sums(best_set)
                    break
        
        # 2. Try 1-swap: remove one, add one
        if not improved and len(current_best) > 0 and missing:
            # Sample some removals to avoid full search
            removals = random.sample(current_best, min(5, len(current_best)))
            for x_remove in removals:
                # Compute sums without x_remove
                sums_without = {0}
                for x in current_best:
                    if x == x_remove:
                        continue
                    w = w_list[x]
                    new_sums = {s + w for s in sums_without}
                    sums_without.update(new_sums)
                
                for x_add in missing:
                    w_add = w_list[x_add]
                    if check_add(sums_without, w_add):
                        new_sums = {s + w_add for s in sums_without}
                        current_sums = sums_without | new_sums
                        current_best = [x for x in current_best if x != x_remove] + [x_add]
                        in_set = set(current_best)
                        improved = True
                        if len(current_best) > best_size:
                            best_size = len(current_best)
                            best_set = list(current_best)
                            best_sums = get_sums(best_set)
                        break
                if improved:
                    break
        
        # 3. Try 2-swap: remove two, add two
        if not improved and len(current_best) > 2 and len(missing) >= 2:
            # Sample a few combinations
            for _ in range(8):
                if time.time() > deadline - 2:
                    break
                x_rem1, x_rem2 = random.sample(current_best, 2)
                x_add1, x_add2 = random.sample(missing, 2)
                
                # Compute sums without the two removed
                sums_without = {0}
                for x in current_best:
                    if x == x_rem1 or x == x_rem2:
                        continue
                    w = w_list[x]
                    new_sums = {s + w for s in sums_without}
                    sums_without.update(new_sums)
                
                # Try adding x_add1 then x_add2
                w1 = w_list[x_add1]
                if not check_add(sums_without, w1):
                    continue
                sums_after1 = sums_without | {s + w1 for s in sums_without}
                
                w2 = w_list[x_add2]
                if not check_add(sums_after1, w2):
                    continue
                
                new_sums = sums_after1 | {s + w2 for s in sums_after1}
                current_sums = new_sums
                current_best = [x for x in current_best if x != x_rem1 and x != x_rem2] + [x_add1, x_add2]
                in_set = set(current_best)
                improved = True
                if len(current_best) > best_size:
                    best_size = len(current_best)
                    best_set = list(current_best)
                    best_sums = get_sums(best_set)
                break
        
        # 4. Try 3-swap: remove three, add three (for better exploration)
        if not improved and len(current_best) > 3 and len(missing) >= 3:
            for _ in range(3):
                if time.time() > deadline - 2:
                    break
                x_rems = random.sample(current_best, 3)
                x_adds = random.sample(missing, 3)
                
                # Compute sums without the three removed
                sums_without = {0}
                for x in current_best:
                    if x in x_rems:
                        continue
                    w = w_list[x]
                    new_sums = {s + w for s in sums_without}
                    sums_without.update(new_sums)
                
                # Try adding the three new elements
                valid = True
                sums_temp = sums_without
                for x_add in x_adds:
                    w = w_list[x_add]
                    if not check_add(sums_temp, w):
                        valid = False
                        break
                    sums_temp = sums_temp | {s + w for s in sums_temp}
                
                if valid:
                    current_sums = sums_temp
                    current_best = [x for x in current_best if x not in x_rems] + x_adds
                    in_set = set(current_best)
                    improved = True
                    if len(current_best) > best_size:
                        best_size = len(current_best)
                        best_set = list(current_best)
                        best_sums = get_sums(best_set)
                    break
        
        # 5. Kick: If stuck, restart from best or random
        if not improved:
            if random.random() < 0.1:
                # Restart from best
                current_best = list(best_set)
                current_sums = get_sums(current_best)
                in_set = set(current_best)
            elif random.random() < 0.05:
                # Random restart
                order = list(all_elements)
                random.shuffle(order)
                s, sums = greedy_with_order(order)
                if len(s) > len(current_best):
                    current_best = s
                    current_sums = sums
                    in_set = set(current_best)

    # Final verification and return
    if best_set:
        return best_set
    return [1]
