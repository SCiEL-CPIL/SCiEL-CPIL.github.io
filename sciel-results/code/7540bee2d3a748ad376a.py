import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0
    
    N = 4096
    
    def greedy_build(order, deadline):
        vals = []
        existing_sums = set()
        for x in order:
            if time.time() > deadline:
                break
            k = len(vals)
            # Check all new triple sums involving x
            conflict = False
            new_sums = []
            
            # 3x
            s = 3 * x
            if s in existing_sums:
                conflict = True
            else:
                new_sums.append(s)
            
            if not conflict:
                # 2x + v for v in vals
                for v in vals:
                    s = 2 * x + v
                    if s in existing_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
            
            if not conflict:
                # x + v1 + v2 for v1 <= v2 in vals
                for i in range(k):
                    v1 = vals[i]
                    for j in range(i, k):
                        v2 = vals[j]
                        s = x + v1 + v2
                        if s in existing_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                    if conflict:
                        break
            
            if not conflict:
                vals.append(x)
                existing_sums.update(new_sums)
        
        return vals
    
    best = []
    
    # Primary: greedy from 1 upward
    order1 = list(range(1, N + 1))
    result = greedy_build(order1, DEADLINE)
    if len(result) > len(best):
        best = result
    
    if time.time() > DEADLINE:
        return best
    
    # Randomized restarts with different orderings
    rng = random.Random(42)
    for attempt in range(200):
        if time.time() > DEADLINE:
            break
        
        # Different strategies for ordering
        strategy = attempt % 4
        
        if strategy == 0:
            # Small bias: random shuffle of [1..N]
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif strategy == 1:
            # Start from a random offset
            offset = rng.randint(0, 200)
            order = list(range(1 + offset, N + 1)) + list(range(1, 1 + offset))
        elif strategy == 2:
            # Two-phase: small numbers first (shuffled), then larger
            small = list(range(1, 200))
            large = list(range(200, N + 1))
            rng.shuffle(small)
            rng.shuffle(large)
            order = small + large
        else:
            # Exponential weighting: pick random starting point, go outward
            center = rng.randint(1, N)
            order = [center]
            for d in range(1, N):
                if center - d >= 1:
                    order.append(center - d)
                if center + d <= N:
                    order.append(center + d)
        
        result = greedy_build(order, DEADLINE)
        if len(result) > len(best):
            best = result
            # If we hit the theoretical max, stop
            if len(best) >= 40:
                break
    
    return best
