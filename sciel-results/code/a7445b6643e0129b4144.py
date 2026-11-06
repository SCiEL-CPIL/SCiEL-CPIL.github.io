"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 78.0
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflict as bitmask for speed
    # Use Python int as bitmask
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(nums[i], nums[j])
            if (nums[i] // g) * nums[j] <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def score_set(s):
        return sum(1.0 / nums[i] for i in s)
    
    def greedy():
        selected = 0  # bitmask
        for i in range(n):
            if not (conflict_mask[i] & selected):
                selected |= (1 << i)
        return selected
    
    def bits(s):
        return [i for i in range(n) if s & (1 << i)]
    
    def local_search(init_mask, max_iters=100):
        best = init_mask
        best_score = sum(1.0 / nums[i] for i in bits(best))
        
        for _ in range(max_iters):
            improved = False
            
            # Try adding one element
            for i in range(n):
                if best & (1 << i):
                    continue
                if conflict_mask[i] & best:
                    continue
                new_score = best_score + 1.0 / nums[i]
                if new_score > best_score:
                    best |= (1 << i)
                    best_score = new_score
                    improved = True
            
            # Try remove-one-add-one
            for i in range(n):
                if not (best & (1 << i)):
                    continue
                without_i = best & ~(1 << i)
                rem_score = best_score - 1.0 / nums[i]
                for j in range(n):
                    if best & (1 << j):
                        continue
                    if conflict_mask[j] & without_i:
                        continue
                    new_score = rem_score + 1.0 / nums[j]
                    if new_score > best_score:
                        best = without_i | (1 << j)
                        best_score = new_score
                        improved = True
                        break
            
            if not improved:
                break
        
        return best, best_score
    
    # Greedy init
    g = greedy()
    best_set = g
    best_score = sum(1.0 / nums[i] for i in bits(g))
    
    # Local search
    ls, ls_score = local_search(g, max_iters=80)
    if ls_score > best_score:
        best_set = ls
        best_score = ls_score
    
    if time.time() > deadline:
        return [nums[i] for i in bits(best_set)]
    
    # Random restarts
    random.seed(7)
    while time.time() < deadline - 3.0:
        r = random.random()
        if r < 0.25:
            # Random order greedy
            order = list(range(n))
            random.shuffle(order)
            order.sort(key=lambda i: nums[i] + random.random() * 80)
            s = 0
            for i in order:
                if not (conflict_mask[i] & s):
                    s |= (1 << i)
        elif r < 0.75:
            # Perturb best
            s = best_set
            k = random.randint(1, min(4, len(bits(s))))
            bl = bits(s)
            for _ in range(k):
                if bl:
                    idx = bl.pop(random.randint(0, len(bl)-1))
                    s &= ~(1 << idx)
        else:
            s = best_set
        
        ls, ls_score = local_search(s, max_iters=40)
        if ls_score > best_score:
            best_set = ls
            best_score = ls_score
    
    return [nums[i] for i in bits(best_set)]
# EVOLVE-BLOCK-END

