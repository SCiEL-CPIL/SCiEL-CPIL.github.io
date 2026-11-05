import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 4096
    max_sum = 3 * n  # 12288
    
    def check_valid_addition(x, selected, used_sums):
        """
        Check if x can be added to selected.
        Returns (True, new_sums_list) if valid, (False, None) otherwise.
        used_sums is a bytearray where used_sums[s] = 1 if sum s is taken.
        """
        new_sums = []
        
        # 1. x + x + x
        s = 3 * x
        if used_sums[s]:
            return False, None
        new_sums.append(s)
        
        # 2. x + x + y for y in selected
        for y in selected:
            s = 2 * x + y
            if used_sums[s]:
                return False, None
            # Check against new_sums (since x+y+x is same as x+x+y, but we just added 3x and 2x+y's)
            # Actually, 2x+y could collide with 3x if y=x, but x is not in selected yet.
            # Could collide with other 2x+y' if y=y'. But y are distinct in selected.
            # Could collide with x+y+z from selected? No, those are already in used_sums.
            # So we only need to check against used_sums and new_sums so far.
            if s in new_sums:
                return False, None
            new_sums.append(s)
        
        # 3. x + y + z for y, z in selected, y <= z
        k = len(selected)
        for i in range(k):
            y = selected[i]
            for j in range(i, k):
                z = selected[j]
                s = x + y + z
                if used_sums[s]:
                    return False, None
                if s in new_sums:
                    return False, None
                new_sums.append(s)
        
        return True, new_sums

    def greedy_build(order):
        used_sums = bytearray(max_sum + 1)
        selected = []
        
        for x in order:
            if time.time() > deadline:
                break
            
            valid, new_sums = check_valid_addition(x, selected, used_sums)
            if valid:
                selected.append(x)
                for s in new_sums:
                    used_sums[s] = 1
        
        return selected

    # Strategy 1: Randomized Greedy
    # Theoretical max size for B_3[1] in [1, N] is roughly (N/2)^(1/3) * constant?
    # Actually, for B_h[1], size is ~ N^(1/h). For h=3, N^(1/3) = 16.
    # Wait, the bound is tighter. 
    # Let's just run many random restarts.
    
    best_S = []
    
    # First, a quick deterministic greedy from small to large to get a baseline
    # This is fast and gives a decent starting point
    S_det = greedy_build(range(1, n + 1))
    best_S = S_det
    
    # Now randomized restarts
    # We need to be careful with time.
    # Each greedy pass is O(N * K^2) where K is size of set.
    # If K ~ 20, K^2 = 400. N=4096. 4096 * 400 = 1.6M ops per pass.
    # Python is slow, maybe 10-50ms per pass.
    # We can do ~1000-5000 passes in 80s.
    
    random.seed(42)
    
    # We can also try to bias the random order.
    # Maybe smaller numbers are "cheaper" in terms of sums?
    # No, smaller numbers produce smaller sums, which are more likely to collide with other small sums.
    # Larger numbers produce larger sums, which are less likely to collide.
    # However, we are bounded by N=4096.
    
    attempts = 0
    while time.time() < deadline - 2.0:
        # Generate a random permutation of [1, 4096]
        # To save time, we can just shuffle a list
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        S_rand = greedy_build(order)
        
        if len(S_rand) > len(best_S):
            best_S = S_rand
        
        attempts += 1
        
        # Optional: if we've been running for a while and not improving, 
        # we might try a different strategy or just keep going.
        # Given the stochastic nature, keeping going is usually best.
        
        # Check time more frequently if the loop is fast
        if attempts % 100 == 0:
             # print(f"Attempt {attempts}, best size {len(best_S)}")
             pass

    # Strategy 2: Local Search / Improvement
    # Try to add elements to best_S by removing others?
    # Or try to fill gaps?
    # Given the time constraint and that greedy is quite effective for these packing problems,
    # simple local search might not yield much if greedy is already near optimal.
    # But let's try a simple "add if possible" pass on the current best set, 
    # checking numbers in a different order (e.g., random or descending).
    
    # Actually, the greedy_build already tries all numbers. If it couldn't add them, 
    # they are blocked by existing sums.
    # To improve, we'd need to swap elements.
    # This is complex. Let's stick to multiple random restarts.
    
    # One more thing: maybe try a biased random order?
    # e.g., prefer numbers that are "sparsely" distributed?
    # Or just stick to uniform random.
    
    # Let's try a few more specific heuristics if time permits
    if time.time() < deadline - 5.0:
        # Try greedy with descending order
        S_desc = greedy_build(range(n, 0, -1))
        if len(S_desc) > len(best_S):
            best_S = S_desc
        
        # Try greedy with random order but biased towards larger numbers?
        # Just shuffle again
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            order = list(range(1, n + 1))
            random.shuffle(order)
            S_rand = greedy_build(order)
            if len(S_rand) > len(best_S):
                best_S = S_rand

    return best_S

# Quick test
if __name__ == "__main__":
    import sys
    start = time.time()
    S = construct()
    end = time.time()
    print(f"Size: {len(S)}, Time: {end-start:.2f}s")
    # Verify validity
    sums = set()
    valid = True
    for i in range(len(S)):
        for j in range(i, len(S)):
            for k in range(j, len(S)):
                s = S[i] + S[j] + S[k]
                if s in sums:
                    valid = False
                    print(f"Collision: {S[i]}+{S[j]}+{S[k]} = {s}")
                sums.add(s)
    print(f"Valid: {valid}")
