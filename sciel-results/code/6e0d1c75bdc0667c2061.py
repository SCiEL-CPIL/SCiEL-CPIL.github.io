import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 128
    TARGETS = list(range(128, 257))  # 129 targets
    
    # Precompute target indices for fast lookup
    target_set = set(TARGETS)
    
    def compute_mult(A_set):
        """Return max multiplicity over targets in [128,256]."""
        max_mult = 0
        for t in TARGETS:
            count = 0
            for a in A_set:
                b = t - a
                if b < a:
                    continue
                if b in A_set:
                    count += 1
            if count == 0:
                return float('inf')  # not covered
            if count > max_mult:
                max_mult = count
        return max_mult

    def is_complete(A_set):
        """Check if all targets are covered."""
        for t in TARGETS:
            found = False
            for a in A_set:
                b = t - a
                if b < a:
                    continue
                if b in A_set:
                    found = True
                    break
            if not found:
                return False
        return True

    def get_missing_targets(A_set):
        missing = []
        for t in TARGETS:
            found = False
            for a in A_set:
                b = t - a
                if b < a:
                    continue
                if b in A_set:
                    found = True
                    break
            if not found:
                missing.append(t)
        return missing

    best_A = None
    best_mult = float('inf')
    
    # Strategy 1: Structured Arithmetic Progressions
    # A = {0, d, 2d, ..., Nd} where Nd <= 128.
    # Sums of two elements: {2i*d, (2i+1)*d, ...}
    # We need to cover [128, 256].
    # If d=1, A={0..128}, sums cover [0, 256]. Mult is high.
    # If d=2, A={0, 2, ..., 128}, sums cover even numbers [0, 256]. Odd targets missing.
    # We need a mix.
    
    # Let's try a hybrid: Start with a structured set that covers most, then repair.
    # A good structured base for covering [128, 256] with low mult:
    # Consider A = {0, 1, 2, ..., 32} U {96, 97, ..., 128}?
    # Sums from low-low: [0, 64]
    # Sums from high-high: [192, 256]
    # Sums from low-high: [96, 160]
    # Union: [0, 64] U [96, 256]. Missing [65, 95].
    
    # Let's try to find a good structured seed via simple heuristics.
    # Candidate 1: All even numbers? No, misses odds.
    # Candidate 2: Random but optimized.
    
    # We will use a "Spectral" or "Fourier" inspired approach? No, too complex for 60s.
    # Let's stick to "Structured Seeds + Local Search".
    
    seeds = []
    
    # Seed 1: Even numbers + some odds to cover gaps
    # If we take A = {0, 2, 4, ..., 128}, we cover all even sums.
    # To cover odd sums, we need at least one odd number in A.
    # If we add 1, we cover sums involving 1.
    # Let's try A = {0, 2, 4, ..., 128} U {1, 3, 5, ...} but that's too big.
    
    # Better Seed: 
    # Let's try to construct a set with roughly k elements such that k^2/2 >= 129 (targets).
    # k ~ 16.
    # If A is random, mult is ~ 129 * C(k,2) / (256-128+1) ? No.
    # Expected mult for random set of size k:
    # Total pairs ~ k^2/2. Targets 129. Avg mult ~ k^2/258.
    # If k=20, avg ~ 400/258 ~ 1.5. Max likely 4-5.
    # If k=30, avg ~ 900/258 ~ 3.5. Max likely 6-8.
    # Current score 6.6 implies max mult around 6 or 7? Or maybe the score is normalized differently?
    # "Minimize the largest representation multiplicity".
    # If best is 6.6, maybe the max mult is 7 and score is 7/6.6? No, score is value.
    # Wait, "search score is seed/value". Seed is 1.0. Value is the max mult.
    # If score is 6.6, value is 1/6.6 = 0.15? No.
    # "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program".
    # If seed program returns max mult M_seed, score = 1.0.
    # If child returns max mult M_child, score = M_seed / M_child.
    # If score is 6.6, then M_seed / M_child = 6.6.
    # This implies M_child is much smaller than M_seed.
    # Or does "value" mean the multiplicity itself?
    # "Minimize the largest representation multiplicity". Let V be this value.
    # Score = Seed_Value / V.
    # If Seed_Value is high (e.g. 20) and V is low (e.g. 3), score = 6.6.
    # Let's check the seed program: `return list(range(64,129))`.
    # A = {64, ..., 128}. Size 65.
    # Sums: [128, 256].
    # For t in [128, 256], how many pairs?
    # t=128: 64+64 (1 pair).
    # t=192: 64+128, 65+127, ..., 96+96. 
    # Number of pairs for t: 
    # a+b=t, 64<=a<=b<=128.
    # a >= t-128, a <= t-64.
    # Also a <= b => a <= t/2.
    # So a in [max(64, t-128), min(128, t-64, t/2)].
    # For t=192: a in [64, 96]. Count = 96-64+1 = 33.
    # So Seed_Value is at least 33.
    # If Score = 6.6, then 33 / V = 6.6 => V = 5.
    # So the current best child has max multiplicity 5.
    # We need to beat 5? Or get closer to 4?
    # Theoretical minimum:
    # If we can get max mult 4, score = 33/4 = 8.25.
    # If we can get max mult 3, score = 11.
    # Is max mult 3 possible?
    # 129 targets. Each pair covers 1 target.
    # If max mult is 3, total pairs >= 129.
    # Number of pairs in A of size k is k(k+1)/2.
    # k(k+1)/2 >= 129 => k^2 ~ 258 => k ~ 16.
    # So a set of size ~16-20 might achieve max mult 3 or 4.
    
    # Strategy: Random search for small sets (k=15 to 25) with strong local search.
    
    def local_search(A_set, max_iters=2000):
        current_mult = compute_mult(A_set)
        if current_mult == float('inf'):
            return A_set, current_mult
            
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Try removing an element
            # If removing an element keeps coverage and doesn't increase max mult, do it?
            # Actually, removing an element reduces total pairs, so max mult might decrease or stay same.
            # It might break coverage.
            for a in list(A_set):
                new_set = A_set - {a}
                if not new_set: continue
                # Check coverage
                if not is_complete(new_set):
                    continue
                new_mult = compute_mult(new_set)
                if new_mult < current_mult:
                    A_set = new_set
                    current_mult = new_mult
                    improved = True
                    break
                elif new_mult == current_mult and len(new_set) < len(A_set):
                    # Prefer smaller set if mult is same?
                    # Might help future moves.
                    A_set = new_set
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try swapping an element
            # Pick a random element in A, swap with random element not in A
            if len(A_set) > 0:
                a_in = random.choice(list(A_set))
                a_out = random.randint(0, N)
                if a_out in A_set:
                    continue
                new_set = A_set - {a_in} | {a_out}
                if not is_complete(new_set):
                    continue
                new_mult = compute_mult(new_set)
                if new_mult < current_mult:
                    A_set = new_set
                    current_mult = new_mult
                    improved = True
            
            if not improved:
                # Stuck. Maybe try adding an element to reduce mult?
                # Adding an element increases pairs. Might increase mult.
                # But if current mult is high, adding an element that "spreads" load?
                # No, adding always adds pairs. It can only increase or keep max mult.
                # Unless the new element creates pairs for low-mult targets and high-mult targets?
                # It adds 1 to mult for each target t such that t-a_out in A.
                # So it increases mult for some targets.
                # So adding elements is generally bad for minimizing max mult, unless we are under-covered.
                # Since we check is_complete, we are covered.
                # So we should only remove or swap.
                
                # If we can't improve, break or random restart?
                # Let's try a few random swaps that maintain coverage and don't increase mult
                # to escape local minima.
                escaped = False
                for _ in range(10):
                    a_in = random.choice(list(A_set))
                    a_out = random.randint(0, N)
                    if a_out in A_set:
                        continue
                    new_set = A_set - {a_in} | {a_out}
                    if not is_complete(new_set):
                        continue
                    new_mult = compute_mult(new_set)
                    if new_mult <= current_mult:
                        A_set = new_set
                        current_mult = new_mult
                        escaped = True
                        break
                if not escaped:
                    break
                    
        return A_set, current_mult

    # Generate diverse seeds
    seeds_generated = 0
    while time.time() < deadline - 10 and seeds_generated < 50:
        seeds_generated += 1
        
        # Random size between 16 and 25
        k = random.randint(16, 25)
        
        # Random initial set
        A_set = set(random.sample(range(0, N+1), k))
        
        # Greedy completion if not complete
        if not is_complete(A_set):
            changed = True
            while changed:
                if time.time() > deadline - 5:
                    break
                changed = False
                missing = get_missing_targets(A_set)
                if not missing:
                    break
                t = missing[0]
                # Find x, y such that x+y=t, x,y in [0,128]
                # Add the one that covers most other missing targets?
                best_x = None
                best_gain = 0
                for x in range(0, N+1):
                    if x in A_set:
                        continue
                    y = t - x
                    if y < 0 or y > N:
                        continue
                    # If we add x, does it cover t? Only if y in A_set.
                    # If y not in A_set, adding x alone doesn't cover t.
                    # So we must add both x and y? Or add x such that y is already there?
                    # If y in A_set, adding x covers t.
                    # Also, adding x might cover other missing targets t' if t'-x in A_set.
                    gain = 0
                    if y in A_set:
                        gain = 1
                    else:
                        gain = 0
                    # Check other missing targets
                    for t2 in missing[1:]:
                        y2 = t2 - x
                        if 0 <= y2 <= N and y2 in A_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_y = y
                
                if best_gain > 0 and best_x is not None:
                    A_set.add(best_x)
                    if best_y not in A_set:
                        A_set.add(best_y)
                    changed = True
                else:
                    # No single addition covers t?
                    # This means for all x not in A, t-x is not in A.
                    # We must add a pair (x, t-x) where neither is in A?
                    # Or add x such that t-x is in A? We checked that.
                    # So we need to add two elements x and y such that x+y=t.
                    x = t // 2
                    y = t - x
                    if x not in A_set:
                        A_set.add(x)
                    if y not in A_set:
                        A_set.add(y)
                    changed = True
        
        if not is_complete(A_set):
            continue
            
        # Local search
        A_set, mult = local_search(A_set, max_iters=500)
        
        if mult < best_mult:
            best_mult = mult
            best_A = set(A_set)
            
        # If we hit a very good score, we can stop early or keep searching for 4?
        # If best_mult <= 4, it's very good.
        if best_mult <= 4:
            # Try to see if 3 is possible with more search?
            # For now, break if we have plenty of time? No, keep searching for 3.
            pass

    # If we didn't find anything, fallback to a known good structure
    if best_A is None:
        # Use the even numbers + 1?
        A_set = set(range(0, 129, 2)) | {1}
        # Check if complete
        if not is_complete(A_set):
            A_set = set(range(0, 129, 2)) | {1, 3}
        # Local search on this
        A_set, mult = local_search(A_set, max_iters=1000)
        best_A = A_set
        best_mult = mult

    return list(best_A)
