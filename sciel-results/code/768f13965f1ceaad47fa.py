import time
import random

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 80.0  # Safety margin

    # Pre-compute all 3-APs and 4-APs structure for fast lookup
    # A 3-AP is defined by (start, diff). Elements: s, s+d, s+2d
    # A 4-AP is defined by (start, diff). Elements: s, s+d, s+2d, s+3d
    
    # For efficiency in local search, we might just check on the fly with bitsets or sets.
    # Given n=256, sets are fast enough.
    
    # Greedy Construction
    # We want to maximize 3-APs. 
    # Strategy: Start with empty set.
    # At each step, pick an element that maximizes the number of new 3-APs formed,
    # subject to not forming any 4-AP.
    # To avoid O(n^2) or worse per step, we can be a bit smarter or just rely on n=256 being small.
    # 256 steps. For each step, check 256 candidates. For each candidate, check 4-AP violations.
    # 4-AP check: iterate d. Max d is ~85. So ~256 * 256 * 85 ~ 5.5M operations. This is fine.
    
    S = set()
    # We can randomize the order of candidates to escape local minima of the greedy choice?
    # Or just try a few different greedy strategies.
    
    def count_3aps(S):
        count = 0
        for x in S:
            for y in S:
                if x < y:
                    z = 2*y - x
                    if z in S and z > y:
                        count += 1
        return count

    # Better: incremental count
    # Let's stick to a robust greedy first.
    
    def greedy_construct(order):
        S = set()
        for x in order:
            # Check if adding x creates a 4-AP
            valid = True
            # x as 4th term: x-3d, x-2d, x-d in S
            # x as 3rd term: x-2d, x-d in S and x+d in S? No, if x+d in S, then {x-2d, x-d, x, x+d} is a 4-AP.
            # Actually, any 4 consecutive terms in AP.
            # If we add x, we need to check if there exists d such that:
            # 1. {x-3d, x-2d, x-d} subset S
            # 2. {x-2d, x-d, x+d} subset S
            # 3. {x-d, x+d, x+2d} subset S
            # 4. {x+d, x+2d, x+3d} subset S
            
            max_d = (n - 1) // 3 + 1
            for d in range(1, max_d + 1):
                if (x - 3*d in S and x - 2*d in S and x - d in S) or \
                   (x - 2*d in S and x - d in S and x + d in S) or \
                   (x - d in S and x + d in S and x + 2*d in S) or \
                   (x + d in S and x + 2*d in S and x + 3*d in S):
                    valid = False
                    break
            
            if valid:
                S.add(x)
        return S

    # Run greedy with natural order
    S1 = greedy_construct(range(1, n + 1))
    score1 = count_3aps(S1)
    
    # Run greedy with random order (few times)
    best_S = S1
    best_score = score1
    
    # Local Search / Polish
    # Try to swap elements to improve 3-AP count.
    # We can also try to add elements if possible (greedy might have missed some if order was bad? No, greedy adds if valid. But maybe adding x blocks adding y which forms more APs).
    # So swap is key.
    
    def get_score(S):
        return count_3aps(S)

    def try_improve(S):
        improved = True
        S = set(S)
        while improved and time.time() < deadline:
            improved = False
            # Try adding any element
            for x in range(1, n + 1):
                if x in S:
                    continue
                # Check 4-AP violation
                valid = True
                max_d = (n - 1) // 3 + 1
                for d in range(1, max_d + 1):
                    if (x - 3*d in S and x - 2*d in S and x - d in S) or \
                       (x - 2*d in S and x - d in S and x + d in S) or \
                       (x - d in S and x + d in S and x + 2*d in S) or \
                       (x + d in S and x + 2*d in S and x + 3*d in S):
                        valid = False
                        break
                if valid:
                    # Calculate gain
                    # New 3-APs involving x
                    gain = 0
                    for y in S:
                        # x is middle: y, x, z => z = 2x - y
                        z = 2*x - y
                        if z in S and z != y:
                            gain += 1
                        # x is end: y, w, x => w = x - (x-y)/2 ... 
                        # Simpler: check all pairs in S that with x form 3-AP
                        # Pairs (a,b) in S such that {a,b,x} is 3-AP
                        # a,b,x or a,x,b or x,a,b
                        # Since we iterate y in S, we cover x as middle (y, x, z) and x as one end (y, x, z where z in S? No, if z in S, it's counted when we pick y=z? No.)
                        
                        # Let's just count explicitly for correctness in this small N
                        # 3-APs containing x:
                        # 1. x is smallest: x, x+d, x+2d. Need x+d, x+2d in S.
                        # 2. x is middle: x-d, x, x+d. Need x-d, x+d in S.
                        # 3. x is largest: x-2d, x-d, x. Need x-2d, x-d in S.
                        
                        # We can just recount or do incremental. Recount is O(|S|^2). |S| ~ 100-150. 150^2 = 22500. 
                        # 256 candidates * 22500 = 5.7M ops per add check. 
                        # 256 adds * 5.7M = 1.4B ops. Too slow for Python if done naively in a loop.
                        
                        # Optimization: Precompute 3-APs?
                        # Let's precompute list of 3-APs.
                        pass
                    # For now, let's assume we can afford a simple recount if we limit iterations or optimize.
                    # Actually, 60s limit. Python is slow.
                    # Let's optimize the score calculation.
                    
                    # Precomputed 3-APs: list of (a,b,c)
                    # For a set S, count is sum(1 for ap in aps_3 if ap[0] in S and ap[1] in S and ap[2] in S)
                    # This is O(|aps_3| * 3). |aps_3| ~ 256*128/2 ~ 16000.
                    # 16000 checks per score calc.
                    # 256 candidates * 16000 = 4M ops per "try add" phase.
                    # This is feasible.
                    
                    pass
                
                # This approach is getting complicated to inline efficiently.
                # Let's use a precomputed list of 3-APs for scoring.
                pass
            # Try swaps
            # Remove y, Add x
            # If score(new) > score(old)
            pass
        return S

    # Let's rewrite with precomputed APs for speed.
    
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps_3.append((a, b, c))
            
    def score_set(S):
        cnt = 0
        for a, b, c in aps_3:
            if a in S and b in S and c in S:
                cnt += 1
        return cnt

    # Re-run greedy with score-awareness? 
    # The previous greedy just picked first valid. 
    # Let's try a greedy that picks the element maximizing immediate 3-AP gain.
    
    def greedy_scored(order):
        S = set()
        for x in order:
            valid = True
            max_d = (n - 1) // 3 + 1
            for d in range(1, max_d + 1):
                if (x - 3*d in S and x - 2*d in S and x - d in S) or \
                   (x - 2*d in S and x - d in S and x + d in S) or \
                   (x - d in S and x + d in S and x + 2*d in S) or \
                   (x + d in S and x + 2*d in S and x + 3*d in S):
                    valid = False
                    break
            if valid:
                # Calculate gain
                gain = 0
                for a, b, c in aps_3:
                    # Check if x is part of this AP and the other two are in S
                    if x == a:
                        if b in S and c in S: gain += 1
                    elif x == b:
                        if a in S and c in S: gain += 1
                    elif x == c:
                        if a in S and b in S: gain += 1
                # If gain > 0, add? Or just add if valid?
                # Adding even with 0 gain might enable future gains.
                # But adding with 0 gain might block future high gains.
                # Simple heuristic: Add if valid. 
                # To be smarter, we could delay adding if gain is low and there are other options?
                # But greedy is myopic.
                S.add(x)
        return S

    # Run multiple random greedy orders
    best_S = S1
    best_score = score_set(S1)
    
    for _ in range(5): # Limit number of random restarts due to time
        if time.time() > deadline - 10: break
        random_order = list(range(1, n + 1))
        random.shuffle(random_order)
        S_cand = greedy_scored(random_order)
        score_cand = score_set(S_cand)
        if score_cand > best_score:
            best_score = score_cand
            best_S = S_cand

    # Local Search: Swap
    # Iterate until no improvement or time limit
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S = set(best_S)
        current_score = best_score
        
        # Try adding elements
        for x in range(1, n + 1):
            if x in S: continue
            # Check 4-AP
            valid = True
            max_d = (n - 1) // 3 + 1
            for d in range(1, max_d + 1):
                if (x - 3*d in S and x - 2*d in S and x - d in S) or \
                   (x - 2*d in S and x - d in S and x + d in S) or \
                   (x - d in S and x + d in S and x + 2*d in S) or \
                   (x + d in S and x + 2*d in S and x + 3*d in S):
                    valid = False
                    break
            if valid:
                S.add(x)
                new_score = score_set(S)
                if new_score > current_score:
                    best_S = S
                    best_score = new_score
                    current_score = new_score
                    improved = True
                else:
                    S.remove(x)
        
        # Try swaps: remove y, add x
        # This is O(N^2) * Cost(Score). 
        # N=256. 256*256 = 65k swaps. 
        # Score cost ~ 16k. 65k * 16k = 1B. Too slow.
        # We need to be smarter. 
        # Only try swapping y if y is "low value" (part of few 3-APs) and x is "high value" (part of many 3-APs with S\{y}).
        
        # Let's just try a limited number of random swaps or specific candidates.
        # Or just rely on the add phase and maybe a simple removal if it helps? 
        # Removal only helps if it allows adding multiple elements.
        
        # Given time constraints, let's stick to the "Add if improves" loop which is O(N * Cost).
        # 256 * 16k = 4M ops. This is fast.
        # The swap is the bottleneck.
        
        # Optimization for Swap:
        # Precompute for each x not in S, the list of 3-APs it forms with S.
        # This is still expensive to update.
        
        # Let's just do a few passes of "Add if improves".
        # And maybe one pass of "Remove if it reduces 4-AP risk for many additions"? Hard.
        
        # Let's try a simple local search: 
        # For each x not in S, if adding x is valid, add it. (Greedy Add)
        # For each y in S, if removing y allows adding at least 2 elements that form more APs?
        # Too complex.
        
        # Let's just run the "Add" loop multiple times.
        pass

    return sorted(best_S)

# EVOLVE-BLOCK-END
