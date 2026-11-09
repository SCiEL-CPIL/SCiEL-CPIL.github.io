import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    MAX_VAL = 128
    
    # We want to cover sums in [128, 256] with a <= b, a,b in A.
    # A sum s can be represented by pairs (a, s-a) where a <= s-a.
    # This implies a <= s/2.
    # Also a >= s - 128 (since s-a <= 128).
    # So for a given s, valid a's are in [max(0, s-128), floor(s/2)].
    
    # Strategy:
    # 1. Start with a "base" set that ensures coverage.
    #    The simplest covering set is A = [0, 128].
    #    Sums: 0+128=128, 128+128=256. Only covers 128 and 256.
    #    We need to cover everything in between.
    #    
    #    Let's use a greedy approach to build A.
    #    We want to minimize the max multiplicity.
    #    Ideally, each sum s in [128, 256] has exactly 1 representation.
    #    This is a perfect matching problem if we view it as:
    #    Can we assign each sum s a unique pair (a_s, b_s) such that 
    #    the set of all a_s and b_s forms A, and no pair is reused?
    #    Actually, the pairs can share elements. A = {a_s, b_s for all s}.
    #    The multiplicity of a sum s is the number of pairs in A that sum to s.
    #    If we construct A such that each s has exactly one pair (a_s, b_s) 
    #    from our chosen set of pairs, then multiplicity is 1 IF no other 
    #    pairs in A sum to s.
    #    
    #    However, if we just pick 129 pairs (one for each s), A might be large.
    #    But we can try to pick pairs such that they don't create extra sums.
    #    This is hard.
    #    
    #    Alternative: 
    #    Use a known structure. 
    #    If A = [0, 128], sums are 0, 1, ..., 128 (from 0+a) and 128, ..., 256 (from 128+a).
    #    Wait, A must be subset of [0, 128].
    #    If A = [0, 128], the sums of two elements are:
    #    0+0=0, 0+128=128, 128+128=256.
    #    Only 3 sums. We need 129 sums (128..256).
    #    
    #    Let's try A = [0, 1, ..., 128].
    #    Sums of two elements from [0, 128]:
    #    Min sum 0, Max sum 256.
    #    Number of representations for sum s:
    #    For s in [0, 128]: s+1 representations (0+s, 1+s-1, ..., s+0). Unordered: floor(s/2)+1?
    #    Let's check s=128. Pairs (0,128), (1,127), ..., (64,64).
    #    Count = 64 + 1 = 65.
    #    For s in [129, 256]: 
    #    Let s = 128 + k, k in [1, 128].
    #    Pairs (a, b) with a+b=s, 0<=a<=b<=128.
    #    a >= s-128 = k. a <= s/2 = 64 + k/2.
    #    Number of values for a is roughly 64.
    #    So peak multiplicity is around 65.
    #    Current score 8.25 implies peak is likely around 8-12? No, score is seed/value.
    #    If seed is 1.0, and score is 8.25, then value = 1/8.25 = 0.12?
    #    Wait, "search score is seed/value, so 1.0 equals the seed program".
    #    If seed program has value V_seed, score = V_seed / V_current.
    #    If score is 8.25, V_current = V_seed / 8.25.
    #    This means the current program is 8.25 times BETTER than the seed.
    #    The seed program likely had a high peak (e.g., 65).
    #    If seed peak was 65, and current score is 8.25, then current peak is 65/8.25 ~ 7.8 -> 8.
    #    So the goal is to get peak multiplicity as low as possible, ideally 1 or 2.
    #    But can we get peak 1?
    #    If peak is 1, every sum in [128, 256] has exactly 1 representation.
    #    This is a Sidon set property for the target interval, but A is not necessarily a Sidon set globally.
    #    It just needs to be a "perfect difference set" or similar for the sumset restricted to [128, 256].
    
    # Let's use a greedy algorithm to construct A.
    # We want to cover all sums 128..256.
    # We can view this as: for each sum s, we need at least one pair (a, b) in A.
    # To minimize max multiplicity, we should try to distribute the representations.
    
    # A powerful heuristic: 
    # Start with A = {0, 128}.
    # Uncovered sums: 129..255.
    # For each uncovered sum s, find a pair (a, b) that covers it.
    # Prefer pairs where a or b is already in A to keep A small? 
    # Or prefer pairs that create few new high-multiplicity sums?
    
    # Better approach: 
    # Use a random restart greedy with local search.
    # But the prompt asks for a STRUCTURAL change.
    # The current program uses Simulated Annealing.
    # Let's use a **Greedy Set Cover with Conflict Minimization**.
    
    # Actually, let's look at the "error_repair" idea.
    # The observation mentions "error_repair" changes the chance of improvement.
    # This suggests that fixing invalid states (uncovered sums) is key.
    
    # New Strategy: **Iterative Greedy Construction with Backtracking-free Repair**
    # 1. Initialize A with 0 and 128.
    # 2. While there are uncovered sums in [128, 256]:
    #    a. Identify the sum s with the fewest available pairs (a,b) from current A + candidates.
    #       Actually, just pick the first uncovered sum.
    #    b. Generate all valid pairs (a, b) for s.
    #    c. For each pair, calculate the "cost" of adding the new element(s) to A.
    #       Cost = increase in max multiplicity or number of new high-multiplicity sums.
    #    d. Pick the pair with minimum cost.
    #    e. Add the new element(s) to A.
    # 3. Once all sums are covered, perform local search to reduce peak multiplicity.
    
    # This is deterministic and structured.
    
    def get_pairs_for_sum(s, A_set):
        """Get all valid pairs (a, b) with a <= b, a+b=s, a,b in [0, 128]."""
        pairs = []
        min_a = max(0, s - 128)
        max_a = s // 2
        for a in range(min_a, max_a + 1):
            b = s - a
            if b <= 128:
                pairs.append((a, b))
        return pairs
    
    def evaluate_peak(A_list):
        """Calculate peak multiplicity and coverage."""
        A_set = set(A_list)
        indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for a in A_set:
            indicator[a] = 1
        
        conv = np.convolve(indicator, indicator, mode='full')
        s_vals = np.arange(N_LOW, N_HIGH + 1, dtype=np.int32)
        conv_slice = conv[N_LOW:N_HIGH+1]
        
        even_mask = (s_vals % 2 == 0)
        counts = np.zeros(len(s_vals), dtype=np.int32)
        
        counts[~even_mask] = conv_slice[~even_mask] // 2
        
        if np.any(even_mask):
            even_s = s_vals[even_mask]
            mids = even_s // 2
            valid_mid = (mids >= 0) & (mids <= MAX_VAL)
            raw = conv_slice[even_mask].copy()
            raw[valid_mid] += indicator[mids[valid_mid]]
            counts[even_mask] = raw // 2
        
        if not np.all(counts > 0):
            return float('inf'), False
        return int(np.max(counts)), True

    best_peak = float('inf')
    best_A = None
    
    # Run multiple random restarts of the greedy process
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        random.seed(42 + restart)
        A = {0, 128}
        
        # Shuffle the order in which we cover sums to introduce diversity
        sums_to_cover = list(range(N_LOW, N_HIGH + 1))
        # 128 and 256 are already covered by {0, 128}
        # 128: 0+128
        # 256: 128+128
        # So we need to cover 129..255
        sums_to_cover = [s for s in sums_to_cover if s not in [128, 256]]
        random.shuffle(sums_to_cover)
        
        for s in sums_to_cover:
            if time.time() > deadline - 5:
                break
                
            # Check if s is already covered by current A
            # We can check this by seeing if there exists a,b in A such that a+b=s
            # For efficiency, we can just try to find a pair.
            covered = False
            for a in A:
                b = s - a
                if b in A and a <= b:
                    covered = True
                    break
            
            if covered:
                continue
                
            # Find best pair to add
            pairs = get_pairs_for_sum(s, A)
            best_pair = None
            best_cost = float('inf')
            
            # Limit candidates to speed up
            random.shuffle(pairs)
            for a, b in pairs[:20]: # Only check top 20 random pairs
                # New elements to add
                new_elements = set()
                if a not in A:
                    new_elements.add(a)
                if b not in A:
                    new_elements.add(b)
                
                if not new_elements:
                    continue # Should not happen if not covered
                
                # Simulate adding new_elements
                temp_A = A.union(new_elements)
                
                # Calculate cost: how many sums in [128, 256] get high multiplicity?
                # Or simply: what is the max multiplicity in the new state?
                # This is expensive. 
                # Heuristic: Count how many existing sums in [128, 256] are affected.
                # A new element x affects sums x+y for y in A.
                # Let's just use a simple heuristic: prefer pairs with smaller "newness"
                # i.e., prefer pairs where one element is already in A.
                
                cost = 0
                if a not in A:
                    cost += 1
                if b not in A:
                    cost += 1
                    
                # Tie-breaker: random
                cost += random.random() * 0.1
                
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (a, b)
                    
            if best_pair:
                A.add(best_pair[0])
                A.add(best_pair[1])
            else:
                # Fallback: add any valid pair
                if pairs:
                    a, b = pairs[0]
                    A.add(a)
                    A.add(b)
                else:
                    break # Should not happen
            
        # After construction, evaluate
        current_peak, is_covered = evaluate_peak(list(A))
        
        if is_covered and current_peak < best_peak:
            best_peak = current_peak
            best_A = list(A)
            
        # Local Search Phase: Try to reduce peak by swapping elements
        if is_covered and time.time() < deadline - 5:
            current_A = set(best_A)
            current_peak = best_peak
            
            # Identify sums with peak multiplicity
            # We can't easily get the counts without re-evaluating, but we can try random swaps.
            for _ in range(50):
                if time.time() > deadline - 2:
                    break
                
                # Pick a random element to remove
                if len(current_A) <= 2:
                    break
                to_remove = random.choice(list(current_A))
                # Pick a random element to add
                to_add = random.randint(0, 128)
                
                if to_remove == to_add:
                    continue
                if to_add in current_A:
                    continue
                    
                new_A = current_A - {to_remove}
                new_A.add(to_add)
                
                new_peak, new_covered = evaluate_peak(list(new_A))
                
                if new_covered and new_peak < current_peak:
                    current_A = new_A
                    current_peak = new_peak
                    best_peak = current_peak
                    best_A = list(current_A)
                elif new_covered and new_peak == current_peak:
                    # Accept neutral moves to escape local minima
                    if random.random() < 0.2:
                        current_A = new_A

    if best_A is None:
        # Fallback: full range
        best_A = list(range(0, 129))
        
    return best_A
