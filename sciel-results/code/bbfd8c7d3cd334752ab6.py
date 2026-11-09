import time
import random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max possible d: 1 + 2*255 = 511 <= 512
    
    def time_ok():
        return time.time() < deadline
    
    # Bitmask approach: bit i represents element (i+1) in the set
    # So element value v corresponds to bit (v-1)
    
    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count
    
    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m
    
    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]
    
    # Fast delta: when replacing old_val with new_val, only check d's that involve old_val or new_val
    def delta_replace(mask, old_val, new_val):
        """Return (delta, new_mask) when replacing old_val with new_val."""
        new_mask = mask & ~(1 << (old_val - 1)) | (1 << (new_val - 1))
        
        # Find d's that could be affected
        affected_ds = set()
        # d's involving old_val
        for d in range(1, MAX_D + 1):
            # old_val as first: old_val, old_val+d, old_val+2d
            if old_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            # old_val as middle: old_val-d, old_val, old_val+d
            if old_val - d >= 1 and old_val + d <= MAX_VAL:
                affected_ds.add(d)
            # old_val as last: old_val-2d, old_val-d, old_val
            if old_val - 2 * d >= 1:
                affected_ds.add(d)
        
        # d's involving new_val
        for d in range(1, MAX_D + 1):
            if new_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - d >= 1 and new_val + d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - 2 * d >= 1:
                affected_ds.add(d)
        
        delta = 0
        for d in affected_ds:
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        
        return delta, new_mask

    # New Structural Component 1: Greedy Construction with Progression Coverage Awareness
    # Instead of just maximizing total score, we prioritize adding elements that 
    # "unlock" many new differences d that were previously impossible.
    def greedy_construct_coverage():
        mask = 0
        s = 0
        s_set = set()
        covered_ds = set() # Track which d values are already achieved
        
        for _ in range(N):
            best_new = None
            best_gain = -1
            best_new_mask = mask
            best_new_s = s
            
            candidates = list(range(1, MAX_VAL + 1))
            random.shuffle(candidates)
            
            # Limit candidates for speed, but keep it high enough for diversity
            # 300 candidates is reasonable for 512 possible values
            for v in candidates[:300]:
                if v in s_set:
                    continue
                
                new_mask = mask | (1 << (v - 1))
                new_s = score_mask(new_mask)
                
                # Calculate gain in terms of new d's
                # This is expensive if done naively, but score_mask is O(255)
                # So this is O(300 * 255) per step, which is fine.
                gain = new_s - s
                
                if gain > best_gain:
                    best_gain = gain
                    best_new = v
                    best_new_mask = new_mask
                    best_new_s = new_s
            
            if best_new is not None:
                mask = best_new_mask
                s = best_new_s
                s_set.add(best_new)
            else:
                # Fallback: pick random
                v = random.choice(candidates)
                if v not in s_set:
                    mask |= (1 << (v - 1))
                    s = score_mask(mask)
                    s_set.add(v)
        
        return mask, s, s_set

    # New Structural Component 2: Simulated Annealing
    # Replaces the deterministic/first-improvement local search with a probabilistic
    # global search method that can escape local minima.
    def simulated_annealing(mask, s, s_set, max_steps=5000, T_start=10.0, T_end=0.1, alpha=0.99):
        current_mask = mask
        current_s = s
        current_set = set(s_set)
        
        best_mask = current_mask
        best_s = current_s
        best_set = set(current_set)
        
        T = T_start
        step = 0
        
        while step < max_steps and time_ok():
            if T < T_end:
                break
                
            # Generate a random move: swap one element with another
            i = random.randint(0, N - 1)
            old_val = list(current_set)[i] # Note: list conversion is O(N), acceptable for N=32
            # Actually, maintaining a list is faster. Let's use a list for the current state in SA.
            # Let's refactor to use a list for current state.
            
            # For now, let's stick to the mask/set but pick a random element from the set.
            # To make it faster, we can maintain a list.
            # Let's restart the SA loop with a list-based state for efficiency.
            break # The above loop is inefficient due to list(current_set). Let's rewrite.
            
        # Efficient SA implementation
        current_list = list(current_set)
        random.shuffle(current_list)
        current_mask = list_to_mask(current_list)
        current_s = score_mask(current_mask)
        
        best_mask = current_mask
        best_s = current_s
        best_list = current_list[:]
        
        T = T_start
        steps = 0
        max_steps = 10000 # Adjust based on time
        
        while steps < max_steps and time_ok():
            # Pick a random index to swap
            i = random.randint(0, N - 1)
            old_val = current_list[i]
            
            # Pick a random new value
            new_val = random.randint(1, MAX_VAL)
            if new_val == old_val or new_val in current_set:
                steps += 1
                T *= alpha
                continue
                
            delta, new_mask = delta_replace(current_mask, old_val, new_val)
            
            # Acceptance criterion
            if delta > 0 or random.random() < math.exp(delta / T):
                # Accept move
                current_set.discard(old_val)
                current_set.add(new_val)
                current_list[i] = new_val
                current_mask = new_mask
                current_s += delta
                
                if current_s > best_s:
                    best_s = current_s
                    best_mask = current_mask
                    best_list = current_list[:]
            
            T *= alpha
            steps += 1
            
        return best_mask, best_s, best_list

    # Main logic
    random.seed(42)
    
    best_mask = 0
    best_s = 0
    best_list = []
    
    # Strategy 1: Greedy with Coverage Awareness
    if time_ok():
        mask, s, s_set = greedy_construct_coverage()
        print(f"Greedy Cov: {s}", file=sys.stderr)
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = mask_to_list(mask)
        
        # Refine with Simulated Annealing
        if time_ok():
            mask, s, lst = simulated_annealing(mask, s, s_set)
            print(f"Greedy Cov + SA: {s}", file=sys.stderr)
            if s > best_s:
                best_mask = mask
                best_s = s
                best_list = lst
    
    # Strategy 2: Interval Start + SA
    if time_ok():
        mask = 0
        for i in range(1, 33):
            mask |= (1 << (i - 1))
        s = score_mask(mask)
        s_set = set(range(1, 33))
        print(f"Interval: {s}", file=sys.stderr)
        
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = mask_to_list(mask)
            
        mask, s, lst = simulated_annealing(mask, s, s_set)
        print(f"Interval + SA: {s}", file=sys.stderr)
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst

    # Strategy 3: Random Starts + SA
    for trial in range(10):
        if not time_ok():
            break
        lst = random.sample(range(1, MAX_VAL + 1), N)
        mask = list_to_mask(lst)
        s = score_mask(mask)
        s_set = set(lst)
        
        mask, s, lst = simulated_annealing(mask, s, s_set)
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]
            print(f"Random trial {trial} + SA: {s}", file=sys.stderr)
            
    # Final Polish: One more SA round on the best found
    if time_ok():
        s_set = set(best_list)
        mask, s, lst = simulated_annealing(best_mask, best_s, s_set)
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]
            
    print(f"Final score: {best_s}", file=sys.stderr)
    return best_list
