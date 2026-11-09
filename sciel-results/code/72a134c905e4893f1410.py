"""Rewritten program with stronger search for 32-element set maximizing 3-AP differences."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    N = 32
    MAX_VAL = 512
    MAX_D = MAX_VAL // 2  # 256, since a + 2*d <= 512
    
    def list_to_mask(lst):
        mask = 0
        for v in lst:
            mask |= (1 << v)
        return mask
    
    def score_mask(mask):
        """Count distinct d > 0 with a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count
    
    def score_list(lst):
        return score_mask(list_to_mask(lst))
    
    def delta_score(mask, pos_add, pos_remove):
        """
        Compute score change when adding pos_add and removing pos_remove from mask.
        pos_add: value to add (or 0 if no add)
        pos_remove: value to remove (or 0 if no remove)
        """
        # Score with just the add
        # We need to count new d values that become valid, minus d values that become invalid
        
        # For adding a value v: a d is "newly valid" if:
        #   - d was not valid before
        #   - after adding v, some (a, a+d, a+2d) is fully present
        # The triples that include v are:
        #   v as first: (v, v+d, v+2d) -> need v+d and v+2d in mask
        #   v as middle: (v-d, v, v+d) -> need v-d and v+d in mask
        #   v as last: (v-2d, v-d, v) -> need v-2d and v-d in mask
        
        # For removing a value w: a d becomes invalid if:
        #   - d was valid before
        #   - the only triple for d involved w
        #   - after removing w, no other triple for d exists
        
        # This is complex; let's just recompute for simplicity when needed
        pass
    
    def score_with_change(mask, add_val, remove_val):
        """Score after adding add_val and removing remove_val."""
        new_mask = mask
        if add_val:
            new_mask |= (1 << add_val)
        if remove_val:
            new_mask &= ~(1 << remove_val)
        return score_mask(new_mask)
    
    def simulated_annealing(lst, max_time=8.0):
        """SA with bitmask evaluation."""
        current = lst[:]
        current_mask = list_to_mask(current)
        current_score = score_mask(current_mask)
        
        best = lst[:]
        best_mask = current_mask
        best_score = current_score
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        t_start = time.time()
        
        while time.time() - t_start < max_time:
            # Choose a random element to move
            idx = random.randint(0, N - 1)
            old_val = current[idx]
            
            # Try replacing with a random new value
            new_val = random.randint(1, MAX_VAL)
            if new_val == old_val:
                continue
            # Check if new_val is already in current (excluding idx)
            if any(current[i] == new_val for i in range(N) if i != idx):
                continue
            
            # Compute new score
            new_mask = current_mask
            new_mask &= ~(1 << old_val)
            new_mask |= (1 << new_val)
            new_score = score_mask(new_mask)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                # Accept
                current[idx] = new_val
                current_mask = new_mask
                current_score = new_score
                
                if new_score > best_score:
                    best_score = new_score
                    best = current[:]
                    best_mask = new_mask
            else:
                # Reject (do nothing)
                pass
            
            T = max(T * cooling, T_min)
            
            # Periodic restart: occasionally try a swap instead
            if random.random() < 0.15:
                i, j = random.sample(range(N), 2)
                new_mask = current_mask
                # Swap doesn't change the set, so score is same
                # Actually swap of two elements in the list doesn't change the set!
                # So this is useless. Let's do a different perturbation.
                # Replace one element with a neighbor of another
                pass
        
        return best, best_score
    
    def greedy_construct():
        """
        Greedy: start with empty set, add elements one by one to maximize
        the number of new differences created.
        """
        mask = 0
        score = 0
        selected = []
        
        # Precompute which d's are currently valid
        valid_ds = set()
        
        for _ in range(N):
            best_val = -1
            best_new_score = -1
            best_new_ds = None
            
            # Try candidates: we'll sample a subset for speed
            candidates = random.sample(range(1, MAX_VAL + 1), min(200, MAX_VAL))
            # Also add some structured candidates
            for d in range(1, 50):
                for a in range(1, MAX_VAL - 2 * d + 1, d):
                    if a not in selected:
                        candidates.append(a)
            candidates = list(set(candidates))
            random.shuffle(candidates)
            candidates = candidates[:300]
            
            for v in candidates:
                if v in selected:
                    continue
                new_mask = mask | (1 << v)
                # Count new valid d's
                new_valid = 0
                for d in range(1, MAX_D + 1):
                    if (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0:
                        new_valid += 1
                if new_valid > best_new_score:
                    best_new_score = new_valid
                    best_val = v
            
            if best_val > 0:
                selected.append(best_val)
                mask |= (1 << best_val)
                score = best_new_score
        
        return selected, score
    
    def local_search(lst, max_time=5.0):
        """Local search: try single moves and swaps."""
        current = lst[:]
        current_mask = list_to_mask(current)
        current_score = score_mask(current_mask)
        best = lst[:]
        best_score = current_score
        
        t_start = time.time()
        no_improve = 0
        
        while time.time() - t_start < max_time:
            improved = False
            
            # Try all single replacements
            for i in range(N):
                if time.time() - t_start > max_time:
                    break
                old_val = current[i]
                # Try a few candidates
                for _ in range(20):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val == old_val:
                        continue
                    if any(current[j] == new_val for j in range(N) if j != i):
                        continue
                    
                    new_mask = current_mask
                    new_mask &= ~(1 << old_val)
                    new_mask |= (1 << new_val)
                    new_score = score_mask(new_mask)
                    
                    if new_score > current_score:
                        current[i] = new_val
                        current_mask = new_mask
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best = current[:]
                        break  # Restart the i loop with improved score
            
            if not improved:
                no_improve += 1
                if no_improve > 10:
                    break
            else:
                no_improve = 0
        
        return best, best_score
    
    # Main search
    best_overall_lst = None
    best_overall_score = 0
    
    # Start with greedy
    greedy_lst, greedy_score = greedy_construct()
    best_overall_score = greedy_score
    best_overall_lst = greedy_lst[:]
    
    # Local search on greedy
    ls_lst, ls_score = local_search(greedy_lst, max_time=6.0)
    if ls_score > best_overall_score:
        best_overall_score = ls_score
        best_overall_lst = ls_lst[:]
    
    # SA from greedy
    if time.time() < deadline - 15.0:
        sa_lst, sa_score = simulated_annealing(greedy_lst, max_time=10.0)
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall_lst = sa_lst[:]
    
    # Multi-start: random and structured initializations
    while time.time() < deadline - 20.0:
        # Generate diverse initial
        r = random.random()
        if r < 0.4:
            # Random
            init = random.sample(range(1, MAX_VAL + 1), N)
        elif r < 0.7:
            # Structured: pick a base step, create AP-like
            step = random.randint(8, 20)
            base = random.randint(1, MAX_VAL - step * N)
            init = [base + i * step for i in range(N)]
            # Ensure all in range
            init = [v for v in init if 1 <= v <= MAX_VAL]
            while len(init) < N:
                v = random.randint(1, MAX_VAL)
                if v not in init:
                    init.append(v)
            init = init[:N]
        else:
            # Two interleaved APs
            step1 = random.randint(6, 15)
            step2 = random.randint(6, 15)
            base1 = random.randint(1, 20)
            base2 = random.randint(20, 40)
            init = []
            for i in range(N // 2):
                v1 = base1 + i * step1
                v2 = base2 + i * step2
                if 1 <= v1 <= MAX_VAL and v1 not in init:
                    init.append(v1)
                if 1 <= v2 <= MAX_VAL and v2 not in init:
                    init.append(v2)
            while len(init) < N:
                v = random.randint(1, MAX_VAL)
                if v not in init:
                    init.append(v)
            init = init[:N]
        
        # SA from this init
        sa_lst, sa_score = simulated_annealing(init, max_time=min(6.0, deadline - time.time() - 10.0))
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall_lst = sa_lst[:]
        
        # Local search on SA result
        if time.time() < deadline - 10.0:
            ls_lst, ls_score = local_search(sa_lst, max_time=min(4.0, deadline - time.time() - 8.0))
            if ls_score > best_overall_score:
                best_overall_score = ls_score
                best_overall_lst = ls_lst[:]
    
    # Final local search on best
    if time.time() < deadline - 3.0:
        final_lst, final_score = local_search(best_overall_lst, max_time=min(4.0, deadline - time.time() - 2.0))
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall_lst = final_lst[:]
    
    # Ensure validity
    result = best_overall_lst[:]
    seen = set()
    clean = []
    for v in result:
        if 1 <= v <= MAX_VAL and v not in seen:
            seen.add(v)
            clean.append(v)
    while len(clean) < N:
        v = random.randint(1, MAX_VAL)
        if v not in seen:
            seen.add(v)
            clean.append(v)
    clean = clean[:N]
    
    return sorted(clean)

# EVOLVE-BLOCK-END
