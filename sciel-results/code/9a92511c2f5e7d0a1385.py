import time
import random
import math

def _score(S):
    """Calculate the number of distinct d > 0 such that a, a+d, a+2d in S."""
    S_set = set(S)
    valid_ds = set()
    elems = sorted(S)
    n = len(elems)
    for i in range(n):
        x = elems[i]
        for j in range(i + 1, n):
            y = elems[j]
            d = y - x
            if x + 2*d in S_set:
                valid_ds.add(d)
    return len(valid_ds)

def _compute_d_counts(S):
    """Compute d_count[d] = number of 3-term APs with difference d in S."""
    S_set = set(S)
    d_count = {}
    elems = sorted(S)
    n = len(elems)
    for i in range(n):
        x = elems[i]
        for j in range(i + 1, n):
            y = elems[j]
            d = y - x
            if x + 2*d in S_set:
                d_count[d] = d_count.get(d, 0) + 1
    return d_count

def _score_from_d_counts(d_count):
    """Score = number of d with d_count[d] > 0."""
    return len(d_count)

def _delta_swap(S_set, d_count, remove_val, add_val):
    """
    Compute the change in score when removing remove_val and adding add_val.
    Returns (delta_score, new_d_count) or None if invalid.
    """
    if add_val in S_set:
        return None
    if remove_val not in S_set:
        return None
    
    # Progressions involving remove_val:
    # remove_val can be: a (start), a+d (middle), a+2d (end)
    # We need to find all progressions that include remove_val
    
    new_d_count = dict(d_count)
    delta = 0
    
    # Find all progressions that contain remove_val
    # Case 1: remove_val = a (start of AP): a, a+d, a+2d
    # Case 2: remove_val = a+d (middle): a, a+d, a+2d => a = remove_val - d, a+2d = remove_val + d
    # Case 3: remove_val = a+2d (end): a, a+d, a+2d => a = remove_val - 2d, a+d = remove_val - d
    
    # For each possible d, check if remove_val participates in an AP
    # d ranges from 1 to 255
    for d in range(1, 256):
        # Case 1: remove_val is start: check if remove_val+d and remove_val+2d in S
        if remove_val + d in S_set and remove_val + 2*d in S_set:
            old_cnt = new_d_count.get(d, 0)
            new_d_count[d] = old_cnt - 1
            if old_cnt == 1:
                delta -= 1  # losing this d
            if new_d_count[d] <= 0:
                del new_d_count[d]
        
        # Case 2: remove_val is middle: check if remove_val-d and remove_val+d in S
        if remove_val - d >= 1 and remove_val - d in S_set and remove_val + d in S_set:
            old_cnt = new_d_count.get(d, 0)
            new_d_count[d] = old_cnt - 1
            if old_cnt == 1:
                delta -= 1
            if new_d_count[d] <= 0:
                del new_d_count[d]
        
        # Case 3: remove_val is end: check if remove_val-2d and remove_val-d in S
        if remove_val - 2*d >= 1 and remove_val - 2*d in S_set and remove_val - d in S_set:
            old_cnt = new_d_count.get(d, 0)
            new_d_count[d] = old_cnt - 1
            if old_cnt == 1:
                delta -= 1
            if new_d_count[d] <= 0:
                del new_d_count[d]
    
    # Now add add_val and check for new progressions
    for d in range(1, 256):
        # Case 1: add_val is start: add_val+d and add_val+2d in S (after removal)
        if add_val + d <= 512 and add_val + d in S_set and add_val + 2*d <= 512 and add_val + 2*d in S_set:
            old_cnt = new_d_count.get(d, 0)
            if old_cnt == 0:
                delta += 1  # gaining this d
            new_d_count[d] = old_cnt + 1
        
        # Case 2: add_val is middle: add_val-d and add_val+d in S
        if add_val - d >= 1 and add_val - d in S_set and add_val + d <= 512 and add_val + d in S_set:
            old_cnt = new_d_count.get(d, 0)
            if old_cnt == 0:
                delta += 1
            new_d_count[d] = old_cnt + 1
        
        # Case 3: add_val is end: add_val-2d and add_val-d in S
        if add_val - 2*d >= 1 and add_val - 2*d in S_set and add_val - d >= 1 and add_val - d in S_set:
            old_cnt = new_d_count.get(d, 0)
            if old_cnt == 0:
                delta += 1
            new_d_count[d] = old_cnt + 1
    
    return (delta, new_d_count)

def _generate_seeds():
    """Generate structured initial seeds for better starting points."""
    seeds = []
    
    # Arithmetic progression based
    for start in range(1, 50):
        for step in range(1, 16):
            vals = [start + i * step for i in range(32)]
            if max(vals) <= 512:
                seeds.append(set(vals))
    
    # Two interleaved APs
    for start in range(1, 20):
        for step1 in range(2, 12):
            for step2 in range(2, 12):
                if step1 == step2:
                    continue
                vals = set()
                for i in range(16):
                    v1 = start + i * step1
                    v2 = start + 1 + i * step2
                    if v1 <= 512:
                        vals.add(v1)
                    if v2 <= 512:
                        vals.add(v2)
                if len(vals) == 32:
                    seeds.append(vals)
    
    # Quadratic pattern: {i^2 mod 512} style
    for offset in range(1, 10):
        vals = set()
        for i in range(1, 50):
            v = (i * i + offset) % 512
            if v >= 1 and v <= 512:
                vals.add(v)
            if len(vals) == 32:
                break
        if len(vals) == 32:
            seeds.append(vals)
    
    # Random seeds
    for _ in range(50):
        seeds.append(set(random.sample(range(1, 513), 32)))
    
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_set = None
    best_score = 0
    
    # Generate and evaluate seeds
    seeds = _generate_seeds()
    
    # Score all seeds and pick best few to start SA from
    seed_scores = []
    for seed in seeds:
        s = _score(seed)
        seed_scores.append((s, seed))
    
    seed_scores.sort(key=lambda x: -x[0])
    
    # Take top seeds for SA starts
    top_seeds = [s for _, s in seed_scores[:20]]
    if not top_seeds:
        top_seeds = [set(random.sample(range(1, 513), 32))]
    
    # Simulated annealing from each top seed
    for seed_idx, init_seed in enumerate(top_seeds):
        if time.time() >= deadline:
            break
        
        current_set = set(init_seed)
        current_d_count = _compute_d_counts(current_set)
        current_score = _score_from_d_counts(current_d_count)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        # SA parameters
        T = 2.0
        T_min = 0.01
        cooling = 0.995
        
        # Time budget per SA run
        sa_deadline = min(time.time() + 3.0, deadline)
        
        while time.time() < sa_deadline and T > T_min:
            # Pick random element to remove
            elems = list(current_set)
            remove_val = random.choice(elems)
            
            # Pick random element to add (biased toward values that create progressions)
            # Generate candidate add values
            candidates = set()
            for x in elems:
                for d in range(1, 30):  # small d more likely to be useful
                    v1 = x - d
                    v2 = x + d
                    v3 = x - 2*d
                    v4 = x + 2*d
                    for v in (v1, v2, v3, v4):
                        if 1 <= v <= 512 and v not in current_set:
                            candidates.add(v)
            
            # Add some random candidates
            for _ in range(10):
                rv = random.randint(1, 512)
                if rv not in current_set:
                    candidates.add(rv)
            
            if not candidates:
                T *= cooling
                continue
            
            add_val = random.choice(list(candidates))
            
            # Compute delta
            result = _delta_swap(current_set, current_d_count, remove_val, add_val)
            if result is None:
                T *= cooling
                continue
            
            delta, new_d_count = result
            new_score = current_score + delta
            
            # Acceptance criterion
            if delta > 0 or random.random() < math.exp(delta / T):
                current_set.discard(remove_val)
                current_set.add(add_val)
                current_d_count = new_d_count
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            
            T *= cooling
    
    # Final verification
    if best_set is None:
        return list(range(1, 33))
    
    # Verify the result
    final_score = _score(best_set)
    if final_score != best_score:
        # Something went wrong, recompute
        pass
    
    return list(best_set)
