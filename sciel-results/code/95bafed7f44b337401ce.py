import time
import random
import math

TARGET = 200
N = 100
_SUM_MASK = (1 << (TARGET + 1)) - 1  # keep only sums 0..200

# EVOLVE-BLOCK-START

def _is_200_free(mask):
    """
    Returns True if no subset of values represented by the set bits of 'mask'
    sums to TARGET (200).
    'mask' is an integer where bit i corresponds to value i+1.
    """
    if mask == 0:
        return False  # Must be non-empty
    
    reach = 1  # Bit 0 set, sum 0 is reachable
    mm = mask
    while mm:
        # Extract lowest set bit
        low = mm & (-mm)
        v = low.bit_length()  # Value is bit_index + 1
        
        # Update reachable sums
        # reach | (reach << v)
        # We only care about sums <= 200, so we mask it
        reach = (reach | (reach << v)) & _SUM_MASK
        
        # Early exit if target sum is reached
        if (reach >> TARGET) & 1:
            return False
            
        mm ^= low
        
    return True

def _cardinality(mask):
    return bin(mask).count("1")

def _init_greedy_desc():
    """Seed strategy: Add 100 down to 1 if feasible."""
    selected = 0
    reach = 1
    for x in range(100, 0, -1):
        new_reach = (reach | (reach << x)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            selected |= (1 << (x - 1))
            reach = new_reach
    return selected

def _init_greedy_asc():
    """Add 1 up to 100 if feasible."""
    selected = 0
    reach = 1
    for x in range(1, 101):
        new_reach = (reach | (reach << x)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            selected |= (1 << (x - 1))
            reach = new_reach
    return selected

def _init_interval(lo, hi):
    """Create mask for interval [lo, hi]."""
    m = 0
    for v in range(lo, hi + 1):
        m |= (1 << (v - 1))
    return m

def _init_random(rng, p=0.5):
    """Create random mask, then repair by dropping largest offenders."""
    m = 0
    for v in range(1, N + 1):
        if rng.random() < p:
            m |= (1 << (v - 1))
    
    # Repair: while not free, drop largest element
    while m and not _is_200_free(m):
        hi = m.bit_length() - 1
        m &= ~(1 << hi)
        
    return m

def _local_search(mask, rng, deadline):
    """
    Perform local search to maximize cardinality.
    Moves: Add, Swap, Remove.
    """
    cur = mask
    cur_c = _cardinality(cur)
    best = mask
    best_c = cur_c
    
    # Temperature parameters
    T_start = 2.0
    T_end = 0.01
    # We don't know total iterations in advance, so use time-based cooling
    # Or just fixed cooling schedule based on loop count, but break on time.
    
    # Precompute set bits and missing bits for faster access? 
    # Iterating 1..100 is fast enough (100 checks).
    
    while time.time() < deadline:
        # Propose a move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to Add a random missing element
            # Pick a random value 1..100
            v = rng.randint(1, N)
            if (cur >> (v - 1)) & 1:
                continue # Already present
            
            cand = cur | (1 << (v - 1))
            if _is_200_free(cand):
                # Feasible add
                cand_c = cur_c + 1
                # Always accept improvements
                cur = cand
                cur_c = cand_c
                if cur_c > best_c:
                    best = cur
                    best_c = cur_c
            else:
                # Infeasible add, try Swap? Or just reject?
                # Let's try a swap in the next iteration or now.
                # For efficiency, if add fails, we might just try a swap immediately.
                # But keeping it simple: if add fails, we might try to swap this v with a random existing u.
                pass # Handled by swap branch below in subsequent iterations or here?
                # Let's handle swap as a separate move type to keep logic clean.
                
        elif move_type < 0.9:
            # Try to Swap: pick existing u, missing v.
            # Pick random v (missing)
            v = rng.randint(1, N)
            if (cur >> (v - 1)) & 1:
                continue # v is present, not a swap candidate (it's an add)
            
            # Pick random u (present)
            # To pick a random present bit, we can generate a random index and check, 
            # or maintain a list. List maintenance is O(N) per step, which is fine.
            # Or just pick random u in 1..100 and check if present.
            u = rng.randint(1, N)
            if not ((cur >> (u - 1)) & 1):
                continue # u is not present
            
            # Swap u out, v in
            cand = (cur & ~(1 << (u - 1))) | (1 << (v - 1))
            
            if _is_200_free(cand):
                cand_c = cur_c # Cardinality unchanged
                # Accept with probability 1 (to explore) or based on T?
                # Since cardinality is same, we accept to change state and maybe enable future adds.
                cur = cand
                # cur_c stays same
                # Update best if same cardinality? No, we want max cardinality.
                # But maybe this state is better? Hard to tell. 
                # We track best cardinality. If cur_c == best_c, we don't update best mask 
                # unless we want to store multiple bests. For simplicity, we only update best if strictly higher.
                
        else:
            # Try to Remove a random present element
            # This decreases cardinality, so we should be cautious.
            # Only do this if stuck? Or with low probability.
            u = rng.randint(1, N)
            if not ((cur >> (u - 1)) & 1):
                continue
            
            cand = cur & ~(1 << (u - 1))
            # Check feasibility (should be feasible if cur was, but removing can't make it infeasible? 
            # Wait, removing an element cannot create a new subset sum. So cand is always feasible if cur is.)
            # So we don't need _is_200_free check for removal.
            
            cand_c = cur_c - 1
            # Acceptance: only if we are "stuck" or using annealing.
            # Since it decreases score, we accept with low probability.
            # Probability = exp(-1 / T) ?
            # Let's use a fixed small probability for now, or tie it to how long we've been stuck.
            # Simpler: Accept with prob 0.1
            if rng.random() < 0.1:
                cur = cand
                cur_c = cand_c
                
        # Optional: Cooling? 
        # Since we don't have a fixed number of iterations, we can't easily cool down.
        # But we can monitor if we've found a new best.
        
    return best, best_c

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    rng = random.Random(42)
    
    # Generate initial candidates
    inits = []
    
    # 1. Greedy Descending (Seed)
    inits.append(_init_greedy_desc())
    
    # 2. Greedy Ascending
    inits.append(_init_greedy_asc())
    
    # 3. Intervals
    # {66..100} is valid (min sum of 3 is 66+67+68=201 > 200, max sum of 2 is 99+100=199 < 200)
    inits.append(_init_interval(66, 100))
    # {60..100}? Min sum of 3: 60+61+62=183. Max sum of 2: 99+100=199.
    # Can 60+61+79 = 200? Yes. So {60..100} is NOT necessarily valid.
    # We need to check validity for any interval.
    
    # 4. Random initializations
    for _ in range(5):
        m = _init_random(rng, p=0.6)
        if m:
            inits.append(m)
            
    # Find best initial
    best_mask = 0
    best_c = 0
    for m in inits:
        # Ensure validity (repair if necessary, though inits should be mostly valid or repaired)
        if not _is_200_free(m):
            # Repair
            mm = m
            while mm and not _is_200_free(mm):
                hi = mm.bit_length() - 1
                mm &= ~(1 << hi)
            m = mm
            
        if m == 0:
            continue
            
        c = _cardinality(m)
        if c > best_c:
            best_c = c
            best_mask = m
            
    # If no valid mask found (shouldn't happen), fallback
    if best_mask == 0:
        best_mask = _init_interval(66, 100)
        best_c = _cardinality(best_mask)
        
    # Run local search from best initial
    # We can also run from other good inits if time permits.
    # For now, run from the single best one to focus effort.
    
    m, c = _local_search(best_mask, rng, deadline)
    
    if c > best_c:
        best_c = c
        best_mask = m
        
    # If time remains, try a second run from a different random start
    if time.time() < deadline - 10.0:
        m2 = _init_random(rng, p=0.7)
        if not _is_200_free(m2):
            mm = m2
            while mm and not _is_200_free(mm):
                hi = mm.bit_length() - 1
                mm &= ~(1 << hi)
            m2 = mm
            
        if m2 and _cardinality(m2) > 0:
            m2, c2 = _local_search(m2, rng, deadline)
            if c2 > best_c:
                best_c = c2
                best_mask = m2

    # Convert mask to list
    out = []
    for v in range(1, N + 1):
        if (best_mask >> (v - 1)) & 1:
            out.append(v)
            
    if not out:
        out = list(range(66, 101))
        
    return out

# EVOLVE-BLOCK-END
