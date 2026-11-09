import time
import random

P = {'n': 12, 'm': 40}
N_MASKS = 1 << P['n']

# Precompute comparability bitsets
# comp[m] is a bitset of all masks x such that x is comparable to m (strict subset/superset)
comp = [0] * N_MASKS

def precompute_comp():
    for m in range(N_MASKS):
        bits = 0
        # Iterate all other masks
        for x in range(N_MASKS):
            if m == x:
                continue
            # Check strict subset or superset
            if (m & x) == m: # m subset x
                bits |= (1 << x)
            elif (m & x) == x: # x subset m
                bits |= (1 << x)
        comp[m] = bits

def mask_to_list(mask):
    return [b for b in range(P['n']) if (mask >> b) & 1]

def score_family(family):
    """Calculate score for a list of masks."""
    # For speed in initial check, we can just use the bitsets
    F_bits = 0
    for m in family:
        F_bits |= (1 << m)
    
    total = 0
    for i in range(len(family)):
        for j in range(i+1, len(family)):
            mi = family[i]
            mj = family[j]
            if (mi & mj) == mi or (mi & mj) == mj:
                total += 1
    return total

def greedy_init():
    family = [0]
    F_bits = 1
    for _ in range(P['m'] - 1):
        best_m = -1
        best_score = -1
        for m in range(N_MASKS):
            if (F_bits >> m) & 1:
                continue
            # Score is number of comparable sets in current family
            s = (comp[m] & F_bits).bit_count()
            if s > best_score:
                best_score = s
                best_m = m
        family.append(best_m)
        F_bits |= (1 << best_m)
    return family

def hill_climb(family, time_limit):
    """
    Perform hill climbing.
    Returns improved family.
    """
    start_time = time.time()
    F_bits = 0
    for m in family:
        F_bits |= (1 << m)
    
    # Calculate initial score for reference (optional, we track deltas)
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        best_delta = 0
        best_i = -1
        best_m = -1
        
        for i in range(P['m']):
            s_i = family[i]
            rest_bits = F_bits ^ (1 << s_i)
            loss = (comp[s_i] & rest_bits).bit_count()
            
            # Find best replacement
            for m in range(N_MASKS):
                if (F_bits >> m) & 1:
                    continue
                gain = (comp[m] & rest_bits).bit_count()
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_m = m
        
        if best_delta > 0:
            # Apply swap
            old_m = family[best_i]
            F_bits ^= (1 << old_m)
            F_bits |= (1 << best_m)
            family[best_i] = best_m
        else:
            break
            
    return family

def construct():
    start_time = time.time()
    deadline = 50.0 # 50 seconds limit
    
    precompute_comp()
    
    best_family = []
    best_score = -1
    
    # Run multiple restarts
    num_restarts = 50 # Arbitrary, limited by time
    
    for r in range(num_restarts):
        if time.time() - start_time > deadline - 5.0: # Leave 5s for final processing
            break
            
        if r == 0:
            # Greedy start
            family = greedy_init()
        else:
            # Random start
            family = random.sample(range(N_MASKS), P['m'])
        
        # Hill climb
        # Allow more time for first restart? No, equal is fine.
        # Since hill climb is fast, we can do many.
        # But random starts are likely poor. Greedy is better.
        # Let's do Greedy + Hill Climb for the first few, then Random + Hill Climb.
        
        family = hill_climb(family, deadline - (time.time() - start_time) - 2.0)
        
        # Calculate score
        s = score_family(family)
        
        if s > best_score:
            best_score = s
            best_family = family[:]
            
    # Convert best_family to list of lists
    result = [mask_to_list(m) for m in best_family]
    
    # Ensure types are correct
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = int(result[i][j])
            
    return result
