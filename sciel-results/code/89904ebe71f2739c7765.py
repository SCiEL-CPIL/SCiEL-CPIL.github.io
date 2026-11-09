import time
import random

# Precomputation
N_POINTS = 16
N_MASKS = 1 << N_POINTS
ALL_MASKS = range(N_MASKS)

# Precompute popcount
POP_COUNT = [0] * N_MASKS
for i in range(1, N_MASKS):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)

# Precompute all triples (masks)
TRIPLES = []
for a in range(N_POINTS):
    for b in range(a + 1, N_POINTS):
        for c in range(b + 1, N_POINTS):
            TRIPLES.append((1 << a) | (1 << b) | (1 << c))

# Precompute supersets for each triple
# supersets[t_index] = list of masks s such that TRIPLES[t_index] is subset of s
# There are 560 triples. Each has 2^(16-3) = 8192 supersets.
SUPERSETS = [None] * len(TRIPLES)
for i, t in enumerate(TRIPLES):
    # Find points NOT in t
    comp = ((1 << N_POINTS) - 1) ^ t
    # Iterate all subsets of comp
    sups = []
    sub = comp
    while True:
        sups.append(t | sub)
        if sub == 0:
            break
        sub = (sub - 1) & comp
    SUPERSETS[i] = sups

# Map mask to index for faster lookup? 
# We store triples as masks. 
# For validity check, we just use masks.

def construct():
    deadline = time.time() + 110.0
    
    # Helper: Check if a list of triple masks is a valid linear system
    # We assume the list is already unique.
    # Linearity: any two share at most 1 point.
    def is_linear(triple_masks):
        n = len(triple_masks)
        for i in range(n):
            t1 = triple_masks[i]
            for j in range(i + 1, n):
                t2 = triple_masks[j]
                if (t1 & t2).bit_count() >= 2:
                    return False
        return True

    # Helper: Check if adding a new triple mask is valid
    def can_add(triple_masks, new_t):
        for t in triple_masks:
            if (t & new_t).bit_count() >= 2:
                return False
        return True

    # Helper: Calculate independence number
    def calc_alpha(triple_masks):
        valid = [True] * N_MASKS
        for t in triple_masks:
            # Find index of t in TRIPLES? 
            # It's better to precompute a dict mapping mask to index
            # Or just iterate supersets directly if we had them per mask.
            # We have SUPERSETS indexed by TRIPLES index.
            # So we need mask_to_index.
            idx = MASK_TO_INDEX[t]
            for s in SUPERSETS[idx]:
                valid[s] = False
        
        # Find max popcount of valid mask
        # Optimization: Check from high popcount down
        # We can pre-sort masks by popcount
        for s in SORTED_MASKS:
            if valid[s]:
                return POP_COUNT[s]
        return 0

    # Precompute MASK_TO_INDEX and SORTED_MASKS
    MASK_TO_INDEX = {t: i for i, t in enumerate(TRIPLES)}
    SORTED_MASKS = sorted(ALL_MASKS, key=lambda x: POP_COUNT[x], reverse=True)
    
    # Re-bind locals for speed in inner loops
    _calc_alpha = calc_alpha
    _can_add = can_add
    _deadline = deadline
    _random = random.Random(42) # Fixed seed for reproducibility? Or just random.
    
    # Initialize with a greedy maximal system
    triples = []
    available = TRIPLES[:]
    _random.shuffle(available)
    for t in available:
        if can_add(triples, t):
            triples.append(t)
            
    current_alpha = _calc_alpha(triples)
    best_alpha = current_alpha
    best_triples = triples[:]
    
    # Local Search
    # We will try to add, remove, or swap triples.
    # To improve alpha, we generally want more triples, but arranged well.
    
    while time.time() < _deadline:
        # Pick a move
        move_type = _random.random()
        new_triples = triples[:]
        
        if move_type < 0.4:
            # Add a triple
            # Pick random candidate from all triples not in current set?
            # Or just pick random from all 560 and check if it's already there?
            # Better: Pick from all, check if valid to add.
            t = _random.choice(TRIPLES)
            if t not in triples: # Simple check, O(m)
                if can_add(triples, t):
                    new_triples.append(t)
                else:
                    continue # Invalid move, skip
            else:
                continue # Already there
                
        elif move_type < 0.7:
            # Remove a triple
            if not triples:
                continue
            idx = _random.randrange(len(triples))
            new_triples.pop(idx)
            
        else:
            # Swap: Remove one, Add one
            if not triples:
                continue
            idx = _random.randrange(len(triples))
            removed_t = new_triples.pop(idx)
            # Try to add a different triple
            # Pick random candidate
            t = _random.choice(TRIPLES)
            if t == removed_t:
                continue
            if t in new_triples:
                continue
            if can_add(new_triples, t):
                new_triples.append(t)
            else:
                # Revert removal if addition fails? 
                # Or just skip this move.
                continue

        # Evaluate new state
        # Quick check: is it linear? 
        # Our moves preserve linearity if we checked can_add.
        # Remove preserves linearity.
        # Swap: we removed one, so remaining are linear. We added one checked against remaining. So linear.
        
        new_alpha = _calc_alpha(new_triples)
        
        # Acceptance
        if new_alpha < current_alpha:
            triples = new_triples
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = triples[:]
        elif new_alpha == current_alpha:
            # Accept with some probability to explore
            if _random.random() < 0.5:
                triples = new_triples
                current_alpha = new_alpha
        else:
            # Reject (Hill Climbing)
            # Could use Simulated Annealing, but Hill Climbing is safer for time.
            pass
            
    # Convert best_triples (list of masks) to list of lists of points
    result = []
    for mask in best_triples:
        pts = []
        for i in range(N_POINTS):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
        
    return result

# Precomputations need to be done before construct is called or inside.
# To avoid overhead on first call, do it at module level.
MASK_TO_INDEX = {t: i for i, t in enumerate(TRIPLES)}
SORTED_MASKS = sorted(ALL_MASKS, key=lambda x: POP_COUNT[x], reverse=True)

# Fix the construct function to use module-level precomputations
def construct():
    deadline = time.time() + 110.0
    
    def calc_alpha(triple_masks):
        valid = [True] * N_MASKS
        for t in triple_masks:
            idx = MASK_TO_INDEX[t]
            for s in SUPERSETS[idx]:
                valid[s] = False
        
        for s in SORTED_MASKS:
            if valid[s]:
                return POP_COUNT[s]
        return 0

    def can_add(triple_masks, new_t):
        for t in triple_masks:
            if (t & new_t).bit_count() >= 2:
                return False
        return True

    _random = random.Random()
    
    # Initialize with a greedy maximal system
    triples = []
    available = TRIPLES[:]
    _random.shuffle(available)
    for t in available:
        if can_add(triples, t):
            triples.append(t)
            
    current_alpha = calc_alpha(triples)
    best_alpha = current_alpha
    best_triples = triples[:]
    
    while time.time() < deadline:
        move_type = _random.random()
        new_triples = triples[:]
        
        if move_type < 0.5:
            # Add
            t = _random.choice(TRIPLES)
            if t not in triples:
                if can_add(triples, t):
                    new_triples.append(t)
                else:
                    continue
            else:
                continue
                
        elif move_type < 0.8:
            # Remove
            if not triples:
                continue
            idx = _random.randrange(len(triples))
            new_triples.pop(idx)
            
        else:
            # Swap
            if not triples:
                continue
            idx = _random.randrange(len(triples))
            removed_t = new_triples.pop(idx)
            t = _random.choice(TRIPLES)
            if t == removed_t or t in new_triples:
                continue
            if can_add(new_triples, t):
                new_triples.append(t)
            else:
                continue

        new_alpha = calc_alpha(new_triples)
        
        if new_alpha < current_alpha:
            triples = new_triples
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = triples[:]
        elif new_alpha == current_alpha:
            if _random.random() < 0.3:
                triples = new_triples
                current_alpha = new_alpha

    result = []
    for mask in best_triples:
        pts = []
        for i in range(N_POINTS):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
        
    return result
