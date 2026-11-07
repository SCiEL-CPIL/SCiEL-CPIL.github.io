import random
import time

def construct():
    n = 128
    # 1. Precompute bad sets as bitmasks
    # A bad set is defined by distinct a,b,c in 1..n such that a+b, a+c, b+c are also in 1..n.
    # The set of values is {a,b,c,a+b,a+c,b+c}.
    # We represent the set of values as a bitmask of length n.
    
    bad_sets_by_elem = [[] for _ in range(n)]
    
    # To avoid duplicates, we can just generate all combinations and add to list.
    # Since we use sets of bitmasks later, duplicates don't matter much for validity check,
    # but for efficiency, unique is better.
    # However, N=128 is small.
    
    all_bad_masks = set()
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                s_ab = a + b
                s_ac = a + c
                s_bc = b + c
                if s_bc > n:
                    break
                
                vals = {a, b, c, s_ab, s_ac, s_bc}
                mask = 0
                for v in vals:
                    mask |= (1 << (v - 1))
                all_bad_masks.add(mask)
                
    # Group masks by element
    # For each mask, for each bit set in mask, add mask to bad_sets_by_elem[bit]
    for mask in all_bad_masks:
        # Iterate through bits in mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bad_sets_by_elem[idx].append(mask)
            m ^= lsb
            
    # Helper to check if adding x (0-indexed) to current set S (mask) is valid
    # It is invalid if there exists a bad mask B containing x such that B \ {x} is subset of S
    def can_add(S, x):
        # x is 0-indexed
        bit_x = 1 << x
        # Check all bad masks containing x
        for B in bad_sets_by_elem[x]:
            # We need B \ {x} to be subset of S
            # B \ {x} is B & ~bit_x
            required = B & ~bit_x
            if (S & required) == required:
                return False
        return True

    # Helper to check if S is valid (no bad mask is subset of S)
    # This is used for swaps or removals where we don't have a single added element context
    # Or we can just check if any bad mask is subset of S
    def is_valid(S):
        for B in all_bad_masks:
            if (S & B) == B:
                return False
        return True

    # Initialize S
    S = 0
    # Greedy initialization: try to add elements in random order
    candidates = list(range(n))
    random.shuffle(candidates)
    for x in candidates:
        if can_add(S, x):
            S |= (1 << x)
            
    best_S = S
    best_size = S.bit_count()
    
    # Local Search
    # We will use a simple hill climbing with random restarts or simulated annealing
    # Given 60s limit, we can run many iterations.
    
    start_time = time.time()
    timeout = 50 # seconds
    
    current_S = S
    current_size = S.bit_count()
    
    while time.time() - start_time < timeout:
        # Generate random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add an element
            # Pick random x not in S
            # Efficient way: pick random index, check if in S
            # Or maintain list of not_in_S?
            # N=128, so just pick random 0..127
            x = random.randint(0, n - 1)
            if (current_S >> x) & 1:
                continue # Already in S
            
            if can_add(current_S, x):
                new_S = current_S | (1 << x)
                new_size = current_size + 1
                # Always accept improvement
                current_S = new_S
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_S = current_S
            else:
                # Optionally, try to swap to escape
                pass
                
        elif move_type < 0.8:
            # Try to swap
            # Pick x in S, y not in S
            # To pick x in S:
            # We can pick random bits until we find one in S
            # Or iterate through set bits
            # Since N is small, let's just pick random x and y and check conditions
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            
            if x == y:
                continue
            if ((current_S >> x) & 1) == 0: # x not in S
                continue
            if ((current_S >> y) & 1) == 1: # y in S
                continue
                
            # Try removing x and adding y
            # New S = (S & ~(1<<x)) | (1<<y)
            # Check validity
            # We can check if new_S is valid
            # But is_valid is O(#bad_sets) which is ~1000-10000?
            # Let's check is_valid
            new_S = (current_S & ~(1 << x)) | (1 << y)
            if is_valid(new_S):
                # Size is same
                # Accept with some probability? Or just accept to explore
                # To avoid cycles, we might need to track visited states or use SA
                # For now, accept 50% of the time
                if random.random() < 0.5:
                    current_S = new_S
                    # current_size unchanged
                else:
                    pass
            else:
                pass
                
        else:
            # Try to remove an element
            # This decreases size, so we should only do this if we expect to gain more later
            # Or just explore
            x = random.randint(0, n - 1)
            if ((current_S >> x) & 1) == 0:
                continue
                
            new_S = current_S & ~(1 << x)
            # Removing always keeps validity (if S was valid, subset is valid)
            # But it reduces size.
            # We should only accept this if we are in a "cooling" phase of SA
            # For simple hill climbing, we don't remove.
            # Let's skip removal for now to keep size high.
            pass
            
    # Convert best_S to list of ints
    result = []
    for i in range(n):
        if (best_S >> i) & 1:
            result.append(i + 1)
            
    return result
