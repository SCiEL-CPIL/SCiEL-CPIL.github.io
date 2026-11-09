import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    points = range(n)
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute K222 configurations
    # A K222 is defined by 6 distinct points partitioned into 3 pairs.
    # There are 15 ways to partition 6 points into 3 pairs.
    # For each such partition, there are 8 transversal triples.
    
    # We need for each triple t, the list of masks of the other 7 triples in each K222 config containing t.
    
    # First, enumerate all K222s.
    # A K222 is a set of 8 triples.
    # We can iterate over all 6-point subsets.
    six_point_subsets = list(itertools.combinations(points, 6))
    
    # For each 6-point subset, generate the 15 pairings.
    # Pairings of 6 elements (a,b,c,d,e,f):
    # (ab,cd,ef), (ab,ce,df), (ab,cf,de),
    # (ac,bd,ef), (ac,be,df), (ac,bf,de),
    # (ad,bc,ef), (ad,be,cf), (ad,bf,ce),
    # (ae,bc,df), (ae,bd,cf), (ae,bd,de) -- wait, typo in thought, let's be precise.
    
    # Let's just generate all 15 pairings programmatically or hardcode logic.
    # Actually, we can just generate all 8 transversals for a given pairing.
    
    # Better approach:
    # Iterate over all 6-point sets.
    # For each, iterate over all 15 pairings.
    # For each pairing, compute the 8 transversal triples.
    # For each of these 8 triples, record that they form a K222 config.
    
    # We will store `configs_by_triple`: dict mapping triple -> list of frozensets of 7 other triples.
    # But for speed, we want bitmasks.
    
    # Let's create a list of all K222 configs as frozensets of 8 triple indices.
    k222_configs = []
    
    for six in six_point_subsets:
        p0, p1, p2, p3, p4, p5 = six
        # Generate 15 pairings
        # A pairing is a set of 3 disjoint pairs covering the 6 points.
        # We can generate these by taking the first point p0, pairing it with p1..p5,
        # then pairing the remaining 4 in 3 ways.
        
        # Pair p0 with p1:
        # Remaining: p2,p3,p4,p5. Pairings: (p2,p3)-(p4,p5), (p2,p4)-(p3,p5), (p2,p5)-(p3,p4)
        # Pair p0 with p2:
        # Remaining: p1,p3,p4,p5. Pairings: (p1,p3)-(p4,p5), (p1,p4)-(p3,p5), (p1,p5)-(p3,p4)
        # Pair p0 with p3:
        # Remaining: p1,p2,p4,p5. Pairings: (p1,p2)-(p4,p5), (p1,p4)-(p2,p5), (p1,p5)-(p2,p4)
        # Pair p0 with p4:
        # Remaining: p1,p2,p3,p5. Pairings: (p1,p2)-(p3,p5), (p1,p3)-(p2,p5), (p1,p5)-(p2,p3)
        # Pair p0 with p5:
        # Remaining: p1,p2,p3,p4. Pairings: (p1,p2)-(p3,p4), (p1,p3)-(p2,p4), (p1,p4)-(p2,p3)
        
        pairings = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)]
        ]
        
        for pairs in pairings:
            (a,b),(c,d),(e,f) = pairs
            # 8 transversals
            transversals = [
                (a,c,e), (a,c,f),
                (a,d,e), (a,d,f),
                (b,c,e), (b,c,f),
                (b,d,e), (b,d,f)
            ]
            # Sort each triple to match combination format
            transversals = [tuple(sorted(t)) for t in transversals]
            # Convert to indices
            idx_set = frozenset(triple_to_idx[t] for t in transversals)
            k222_configs.append(idx_set)
    
    # Now, for each triple, build the list of conflict masks.
    # conflict_masks[i] is a list of masks. Each mask is an int with bits set for the other 7 triples in a K222 config containing i.
    conflict_masks = [[] for _ in range(num_triples)]
    
    for config in k222_configs:
        # config is a frozenset of 8 indices
        config_list = list(config)
        for i in range(8):
            t_idx = config_list[i]
            # The other 7 are config_list[j] for j != i
            mask = 0
            for j in range(8):
                if i != j:
                    mask |= (1 << config_list[j])
            conflict_masks[t_idx].append(mask)
    
    # Helper function to check if triple i can be added to current_mask
    def can_add(i, current_mask):
        for m in conflict_masks[i]:
            if (current_mask & m) == m:
                return False
        return True
    
    # Local Search
    def local_search(initial_mask, max_time):
        current_mask = initial_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        
        # List of all triple indices
        all_indices = list(range(num_triples))
        
        while time.time() < max_time:
            # Try to add a random triple
            # To be efficient, we can pick from a list of "candidates" or just random.
            # Random is fine.
            
            # Pick a random triple not in current_mask
            # To avoid O(N) scan, we can just pick random index and check if bit is set.
            # If set, try again (limited retries) or pick from a precomputed list of absent triples.
            # Maintaining a list of absent triples is better.
            
            # Let's maintain a list of absent triples.
            # This update is O(N) per step if we rebuild, but we can update incrementally.
            # However, N=560 is small. Rebuilding a list of 560 items is fast.
            
            absent = [i for i in all_indices if not (current_mask & (1 << i))]
            
            if not absent:
                break # Full
            
            # Try adding
            success = False
            # Try a few random absent triples
            random.shuffle(absent)
            for i in absent[:10]: # Try up to 10
                if can_add(i, current_mask):
                    current_mask |= (1 << i)
                    current_size += 1
                    success = True
                    break
            
            if success:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                continue
            
            # If no add was possible, try a swap
            # Pick a random present triple to remove
            present = [i for i in all_indices if (current_mask & (1 << i))]
            if not present:
                break
            
            # Try to find a swap that maintains or increases size?
            # Actually, just remove one to escape local minimum.
            # Or, try to remove one that is "blocking" an add.
            
            # Strategy: Pick a random absent triple i.
            # It failed to add. Find why.
            # Find a conflict mask m in conflict_masks[i] such that (current_mask & m) == m.
            # This means all 7 triples in m are present.
            # Pick one of these 7, say j, and remove it.
            # Then try to add i.
            
            # Pick random absent i
            i = random.choice(absent)
            
            # Find blocking mask
            blocking_mask = None
            for m in conflict_masks[i]:
                if (current_mask & m) == m:
                    blocking_mask = m
                    break
            
            if blocking_mask:
                # Pick a random triple from the blocking mask to remove
                # Get indices from mask
                # Since mask has 7 bits set, we can extract them
                bits = []
                temp = blocking_mask
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    temp ^= lsb
                
                j = random.choice(bits)
                
                # Remove j
                current_mask &= ~(1 << j)
                current_size -= 1
                
                # Add i
                # It should be valid now because we broke the conflict
                # But check just in case
                if can_add(i, current_mask):
                    current_mask |= (1 << i)
                    current_size += 1
                else:
                    # Should not happen, but revert
                    current_mask |= (1 << j)
                    current_size += 1
            
            # If we did a swap, size is same.
            # If we did nothing, we are stuck.
            
        return best_mask, best_size

    # Initialization Strategies
    def greedy_init(order):
        mask = 0
        size = 0
        for i in order:
            if can_add(i, mask):
                mask |= (1 << i)
                size += 1
        return mask, size

    best_mask = 0
    best_size = 0
    
    # Strategy 1: Random Greedy
    random.seed(42)
    for _ in range(5):
        order = list(range(num_triples))
        random.shuffle(order)
        mask, size = greedy_init(order)
        if size > best_size:
            best_mask = mask
            best_size = size
    
    # Strategy 2: Sorted by conflict count (fewer conflicts first)
    conflict_counts = [len(cm) for cm in conflict_masks]
    order_sorted = sorted(range(num_triples), key=lambda x: conflict_counts[x])
    mask, size = greedy_init(order_sorted)
    if size > best_size:
        best_mask = mask
        best_size = size
    
    # Strategy 3: Reverse sorted
    order_rev = sorted(range(num_triples), key=lambda x: -conflict_counts[x])
    mask, size = greedy_init(order_rev)
    if size > best_size:
        best_mask = mask
        best_size = size

    # Local Search on best
    # Allocate remaining time
    current_time = time.time()
    time_left = deadline - current_time
    
    # Run local search for half the time
    ls_mask, ls_size = local_search(best_mask, current_time + time_left * 0.5)
    if ls_size > best_size:
        best_mask = ls_mask
        best_size = ls_size
    
    # Run local search again on the result for the rest of the time
    current_time = time.time()
    time_left = deadline - current_time
    if time_left > 5:
        ls_mask, ls_size = local_search(best_mask, current_time + time_left * 0.8)
        if ls_size > best_size:
            best_mask = ls_mask
            best_size = ls_size

    # Convert mask back to list of triples
    result = []
    for i in range(num_triples):
        if best_mask & (1 << i):
            result.append(list(all_triples[i]))
    
    return result
