import time
import random

def _build_masks(n):
    masks = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i == j:
                continue
            if (i * j) % (i + j) != 0:
                mask |= (1 << j)
        masks[i] = mask
    return masks

def _popcount(x):
    return bin(x).count("1")

def _get_bits(x):
    while x:
        # Get lowest set bit
        lsb = x & -x
        yield lsb.bit_length() - 1
        x ^= lsb

def _search(n, masks, deadline, seed):
    rng = random.Random(seed)
    nodes = list(range(1, n + 1))
    rng.shuffle(nodes)
    
    # Greedy construction with random tie-breaking
    S_mask = 0
    S_list = []
    available = (1 << (n + 1)) - 2 # bits 1..n set
    
    # We will greedily add nodes. To avoid always picking the "best" one, we can shuffle the iteration order of candidates.
    # A simple greedy: pick node with max compatibility in current context?
    # Or just random shuffle and pick first compatible?
    # Let's try: iterate through shuffled nodes, if compatible with current S, add it.
    
    # Optimization: Instead of adding one by one which might get stuck, 
    # let's do a standard local search from this greedy start.
    
    # Refine Greedy: 
    # Sort nodes by degree in conflict graph? 
    # Actually, let's just use the shuffled order and pick if compatible.
    
    for node in nodes:
        if available & (1 << node):
            # Check if node is compatible with all in S
            # S is small initially. 
            # Efficient check: (intersection of masks of all in S) has bit node?
            # Let's maintain `compatible_with_S` mask.
            # But we don't have it in this simple loop.
            # Let's just check against S_list which is small.
            ok = True
            for s in S_list:
                if not (masks[node] & (1 << s)):
                    ok = False
                    break
            if ok:
                S_mask |= (1 << node)
                S_list.append(node)
                available ^= (1 << node)
                
    best_S = S_mask
    best_size = _popcount(S_mask)
    
    # Local Search
    # We can do 1-out 2-in moves.
    # State: S_mask, available
    # Try to improve size.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1-out, 2-in move
        # Remove r from S.
        # Find a, b in available such that:
        # 1. a != b
        # 2. a, b compatible with S \ {r}
        # 3. a compatible with b
        
        # To speed up, we can precompute for each r, the set of candidates in available compatible with S\{r}.
        # But S changes.
        
        # Heuristic: 
        # For each r in S:
        #   new_available = available | (1 << r)
        #   # Remove r from S
        #   S_temp = S_mask ^ (1 << r)
        #   # Find all in new_available compatible with S_temp
        #   # This requires intersecting masks of all in S_temp.
        #   # If |S| is large, this is expensive.
        
        # Alternative:
        # Just try swapping one element out for two specific ones.
        # This is O(|S| * |A|^2). 256 * 256^2 is too slow in Python per iteration.
        
        # Better:
        # Randomized local search.
        # Pick a random r in S.
        # Pick two random candidates from available (or top candidates).
        # Check validity.
        
        # Let's try a smarter heuristic for 1-2 move:
        # 1. Identify "bad" elements in S (those with many conflicts with available).
        # 2. Identify "good" elements in available (those with many compatibilities).
        
        # Given time constraints, a robust approach is:
        # 1. Greedy Start (multiple variants)
        # 2. Hill Climbing with 1-swap (size neutral) and 2-swap (size +1)
        
        # To make 2-swap faster:
        # For a fixed r removed:
        # Let R = S \ {r}.
        # Let C = {x in available | x compatible with all y in R}.
        # We need to find a pair (a, b) in C such that a compatible with b.
        # C is computed by:
        #   mask_C = (1 << (n+1)) - 2
        #   for y in R:
        #       mask_C &= masks[y]
        #   mask_C &= available
        #   # Now mask_C contains all candidates compatible with R and in available.
        #   # We need to check if there is an edge in the compatibility graph between any two in mask_C.
        #   # i.e., do we exist a, b in mask_C such that masks[a] & (1 << b) != 0?
        #   # This is checking if the subgraph induced by mask_C on the compatibility graph has an edge.
        #   # Since we want MAX INDEPENDENT SET in conflict graph, this is MAX CLIQUE in compatibility graph.
        #   # Wait, if a and b are compatible, they form an edge in compat graph.
        #   # We just need ANY pair.
        
        # So, for each r:
        #   Compute mask_C.
        #   If popcount(mask_C) >= 2:
        #      Check if any two in mask_C are compatible.
        #      This can be done by iterating a in mask_C:
        #         if masks[a] & mask_C: return True (found b)
        
        # Cost per r: O(|S| * N) for mask_C computation + O(|C|^2) worst case for pair check.
        # With N=256, this is doable if we optimize inner loops.
        
        S_list_temp = [i for i in _get_bits(S_mask)]
        for r in S_list_temp:
            if time.time() > deadline: return best_S, best_size
            
            # Compute mask of elements in available compatible with S \ {r}
            mask_cand = available
            # Iterate S_mask without r
            temp_s = S_mask ^ (1 << r)
            t = temp_s
            while t:
                lsb = t & -t
                idx = lsb.bit_length() - 1
                mask_cand &= masks[idx]
                t ^= lsb
            
            # Now mask_cand has bits for elements in available compatible with S\{r}
            if _popcount(mask_cand) < 2:
                continue
                
            # Check if there exists a, b in mask_cand such that a and b are compatible.
            # Iterate a in mask_cand
            t = mask_cand
            found_pair = False
            while t:
                lsb_a = t & -t
                a = lsb_a.bit_length() - 1
                # Check if any b in mask_cand (excluding a) is compatible with a
                # masks[a] & mask_cand will contain a itself? No, masks[a] has 0 at a.
                # So masks[a] & mask_cand will contain b if b is compatible with a and b is in mask_cand.
                if masks[a] & mask_cand:
                    found_pair = True
                    break
                t ^= lsb_a
                
            if found_pair:
                # We found a move. Apply it.
                # We need to pick specific a and b.
                # Let's find them.
                t = mask_cand
                a_val = -1
                b_val = -1
                while t:
                    lsb_a = t & -t
                    a = lsb_a.bit_length() - 1
                    if masks[a] & mask_cand:
                        a_val = a
                        # Find b
                        m_b = masks[a] & mask_cand
                        lsb_b = m_b & -m_b
                        b_val = lsb_b.bit_length() - 1
                        break
                    t ^= lsb_a
                
                if a_val != -1:
                    # Apply move: remove r, add a, b
                    S_mask ^= (1 << r)
                    S_mask |= (1 << a_val)
                    S_mask |= (1 << b_val)
                    available ^= (1 << a_val)
                    available ^= (1 << b_val)
                    available |= (1 << r)
                    
                    best_size += 1
                    if best_size > _popcount(best_S):
                        best_S = S_mask
                    improved = True
                    break # Restart loop as S changed
                    
    return best_S, best_size

def construct():
    n = 256
    # Precompute masks
    masks = _build_masks(n)
    
    start_time = time.time()
    deadline = start_time + 50.0 # Leave 10s margin
    
    best_S = 0
    best_size = 0
    
    # Run multiple restarts
    # The quality of the greedy start matters.
    # We can vary the greedy strategy.
    
    # Strategy 1: Random shuffle greedy
    # Strategy 2: Degree-based greedy (pick node with lowest degree in conflict graph first? or highest?)
    # Actually, for Max Independent Set, picking nodes with high degree in conflict graph (few compatible) first might prune the space, but for constructive, we want to place hard-to-place nodes first.
    
    # Let's just do many random restarts.
    # Each restart takes ~5-10ms? 50s / 10ms = 5000 restarts.
    
    # To make restarts more effective, we can introduce randomness into the local search too.
    # E.g., random choice of which r to try in 1-2 move, or random tie-breaking in greedy.
    
    for i in range(10000):
        if time.time() > deadline:
            break
        S, size = _search(n, masks, deadline, seed=i)
        if size > best_size:
            best_size = size
            best_S = S
            
    # Convert best_S to list
    result = list(_get_bits(best_S))
    
    # Final safety check
    for a in result:
        for b in result:
            if a == b: continue
            if (a * b) % (a + b) == 0:
                return list(range(1, 257, 2)) # Fallback to odds
        
    return result
