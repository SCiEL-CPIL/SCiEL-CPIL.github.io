import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples as tuples of sorted points
    all_triples = list(combinations(points, 3))
    triple_to_mask = {}
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_to_mask[t] = mask
        
    # Helper to calculate alpha exactly
    def calc_alpha(hypergraph):
        if not hypergraph:
            return n
        
        # Convert to bitmasks for speed
        triple_masks = []
        for t in hypergraph:
            triple_masks.append(triple_to_mask[t])
        
        # Check from largest k down
        for k in range(n, 0, -1):
            found = False
            # Use combinations for subsets of size k
            for subset in combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is contained in mask
                is_independent = True
                for tm in triple_masks:
                    if (mask & tm) == tm:
                        is_independent = False
                        break
                if is_independent:
                    found = True
                    break
            if found:
                return k
        return 1

    # Greedy construction to get a good starting point
    def greedy_construct():
        used_pairs = set()
        hypergraph = []
        
        # Create a shuffled list of all triples
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        for t in candidates:
            p0, p1, p2 = t
            pairs = [(p0, p1), (p0, p2), (p1, p2)]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            
            if valid:
                hypergraph.append(t)
                used_pairs.update(pairs)
                
        return hypergraph, used_pairs

    # Local search to maximize number of triples
    def local_search_max_m(hypergraph, used_pairs, time_limit=5.0):
        h = [list(t) for t in hypergraph]
        up = set(used_pairs)
        
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        start_time = time.time()
        current_m = len(h)
        best_m = current_m
        best_h = [t[:] for t in h]
        best_up = set(up)
        
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < start_time + time_limit:
            if no_improve_count > max_no_improve:
                break
                
            if not h:
                break
                
            # Pick a random triple to remove
            idx = random.randint(0, len(h) - 1)
            t_out = h[idx]
            h.pop(idx)
            p0, p1, p2 = t_out
            up.discard((p0, p1))
            up.discard((p0, p2))
            up.discard((p1, p2))
            
            # Try to add a random triple
            added = False
            # Try a few random candidates
            for _ in range(10):
                if not candidates: break
                t_in = random.choice(candidates)
                p_a, p_b, p_c = t_in
                pairs = [(p_a, p_b), (p_a, p_c), (p_b, p_c)]
                if all(p not in up for p in pairs):
                    # Add it
                    h.append(t_in)
                    up.update(pairs)
                    new_m = len(h)
                    
                    if new_m > best_m:
                        best_m = new_m
                        best_h = [t[:] for t in h]
                        best_up = set(up)
                        no_improve_count = 0
                    else:
                        # Revert
                        h.pop()
                        for p in pairs:
                            up.discard(p)
                    added = True
                    break
            
            if not added:
                # Revert removal
                h.append(t_out)
                up.update([(p0, p1), (p0, p2), (p1, p2)])
                no_improve_count += 1
            else:
                no_improve_count += 1 # Increment even if added but not improved, to allow shaking?
                # Actually, if we added but m didn't increase, we reverted.
                # If we added and m increased, we kept it.
                # So if we kept it, m is current_m + 1? No, we removed one and added one. m is same.
                # Wait, if we remove one and add one, m stays same.
                # To increase m, we need to add a triple WITHOUT removing one?
                # But that's not a swap. That's just adding.
                # If we can add a triple without removing, m increases.
                
                # Let's modify the move:
                # 1. Try to add a random triple. If valid, add it. (m increases)
                # 2. If cannot add, try a swap: remove one, add one. (m same)
                
                # Let's restart the logic for local_search_max_m
                
        return best_h, best_up, best_m

    # Revised Local Search for Max M
    def local_search_max_m_v2(hypergraph, used_pairs, time_limit=5.0):
        h = [list(t) for t in hypergraph]
        up = set(used_pairs)
        
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        start_time = time.time()
        best_m = len(h)
        best_h = [t[:] for t in h]
        best_up = set(up)
        
        stall = 0
        max_stall = 1000
        
        while time.time() < start_time + time_limit:
            if stall > max_stall:
                break
            
            moved = False
            
            # 1. Try to add a new triple (increases m)
            # Pick a random candidate
            if candidates:
                t_in = random.choice(candidates)
                p_a, p_b, p_c = t_in
                pairs = [(p_a, p_b), (p_a, p_c), (p_b, p_c)]
                if all(p not in up for p in pairs):
                    h.append(t_in)
                    up.update(pairs)
                    moved = True
                    if len(h) > best_m:
                        best_m = len(h)
                        best_h = [t[:] for t in h]
                        best_up = set(up)
                        stall = 0
                    else:
                        stall += 1
                else:
                    # Cannot add, try swap
                    if h:
                        idx = random.randint(0, len(h) - 1)
                        t_out = h[idx]
                        h.pop(idx)
                        p0, p1, p2 = t_out
                        up.discard((p0, p1))
                        up.discard((p0, p2))
                        up.discard((p1, p2))
                        
                        # Try to add a different triple
                        for _ in range(5):
                            if not candidates: break
                            t_try = random.choice(candidates)
                            p_a, p_b, p_c = t_try
                            pairs = [(p_a, p_b), (p_a, p_c), (p_b, p_c)]
                            if all(p not in up for p in pairs):
                                h.append(t_try)
                                up.update(pairs)
                                moved = True
                                # m is same as before swap
                                # Check if this swap leads to better potential?
                                # For now, just accept swap to explore.
                                break
                        if not moved:
                            # Revert swap
                            h.append(t_out)
                            up.update([(p0, p1), (p0, p2), (p1, p2)])
                            stall += 1
            
            if not moved:
                stall += 1
            else:
                if len(h) > best_m:
                    stall = 0
                else:
                    stall += 1

        return best_h, best_up, best_m

    # Local search to minimize Alpha (keeping m constant)
    def local_search_min_alpha(hypergraph, used_pairs, time_limit=5.0):
        h = [list(t) for t in hypergraph]
        up = set(used_pairs)
        
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        start_time = time.time()
        current_alpha = calc_alpha(h)
        best_alpha = current_alpha
        best_h = [t[:] for t in h]
        best_up = set(up)
        
        stall = 0
        max_stall = 500
        
        while time.time() < start_time + time_limit:
            if stall > max_stall:
                break
            
            if not h:
                break
            
            # Perform a swap
            idx = random.randint(0, len(h) - 1)
            t_out = h[idx]
            h.pop(idx)
            p0, p1, p2 = t_out
            up.discard((p0, p1))
            up.discard((p0, p2))
            up.discard((p1, p2))
            
            added = False
            for _ in range(10):
                if not candidates: break
                t_in = random.choice(candidates)
                p_a, p_b, p_c = t_in
                pairs = [(p_a, p_b), (p_a, p_c), (p_b, p_c)]
                if all(p not in up for p in pairs):
                    h.append(t_in)
                    up.update(pairs)
                    # Check alpha
                    # To save time, only check every few steps or if swap is "good"?
                    # We don't know if it's good without checking.
                    # Let's check alpha.
                    new_alpha = calc_alpha(h)
                    
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_h = [t[:] for t in h]
                        best_up = set(up)
                        stall = 0
                    else:
                        # Revert
                        h.pop()
                        for p in pairs:
                            up.discard(p)
                    added = True
                    break
            
            if not added:
                h.append(t_out)
                up.update([(p0, p1), (p0, p2), (p1, p2)])
                stall += 1
            else:
                stall += 1 # Even if we swapped, if alpha didn't improve, we stalled?
                # If we improved, stall=0. If not, stall++.
                # This might be too strict. Let's allow some random walks.
                # If we don't improve for max_stall, break.

        return best_h, best_up, best_alpha

    # Main loop
    best_alpha = 16
    best_hypergraph = []
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        # Greedy start
        h, up = greedy_construct()
        m = len(h)
        
        # Phase 1: Maximize m
        if time.time() < deadline - 10.0:
            h, up, m = local_search_max_m_v2(h, up, time_limit=2.0)
            
        # Phase 2: Minimize alpha
        if time.time() < deadline - 5.0:
            h, up, a = local_search_min_alpha(h, up, time_limit=3.0)
            
            if a < best_alpha:
                best_alpha = a
                best_hypergraph = h
        
        restarts += 1
        
    return [list(t) for t in best_hypergraph]

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
