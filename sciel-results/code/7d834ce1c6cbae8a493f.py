import time
import random
import itertools
import sys

def construct():
    """
    Constructs a linear triple system on 16 points with a small independence number.
    
    Strategy:
    1. Generate a random valid linear triple system (packing).
    2. Use local search (simulated annealing-like) to minimize the exact independence number.
    3. Moves are 1-swaps (remove one triple, add one valid triple).
    4. Exact independence number is calculated for each candidate system.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    n_points = 16
    points = list(range(n_points))
    
    # Pre-compute all possible triples and their pairs
    all_triples = list(itertools.combinations(points, 3))
    
    # Map pair to index for fast access? Or just use set of tuples.
    # For 16 points, there are 16*15/2 = 120 pairs.
    # We can map pair (a,b) to a single integer id for faster counting.
    pair_id_map = {}
    pair_list = []
    idx = 0
    for a in range(n_points):
        for b in range(a+1, n_points):
            pair_id_map[(a,b)] = idx
            pair_list.append((a,b))
            idx += 1
            
    # Pre-compute pairs for each triple
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = [pair_id_map[(a,b)], pair_id_map[(a,c)], pair_id_map[(b,c)]]
        
    # Function to calculate exact independence number
    def get_exact_alpha(triples):
        # triples is a set of tuples
        if not triples:
            return n_points
            
        # Convert triples to bitmasks for fast subset check
        triple_masks = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mask)
            
        # We want the largest subset of points that contains NO triple.
        # Iterate k from n_points down to 1
        # Check if there is an independent set of size k.
        
        # Optimization: If we have many triples, alpha is small.
        # We can use a backtracking search for Max Independent Set which is often faster.
        # But for n=16, even brute force over subsets of size k is fast.
        
        # Let's use a recursive backtracking with pruning.
        # State: current_mask, candidates_mask
        # candidates_mask contains points that can still be added.
        # If we add point p, we remove p and any point q such that {p,q} is in a triple?
        # No, {p,q,r} is a triple.
        # If we pick p, we cannot pick any pair {q,r} such that {p,q,r} is a triple.
        # This is complex to prune simply.
        
        # Let's stick to checking subsets of size k.
        # Precompute all masks of size k? No, generate on the fly.
        
        # Actually, a very fast way for n=16:
        # Iterate through all 2^16 masks? 65536 masks.
        # For each mask, check if it's independent.
        # Check: for each triple_mask, if (mask & triple_mask) == triple_mask, then not independent.
        # This is 65536 * len(triples) operations.
        # len(triples) ~ 40.
        # 65536 * 40 = 2.6 million operations.
        # In Python, this might take ~0.1s - 0.2s.
        # If we do 1000 local search steps, that's 100-200s. Too slow.
        
        # We need a faster alpha calculation.
        # Backtracking for Max Independent Set in Hypergraph:
        
        def max_independent_set_size(triple_masks, n):
            # Use a simple branch and bound
            # Order points by degree (number of triples they are in) descending?
            # Points in many triples are "bad" for independent set, so try to exclude them?
            # Or include them?
            # If we include a point, we lose more options.
            # So we should try to EXCLUDE points with high degree first?
            # No, we want to MAXIMIZE the set.
            # So we should try to INCLUDE points that have low "cost".
            
            # Let's use a standard MIS algorithm on the "conflict graph"?
            # The conflict graph has an edge (u,v) if u and v are in a triple?
            # No. {u,v,w} is a triple.
            # An independent set in the hypergraph is a set with no hyperedge.
            # This is NOT the same as independent set in the 2-section graph.
            # In the 2-section graph, {u,v} is an edge if they are in a triple.
            # An independent set in the hypergraph CAN contain {u,v} if w is not in the set.
            # So the 2-section graph is not directly useful for MIS in the hypergraph sense.
            
            # Backtracking:
            # Sort points.
            # At each step, pick a point p.
            # Branch 1: Include p.
            #    - Remove p from candidates.
            #    - For every triple containing p, remove the other 2 points from candidates?
            #      No, if we include p, we cannot include ANY pair {q,r} such that {p,q,r} is a triple.
            #      This means q and r cannot BOTH be in the set.
            #      This is a 2-SAT like constraint?
            
            # Given n=16, let's just use the subset enumeration but optimize the check.
            # We can precompute for each point the list of triples it's in.
            # When checking a mask, we can be smarter?
            
            # Alternative: Iterate k from 16 down.
            # For a fixed k, generate all combinations of size k.
            # Check if any is independent.
            # Stop at first k where we find one.
            
            # To speed up the check:
            # For a candidate set S (as a set of points),
            # We need to ensure no triple is subset of S.
            
            # Let's try the 2^16 iteration but with a tighter loop.
            best = 0
            # Iterate masks in descending order of popcount?
            # Or just iterate all masks, track max popcount of independent masks.
            
            # 65536 iterations.
            # Inner loop: check against ~40 triple masks.
            # We can break early if a triple is found.
            
            # Let's try this. If it's too slow, we'll optimize.
            # We can also cache results? No, triples change.
            
            # Optimization:
            # If a mask is independent, any subset of it is independent.
            # So we can use a DP or just iterate.
            
            # Let's use the recursive backtracking with a strong pruning heuristic:
            # 1. Sort points by degree (number of triples containing them).
            # 2. Try to include the point with LOWEST degree first?
            #    If we include a low-degree point, we block few other points.
            #    This should lead to larger independent sets.
            
            point_triples = [[] for _ in range(n)]
            for t in triple_masks:
                for p in range(n):
                    if t & (1 << p):
                        point_triples[p].append(t)
                        
            # Sort points by degree
            sorted_points = sorted(range(n), key=lambda x: len(point_triples[x]))
            
            best_size = 0
            current_mask = 0
            current_size = 0
            candidates_mask = (1 << n) - 1
            
            # Recursive search
            # We can use a simple DFS
            def dfs(idx, current_mask, current_size, candidates_mask):
                nonlocal best_size
                if current_size + bin(candidates_mask).count('1') <= best_size:
                    return
                    
                if idx == n:
                    if current_size > best_size:
                        best_size = current_size
                    return
                    
                p = sorted_points[idx]
                p_bit = 1 << p
                
                if not (candidates_mask & p_bit):
                    # p is not in candidates, skip
                    dfs(idx + 1, current_mask, current_size, candidates_mask)
                else:
                    # Option 1: Exclude p
                    dfs(idx + 1, current_mask, current_size, candidates_mask)
                    
                    # Option 2: Include p
                    # If we include p, we must ensure no triple containing p is fully in the final set.
                    # This means for any triple {p, q, r}, we cannot have both q and r in the set.
                    # This is a constraint on the remaining choices.
                    # It's not simply removing q and r from candidates.
                    # It's that q and r become "incompatible" with each other.
                    # This makes the state space larger (we need to track incompatibilities).
                    
                    # Given the complexity, and n=16, let's stick to the 2^16 iteration for alpha,
                    # but optimize the Python loop.
                    pass
            
            # Let's revert to 2^16 iteration. It's simple and predictable.
            # We can speed it up by using integer operations.
            
            # Precompute triple_masks as a list of ints.
            # Iterate mask from 0 to 2^n - 1?
            # We want max popcount.
            # We can iterate k from n down to 1 and check if ANY mask of size k is independent.
            
            # To check if ANY mask of size k is independent:
            # Generate all masks of size k?
            # itertools.combinations(range(n), k) -> convert to mask.
            
            # For k=16: 1 mask.
            # For k=15: 16 masks.
            # ...
            # For k=8: 12870 masks.
            
            # This is much faster than 65536 if alpha is small (which it should be).
            
            for k in range(n, 0, -1):
                found = False
                for combo in itertools.combinations(range(n), k):
                    mask = 0
                    for p in combo:
                        mask |= (1 << p)
                    
                    is_independent = True
                    for tm in triple_masks:
                        if (mask & tm) == tm:
                            is_independent = False
                            break
                            
                    if is_independent:
                        return k
                # If no independent set of size k found, try k-1
            return 0
            
    # Initial Construction: Greedy
    def greedy_construct():
        random.shuffle(all_triples)
        triples = set()
        pair_count = [0] * len(pair_list)
        
        for t in all_triples:
            pairs = triple_pairs[t]
            valid = True
            for p_id in pairs:
                if pair_count[p_id] > 0:
                    valid = False
                    break
            if valid:
                triples.add(t)
                for p_id in pairs:
                    pair_count[p_id] += 1
        return triples, pair_count
    
    # Local Search
    best_triples, best_pair_count = greedy_construct()
    best_alpha = get_exact_alpha(best_triples)
    
    current_triples = best_triples
    current_pair_count = best_pair_count
    current_alpha = best_alpha
    
    # List of all triples for random selection
    all_triples_list = all_triples
    
    # Local search loop
    while time.time() < deadline:
        # Pick a triple to remove
        if not current_triples:
            break
        t_out = random.choice(list(current_triples))
        
        # Pick a triple to add
        # To ensure we don't pick t_out, we can just pick randomly and check
        t_in = random.choice(all_triples_list)
        
        if t_in == t_out:
            continue
            
        # Check if t_in is valid given t_out is removed
        # Pairs of t_out will be freed.
        # Pairs of t_in will be used.
        # Current pair_count reflects current_triples.
        # After removing t_out, pair_count for t_out's pairs decreases by 1.
        # Then we add t_in, so pair_count for t_in's pairs increases by 1.
        
        p_out = triple_pairs[t_out]
        p_in = triple_pairs[t_in]
        
        valid = True
        # Check p_in
        # For each pair in p_in, the new count will be:
        # current_pair_count[p] - (1 if p in p_out else 0) + 1
        # This must be <= 1.
        
        for p_id in p_in:
            new_count = current_pair_count[p_id]
            if p_id in p_out:
                new_count -= 1
            new_count += 1
            if new_count > 1:
                valid = False
                break
                
        if not valid:
            continue
            
        # Perform swap
        # Update pair_count
        for p_id in p_out:
            current_pair_count[p_id] -= 1
        for p_id in p_in:
            current_pair_count[p_id] += 1
            
        current_triples.remove(t_out)
        current_triples.add(t_in)
        
        new_alpha = get_exact_alpha(current_triples)
        
        # Acceptance
        if new_alpha < current_alpha:
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = set(current_triples)
                best_pair_count = current_pair_count[:]
        elif new_alpha == current_alpha:
            # Accept with some probability to explore
            if random.random() < 0.1:
                pass # Keep
            else:
                # Revert
                for p_id in p_out:
                    current_pair_count[p_id] += 1
                for p_id in p_in:
                    current_pair_count[p_id] -= 1
                current_triples.remove(t_in)
                current_triples.add(t_out)
        else:
            # Revert
            for p_id in p_out:
                current_pair_count[p_id] += 1
            for p_id in p_in:
                current_pair_count[p_id] -= 1
            current_triples.remove(t_in)
            current_triples.add(t_out)
            
    return list(best_triples)
