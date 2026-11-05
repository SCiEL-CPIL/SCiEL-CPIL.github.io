import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n = len(all_triples)
    
    # Precompute pair sets for each triple
    triple_pairs = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)
        
    # Precompute for each triple, which other triples conflict with it (share a pair)
    # Conflict: share >= 2 points.
    # We can precompute a list of conflicting triples for each triple.
    # Or better, for each pair, list of triples containing it.
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for p in triple_pairs[i]:
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(i)
            
    # Build a fast conflict check
    # If we have a set of used pairs, a new triple is valid if none of its pairs are in used_pairs.
    
    def get_alpha(triple_list):
        # Returns the size of the largest independent set
        # We check from 16 down to 1
        # Optimization: Use bitmask for triples
        # If a subset of points is independent, it contains no triple.
        # This is equivalent to: The set of points is NOT covered by any triple in the list.
        # Wait, "independent set" in the context of hypergraph usually means a set of vertices containing no edge.
        # Yes, "largest point set containing no triple".
        
        # We can use a recursive backtracking or just check all subsets if n is small.
        # n=16, 2^16 = 65536 subsets.
        # For each subset, check if any triple is contained in it.
        # Precompute triple masks.
        masks = []
        for t in triple_list:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            masks.append(mask)
            
        # Sort masks by popcount? No, they are all 3.
        # To speed up, we can iterate subsets in decreasing order of size.
        # Or use a DP / BFS?
        # Simple loop is probably fast enough for 65536 * len(triples).
        # len(triples) is at most 16*15/2 / 3 * ... roughly 56 triples max?
        # 65536 * 56 ~ 3.6M ops. Very fast in Python if optimized? Maybe 0.1s.
        # We call this many times.
        
        # Better approach:
        # An independent set is a set of vertices that does NOT contain any triple.
        # Let S be a subset. S is independent iff for all t in triple_list, t is not a subset of S.
        
        # We can compute the "forbidden" sets.
        # Actually, let's just use the bitmask check.
        
        # To make it faster, we can precompute the "cover" of a point? No.
        
        # Let's try to find the maximum k such that there exists a subset of size k with no triple.
        # We can iterate k from 16 down to 1.
        # For a fixed k, we iterate all subsets of size k.
        # Number of subsets C(16, k).
        # Max C(16, 8) = 12870.
        # Total subsets = 65536.
        
        # We can precompute a list of all subsets? No, too big.
        
        # We'll stick to the loop but optimize the inner check.
        # Instead of checking all triples for a subset, we can check if the subset mask AND triple mask == triple mask.
        
        # Even better:
        # If we know the triples, we can compute the "bad" sets?
        
        # Let's just implement the loop efficiently.
        
        # Pre-compute masks for the current triple_list
        # We do this outside the k-loop
        
        # To speed up further, we can use the fact that if a subset of size k is independent, 
        # any subset of it is also independent.
        # So we can do a DFS/BFS on the subset lattice?
        
        # For now, the simple check is:
        # for k in range(16, 0, -1):
        #    for subset in combinations(points, k):
        #       mask = ...
        #       if not any(mask & tm == tm for tm in masks):
        #           return k
        # return 0
        
        # Let's optimize the mask generation.
        
        # We can precompute the masks for all 2^16 subsets? 
        # 65536 integers. 
        # Then for each triple mask, mark all supersets as "bad"?
        # Supersets of a triple: 2^(16-3) = 8192 per triple.
        # 56 triples * 8192 ~ 460k operations.
        # Then we just scan the "bad" array to find the largest subset that is NOT bad.
        # This is much faster!
        
        # bad_mask = [False] * (1 << 16)
        # for tm in masks:
        #    # Mark all supersets of tm as bad
        #    # Supersets are of the form (tm | rest) where rest is subset of (~tm) & ((1<<16)-1)
        #    # We can iterate through submasks of the complement?
        #    # Or just:
        #    # for sup in range(1<<16):
        #    #    if (sup & tm) == tm: bad_mask[sup] = True
        #    # This is 65536 * 56 ~ 3.6M ops.
        #    # This is acceptable if done once per evaluation of alpha.
        #    # But we evaluate alpha many times.
        
        # Alternative:
        # Maintain a count array?
        # Let's try the "Mark Supersets" approach.
        
        size = 1 << 16
        # We can reuse a global array? No, it changes.
        # Creating a list of 65536 False is fast.
        is_bad = [False] * size
        
        # Optimization: Only mark if the triple is actually present.
        for tm in masks:
            # Iterate all supersets of tm
            # A superset S satisfies S | tm == S  =>  (S & tm) == tm
            # We can iterate S from 0 to 2^16-1 and check?
            # Or generate supersets.
            # The number of supersets is 2^(16-3) = 8192.
            # How to iterate supersets of tm?
            # Let free = (~tm) & (size - 1)
            # sub = free
            # while True:
            #    sup = tm | sub
            #    is_bad[sup] = True
            #    if sub == 0: break
            #    sub = (sub - 1) & free
            
            free = (~tm) & (size - 1)
            sub = free
            while True:
                sup = tm | sub
                is_bad[sup] = True
                if sub == 0:
                    break
                sub = (sub - 1) & free
                
        # Now find the largest subset that is NOT bad.
        # We can iterate by size k from 16 down to 1.
        # Or just iterate all masks from 65535 down to 0 and check popcount?
        # We want the maximum popcount among non-bad masks.
        
        max_alpha = 0
        # Iterate all masks
        for m in range(size):
            if not is_bad[m]:
                # Count bits
                # Python 3.8+ has int.bit_count()
                bc = m.bit_count()
                if bc > max_alpha:
                    max_alpha = bc
                    if max_alpha == 16:
                        return 16
        return max_alpha

    def has_independent_set_size(triple_list, k):
        # This is now redundant if we use get_alpha, but might be used for early exit.
        # We can just use get_alpha.
        return get_alpha(triple_list) >= k

    # MILP part: Keep it but limit time?
    # The MILP might be slow. I will wrap it in a try-except and a time check.
    
    best_alpha = 16
    best_triples = []
    
    # Helper to validate and update best
    def update_best(triples):
        nonlocal best_alpha, best_triples
        if not triples:
            return
        # Check validity (linearity)
        used_pairs = set()
        for t in triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            for p in pairs:
                if p in used_pairs:
                    return # Invalid
                used_pairs.add(p)
                
        alpha = get_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples

    # 1. Try MILP (short timeout)
    try:
        # Build constraint matrix
        all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
        pair_to_idx = {p: idx for idx, p in enumerate(all_pairs)}
        n_pairs = len(all_pairs)
        
        A = np.zeros((n_pairs, n), dtype=np.int8)
        for i, t in enumerate(all_triples):
            for p in triple_pairs[i]:
                A[pair_to_idx[p], i] = 1
                
        c = -np.ones(n)
        constraints = [LinearConstraint(A, 0, 1)]
        bounds = Bounds(0, 1)
        integrality = np.ones(n)
        
        # Use a time limit? milp doesn't have a direct time limit param in scipy?
        # We can just call it and hope. Or skip it if it's too slow.
        # Given the deadline, I will run it once.
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        if res.status == 0 and res.x is not None:
            selected = [i for i in range(n) if res.x[i] > 0.5]
            current_triples = [list(all_triples[i]) for i in selected]
            update_best(current_triples)
    except Exception:
        pass

    # 2. Local Search
    # We will perform multiple restarts.
    # Each restart:
    # a. Greedy construction
    # b. Local search (Hill Climbing / Simulated Annealing)
    
    num_restarts = 5000
    for i in range(num_restarts):
        if time.time() > deadline - 2: # Safety margin
            break
            
        # Greedy Construction
        # Randomize the order of triples
        indices = list(range(n))
        random.shuffle(indices)
        
        current_triples = []
        used_pairs = set()
        
        for idx in indices:
            t = all_triples[idx]
            pairs = triple_pairs[idx]
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
            if not conflict:
                current_triples.append(t)
                for p in pairs:
                    used_pairs.add(p)
        
        update_best(current_triples)
        
        # Local Search Phase
        # We want to minimize alpha.
        # Move: Remove one triple, Add one triple.
        # Constraint: The set of triples must remain linear.
        
        # We maintain:
        # - current_triples (list of tuples)
        # - used_pairs (set of pairs)
        # - current_alpha
        
        if not current_triples:
            continue
            
        curr_alpha = get_alpha(current_triples)
        
        # If we already have alpha 4, we can try to find 3.
        # If we have 5, try 4.
        
        local_time_limit = 0.1 # 100ms per restart?
        local_start = time.time()
        
        # Precompute conflict info for faster updates?
        # When we remove a triple T_out:
        # - We free up its 3 pairs.
        # - We can add any triple T_in that does NOT conflict with (current_triples - {T_out}).
        # - T_in conflicts with (current_triples - {T_out}) if it shares a pair with any of them.
        # - T_in can share pairs with T_out? No, T_out is removed.
        # - So T_in is valid if none of its pairs are in (used_pairs - pairs_of_T_out).
        
        # To speed up, we can precompute for each triple T_in, the set of triples it conflicts with.
        # But the set of "current" triples changes.
        
        # Simple approach:
        # 1. Pick a random T_out from current_triples.
        # 2. Calculate available_pairs = used_pairs - pairs(T_out).
        # 3. Find all T_in such that pairs(T_in) are disjoint from available_pairs.
        # 4. Among valid T_in, pick one that minimizes new_alpha (or maximizes overlap with MIS).
        
        # To find valid T_in fast:
        # We can iterate all triples? 560 triples.
        # For each T_in, check if any pair in T_in is in available_pairs.
        # This is 560 * 3 checks. ~1680 ops.
        # Very fast.
        
        # We need to calculate new_alpha.
        # get_alpha is ~0.1ms?
        # If we do 1000 local steps, that's 100ms.
        # 5000 restarts * 100ms = 500s. Too slow.
        
        # We need to be smarter or limit local search steps.
        # Or make get_alpha faster.
        
        # Let's limit local search to a few steps or use a cheaper heuristic.
        # Heuristic: Pick T_in that covers the most points in the current MIS.
        # We don't need to compute new_alpha for every candidate.
        # We just pick the best candidate based on heuristic, then compute new_alpha once.
        
        # Let's implement this.
        
        # Get current MIS
        # We need a function to get an actual MIS, not just the size.
        def get_mis(triples):
            masks = []
            for t in triples:
                mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
                masks.append(mask)
                
            size = 1 << 16
            is_bad = [False] * size
            for tm in masks:
                free = (~tm) & (size - 1)
                sub = free
                while True:
                    sup = tm | sub
                    is_bad[sup] = True
                    if sub == 0:
                        break
                    sub = (sub - 1) & free
                    
            max_alpha = 0
            best_mask = 0
            for m in range(size):
                if not is_bad[m]:
                    bc = m.bit_count()
                    if bc > max_alpha:
                        max_alpha = bc
                        best_mask = m
            return best_mask

        # In the local search loop:
        # 1. Get MIS mask.
        # 2. Pick random T_out.
        # 3. Find valid T_in candidates.
        # 4. Score candidates by popcount(T_in_mask & MIS_mask).
        # 5. Pick best candidate.
        # 6. Update state.
        # 7. Compute new_alpha.
        # 8. If new_alpha < curr_alpha, accept and update curr_alpha.
        #    If new_alpha == curr_alpha, accept with probability p (or just accept to move around).
        #    If new_alpha > curr_alpha, reject (or accept with low prob).
        
        # Let's try simple hill climbing (only accept if alpha decreases or stays same).
        # To avoid getting stuck, we can randomize the choice among tied candidates.
        
        for step in range(20): # 20 local steps per restart
            if time.time() > deadline - 2:
                break
                
            if len(current_triples) == 0:
                break
                
            # Get MIS
            mis_mask = get_mis(current_triples)
            
            # Pick T_out
            out_idx = random.randint(0, len(current_triples) - 1)
            t_out = current_triples[out_idx]
            pairs_out = triple_pairs[all_triples.index(t_out)] # This is slow!
            
            # Optimization: Store indices in current_triples?
            # Or map tuple to index?
            # Let's store indices in the list.
            
            # I need to refactor current_triples to store indices.
            # Let's restart the local search logic with indices.
            
            # Refactor:
            # current_indices = [list of indices into all_triples]
            # used_pairs = set
            
            # Let's re-initialize the local search state properly.
            
            break # Break out of the flawed local search implementation above
            
        # Corrected Local Search Implementation
        
        # Re-initialize state for local search
        curr_indices = []
        curr_used_pairs = set()
        
        # Re-run greedy to get indices
        indices = list(range(n))
        random.shuffle(indices)
        for idx in indices:
            t = all_triples[idx]
            pairs = triple_pairs[idx]
            conflict = False
            for p in pairs:
                if p in curr_used_pairs:
                    conflict = True
                    break
            if not conflict:
                curr_indices.append(idx)
                for p in pairs:
                    curr_used_pairs.add(p)
                    
        if not curr_indices:
            continue
            
        curr_alpha = get_alpha([all_triples[i] for i in curr_indices])
        update_best([all_triples[i] for i in curr_indices])
        
        local_start = time.time()
        
        for step in range(50): # 50 local steps
            if time.time() > deadline - 2:
                break
                
            if len(curr_indices) == 0:
                break
                
            # Get MIS
            # We need the triples for get_mis
            curr_triples_list = [all_triples[i] for i in curr_indices]
            mis_mask = get_mis(curr_triples_list)
            
            # Pick T_out
            out_pos = random.randint(0, len(curr_indices) - 1)
            out_idx = curr_indices[out_pos]
            t_out = all_triples[out_idx]
            pairs_out = triple_pairs[out_idx]
            
            # Remove T_out
            curr_indices.pop(out_pos)
            for p in pairs_out:
                curr_used_pairs.remove(p)
                
            # Find valid T_in
            # A triple T_in is valid if none of its pairs are in curr_used_pairs.
            # We want to maximize overlap with MIS.
            
            best_in_idx = -1
            best_score = -1
            candidates = []
            
            for in_idx in range(n):
                t_in = all_triples[in_idx]
                pairs_in = triple_pairs[in_idx]
                
                # Check conflict
                valid = True
                for p in pairs_in:
                    if p in curr_used_pairs:
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Score: overlap with MIS
                mask_in = (1 << t_in[0]) | (1 << t_in[1]) | (1 << t_in[2])
                score = (mask_in & mis_mask).bit_count()
                
                if score > best_score:
                    best_score = score
                    best_in_idx = in_idx
                    candidates = [in_idx]
                elif score == best_score:
                    candidates.append(in_idx)
                    
            if best_in_idx == -1:
                # No valid triple to add? Restore T_out
                curr_indices.append(out_idx)
                for p in pairs_out:
                    curr_used_pairs.add(p)
                continue
                
            # Pick randomly from best candidates
            chosen_in_idx = random.choice(candidates)
            
            # Add T_in
            curr_indices.append(chosen_in_idx)
            t_in = all_triples[chosen_in_idx]
            for p in triple_pairs[chosen_in_idx]:
                curr_used_pairs.add(p)
                
            # Evaluate new alpha
            new_triples = [all_triples[i] for i in curr_indices]
            new_alpha = get_alpha(new_triples)
            
            if new_alpha < curr_alpha:
                curr_alpha = new_alpha
                update_best(new_triples)
            elif new_alpha == curr_alpha:
                # Accept to explore
                pass
            else:
                # Reject: Restore T_out and remove T_in
                curr_indices.pop() # Remove T_in
                for p in triple_pairs[chosen_in_idx]:
                    curr_used_pairs.remove(p)
                    
                curr_indices.append(out_idx) # Restore T_out
                for p in pairs_out:
                    curr_used_pairs.add(p)
                    
    return best_triples
