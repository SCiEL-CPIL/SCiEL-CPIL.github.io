import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    
    # Precompute masks for all triples
    triple_masks = {}
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[t] = mask

    def is_valid(triple, used_pairs):
        # Check if triple can be added given the set of used pairs
        # A triple is valid if none of its 3 pairs are in used_pairs
        a, b, c = triple
        if (a, b) in used_pairs or (a, c) in used_pairs or (b, c) in used_pairs:
            return False
        return True

    def calc_alpha(triples):
        """
        Calculate the independence number (max size of a set containing no full triple).
        Returns the size.
        """
        if not triples:
            return N
        
        t_masks = [triple_masks[t] for t in triples]
        
        # We check subsets from largest to smallest
        # Optimization: If we find an independent set of size k, we don't need to check smaller ones
        # for the *max* size, but we need to be sure we found the max.
        # Iterating all 2^16 masks is 65536. Checking each against ~40 triples is fast.
        
        best_alpha = 0
        # Iterate all non-empty subsets
        for mask in range(1, 1 << N):
            if mask.bit_count() <= best_alpha:
                continue
            
            is_indep = True
            for tm in t_masks:
                # If the subset contains the triple (mask covers tm), it's not independent
                if (mask & tm) == tm:
                    is_indep = False
                    break
            
            if is_indep:
                best_alpha = mask.bit_count()
                
        return best_alpha

    def find_max_independent_set(triples):
        """
        Find one specific maximum independent set.
        Returns a list of points in the set.
        """
        if not triples:
            return list(range(N))
        
        t_masks = [triple_masks[t] for t in triples]
        
        best_set = None
        best_size = 0
        
        # To speed up, we can iterate masks in descending order of bit count?
        # No, just iterate all and keep the best.
        # Or, better: iterate k from N down to 1, check if any set of size k is independent.
        # This can stop early.
        
        # Let's use the property that we just need *one* max set.
        # We'll iterate all masks, but we can prune if bit_count < current best.
        
        for mask in range(1, 1 << N):
            bc = mask.bit_count()
            if bc <= best_size:
                continue
            
            is_indep = True
            for tm in t_masks:
                if (mask & tm) == tm:
                    is_indep = False
                    break
            
            if is_indep:
                best_size = bc
                # Convert mask to list of points
                pts = []
                for i in range(N):
                    if mask & (1 << i):
                        pts.append(i)
                best_set = pts
                
        if best_set is None:
            return []
        return best_set

    # 1. Greedy Initialization
    # We want a dense packing. Random shuffling helps avoid bad local structures.
    best_init_triples = []
    best_init_alpha = N + 1
    
    init_deadline = start_time + 5.0
    while time.time() < init_deadline:
        random.shuffle(all_triples)
        triples = []
        used_pairs = set()
        for t in all_triples:
            if is_valid(t, used_pairs):
                triples.append(t)
                a, b, c = t
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
        
        if len(triples) > len(best_init_triples):
            best_init_triples = triples[:]
            # Don't calc alpha here, it's expensive. Just track size for init.
            # We will calc alpha after init loop.
            if len(triples) >= 38: # Theoretical max is close to this
                break

    current_triples = best_init_triples
    current_alpha = calc_alpha(current_triples)
    best_triples = current_triples[:]
    best_alpha = current_alpha
    
    T = 1.0
    no_improve_counter = 0
    
    while time.time() < deadline:
        # Re-evaluate alpha to ensure it's current (it changes only if we accept a move)
        # But we should probably find the specific witness set to guide the move
        
        witness_set = find_max_independent_set(current_triples)
        if not witness_set:
            break
            
        witness_mask = 0
        for p in witness_set:
            witness_mask |= (1 << p)
            
        # Generate candidates that lie entirely within the witness set
        # This helps "break" the independence of the witness set
        candidates = []
        for t in all_triples:
            if (triple_masks[t] & witness_mask) == triple_masks[t]:
                candidates.append(t)
                
        random.shuffle(candidates)
        
        move_applied = False
        new_triples = current_triples[:]
        used_pairs = set()
        for t in current_triples:
            a, b, c = t
            used_pairs.add((a, b))
            used_pairs.add((a, c))
            used_pairs.add((b, c))
            
        # Strategy: Try to add a triple from the witness set
        # If we can't, try to swap a triple with one from the witness set
        # If we can't, do a random move (SA exploration)
        
        added = False
        for t_new in candidates:
            if is_valid(t_new, used_pairs):
                new_triples.append(t_new)
                added = True
                move_applied = True
                break
                
        if not added:
            # Try swap: Remove a random triple, see if we can add a witness triple
            # Better: Identify triples that block the witness candidates.
            # For now, simple random swap with witness candidate
            if current_triples:
                # Pick a random triple to remove
                idx = random.randint(0, len(current_triples) - 1)
                t_old = current_triples[idx]
                
                # Remove t_old
                new_triples.pop(idx)
                a, b, c = t_old
                used_pairs.discard((a, b))
                used_pairs.discard((a, c))
                used_pairs.discard((b, c))
                
                # Try to add a witness triple
                swap_success = False
                for t_new in candidates:
                    if is_valid(t_new, used_pairs):
                        new_triples.append(t_new)
                        swap_success = True
                        move_applied = True
                        break
                        
                if not swap_success:
                    # Revert
                    new_triples.append(t_old)
                    used_pairs.add((a, b))
                    used_pairs.add((a, c))
                    used_pairs.add((b, c))
                    # If we failed to swap, maybe just do a pure random move?
                    # Or try a different random triple to remove?
                    # Let's try a generic random move to escape local minima
                    if random.random() < 0.2: # 20% chance to do generic random move
                         # Generic random swap
                         idx2 = random.randint(0, len(current_triples) - 1)
                         t_old2 = current_triples[idx2]
                         new_triples2 = current_triples[:]
                         new_triples2.pop(idx2)
                         used_pairs2 = set()
                         for t in new_triples2:
                             a,b,c = t
                             used_pairs2.add((a,b)); used_pairs2.add((a,c)); used_pairs2.add((b,c))
                         
                         t_rand = random.choice(all_triples)
                         if is_valid(t_rand, used_pairs2):
                             new_triples2.append(t_rand)
                             new_triples = new_triples2
                             move_applied = True

        if not move_applied:
            # If no move applied, cool down slightly and continue
            T = max(T * 0.99, 0.1)
            continue
            
        new_alpha = calc_alpha(new_triples)
        
        if new_alpha < current_alpha:
            current_triples = new_triples
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = current_triples[:]
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            T = max(T * 0.95, 0.1)
        else:
            delta = new_alpha - current_alpha
            if delta > 0:
                prob = pow(2.71828, -delta / T)
                if random.random() < prob:
                    current_triples = new_triples
                    current_alpha = new_alpha
                    no_improve_counter += 1
            else:
                # delta == 0, accept to keep diversity?
                if random.random() < 0.5:
                    current_triples = new_triples
                    current_alpha = new_alpha
                    no_improve_counter += 1
            
            T = max(T * 0.99, 0.1)
            
            # If stuck for too long, restart temperature or randomize?
            if no_improve_counter > 1000:
                T = 1.0
                no_improve_counter = 0

    return [list(t) for t in best_triples]
