import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Points 0..15
    N = 16
    points = list(range(N))
    
    # All possible triples
    all_triples = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                all_triples.append((a, b, c))
                
    random.shuffle(all_triples)
    
    best_alpha = N
    best_system = []
    
    # Helper to check if a triple can be added
    def can_add(triple, current_triples, used_pairs):
        a, b, c = triple
        if (a, b) in used_pairs or (b, a) in used_pairs: return False
        if (a, c) in used_pairs or (c, a) in used_pairs: return False
        if (b, c) in used_pairs or (c, b) in used_pairs: return False
        return True

    def add_triple(triple, current_triples, used_pairs):
        a, b, c = triple
        current_triples.append(triple)
        used_pairs.add((a, b))
        used_pairs.add((b, a))
        used_pairs.add((a, c))
        used_pairs.add((c, a))
        used_pairs.add((b, c))
        used_pairs.add((c, b))

    def remove_triple(triple, current_triples, used_pairs):
        a, b, c = triple
        current_triples.remove(triple)
        used_pairs.discard((a, b))
        used_pairs.discard((b, a))
        used_pairs.discard((a, c))
        used_pairs.discard((c, a))
        used_pairs.discard((b, c))
        used_pairs.discard((c, b))

    def calc_alpha(system):
        if not system:
            return N
        # Convert to masks for speed
        masks = [ (1<<a) | (1<<b) | (1<<c) for a,b,c in system ]
        max_alpha = 0
        # Iterate over all subsets
        # Optimization: only check subsets of size > current best
        for mask in range(1 << N):
            # Count bits
            # If popcount <= max_alpha, skip? 
            # Actually we want max size.
            # We can iterate sizes from N down to 0?
            # Or just check all and keep max.
            # 65536 iterations is fine.
            is_indep = True
            for m in masks:
                if (mask & m) == m:
                    is_indep = False
                    break
            if is_indep:
                # Count bits in mask
                cnt = bin(mask).count('1')
                if cnt > max_alpha:
                    max_alpha = cnt
        return max_alpha

    # Greedy Construction
    for _ in range(100): # Try multiple random starts
        if time.time() > deadline: break
        
        current_triples = []
        used_pairs = set()
        
        # Greedy fill
        for t in all_triples:
            if can_add(t, current_triples, used_pairs):
                add_triple(t, current_triples, used_pairs)
        
        # Local Search to improve (increase count or reduce alpha)
        # Try swapping
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomly pick a triple to remove
            if not current_triples: continue
            t_to_remove = random.choice(current_triples)
            remove_triple(t_to_remove, current_triples, used_pairs)
            
            # Try to add as many as possible from the freed pairs
            # The freed pairs are the 3 pairs in t_to_remove
            # We can try to add triples that use these pairs
            # But simpler: just run greedy again on the remaining available triples?
            # Or just try to add any valid triple.
            
            # Let's try to add triples that share pairs with the removed one
            # Actually, just iterate through all_triples and add if valid
            # This is O(M * N^3) which is fine.
            
            # To make it faster, we can just try to add random valid triples
            # But greedy is better for density.
            
            # Let's restart greedy from the current state?
            # No, just try to add triples that are currently valid.
            
            # A better move: 
            # Remove 1, add 2?
            
            # Let's just do a simple hill climb on alpha.
            # Calculate alpha before move? We did it at the end.
            # Let's calculate alpha for the new state.
            
            # Actually, let's just maximize the number of triples.
            # If we can add more triples, we do it.
            
            # Try to add triples
            added_any = False
            # Shuffle to randomize
            for t in all_triples:
                if t == t_to_remove: continue
                if can_add(t, current_triples, used_pairs):
                    add_triple(t, current_triples, used_pairs)
                    added_any = True
                    # If we added at least one, we improved density
                    # But we removed one. If we added 1, count is same.
                    # If we added >1, count increased.
                    # If we added 0, count decreased.
                    # We should only accept if count increased or alpha decreased.
                    # This is getting complicated.
            
            # Simpler: 
            # If len(current_triples) < len(previous_triples), revert?
            # Let's store the previous state.
            
            # This loop is messy. Let's use a cleaner local search.
            
            break # Skip this complex local search for now, just use multiple random greedy runs.
            
        # Calculate alpha for this system
        alpha = calc_alpha(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = [list(t) for t in current_triples]
            
            if best_alpha <= 5: # Likely optimal or near optimal
                break

    return best_system
