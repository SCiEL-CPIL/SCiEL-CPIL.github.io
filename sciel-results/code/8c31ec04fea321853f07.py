import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Beam search
    # State: (sequence, all_products_set, suffix_products_list)
    # We want to maximize length.
    
    beam = [([], set(), [])]
    best_seq = []
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        for seq, prods, suffixes in beam:
            # Calculate new suffixes and products
            # New suffixes: x, x * s for s in suffixes
            # Note: suffixes contains products of suffixes of seq.
            # If seq is empty, suffixes is [].
            # New suffix product for [x] is x.
            # New suffix products for [seq..., x] are x * (product of suffix of seq).
            
            new_suffixes = [x]
            for s in suffixes:
                new_suffixes.append(x * s)
            
            new_prods = set(new_suffixes)
            
            # Check collision
            if prods.intersection(new_prods):
                continue
            
            # Valid extension
            new_seq = seq + [x]
            new_all_prods = prods.union(new_prods)
            
            new_beam.append((new_seq, new_all_prods, new_suffixes))
        
        if not new_beam:
            # No valid extensions? This shouldn't happen as we can always stop, 
            # but if we must extend, we might be stuck. 
            # However, the problem asks to return a sequence. 
            # If no extension is valid for x, we just skip x for all current beam states.
            # The beam remains the same? No, we are iterating x. 
            # If we skip x, the sequences in the beam don't change.
            # But we are moving to the next x.
            # So we keep the old beam? 
            # Actually, if we can't add x to any state, we just don't add any new states.
            # The existing states are still valid sequences ending at some x' < x.
            # But we want to maximize length.
            # We should keep the best states from the previous step if no new ones are generated?
            # Or just let the beam shrink?
            # If we let the beam shrink, we might lose good prefixes.
            # It's better to keep the old states as candidates too?
            # No, the states in `beam` are sequences ending at some value < x.
            # If we don't extend them with x, they remain valid sequences.
            # But we are looking for the longest sequence overall.
            # We should update `best_seq` with the longest in `beam` before replacing it.
            
            # Actually, the standard beam search for this problem usually just replaces the beam.
            # If a state can't be extended, it dies.
            # But a state that ends at 10 is a valid sequence.
            # We just need to record the best one seen so far.
            
            # Let's update best_seq with the longest in the current beam before processing x?
            # Or after?
            # The current beam contains sequences ending at values < x.
            # So they are complete valid sequences.
            
            for s, _, _ in beam:
                if len(s) > len(best_seq):
                    best_seq = s
            
            # If new_beam is empty, we can't proceed with any extension for x.
            # We should probably break or continue with an empty beam?
            # If we continue with empty beam, we stop searching.
            # But maybe a different x later could have been added? No, we must pick increasing integers.
            # If we can't add x to any current prefix, can we add x+1?
            # Yes, if the prefix is shorter.
            # But the prefixes in `beam` are the best ones.
            # If they can't be extended by x, they might be extended by x+1?
            # Wait, the sequence must be strictly increasing.
            # If we have prefix [2, 3], and x=4 doesn't work, maybe x=5 works?
            # Yes.
            # So we should NOT discard the beam just because x failed.
            # We should keep the beam as is, and try the next x.
            
            # So: if new_beam is empty, we just skip x.
            # But we should update best_seq first.
            
            # However, if we keep the beam, we might try to extend the same prefixes with x+1.
            # This is correct.
            
            # So:
            # 1. Update best_seq from current beam.
            # 2. If new_beam is not empty, replace beam with top K of new_beam.
            #    Else, keep beam as is?
            #    Actually, if new_beam is empty, it means no state in beam could be extended by x.
            #    The states in beam are still valid candidates for extension by x+1.
            #    So we keep `beam` unchanged.
            
            # But wait, if we keep `beam`, we might have duplicate states or inefficient memory?
            # No, it's fine.
            
            # Let's restructure:
            # 1. Update best_seq from `beam`.
            # 2. Generate `new_beam` from `beam` + x.
            # 3. If `new_beam` is not empty, sort and truncate. `beam = new_beam`.
            #    Else, `beam` remains the same.
            
            pass # Logic handled below

        # Update best_seq with sequences in beam (which end at < x)
        # Actually, the sequences in `beam` are the ones we just tried to extend.
        # They are valid sequences.
        for s, _, _ in beam:
            if len(s) > len(best_seq):
                best_seq = s

        if new_beam:
            # Sort by length, then maybe by last element (smaller is better for flexibility?)
            # Length is the primary objective.
            new_beam.sort(key=lambda item: (len(item[0]), item[0][-1]), reverse=True)
            beam = new_beam[:50]
        # else: beam stays the same
        
    # Final check
    for s, _, _ in beam:
        if len(s) > len(best_seq):
            best_seq = s
            
    return best_seq

# EVOLVE-BLOCK-START
# The code above is the new design.
# We need to wrap it in the required function signature.
# EVOLVE-BLOCK-END
