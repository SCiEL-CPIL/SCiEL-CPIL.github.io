import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Beam search state: (seq_tuple, products_set, suffix_products_list)
    # Initial state
    initial_state = ((), set(), [])
    beam = [initial_state]
    
    best_seq = ()
    best_len = 0

    # We iterate through possible values x from 2 to 128
    # At each step, we decide to include x or not for each state in the beam
    # However, standard beam search over "steps" (values) works well here.
    
    # To improve efficiency, we can prune states that cannot possibly beat the current best length
    # Max possible additional length from value x is (128 - x + 1)
    
    for x in range(2, 129):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
            
        new_beam = []
        
        for state in beam:
            seq, prods, suffixes = state
            last_elem = seq[-1] if seq else 1
            
            # Option 1: Skip x
            # Only keep this state if it's potentially useful
            # We will deduplicate and prune later
            new_beam.append(state)
            
            # Option 2: Take x
            if x > last_elem:
                # Calculate new suffix products
                # New suffixes: [x, suffixes[0]*x, suffixes[1]*x, ...]
                # Note: suffixes[0] is the last element of seq (product of length 1 suffix)
                # suffixes[1] is product of last 2 elements, etc.
                
                new_suffixes = [x]
                # Check collision for x
                if x in prods:
                    valid = False
                else:
                    valid = True
                    for s in suffixes:
                        ns = s * x
                        new_suffixes.append(ns)
                        if ns in prods:
                            valid = False
                            break
                
                if valid:
                    # All new suffixes are distinct from each other and from old prods
                    new_prods = prods.union(new_suffixes)
                    new_seq = seq + (x,)
                    new_state = (new_seq, new_prods, new_suffixes)
                    
                    # Update best
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
                        
                    new_beam.append(new_state)
        
        if not new_beam:
            break
            
        # Prune the beam
        # Score:
        # 1. Length (higher is better)
        # 2. Flexibility: Smaller last element is better (allows more future additions)
        # 3. Smaller max product? (might help avoid collisions)
        
        def state_score(state):
            seq, prods, suffixes = state
            L = len(seq)
            last = seq[-1] if seq else 1
            # Prefer higher length
            # Then prefer smaller last element
            # Then prefer smaller number of products? (all same length have same number of products k(k+1)/2)
            # Maybe prefer smaller max product to keep numbers small?
            # Let's try: (L, -last, -max(prods) if prods else 0)
            # max(prods) calculation is O(k^2). Might be slow.
            # Let's just use (L, -last) for now.
            return (L, -last)
            
        # Sort and keep top K
        # Beam width strategy:
        # Early on, wider beam. Later, narrower?
        # Or fixed width?
        # Let's use a width that adapts to time.
        
        elapsed = time.time() - start_time
        time_frac = elapsed / 55.0
        if time_frac < 0.5:
            beam_width = 300
        elif time_frac < 0.8:
            beam_width = 150
        else:
            beam_width = 50
            
        # Sort by score descending
        new_beam.sort(key=state_score, reverse=True)
        
        # Deduplicate?
        # States with same seq are identical.
        # We can use a set of seq tuples to avoid duplicates in the beam.
        seen_seqs = set()
        unique_beam = []
        for s in new_beam:
            if s[0] not in seen_seqs:
                seen_seqs.add(s[0])
                unique_beam.append(s)
            if len(unique_beam) >= beam_width:
                break
                
        beam = unique_beam
        
        # Early termination check
        # If the max length in beam + remaining numbers <= best_len, we can stop?
        # Max possible extension from current x is (128 - x + 1)
        if beam:
            max_len_in_beam = beam[0][0] and len(beam[0][0]) or 0
            if max_len_in_beam + (128 - x + 1) <= best_len:
                break

    return list(best_seq)

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
