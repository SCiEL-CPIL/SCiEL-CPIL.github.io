import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Beam search parameters
    beam_width = 20
    
    # State: (sequence, set_of_sums, last_element)
    # We store the sequence and the set of sums.
    # To speed up, we can store the set of sums as a Python set.
    
    # Initial state
    # We start with empty sequence.
    # But we must pick the first element.
    # Let's just run greedy first to get a baseline.
    
    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values
    
    best_seq = greedy()
    best_len = len(best_seq)
    
    # Beam search to try to improve
    # Each beam item is a tuple (seq, sums_set, last_val)
    # We initialize with the first few elements from greedy?
    # Or start from scratch.
    
    # Let's start from scratch with beam search.
    # Beam contains tuples (seq, sums_set)
    beam = [([], set())]
    
    for step in range(20): # Max length 20 is plenty
        if time.time() > deadline:
            break
            
        next_beam = []
        for seq, sums_set in beam:
            last_val = seq[-1] if seq else 0
            
            # Try candidates from last_val + 1 to 256
            # To be efficient, we can limit the range or break early if we find enough good candidates
            # But we want the "best" candidates.
            # Heuristic: prefer smaller values?
            
            candidates = []
            for x in range(last_val + 1, 257):
                # Check collision
                tails, total = [x], x
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                
                # Check intersection
                # Optimization: if any tail is in sums_set, skip
                # Since tails are increasing, and sums_set is a set, we can just check.
                # If len(tails) is small, it's fast.
                
                is_valid = True
                for t in tails:
                    if t in sums_set:
                        is_valid = False
                        break
                
                if is_valid:
                    new_sums = sums_set.union(tails)
                    next_beam.append((seq + [x], new_sums))
                    
                    # If we have enough candidates, we can stop looking for this seq
                    # to save time, but we want diversity.
                    # Let's collect all valid candidates for this seq, then sort.
            
            # Sort candidates for this seq by x (smallest first)
            # We already generated them in increasing order of x.
            # So next_beam has them in order.
            
        # Now we have a list of all possible extensions for all beams.
        # We need to select the top `beam_width` candidates.
        # Heuristic: Prefer sequences that allow for longer extensions.
        # A simple heuristic is to prefer smaller last elements, as they leave more room.
        # Or prefer smaller total sum?
        # Let's sort by last element, then by total sum.
        
        if not next_beam:
            break
            
        # Sort by last element, then by length of sequence (should be same), then by total sum
        # Actually, all sequences in next_beam have length step+1.
        # So sort by last element.
        next_beam.sort(key=lambda item: item[0][-1])
        
        # Keep top beam_width
        beam = next_beam[:beam_width]
        
        # Update best
        for seq, _ in beam:
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq
        
        # If we reached max possible length or no more extensions, break
        if len(beam) == 0:
            break
            
        # If the smallest last element in the beam is 256, we can't extend further
        if beam[0][0][-1] == 256:
            break

    return best_seq
