import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    best_seq = []
    
    def count_valid_nexts(seq, sums, min_val):
        # Count how many x > min_val are valid for seq
        count = 0
        for x in range(min_val, 257):
            if x in sums:
                continue
            
            s = 0
            valid = True
            # Check x
            # Check x + suffix
            # We don't need to build the set, just check validity
            # x
            if x in sums: continue # Redundant
            
            # x + suffix
            # Note: x is not in sums (checked).
            # We need to check if x + suffix_sum is in sums.
            
            # Also, do we need to check if x + suffix_sum creates duplicates among themselves?
            # No, seq is increasing, so suffix sums are increasing.
            # So x + suffix sums are distinct.
            # We only care about collision with existing `sums`.
            
            for i in range(len(seq) - 1, -1, -1):
                s += seq[i]
                if (x + s) in sums:
                    valid = False
                    break
            
            if valid:
                count += 1
                
        return count

    # Beam Search
    # State: (sequence, sums_set, last_val)
    beam = [([], set(), 0)]
    beam_width = 20
    
    while True:
        if time.time() > deadline:
            break
            
        next_beam = []
        
        for seq, sums, last_val in beam:
            # Generate candidates
            for x in range(last_val + 1, 257):
                if x in sums:
                    continue
                
                # Check validity
                s = 0
                valid = True
                new_sums_set = set(sums) # We need this for the next state? 
                # Actually, we can just store the set.
                # But copying sets is expensive.
                # We can store the set and update it? No, immutable is safer.
                # Let's create the new set.
                
                # Optimization:
                # We can check validity first.
                
                # x
                # x + suffix
                
                # If valid, create new sums set
                # new_sums = sums | {x, x+seq[-1], ...}
                
                # To avoid creating the set if invalid, check first.
                
                s = 0
                added = [x]
                for i in range(len(seq) - 1, -1, -1):
                    s += seq[i]
                    val = x + s
                    added.append(val)
                    if val in sums:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Also check if x in sums (done)
                # Check if any added val in sums (done in loop for suffixes, but x itself?)
                # x is in added[0]. We checked x in sums before loop.
                
                # Create new sums set
                new_sums = sums.union(added)
                
                # Calculate flexibility score
                # How many y > x are valid for new_seq?
                # This is expensive if done for every candidate.
                # But we only have ~20-50 candidates in next_beam.
                # So we can calculate it.
                
                # However, we can delay the scoring until we have the full next_beam.
                # Store (x, new_seq, new_sums)
                next_beam.append((x, seq + [x], new_sums))
        
        if not next_beam:
            break
        
        # Score and sort next_beam
        # Score: (valid_next_count, -last_val)
        # We want to maximize valid_next_count.
        # And minimize last_val.
        
        scored_beam = []
        for x, seq, sums in next_beam:
            # Count valid nexts for this state
            # min_val for next is x + 1
            flex = count_valid_nexts(seq, sums, x + 1)
            scored_beam.append((flex, -x, seq, sums))
            
        # Sort by flex (desc), then x (asc)
        scored_beam.sort(key=lambda item: (item[0], item[1]), reverse=True)
        
        # Keep top beam_width
        # Note: reverse=True sorts by flex desc, then -x desc (which is x asc).
        # Wait. If I want x asc, I should sort by -x asc?
        # No.
        # Key: (flex, -x)
        # Sort reverse=True:
        # High flex first.
        # For same flex, high -x first -> Low x first.
        # Yes, this works.
        
        beam = [(seq, sums, -neg_x) for flex, neg_x, seq, sums in scored_beam[:beam_width]]
        # Wait, I stored -x as neg_x. So last_val is -neg_x.
        # Let's restructure to avoid confusion.
        
        # Let's just store (seq, sums) and re-derive last_val?
        # Or store (seq, sums, last_val).
        
        # Let's redo the storage in scored_beam.
        # scored_beam contains (flex, x, seq, sums)
        # Sort by flex desc, x asc.
        
        # Python sort is stable.
        # We can sort by x first, then flex?
        # No, we want flex to be primary.
        
        # Let's just use a custom sort key.
        # We want max flex, min x.
        # So key = (-flex, x)
        # Sort ascending.
        
        scored_beam2 = []
        for x, seq, sums in next_beam:
            flex = count_valid_nexts(seq, sums, x + 1)
            scored_beam2.append((-flex, x, seq, sums))
            
        scored_beam2.sort(key=lambda item: (item[0], item[1]))
        
        beam = [(seq, sums, x) for _, x, seq, sums in scored_beam2[:beam_width]]
        
        # Update best_seq
        current_len = len(beam[0][0])
        if current_len > len(best_seq):
            best_seq = beam[0][0]
            
        # Optional: Adjust beam width
        # If we are stuck (no length increase), maybe increase width?
        # Or if we have time, increase width.
        
        if time.time() < deadline - 10:
             beam_width = min(beam_width + 5, 100)
        else:
             beam_width = max(beam_width - 10, 10) # Tighten up?
             
    return best_seq
