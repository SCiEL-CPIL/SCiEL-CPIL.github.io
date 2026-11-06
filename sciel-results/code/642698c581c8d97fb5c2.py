import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    best_seq = []
    best_len = 0
    beam_width = 500  # Increased beam width for better exploration
    
    # beams is a list of tuples: (seq_list, sum_mask, last_val, suffix_sums_list)
    # suffix_sums_list[i] is the sum of the last i+1 elements.
    current_beams = [([], 0, 0, [])]
    
    while current_beams:
        if time.time() > deadline:
            break
            
        next_beams = []
        
        for seq, sums, last_val, suffix_sums in current_beams:
            # Generate all valid extensions for this beam
            for x in range(last_val + 1, 257):
                # Check sum x
                if sums & (1 << x):
                    continue
                    
                new_sum_bits = 1 << x
                conflict = False
                s = x
                new_suffix = [x]
                
                # Check sums with previous suffixes
                for s_suffix in suffix_sums:
                    s += s_suffix
                    if sums & (1 << s):
                        conflict = True
                        break
                    new_sum_bits |= (1 << s)
                    new_suffix.append(s)
                    
                if conflict:
                    continue
                
                # Valid extension
                new_seq = seq + [x]
                new_sums = sums | new_sum_bits
                new_last = x
                
                next_beams.append((new_seq, new_sums, new_last, new_suffix))
                
                # Update best if this sequence is longer
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
        
        if not next_beams:
            break
            
        # Prune next_beams to keep only the most promising ones
        # Heuristic: Smaller last_val is better (leaves more room)
        next_beams.sort(key=lambda x: x[2])
        
        current_beams = next_beams[:beam_width]
        
    return best_seq
