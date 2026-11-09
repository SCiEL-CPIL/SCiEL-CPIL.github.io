import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 50  # Safety margin
    
    def compute_new_sums(seq, x):
        new_sums = set()
        acc = x
        for v in reversed(seq):
            new_sums.add(acc)
            acc += v
        return new_sums

    def is_valid_extension(seq, sums_set, x):
        new_sums = compute_new_sums(seq, x)
        if new_sums & sums_set:
            return False, new_sums
        return True, new_sums

    # Beam Search
    # State: (sequence_tuple, sums_set, length)
    beam = [([], set(), 0)]
    best_seq = []
    best_len = 0
    
    BEAM_WIDTH = 100
    MAX_CANDIDATES_PER_STATE = 20  # Limit branching factor
    
    while time.time() < DEADLINE:
        next_beam = []
        
        for seq, sums, length in beam:
            last_val = seq[-1] if seq else 0
            found = 0
            
            # Try to add next values, starting from smallest (greedy bias)
            for x in range(last_val + 1, 257):
                if found >= MAX_CANDIDATES_PER_STATE:
                    break
                
                valid, new_sums = is_valid_extension(seq, sums, x)
                if valid:
                    new_seq = seq + (x,)
                    new_sums_set = sums | new_sums
                    new_len = length + 1
                    next_beam.append((new_seq, new_sums_set, new_len))
                    found += 1
        
        if not next_beam:
            break
            
        # Sort by length descending
        next_beam.sort(key=lambda x: x[2], reverse=True)
        
        # Prune to BEAM_WIDTH
        beam = next_beam[:BEAM_WIDTH]
        
        # Update best
        if beam:
            current_best_len = beam[0][2]
            if current_best_len > best_len:
                best_len = current_best_len
                best_seq = list(beam[0][0])
        
        if time.time() > DEADLINE - 2:
            break

    return best_seq
