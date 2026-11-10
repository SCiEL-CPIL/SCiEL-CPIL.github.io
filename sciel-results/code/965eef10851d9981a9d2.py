import time
import sys
import heapq

def construct():
    start_time = time.time()
    time_limit = 50.0
    
    # Beam search parameters
    beam_width = 200
    max_val = 256
    
    # State: (negative_score, seq, sums, suffix_sums, last_val)
    # We use a max-heap for the beam, so we negate the score.
    # Score: We want to maximize min_gap.
    # If min_gap is 0, it's bad.
    
    # Initial state
    # seq: []
    # sums: set()
    # suffix_sums: []
    # last_val: 0
    
    current_beam = [(0, [], set(), [], 0)] # (neg_score, seq, sums, suffix_sums, last_val)
    # Actually, for the first step, score is arbitrary, let's use 0.
    
    best_seq = []
    best_len = 0
    
    # Helper to calculate min gap
    def calc_min_gap(sums):
        if len(sums) < 2:
            return 10000 # Large gap if few sums
        sorted_sums = sorted(sums)
        min_g = 10000
        for i in range(len(sorted_sums) - 1):
            g = sorted_sums[i+1] - sorted_sums[i]
            if g < min_g:
                min_g = g
        return min_g

    # We will iterate levels
    # Level k means sequence length is k
    
    while time.time() - start_time < time_limit:
        next_beam = []
        
        for neg_score, seq, sums, suffix_sums, last_val in current_beam:
            # Try to extend
            for x in range(last_val + 1, max_val + 1):
                # Check time
                if time.time() - start_time > time_limit:
                    break
                
                # Calculate new suffix sums
                new_suffix_sums = [x]
                # Check collision with existing sums
                if x in sums:
                    continue
                
                valid = True
                for s in suffix_sums:
                    ns = x + s
                    new_suffix_sums.append(ns)
                    if ns in sums:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                new_sums = sums.union(new_suffix_sums)
                new_seq = seq + [x]
                new_len = len(new_seq)
                
                if new_len > best_len:
                    best_len = new_len
                    best_seq = new_seq
                
                # Calculate heuristic score
                # We want to maximize the minimum gap between sums.
                # Also, we prefer smaller last_val to keep options open?
                # Or larger last_val to avoid collisions with small sums?
                # Let's try: score = min_gap * 100 - last_val
                # This encourages sparse sums and smaller values.
                
                min_g = calc_min_gap(new_sums)
                # If min_g is 0, it's invalid (but we checked disjointness, so min_g >= 1)
                score = min_g * 100 - last_val
                
                # Add to next beam
                next_beam.append((-score, new_seq, new_sums, new_suffix_sums, x))
        
        if not next_beam:
            break
            
        # Select top beam_width states
        # Sort by score (ascending, so most negative is best)
        next_beam.sort(key=lambda item: item[0])
        current_beam = next_beam[:beam_width]
        
        # Optional: If we have a very long sequence, we can stop expanding
        # if the theoretical max is reached.
        # Max possible length is roughly 256 / (avg step).
        # If best_len is high, say 20, and we are at level 20,
        # we can check if any state in beam can be extended.
        # If not, we are done.
        
    return best_seq

if __name__ == "__main__":
    print(construct())
