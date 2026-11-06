import time
import random

def construct():
    start_time = time.time()
    # The evaluator kills after 90s. We should stop a bit earlier to be safe.
    # The previous program used 55s. Let's use 85s to maximize search time.
    deadline = start_time + 85.0
    
    MAX_VAL = 256
    best_seq = []
    best_len = 0
    
    def try_randomized_greedy():
        """
        Performs a single randomized greedy search.
        Returns the sequence found.
        """
        seq = []
        suffix_sums = []
        used_sums = set()
        last_val = 0
        
        while True:
            # Find valid candidates
            # We scan from last_val + 1 to MAX_VAL
            # We collect a limited number of valid candidates to keep it fast
            valid_candidates = []
            
            # Optimization: We can break early if we have enough candidates
            # but we want to know if there are ANY.
            # Scanning 256 items is fast.
            
            for x in range(last_val + 1, MAX_VAL + 1):
                # Check if x is valid
                # New sums will be x and x + s for all s in suffix_sums
                if x in used_sums:
                    continue
                
                valid = True
                for s in suffix_sums:
                    if x + s in used_sums:
                        valid = False
                        break
                
                if valid:
                    valid_candidates.append(x)
                    # Limit the number of candidates we consider to keep the step fast
                    # and to bias towards smaller numbers if we just take the first few.
                    # However, taking too few might miss good jumps.
                    # Let's take up to 10 candidates.
                    if len(valid_candidates) >= 10:
                        break
                        
            if not valid_candidates:
                break
            
            # Select a candidate
            # Strategy: Heavily bias towards the smallest valid candidate,
            # but occasionally pick a larger one to explore different branches.
            r = random.random()
            if r < 0.8:
                # Pick the smallest
                x = valid_candidates[0]
            elif r < 0.95:
                # Pick from the first 3
                x = valid_candidates[random.randint(0, min(2, len(valid_candidates)-1))]
            else:
                # Pick from all found (up to 10)
                x = valid_candidates[random.randint(0, len(valid_candidates)-1)]
            
            # Update state
            seq.append(x)
            last_val = x
            
            # Calculate new suffix sums
            new_suffix = [x]
            for s in suffix_sums:
                new_suffix.append(x + s)
            
            # Add to used sums
            used_sums.update(new_suffix)
            suffix_sums = new_suffix
            
            # Check time
            if time.time() > deadline:
                return seq
                
        return seq

    # Run multiple trials
    # We can run a few "pure greedy" trials first (with r=0.0 logic, i.e., always pick smallest)
    # but the randomized version with high bias covers that.
    
    while time.time() < deadline:
        seq = try_randomized_greedy()
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            
        # Optional: If we find a very long sequence, we might try to extend it further
        # or do a local search. But for now, just keep running randomized trials.
        
        # If we have plenty of time, we can increase the "randomness" or candidate pool.
        # But the current heuristic is simple and robust.
        
    return best_seq
