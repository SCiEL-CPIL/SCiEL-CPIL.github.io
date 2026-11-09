import sys
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0

    def max_length_sequence():
        # We will use a recursive backtracking search.
        # State: current sequence (list), set of all consecutive block sums.
        # Heuristic: Try larger numbers first.
        
        best_seq = []
        
        def backtrack(seq, sums, last_val):
            nonlocal best_seq
            
            # Update best if current is longer
            if len(seq) > len(best_seq):
                best_seq = seq[:]
                
            # Check time
            if time.time() - start_time > TIME_LIMIT:
                return

            # Pruning: if we can't add any more numbers, return
            # Max possible length is 256 - last_val
            # If we are close to the theoretical max, we can stop? 
            # No, we want to explore.
            
            # Generate candidates: last_val + 1 to 256
            # We try them in DESCENDING order.
            # Why? To spread out the sums.
            
            # Optimization: Pre-calculate suffix sums for current seq?
            # We can maintain the suffix sums of the current sequence to check conflicts faster.
            # Let's pass the current suffix sums ending at the last element?
            # Actually, checking conflicts requires checking all new suffixes against `sums`.
            # The new suffixes for a new element x are:
            # x, x + seq[-1], x + seq[-1] + seq[-2], ...
            
            # To speed up, we can just iterate. N is small.
            
            for x in range(256, last_val, -1):
                # Calculate new sums
                new_sums = [x]
                s = x
                for a in reversed(seq):
                    s += a
                    new_sums.append(s)
                    
                # Check for conflicts
                conflict = False
                for ns in new_sums:
                    if ns in sums:
                        conflict = True
                        break
                        
                if conflict:
                    continue
                
                # Accept
                seq.append(x)
                sums.update(new_sums)
                
                backtrack(seq, sums, x)
                
                # Backtrack
                sums.difference_update(new_sums)
                seq.pop()
                
                # If we found a sequence of length 256 (impossible), we could stop.
                # If we found a very long one, we might want to stop early?
                # No, we want the maximum. But if we find one of length 10, 
                # and we know 10 is hard, maybe we continue.
                
                # Time check inside loop to avoid deep recursion timeout
                if time.time() - start_time > TIME_LIMIT:
                    return

        # Initial call
        # Start with empty sequence.
        # To help the search, we can force the first few elements?
        # No, let the heuristic do it.
        
        backtrack([], set(), 0)
        
        return best_seq

    seq = max_length_sequence()
    
    # Fallback if the search didn't find anything or timed out with empty
    if not seq:
        # Simple greedy as fallback
        values, sums = [], set()
        for x in range(1, 257):
            new_sums = [x]
            s = x
            for a in reversed(values):
                s += a
                new_sums.append(s)
            if not sums.intersection(new_sums):
                values.append(x)
                sums.update(new_sums)
        seq = values
        
    return seq
