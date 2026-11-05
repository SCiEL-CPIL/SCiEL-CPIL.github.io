import time

def construct():
        start_time = time.time()
        deadline = start_time + 55  # Safety margin
        
        best_seq = []
        
        def dfs(seq, used_sums, suffix_sums, last_val):
            nonlocal best_seq
            
            # Update best
            if len(seq) > len(best_seq):
                best_seq = list(seq)
                
            # Check time
            if time.time() > deadline:
                return
                
            # Try candidates
            # Heuristic: Largest first
            # Range: last_val + 1 to 256
            for x in range(256, last_val, -1):
                # Calculate new sums
                # New suffix sums are x, x+last, x+last+prev...
                # Which is x + s for s in [0] + suffix_sums
                new_suffix_sums = [x + s for s in [0] + suffix_sums]
                
                # Check for collisions
                # If any new sum is in used_sums, invalid
                valid = True
                for s in new_suffix_sums:
                    if s in used_sums:
                        valid = False
                        break
                        
                if valid:
                    new_used_sums = used_sums.union(new_suffix_sums)
                    dfs(seq + [x], new_used_sums, new_suffix_sums, x)
                    
                    # Optional: If we found a very long sequence, we might stop early?
                    # No, we want the max. But if we hit time, we stop.
                    
        dfs([], set(), [], 0)
        return best_seq
