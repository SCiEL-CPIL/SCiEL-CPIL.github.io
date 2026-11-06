import time
import random

def construct():
        start_time = time.time()
        time_limit = 55.0
        
        best_seq = []
        
        def get_new_sums(seq, x):
            sums = []
            s = x
            sums.append(s)
            for a in reversed(seq):
                s += a
                sums.append(s)
            return sums

        def dfs(seq, sums_set, last_val, depth, deadline):
            nonlocal best_seq
            if time.time() > deadline:
                return
            
            if len(seq) > len(best_seq):
                best_seq = seq.copy()
            
            # Try to extend
            # Candidates are last_val + 1 to 256
            # To add diversity, we can shuffle candidates? 
            # Or just iterate.
            
            # Pruning: 
            # If we have used many values, we might not have enough room.
            
            valid_candidates = []
            for x in range(last_val + 1, 257):
                new_sums = get_new_sums(seq, x)
                # Check collision
                if any(s in sums_set for s in new_sums):
                    continue
                valid_candidates.append((x, new_sums))
            
            if not valid_candidates:
                return
            
            # Heuristic: 
            # If we want to go deep, we might want to pick candidates that leave "room".
            # Smaller x leaves more numbers available.
            # Larger x creates larger sums, which might be safer from collisions with small sums?
            # But we are constrained to 256.
            
            # Let's try a random choice to explore different branches.
            # Or, we can iterate through them, but if we find a dead end, we backtrack.
            # To ensure we explore different "shapes", we can randomize the order.
            
            random.shuffle(valid_candidates)
            
            for x, new_sums in valid_candidates:
                if time.time() > deadline:
                    return
                
                seq.append(x)
                sums_set.update(new_sums)
                
                dfs(seq, sums_set, x, depth + 1, deadline)
                
                # Backtrack
                seq.pop()
                for s in new_sums:
                    sums_set.remove(s)
                
                # If we found a longer sequence, we might stop this branch?
                # No, we want to keep searching for even longer ones.
                # But if we hit the time limit, we return.
        
        # Multi-start
        # We can start with different initial "seeds" or just randomize the DFS.
        # Since the DFS is deterministic if we don't shuffle, shuffling makes it explore different paths.
        
        # To make it more effective, we can run multiple independent DFS runs with different random seeds.
        
        for i in range(1000): # Max runs, but limited by time
            if time.time() > start_time + time_limit:
                break
            
            # Reset best_seq? No, we want the global best.
            # But we should probably clear the current path.
            # The DFS starts from empty.
            
            # To diversify, we can bias the shuffle?
            # Or just let random.shuffle do its work.
            
            dfs([], set(), 0, 0, start_time + time_limit)
            
            # If best_seq is very long, maybe we can stop?
            # No, keep trying.
            
        return best_seq
