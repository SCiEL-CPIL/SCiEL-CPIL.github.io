import time
import sys

# Increase recursion limit just in case, though depth is limited by 256
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin below 60s
    
    best_seq = []
    
    # Helper to check if a sequence is valid and return its sums
    # This is used for the initial greedy result as a baseline
    def get_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    # Initial greedy solution as baseline
    values, sums = [], set()
    for x in range(1, 257):
        tails = []
        total = 0
        # Calculate sums ending at x
        # Actually, for greedy, we just check if adding x creates a conflict with existing tails
        # But the standard greedy in the prompt is:
        # for x in 1..256:
        #   tails = [x]
        #   total = x
        #   for a in reversed(values):
        #      total += a
        #      tails.append(total)
        #   if no conflict: add x
        
    # Let's re-implement the baseline quickly to have a starting point
    base_values, base_sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(base_values):
            total += a
            tails.append(total)
        if not base_sums.intersection(tails):
            base_values.append(x)
            base_sums.update(tails)
            best_seq = base_values[:]
            
    def dfs(current_seq, current_sums, next_min):
        nonlocal best_seq, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        # Update best if current is longer
        if len(current_seq) > len(best_seq):
            best_seq = current_seq[:]
            
        # Pruning: If we can't fit any more elements, return
        # Max possible length is bounded by 256, but realistically much smaller
        # If next_min > 256, return
        if next_min > 256:
            return
            
        # Try to add a new element
        for x in range(next_min, 257):
            # Calculate new sums ending at x
            new_sums = []
            total = x
            new_sums.append(total)
            # Add sums with previous elements
            for a in reversed(current_seq):
                total += a
                new_sums.append(total)
                
            # Check for conflicts
            conflict = False
            for s in new_sums:
                if s in current_sums:
                    conflict = True
                    break
                    
            if conflict:
                continue
                
            # No conflict, proceed
            # Update state
            current_seq.append(x)
            current_sums.update(new_sums)
            
            dfs(current_seq, current_sums, x + 1)
            
            # Backtrack
            current_seq.pop()
            for s in new_sums:
                current_sums.remove(s)
                
            # Early exit if time limit reached
            if time.time() - start_time > time_limit:
                return

    dfs(base_values, base_sums, base_values[-1] + 1 if base_values else 1)
    
    return best_seq
