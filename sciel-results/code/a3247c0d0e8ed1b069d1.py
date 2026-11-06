import time
import sys

def construct():
        start_time = time.time()
        deadline = start_time + 50  # 50s safety margin
        
        best_seq = []
        
        # Initial greedy to get a baseline
        # Greedy: pick smallest valid x
        curr_seq = []
        curr_sums = set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(curr_seq):
                total += a
                tails.append(total)
            if not curr_sums.intersection(tails):
                curr_seq.append(x)
                curr_sums.update(tails)
        best_seq = curr_seq[:]
        
        # DFS to improve
        # We will try to extend the greedy solution or find better ones.
        # Actually, let's run a DFS from scratch but with pruning.
        # To speed up, we can use the greedy solution as a lower bound for pruning.
        
        def dfs(seq, sums, last_val):
            nonlocal best_seq
            if time.time() > deadline:
                return
            
            if len(seq) > len(best_seq):
                best_seq = seq[:]
                
            # Pruning: 
            # Max possible additional elements?
            # If we add k elements, we need k*(k+1)/2 new sums + k*len(seq) cross sums.
            # Total sums will be (len(seq)+k)*(len(seq)+k+1)/2.
            # These must fit in the range of possible sums.
            # Max sum is sum(seq) + sum(next k elements).
            # Min sum is min(seq).
            # This bound is complex. A simple heuristic:
            # If len(seq) + (256 - last_val) <= len(best_seq): return
            if len(seq) + (256 - last_val) <= len(best_seq):
                return
            
            # Try next values
            # Heuristic: Try values that are "far" from existing sums?
            # Or just iterate from last_val + 1 to 256.
            # To find longer sequences, smaller values might be better as they allow more numbers.
            # But powers of 2 work.
            # Let's iterate x from last_val + 1 to 256.
            
            for x in range(last_val + 1, 257):
                if time.time() > deadline:
                    return
                
                # Calculate new sums
                # Sums ending at x
                new_sums = []
                total = x
                new_sums.append(total)
                for a in reversed(seq):
                    total += a
                    new_sums.append(total)
                    
                # Check for conflicts
                conflict = False
                for s in new_sums:
                    if s in sums:
                        conflict = True
                        break
                        
                if not conflict:
                    # Update sums
                    # Creating a new set is expensive. 
                    # We can use a set and update, but we need to backtrack.
                    # Since we are in Python, copying a set of size ~50-100 is fast.
                    new_sums_set = sums.union(new_sums)
                    
                    dfs(seq + [x], new_sums_set, x)
                    
                    # Backtrack is implicit due to passing new objects
                    # But creating new lists and sets is O(N).
                    # For N=10, it's fine.
        
        # Start DFS
        # To get a good start, we can start from empty, but that's slow.
        # We can start from the greedy solution and try to modify/extend?
        # Or just run DFS from empty with the pruning.
        # The greedy solution is length ~9.
        # DFS from empty:
        # Depth 1: 256 options
        # Depth 2: ~250 options
        # ...
        # This might be too slow if branching factor is high.
        
        # Better strategy:
        # 1. Run Greedy to get baseline.
        # 2. Use Local Search / Hill Climbing?
        #    - Start with greedy sequence.
        #    - Try to swap elements to free up space?
        #    - Or insert elements?
        
        # Let's try a smarter DFS:
        # Instead of starting from empty, we can use the fact that the sequence is increasing.
        # We can fix the first few elements to small values to allow more elements later.
        
        # Let's try to find a sequence of length 10.
        # If DFS is too slow, we might need to optimize the check.
        
        # Optimization:
        # The check `if s in sums` is O(1).
        # The bottleneck is the recursion depth and branching.
        # Let's limit the search to finding length > len(best_seq).
        
        # Let's try a different approach:
        # Since the range is small (256), we can use bitmasks? No, sums are up to ~2000.
        
        # Let's stick to DFS but with a strong heuristic.
        # Heuristic: Prefer smaller x?
        # If we prefer smaller x, we might get stuck in a local minimum of "small numbers".
        # But small numbers have smaller sums, which might collide more?
        # Actually, powers of 2 are "large" relative to their position.
        
        # Let's try to run the DFS from the greedy result, trying to replace the last element with a smaller one to allow more extensions?
        # Or just run a full DFS with a time limit.
        
        # To make DFS faster, we can pre-compute valid transitions?
        # No, sums depend on the whole prefix.
        
        # Let's try this:
        # 1. Get greedy solution.
        # 2. Try to extend it by inserting elements?
        #    - The sequence must be strictly increasing.
        #    - We can try to insert a new element $x$ between $a_i$ and $a_{i+1}$.
        #    - This changes all sums involving $x$ and subsequent elements.
        #    - This is complex.
        
        # Alternative:
        # Randomized Local Search.
        # 1. Start with greedy.
        # 2. Randomly pick an index $i$ and a new value $x \in (a_{i-1}, a_{i+1})$.
        # 3. Check if the new sequence is valid.
        # 4. If valid and longer (if we added) or same length (if we swapped), keep it?
        #    - We want to maximize length.
        #    - So we should try to ADD elements.
        #    - Start with a short sequence, try to add.
        
        # Let's combine:
        # 1. Greedy to get base.
        # 2. DFS with pruning, but start from a "promising" prefix.
        #    - What is a promising prefix?
        #    - Small numbers?
        #    - Let's try DFS from scratch but with a limit on the number of nodes visited, and random restarts if needed?
        #    - No, deterministic is better for reproducibility in scoring.
        
        # Let's optimize the DFS check.
        # Instead of creating a new set every time, we can pass the set and update it in place?
        # Python sets are mutable.
        # `sums.update(new_sums)`
        # `dfs(...)`
        # `sums.difference_update(new_sums)`
        # This avoids copying the set.
        
        def dfs_fast(seq, sums, last_val):
            nonlocal best_seq
            if time.time() > deadline:
                return
            
            if len(seq) > len(best_seq):
                best_seq = seq[:]
                
            if len(seq) + (256 - last_val) <= len(best_seq):
                return
            
            for x in range(last_val + 1, 257):
                if time.time() > deadline:
                    return
                
                # Check conflicts
                new_sums_list = []
                total = x
                new_sums_list.append(total)
                for a in reversed(seq):
                    total += a
                    new_sums_list.append(total)
                    
                conflict = False
                for s in new_sums_list:
                    if s in sums:
                        conflict = True
                        break
                        
                if not conflict:
                    # Add to set
                    sums.update(new_sums_list)
                    seq.append(x)
                    
                    dfs_fast(seq, sums, x)
                    
                    # Backtrack
                    seq.pop()
                    sums.difference_update(new_sums_list)
                    
        # Start with empty
        dfs_fast([], set(), 0)
        
        return best_seq
