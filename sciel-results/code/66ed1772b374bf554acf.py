def construct():
            import time
            import numpy as np
            import random

            start_time = time.time()
            deadline = start_time + 85.0
            
            N = 17
            M = 1 << N
            min_dist = 6
            max_radius = min_dist - 1  # 5
            
            # Precompute offsets (masks) of weight 1 to max_radius
            # This is done once.
            # Number of masks is ~9400.
            offsets_list = []
            # Generate masks
            # We can use itertools.combinations for positions
            from itertools import combinations
            for r in range(1, max_radius + 1):
                for comb in combinations(range(N), r):
                    mask = 0
                    for pos in comb:
                        mask |= (1 << pos)
                    offsets_list.append(mask)
            offsets = np.array(offsets_list, dtype=np.uint32)
            
            best_code = []
            best_size = 0
            
            # Multiple restarts
            # The greedy choice order affects the result.
            # We can shuffle the initial available_list.
            
            for restart in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                # Reset state
                conflict_counts = np.zeros(M, dtype=np.int16)
                available_mask = np.ones(M, dtype=bool)
                available_list = list(range(M))
                random.shuffle(available_list)
                
                code = []
                
                # Main loop
                # We pop from available_list.
                # Lazy deletion: if popped item is not in available_mask, skip.
                
                idx = 0
                # We can't just use a while loop with pop(0) as it's O(N).
                # Use a pointer or a heap?
                # Actually, since we just want *any* valid word, and the list is shuffled,
                # we can just iterate through the shuffled list?
                # No, because items get invalidated.
                # Using a list and a pointer `idx` is good.
                # If available_list[idx] is invalid, increment idx.
                # This is O(1) amortized per check.
                
                while idx < len(available_list) and time.time() < deadline - 5.0:
                    x = available_list[idx]
                    idx += 1
                    
                    if not available_mask[x]:
                        continue
                    
                    # Accept x
                    code.append(x)
                    available_mask[x] = False
                    
                    # Update conflicts
                    # neighbors = x ^ offsets
                    # x is scalar, offsets is array.
                    neighbors = x ^ offsets
                    # Update counts
                    conflict_counts[neighbors] += 1
                    # Find newly blocked
                    newly_blocked_mask = (conflict_counts[neighbors] == 1)
                    newly_blocked = neighbors[newly_blocked_mask]
                    if len(newly_blocked) > 0:
                        available_mask[newly_blocked] = False
                        
                if len(code) > best_size:
                    best_size = len(code)
                    best_code = code[:]
                    
            return best_code
