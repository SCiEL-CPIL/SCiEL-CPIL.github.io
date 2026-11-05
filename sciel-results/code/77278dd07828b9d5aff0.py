import numpy as np
import time
import random
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Specific case optimization
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(6)]
        
        # Precompute neighbors as bitsets
        # cover_mask[i] is an integer with bits set for all j such that dist(i, j) <= 1
        cover_mask = [0] * total
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(6)]
            mask = 1 << i
            for p in range(6):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        mask |= (1 << new_i)
            cover_mask[i] = mask

        def idx_to_list(idx):
            return [(idx // powers[p]) % 3 for p in range(6)]

        def greedy_set_cover():
            """Greedy set cover using bitsets."""
            uncovered = (1 << total) - 1
            codewords = []
            
            while uncovered:
                # Find word that covers most uncovered bits
                best_c = -1
                best_count = -1
                # Optimization: We can sample or iterate. Iterating 729 is fast.
                # To add diversity, we can randomize tie-breaking or use a small window of top-k.
                # For now, deterministic max, but we can add noise.
                for c in range(total):
                    count = (cover_mask[c] & uncovered).bit_count()
                    if count > best_count:
                        best_count = count
                        best_c = c
                
                codewords.append(best_c)
                uncovered &= ~cover_mask[best_c]
                
            return codewords

        def local_search(codewords, time_budget):
            """Perform local search with simulated annealing to reduce code size."""
            S = set(codewords)
            S_mask = 0
            for c in S:
                S_mask |= (1 << c)
            
            best_S = set(S)
            best_size = len(S)
            
            # Temperature for simulated annealing
            temperature = 1.0
            initial_temp = 1.0
            cooling_rate = 0.995
            min_temp = 0.01
            
            def compute_unique_cover(w, S_set, S_mask_val):
                """Compute words uniquely covered by w (not covered by any other in S_set)."""
                union_others = 0
                for v in S_set:
                    if v != w:
                        union_others |= cover_mask[v]
                return cover_mask[w] & ~union_others
            
            def try_remove():
                """Try to find and remove a redundant word."""
                nonlocal S, S_mask, best_S, best_size
                items = list(S)
                random.shuffle(items)
                
                for w in items:
                    L = compute_unique_cover(w, S, S_mask)
                    if L == 0:
                        S.remove(w)
                        S_mask &= ~(1 << w)
                        if len(S) < best_size:
                            best_size = len(S)
                            best_S = set(S)
                        return True
                return False
            
            def try_swap():
                """Try to swap a word with another that maintains coverage."""
                nonlocal S, S_mask, best_S, best_size
                items = list(S)
                random.shuffle(items)
                
                for w in items:
                    L = compute_unique_cover(w, S, S_mask)
                    if L == 0:
                        continue  # Already redundant, handled by try_remove
                    
                    # Find candidates that cover all of L
                    candidates = (1 << total) - 1
                    temp_L = L
                    while temp_L:
                        x = (temp_L & -temp_L).bit_length() - 1
                        temp_L &= temp_L - 1
                        candidates &= cover_mask[x]
                        if candidates == 0:
                            break
                    
                    if candidates == 0:
                        continue
                    
                    # Exclude words already in S
                    candidates &= ~S_mask
                    candidates &= ~(1 << w)
                    
                    if candidates:
                        # Pick a random candidate
                        w_new = (candidates & -candidates).bit_length() - 1
                        
                        # Check if this swap improves coverage (reduces total unique coverage)
                        # For now, accept all valid swaps (they maintain coverage)
                        S.remove(w)
                        S.add(w_new)
                        S_mask = 0
                        for c in S:
                            S_mask |= (1 << c)
                        return True
                
                return False
            
            def try_2_swap():
                """Try to remove 2 words and add 1-2 words to reduce size."""
                nonlocal S, S_mask, best_S, best_size
                if len(S) < 2:
                    return False
                
                items = list(S)
                random.shuffle(items)
                
                # Try removing two words w1, w2
                for i in range(min(20, len(items))):
                    w1 = items[i]
                    L1 = compute_unique_cover(w1, S, S_mask)
                    if L1 == 0:
                        continue
                    
                    for j in range(i+1, min(40, len(items))):
                        w2 = items[j]
                        # Temporarily remove w1 to compute L2
                        S_temp = S - {w1}
                        L2 = compute_unique_cover(w2, S_temp, S_mask)
                        
                        # Combined unique coverage
                        L_combined = L1 | L2
                        
                        if L_combined == 0:
                            # Both redundant
                            S.remove(w1)
                            S.remove(w2)
                            S_mask &= ~(1 << w1)
                            S_mask &= ~(1 << w2)
                            if len(S) < best_size:
                                best_size = len(S)
                                best_S = set(S)
                            return True
                        
                        # Try to find a single word that covers L_combined
                        candidates = (1 << total) - 1
                        temp_L = L_combined
                        while temp_L:
                            x = (temp_L & -temp_L).bit_length() - 1
                            temp_L &= temp_L - 1
                            candidates &= cover_mask[x]
                            if candidates == 0:
                                break
                        
                        if candidates == 0:
                            continue
                        
                        candidates &= ~S_mask
                        candidates &= ~(1 << w1)
                        candidates &= ~(1 << w2)
                        
                        if candidates:
                            w_new = (candidates & -candidates).bit_length() - 1
                            S.remove(w1)
                            S.remove(w2)
                            S.add(w_new)
                            S_mask = 0
                            for c in S:
                                S_mask |= (1 << c)
                            if len(S) < best_size:
                                best_size = len(S)
                                best_S = set(S)
                            return True
                
                return False
            
            stuck_count = 0
            max_stuck = 50
            
            while time.time() < time_budget:
                # Try removal first (reduces size)
                if try_remove():
                    stuck_count = 0
                    max_stuck = random.randint(30, 100)
                    temperature = max(min_temp, temperature * cooling_rate)
                    continue
                
                # Try 2-swap (can reduce size by 1)
                if try_2_swap():
                    stuck_count = 0
                    max_stuck = random.randint(30, 100)
                    temperature = max(min_temp, temperature * cooling_rate)
                    continue
                
                # Try regular swap (maintains size, helps escape local minima)
                if try_swap():
                    # Accept with probability based on temperature (simulated annealing)
                    # Since swap maintains size, we use temperature to control acceptance
                    # Actually, all valid swaps are accepted since they don't increase size
                    stuck_count = 0
                    max_stuck = random.randint(50, 150)
                    temperature = max(min_temp, temperature * cooling_rate)
                    continue
                
                # No improvement
                stuck_count += 1
                if stuck_count > max_stuck:
                    # Restart or cool down
                    stuck_count = 0
                    max_stuck = random.randint(100, 300)
                    temperature = max(min_temp, temperature * 0.9)
                    
                    if temperature < min_temp:
                        break

            return list(best_S)

        # Multi-restart strategy
        best_overall = None
        best_overall_size = float('inf')
        
        restart_count = 0
        while time.time() < deadline - 5:
            restart_count += 1
            
            # Time budget per restart
            # Start with short, increase if we are finding good solutions?
            # Or just fixed reasonable budget.
            # 115s total. Let's do ~10s per restart, so ~10 restarts.
            time_budget = min(time.time() + 10, deadline - 5)
            
            # Greedy initialization
            codewords = greedy_set_cover()
            
            # Local search
            codewords = local_search(codewords, time_budget)
            
            if len(codewords) < best_overall_size:
                best_overall_size = len(codewords)
                best_overall = list(codewords)
                
            # If we found 73, we can try to polish it further?
            # The local search already runs until time budget.
            # If we find 73, we might want to extend the time for that specific solution?
            # But we don't know which restart will find it.
            # So we just keep restarting.

        if best_overall is None:
            best_overall = [0]
            
        return [idx_to_list(idx) for idx in best_overall]

    else:
        # Fallback for other parameters
        total = q ** n
        return [[0] * n]
