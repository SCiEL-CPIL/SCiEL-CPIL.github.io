import time
import sys
sys.setrecursionlimit(10000)

def construct():
        start_time = time.time()
        deadline = start_time + 110
        
        fixed = [1, 2, 5, 11]
        
        def get_sum_mask(elements):
            mask = 0
            for i in range(len(elements)):
                for j in range(i, len(elements)):
                    s = elements[i] + elements[j]
                    mask |= (1 << s)
            return mask

        # 1. Greedy to find a good lower bound
        def greedy():
            current = list(fixed)
            mask = get_sum_mask(current)
            candidates = [x for x in range(1, 285) if x not in fixed]
            
            # Sort by some heuristic? 
            # Let's try sorting by "popularity" of sums? 
            # Or just random restarts.
            import random
            random.seed(1)
            best_greedy = current[:]
            best_mask = mask
            
            for _ in range(100): # 100 restarts
                cur = list(fixed)
                m = get_sum_mask(fixed)
                cands = [x for x in range(1, 285) if x not in fixed]
                random.shuffle(cands)
                
                for x in cands:
                    if time.time() > deadline:
                        break
                    valid = True
                    if (m >> (x+x)) & 1:
                        valid = False
                    else:
                        for y in cur:
                            if (m >> (x+y)) & 1:
                                valid = False
                                break
                    if valid:
                        cur.append(x)
                        m |= (1 << (x+x))
                        for y in cur[:-1]:
                            m |= (1 << (x+y))
                
                if len(cur) > len(best_greedy):
                    best_greedy = cur
                    best_mask = m
                    
            return best_greedy, best_mask

        greedy_sol, greedy_mask = greedy()
        best_size = len(greedy_sol)
        best_set = list(greedy_sol)
        
        # 2. Branch and Bound
        candidates = [x for x in range(1, 285) if x not in fixed]
        
        # Use a global list for current elements to avoid copying
        current_elements = list(fixed)
        
        def search(current_mask, current_count, start_idx, candidates_list):
            nonlocal best_size, best_set
            
            if time.time() > deadline:
                return

            remaining = len(candidates_list) - start_idx
            if current_count + remaining <= best_size:
                return

            if current_count > best_size:
                best_size = current_count
                best_set = list(current_elements)
                
            for i in range(start_idx, len(candidates_list)):
                if time.time() > deadline:
                    return
                
                x = candidates_list[i]
                
                # Check validity
                if (current_mask >> (x+x)) & 1:
                    continue
                    
                valid = True
                new_sums = []
                for y in current_elements:
                    s = x + y
                    if (current_mask >> s) & 1:
                        valid = False
                        break
                    new_sums.append(s)
                
                if not valid:
                    continue
                
                # Add x
                current_elements.append(x)
                new_mask = current_mask | (1 << (x+x))
                for s in new_sums:
                    new_mask |= (1 << s)
                    
                search(new_mask, current_count + 1, i + 1, candidates_list)
                
                # Backtrack
                current_elements.pop()

        # Try sorted candidates
        search(greedy_mask, len(fixed), 0, sorted(candidates))
        
        return best_set
