"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # 2s safety margin
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    def count_bad_triples_for(x, selected_set):
        """
        Count how many bad triples involve x.
        A bad triple is (a, b, x) where a, b are in selected_set 
        and gcd(a,b) == gcd(a,x) == gcd(b,x).
        """
        count = 0
        lst = list(selected_set)
        k = len(lst)
        for i in range(k):
            a = lst[i]
            g_ax = gcd_table[a][x]
            for j in range(i+1, k):
                b = lst[j]
                g_bx = gcd_table[b][x]
                if g_ax == g_bx:
                    if gcd_table[a][b] == g_ax:
                        count += 1
        return count

    def greedy_removal(seed=0):
        """
        Start with all elements, iteratively remove the element 
        involved in the most bad triples.
        """
        selected = set(range(1, N+1))
        rng = random.Random(seed)
        
        while True:
            # Count bad triples for each element in selected
            counts = {}
            for x in selected:
                counts[x] = count_bad_triples_for(x, selected)
                
            # Find max count
            max_count = max(counts.values()) if counts else 0
            
            if max_count == 0:
                break
            
            # Find all elements with max count
            candidates = [x for x, c in counts.items() if c == max_count]
            
            # Pick one to remove
            if seed == 0:
                to_remove = min(candidates)
            elif seed == 1:
                to_remove = max(candidates)
            else:
                to_remove = rng.choice(candidates)
                
            selected.remove(to_remove)
            
            # Early termination if time is up
            if time.time() > deadline - 8:
                break
                
        return selected

    def improved_local_search(selected, time_limit):
        """
        More effective local search: try multiple swap strategies.
        """
        best = set(selected)
        best_size = len(best)
        
        # Helper: try to greedily add elements to a set
        def try_add_elements(base_set, candidates):
            """Try to add elements from candidates to base_set, one by one."""
            result = set(base_set)
            for x in candidates:
                if x not in result and count_bad_triples_for(x, result) == 0:
                    result.add(x)
            return result
        
        while time.time() < time_limit:
            improved = False
            
            # Strategy 1: Try to add any single element
            for x in range(1, N+1):
                if x in best:
                    continue
                if count_bad_triples_for(x, best) == 0:
                    best.add(x)
                    best_size = len(best)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Strategy 2: Try removing one element and adding two
            for x in sorted(best):
                if time.time() > time_limit:
                    break
                temp = best - {x}
                
                # Find all elements that can be added to temp
                addable = []
                for y in range(1, N+1):
                    if y in temp or y == x:
                        continue
                    if count_bad_triples_for(y, temp) == 0:
                        addable.append(y)
                
                # Try pairs from addable
                for i in range(len(addable)):
                    if time.time() > time_limit:
                        break
                    y = addable[i]
                    temp_with_y = temp | {y}
                    
                    for j in range(i+1, len(addable)):
                        if time.time() > time_limit:
                            break
                        z = addable[j]
                        if count_bad_triples_for(z, temp_with_y) == 0:
                            new_set = temp_with_y | {z}
                            if len(new_set) > best_size:
                                best = new_set
                                best_size = len(best)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            # Strategy 3: Try removing two elements and adding three
            if not improved:
                best_list = sorted(best)
                for i in range(len(best_list)):
                    if time.time() > time_limit:
                        break
                    x1 = best_list[i]
                    for j in range(i+1, len(best_list)):
                        if time.time() > time_limit:
                            break
                        x2 = best_list[j]
                        temp = best - {x1, x2}
                        
                        # Find all elements that can be added to temp
                        addable = []
                        for y in range(1, N+1):
                            if y in temp or y in (x1, x2):
                                continue
                            if count_bad_triples_for(y, temp) == 0:
                                addable.append(y)
                        
                        # Try triples from addable
                        found = False
                        for a in range(len(addable)):
                            if found or time.time() > time_limit:
                                break
                            y1 = addable[a]
                            temp_with_y1 = temp | {y1}
                            
                            for b in range(a+1, len(addable)):
                                if found or time.time() > time_limit:
                                    break
                                y2 = addable[b]
                                temp_with_y1y2 = temp_with_y1 | {y2}
                                
                                for c in range(b+1, len(addable)):
                                    if found or time.time() > time_limit:
                                        break
                                    y3 = addable[c]
                                    if count_bad_triples_for(y3, temp_with_y1y2) == 0:
                                        new_set = temp_with_y1y2 | {y3}
                                        if len(new_set) > best_size:
                                            best = new_set
                                            best_size = len(best)
                                            improved = True
                                            found = True
                                            break
                        
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
                
        return best

    best_result = set()
    
    # Run multiple greedy removals with different seeds
    for seed in range(30):
        if time.time() > deadline - 12:
            break
        
        current = greedy_removal(seed)
        
        if len(current) > len(best_result):
            best_result = current
        
        # If we have time, do local search on the best one found so far
        if time.time() < deadline - 6:
            improved = improved_local_search(best_result, deadline - 3)
            if len(improved) > len(best_result):
                best_result = improved

    # Final local search on the absolute best
    if time.time() < deadline - 1:
        final_result = improved_local_search(best_result, deadline - 0.5)
        if len(final_result) > len(best_result):
            best_result = final_result

    if not best_result:
        best_result = {1}
        
    return list(best_result)

# EVOLVE-BLOCK-END
