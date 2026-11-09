"""Rewritten program: maximize valid subset of [1,256] where (a*b)%(a+b)!=0 for all distinct a,b."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s kill
    
    # Precompute conflict matrix: conflict[i][j] = True if (i*j) % (i+j) == 0
    n = 256
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict[i][j] = True
                conflict[j][i] = True
    
    def is_valid_set(indices):
        """Check if the set of indices is valid (no conflicts)."""
        s = set(indices)
        for i in s:
            for j in s:
                if i != j and conflict[i][j]:
                    return False
        return True
    
    def count_conflicts(indices):
        """Count number of conflicting pairs in the set."""
        cnt = 0
        s = list(indices)
        for k in range(len(s)):
            for m in range(k + 1, len(s)):
                if conflict[s[k]][s[m]]:
                    cnt += 1
        return cnt
    
    def greedy_build(order):
        """Greedily add elements in the given order if they don't conflict with existing set."""
        selected = []
        selected_set = set()
        for idx in order:
            can_add = True
            for s in selected_set:
                if conflict[idx][s]:
                    can_add = False
                    break
            if can_add:
                selected.append(idx)
                selected_set.add(idx)
        return selected, selected_set
    
    def local_search(selected, selected_set, max_iters=5000):
        """Try to improve the set by swapping or adding/removing elements."""
        best_size = len(selected)
        selected = list(selected)
        selected_set = set(selected)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try adding an element
            for idx in range(n):
                if idx in selected_set:
                    continue
                can_add = True
                for s in selected_set:
                    if conflict[idx][s]:
                        can_add = False
                        break
                if can_add:
                    selected.append(idx)
                    selected_set.add(idx)
                    best_size = len(selected)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            for rem_idx in list(selected_set):
                for add_idx in range(n):
                    if add_idx in selected_set:
                        continue
                    # Check if add_idx conflicts with any remaining element
                    new_set = selected_set - {rem_idx}
                    can_add = True
                    for s in new_set:
                        if conflict[add_idx][s]:
                            can_add = False
                            break
                    if can_add:
                        selected_set = new_set | {add_idx}
                        selected = list(selected_set)
                        if len(selected_set) > best_size:
                            best_size = len(selected_set)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return selected, selected_set, best_size
    
    # Strategy 1: Power-of-2 prioritized greedy (baseline)
    # Powers of 2 have special properties: if a is power of 2, a+b divides a*b only in specific cases
    # Let's try multiple orderings
    
    best_result = None
    best_size = 0
    
    # Order 1: All odd numbers first (they're the seed)
    odds = [i for i in range(n) if (i + 1) % 2 == 1]
    evens = [i for i in range(n) if (i + 1) % 2 == 0]
    order1 = odds + evens
    sel1, set1 = greedy_build(order1)
    sel1, set1, sz1 = local_search(sel1, set1, max_iters=3000)
    if sz1 > best_size:
        best_size = sz1
        best_result = list(set1)
    
    # Order 2: Powers of 2 first, then others
    powers_of_2 = [i for i in range(n) if (i + 1) & (i) == 0]  # i+1 is power of 2
    others = [i for i in range(n) if i not in set(powers_of_2)]
    order2 = powers_of_2 + others
    sel2, set2 = greedy_build(order2)
    sel2, set2, sz2 = local_search(sel2, set2, max_iters=3000)
    if sz2 > best_size:
        best_size = sz2
        best_result = list(set2)
    
    # Order 3: Descending order
    order3 = list(range(n - 1, -1, -1))
    sel3, set3 = greedy_build(order3)
    sel3, set3, sz3 = local_search(sel3, set3, max_iters=3000)
    if sz3 > best_size:
        best_size = sz3
        best_result = list(set3)
    
    # Order 4: Ascending order
    order4 = list(range(n))
    sel4, set4 = greedy_build(order4)
    sel4, set4, sz4 = local_search(sel4, set4, max_iters=3000)
    if sz4 > best_size:
        best_size = sz4
        best_result = list(set4)
    
    # Randomized restarts
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Random ordering with some bias
        order = list(range(n))
        random.shuffle(order)
        
        sel, sel_set = greedy_build(order)
        sel, sel_set, sz = local_search(sel, sel_set, max_iters=2000)
        if sz > best_size:
            best_size = sz
            best_result = list(sel_set)
    
    # More targeted local search on best result found
    if best_result is not None:
        best_set = set(best_result)
        best_result, best_set, final_size = local_search(best_result, best_set, max_iters=10000)
        if final_size > best_size:
            best_size = final_size
    
    # Final: try to add any remaining elements
    if best_result is not None:
        best_set = set(best_result)
        for idx in range(n):
            if idx in best_set:
                continue
            can_add = True
            for s in best_set:
                if conflict[idx][s]:
                    can_add = False
                    break
            if can_add:
                best_result.append(idx)
                best_set.add(idx)
    
    if best_result is None or len(best_result) == 0:
        # Fallback: just odd numbers
        return list(range(1, 257, 2))
    
    # Convert indices back to values (index i corresponds to value i+1)
    result = [i + 1 for i in best_result]
    result.sort()
    
    return result
