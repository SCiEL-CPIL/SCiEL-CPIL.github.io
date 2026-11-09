"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 80.0
    
    N = 4096
    
    def try_add(x, selected, triple_sums):
        """Check if x can be added. Returns (ok, new_sums_list)."""
        n = len(selected)
        new_sums = set()
        
        # Triples with one x: x + a_i + a_j for i <= j
        for i in range(n):
            ai = selected[i]
            for j in range(i, n):
                s = x + ai + selected[j]
                if s in new_sums or s in triple_sums:
                    return False, []
                new_sums.add(s)
        
        # Triples with two x's: 2*x + a_i
        for i in range(n):
            s = 2 * x + selected[i]
            if s in new_sums or s in triple_sums:
                return False, []
            new_sums.add(s)
        
        # Triple with three x's
        s = 3 * x
        if s in new_sums or s in triple_sums:
            return False, []
        new_sums.add(s)
        
        return True, list(new_sums)
    
    def build_greedy(start=1, step=1):
        """Greedy: add numbers in order with given start and step."""
        selected = []
        triple_sums = set()
        
        for x in range(start, N + 1, step):
            if time.time() > DEADLINE - 10:
                break
            ok, new_sums = try_add(x, selected, triple_sums)
            if ok:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected
    
    def compute_triple_sums(arr):
        """Compute all unordered triple sums with repetition."""
        m = len(arr)
        sums = set()
        for i in range(m):
            for j in range(i, m):
                for k in range(j, m):
                    sums.add(arr[i] + arr[j] + arr[k])
        return sums
    
    # Phase 1: Basic greedy from 1
    result = build_greedy(1, 1)
    best_size = len(result)
    
    # Phase 2: Try various greedy strategies
    if time.time() < DEADLINE - 15:
        strategies = [
            (1, 2), (1, 3), (2, 1), (2, 2), (3, 1), (3, 3),
            (1, 4), (5, 1), (7, 1), (1, 5)
        ]
        for start, step in strategies:
            if time.time() > DEADLINE - 15:
                break
            candidate = build_greedy(start, step)
            if len(candidate) > best_size:
                result = candidate
                best_size = len(candidate)
    
    # Phase 3: Local search - remove one element, greedily refill
    if time.time() < DEADLINE - 12:
        for _ in range(3):
            if time.time() > DEADLINE - 12:
                break
            improved = False
            for i in range(len(result)):
                if time.time() > DEADLINE - 12:
                    break
                temp = result[:i] + result[i+1:]
                triple_sums = compute_triple_sums(temp)
                
                new_list = list(temp)
                for x in range(1, N + 1):
                    if x in new_list:
                        continue
                    if time.time() > DEADLINE - 12:
                        break
                    ok, new_sums = try_add(x, new_list, triple_sums)
                    if ok:
                        new_list.append(x)
                        triple_sums.update(new_sums)
                
                if len(new_list) > best_size:
                    result = new_list
                    best_size = len(result)
                    improved = True
                    break
    
    return result
# EVOLVE-BLOCK-END
