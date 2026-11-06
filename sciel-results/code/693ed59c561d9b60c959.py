import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(x, S_set):
        """Check if x can be added to S_set while maintaining sum-free property."""
        for a in S_set:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
        if 2 * x in S_set:
            return False
        return True

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_add(candidates, current_set, rng):
        """Greedy add elements from candidates to current_set."""
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        for e in shuffled:
            if can_add(e, current_set):
                current_set.add(e)
        return current_set

    def local_search(current_set, S, rng, max_iters=30):
        """Local search: try removing elements and greedily adding back."""
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            improved = False
            elements_to_try = list(current_set)
            rng.shuffle(elements_to_try)
            
            for u in elements_to_try:
                if time.time() > deadline - 10:
                    break
                    
                test_set = set(current_set)
                test_set.remove(u)
                
                # Try to add back any element from S not in test_set
                candidates = [e for e in S if e not in test_set]
                rng.shuffle(candidates)
                
                for e in candidates:
                    if can_add(e, test_set):
                        test_set.add(e)
                        
                if len(test_set) > len(current_set):
                    current_set = test_set
                    improved = True
                    break
                    
            if not improved:
                break
                
        return current_set

    def solve_instance(S):
        S_set = set(S)
        odds = [x for x in S if x % 2 != 0]
        evens = [x for x in S if x % 2 == 0]
        
        best_set = set(odds)
        best_size = len(best_set)
        
        num_restarts = 15
        rng = Random(42)
        
        for r in range(num_restarts):
            if time.time() > deadline - 15:
                break
                
            current_set = set(odds)
            
            # Greedy add evens
            shuffled_evens = evens[:]
            rng.shuffle(shuffled_evens)
            
            for e in shuffled_evens:
                if time.time() > deadline - 15:
                    break
                if can_add(e, current_set):
                    current_set.add(e)
            
            # Local search
            current_set = local_search(current_set, S, rng, max_iters=20)
            
            if len(current_set) > best_size:
                best_set = current_set
                best_size = len(best_set)
                
        return list(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Fallback if time is tight
        if time.time() > deadline - 10:
             results.append([x for x in S if x % 2])
             continue
             
        res = solve_instance(S)
        
        # Verify
        if not is_sum_free(res):
            # Fallback to parity
            res = [x for x in S if x % 2]
            
        results.append(res)
        
    return results
