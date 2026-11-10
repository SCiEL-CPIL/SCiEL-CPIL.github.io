import time
import random
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    best_set = []
    best_len = 0
    
    def can_add(chosen, counts, x):
        """Check if x can be added without violating B2[2] constraint."""
        if counts[2 * x] >= 2:
            return False
        for y in chosen:
            if counts[x + y] == 2:
                return False
        return True
        
    def add_element(chosen, counts, x):
        """Add x to chosen set and update counts."""
        for y in chosen:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen.append(x)
        
    def remove_element(chosen, counts, x):
        """Remove x from chosen set and update counts."""
        for y in chosen:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen.remove(x)
        
    def greedy_with_order(order='random', seed=None):
        """Run greedy construction with specified ordering."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
            
        if order == 'random':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif order == 'squares':
            squares = [i*i for i in range(1, int(np.sqrt(N)) + 1) if i*i <= N]
            others = [x for x in range(1, N + 1) if x not in squares]
            rng.shuffle(others)
            candidates = squares + others
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
        chosen = []
        counts = [0] * (max_sum + 1)
        
        for x in candidates:
            if can_add(chosen, counts, x):
                add_element(chosen, counts, x)
                
        return chosen, counts
        
    def local_search(chosen, counts, max_rounds=100, time_limit=None, start_time=None):
        """Improved local search."""
        chosen_set = set(chosen)
        improved = True
        rounds = 0
        no_improve_count = 0
        
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            no_improve_count += 1
            
            if time_limit and start_time:
                if time.time() - start_time > time_limit:
                    break
                    
            # 1. Try to add any element
            for x in range(1, N + 1):
                if x not in chosen_set and can_add(chosen, counts, x):
                    add_element(chosen, counts, x)
                    chosen_set.add(x)
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue
                
            # 2. If no add, try swaps to escape local max
            if len(chosen) > 0:
                x_to_remove = random.choice(chosen)
                remove_element(chosen, counts, x_to_remove)
                chosen_set.discard(x_to_remove)
                
                # Try to add any element
                added = False
                for y in range(1, N + 1):
                    if y not in chosen_set and can_add(chosen, counts, y):
                        add_element(chosen, counts, y)
                        chosen_set.add(y)
                        added = True
                        break
                        
                if not added:
                    # Put back
                    add_element(chosen, counts, x_to_remove)
                    chosen_set.add(x_to_remove)
                else:
                    # We swapped. Size is same. 
                    improved = True
                    # no_improve_count is not reset, so consecutive swaps will eventually break the loop
                    # unless a swap leads to a state where an add is possible in the next round
                    # But wait, if we swap, we change the set. The next round will try to add again.
                    # If the swap opened up a spot, the next round's "add" phase will succeed.
                    # In that case, no_improve_count will reset.
                    # If the swap didn't help, no_improve_count keeps increasing.
                    
        return chosen, counts
        
    start_time = time.time()
    timeout = 110
    
    # Strategy 1: Squares initialization
    current_set, current_counts = greedy_with_order('squares', 42)
    current_set, current_counts = local_search(current_set, current_counts, max_rounds=100, time_limit=timeout - (time.time() - start_time), start_time=start_time)
    if len(current_set) > best_len:
        best_len = len(current_set)
        best_set = current_set[:]
        
    # Strategy 2: Random restarts
    while time.time() - start_time < timeout - 2:
        current_set, current_counts = greedy_with_order('random', None)
        current_set, current_counts = local_search(current_set, current_counts, max_rounds=50, time_limit=timeout - (time.time() - start_time), start_time=start_time)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
    return best_set
