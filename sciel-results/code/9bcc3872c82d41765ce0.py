import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    all_vals = list(range(1, max_val + 1))
    
    def init_sums(S):
        Sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                Sums.add(S[i] + S[j])
        return Sums

    def can_add(x, S, Sums):
        if 2 * x in Sums:
            return False
        for y in S:
            if x + y in Sums:
                return False
        return True

    def add_element(x, S, Sums):
        Sums.add(2 * x)
        for y in S:
            Sums.add(x + y)
        S.append(x)
        S.sort()

    def remove_element(x, S, Sums):
        S.remove(x)
        Sums.discard(2 * x)
        for y in S:
            Sums.discard(x + y)

    def greedy_fill(S, Sums):
        """Fill S greedily from smallest available x."""
        S_set = set(S)
        for x in range(1, max_val + 1):
            if x in S_set:
                continue
            if can_add(x, S, Sums):
                add_element(x, S, Sums)
                S_set.add(x)
        return S, Sums

    def local_search(S, Sums, max_iters=200):
        """Local search: try adds, then swaps. Returns (S, Sums, improved)."""
        S_set = set(S)
        improved = False
        iters = 0
        
        while iters < max_iters and time.time() < deadline:
            iters += 1
            # Try to add a random element
            candidates = [x for x in all_vals if x not in S_set]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if can_add(x, S, Sums):
                    add_element(x, S, Sums)
                    S_set.add(x)
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # If no add possible, try to swap
            removable = [x for x in S if x not in prescribed_set]
            if not removable:
                break
            
            random.shuffle(removable)
            swap_done = False
            for r in removable:
                # Temporarily remove r
                remove_element(r, S, Sums)
                S_set.discard(r)
                
                # Try to add a random element
                candidates = [x for x in all_vals if x not in S_set]
                random.shuffle(candidates)
                
                for x in candidates:
                    if can_add(x, S, Sums):
                        add_element(x, S, Sums)
                        S_set.add(x)
                        swap_done = True
                        break
                
                if swap_done:
                    improved = True
                    break
                else:
                    # Restore r
                    add_element(r, S, Sums)
                    S_set.add(r)
            
            if not swap_done:
                # Try a double swap: remove two, add two
                if len(removable) >= 2:
                    random.shuffle(removable)
                    r1, r2 = removable[0], removable[1]
                    remove_element(r1, S, Sums)
                    S_set.discard(r1)
                    remove_element(r2, S, Sums)
                    S_set.discard(r2)
                    
                    candidates = [x for x in all_vals if x not in S_set]
                    random.shuffle(candidates)
                    
                    added_count = 0
                    added_elems = []
                    for x in candidates:
                        if can_add(x, S, Sums):
                            add_element(x, S, Sums)
                            S_set.add(x)
                            added_elems.append(x)
                            added_count += 1
                            if added_count >= 2:
                                break
                    
                    if added_count >= 2:
                        improved = True
                    else:
                        # Restore
                        for x in added_elems:
                            remove_element(x, S, Sums)
                            S_set.discard(x)
                        add_element(r1, S, Sums)
                        S_set.add(r1)
                        add_element(r2, S, Sums)
                        S_set.add(r2)
                else:
                    break
        
        return S, Sums, improved

    best_size = 0
    best_S = []
    
    # Run multiple restarts
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        
        # Randomized greedy initialization
        S = list(prescribed)
        S.sort()
        Sums = init_sums(S)
        S, Sums = greedy_fill(S, Sums)
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
        
        # Local search
        S, Sums, improved = local_search(S, Sums, max_iters=100)
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
        
        # If we reached 20, stop
        if best_size >= 20:
            break
        
        # If we've been stuck for a while, try a different strategy
        # Occasionally do a random restart with different greedy order
        if restart_count % 10 == 0:
            # Try a different random ordering for greedy
            S = list(prescribed)
            S.sort()
            Sums = init_sums(S)
            S_set = set(S)
            
            # Randomly select candidates to try adding
            remaining = [x for x in all_vals if x not in S_set]
            random.shuffle(remaining)
            
            for x in remaining:
                if can_add(x, S, Sums):
                    add_element(x, S, Sums)
                    S_set.add(x)
            
            S, Sums, improved = local_search(S, Sums, max_iters=100)
            current_size = len(S)
            if current_size > best_size:
                best_size = current_size
                best_S = S.copy()
            
            if best_size >= 20:
                break

    return best_S
