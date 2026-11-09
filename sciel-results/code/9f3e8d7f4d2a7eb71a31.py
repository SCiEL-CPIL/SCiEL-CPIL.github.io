import random
import time

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = set()
    best_size = 0
    
    def is_valid_addition(x, S):
        for d in range(1, 171):
            if x + 3*d <= N and (x+d in S and x+2*d in S and x+3*d in S):
                return False
            if x - d >= 1 and (x-d in S and x+d in S and x+2*d in S):
                return False
            if x - 2*d >= 1 and (x-2*d in S and x-d in S and x+d in S):
                return False
            if x - 3*d >= 1 and (x-3*d in S and x-2*d in S and x-d in S):
                return False
        return True

    def greedy_init():
        S = set()
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_addition(x, S):
                S.add(x)
        return S

    def local_search(S):
        current_size = len(S)
        # Phase 1: Exhaustive Addition
        while time.time() < deadline:
            added = False
            for x in range(1, N+1):
                if x not in S:
                    if is_valid_addition(x, S):
                        S.add(x)
                        current_size += 1
                        added = True
                        break
            if not added:
                break
        
        # Phase 2: Swaps to escape local optima
        # Try random swaps. If a swap leads to a state where an addition is possible, accept it.
        # We limit the number of swap attempts to avoid infinite loops.
        swap_attempts = 0
        max_swaps = 5000
        while swap_attempts < max_swaps and time.time() < deadline:
            swap_attempts += 1
            # Pick random x in S, y not in S
            x = random.choice(list(S))
            y = random.randint(1, N)
            if y in S:
                continue
                
            # Try swap
            S.remove(x)
            if is_valid_addition(y, S):
                S.add(y)
                # Now try to add more
                size_before = len(S)
                # Run addition loop
                added_any = False
                for x_add in range(1, N+1):
                    if x_add not in S:
                        if is_valid_addition(x_add, S):
                            S.add(x_add)
                            added_any = True
                            break
                            
                if added_any:
                    # Swap was beneficial
                    # Continue search
                    swap_attempts = 0 # Reset counter
                    # Loop back to Phase 1 effectively by continuing the while
                else:
                    # Swap didn't help immediately, revert
                    S.remove(y)
                    S.add(x)
            else:
                # y couldn't be added even without x
                # Revert (x is already removed, so just put it back)
                S.add(x)
                
        return current_size, S

    for restart in range(200):
        if time.time() > deadline:
            break
        S = greedy_init()
        size, S = local_search(S)
        if size > best_size:
            best_size = size
            best_set = set(S)
            
    return sorted(best_set)
