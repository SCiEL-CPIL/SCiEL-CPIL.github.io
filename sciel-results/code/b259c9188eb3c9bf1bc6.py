import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_set = set()
    
    def can_add(S, x):
        # Check if adding x to S creates a forbidden triple
        # Forbidden: exists a, b in S, a != b such that a+b, a+x, b+x are all in S
        items = list(S)
        n = len(items)
        for i in range(n):
            a = items[i]
            ax = a + x
            if ax > 128:
                continue
            for j in range(i + 1, n):
                b = items[j]
                bx = b + x
                if bx > 128:
                    continue
                if (a + b in S) and (ax in S) and (bx in S):
                    return False
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S, time_limit):
        current_set = set(S)
        current_size = len(current_set)
        best_local = set(current_set)
        
        all_elements = set(range(1, 129))
        
        T = 1.0
        cooling = 0.999
        min_T = 0.05
        
        start_ls = time.time()
        
        while time.time() - start_ls < time_limit and T > min_T:
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_set = set(current_set)
            valid_move = False
            size_change = 0
            
            if move_type == 'add':
                candidates = list(all_elements - current_set)
                if candidates:
                    x = random.choice(candidates)
                    if can_add(current_set, x):
                        new_set.add(x)
                        valid_move = True
                        size_change = 1
            elif move_type == 'remove':
                if current_set:
                    y = random.choice(list(current_set))
                    new_set.remove(y)
                    valid_move = True
                    size_change = -1
            elif move_type == 'swap':
                candidates_out = list(all_elements - current_set)
                candidates_in = list(current_set)
                if candidates_out and candidates_in:
                    x = random.choice(candidates_out)
                    y = random.choice(candidates_in)
                    temp_set = current_set.copy()
                    temp_set.remove(y)
                    if can_add(temp_set, x):
                        new_set.add(x)
                        new_set.remove(y)
                        valid_move = True
                        size_change = 0
            
            if valid_move:
                new_size = len(new_set)
                delta_size = new_size - current_size
                
                if delta_size > 0:
                    accept = True
                elif delta_size == 0:
                    accept = True
                else:
                    # delta_size < 0
                    p = 2.718281828 ** (delta_size / T)
                    accept = random.random() < p
                
                if accept:
                    current_set = new_set
                    current_size = new_size
                    if current_size > len(best_local):
                        best_local = set(current_set)
            
            T *= cooling
        
        return best_local

    # Initial greedy
    initial_order = list(range(128, 0, -1))
    best_set = greedy_construct(initial_order)
    
    # Restart loop
    while time.time() < deadline - 5:
        # Random order
        order = list(range(1, 129))
        random.shuffle(order)
        
        S = greedy_construct(order)
        S = local_search(S, 2.0)
        
        if len(S) > len(best_set):
            best_set = S
            
    # Final polish on best_set
    # Run local search again on the best found to squeeze out more
    if time.time() < deadline - 1:
        best_set = local_search(best_set, deadline - time.time() - 1)

    result = [int(x) for x in best_set]
    return result
