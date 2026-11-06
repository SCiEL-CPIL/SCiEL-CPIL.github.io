import random
import time
import math

def _init_greedy():
    """
    Greedy construction to get a valid starting set.
    """
    S = set()
    counts = [0] * 2001
    order = list(range(1, 1001))
    random.shuffle(order)
    
    for x in order:
        valid = True
        if counts[2*x] >= 2:
            valid = False
        else:
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
        
        if valid:
            S.add(x)
            counts[2*x] += 1
            for y in S:
                if y != x:
                    counts[x+y] += 1
    return list(S), counts

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Initialize
    S_list, counts = _init_greedy()
    S_set = set(S_list)
    best_S = S_list[:]
    best_size = len(S_list)
    
    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.99995
    no_improve_steps = 0
    max_no_improve = 1000
    
    while time.time() < deadline and T > T_min:
        # 1. Try to add a random element
        y = random.randint(1, 1000)
        
        if y not in S_set:
            # Try to add y
            valid = True
            if counts[2*y] >= 2:
                valid = False
            else:
                for z in S_list:
                    if counts[z+y] >= 2:
                        valid = False
                        break
            
            if valid:
                S_list.append(y)
                S_set.add(y)
                counts[2*y] += 1
                for z in S_list:
                    if z != y:
                        counts[y+z] += 1
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S = S_list[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
        else:
            # y is in S, try a swap
            if S_list:
                x = random.choice(S_list)
                if x == y:
                    no_improve_steps += 1
                    continue
                
                # Remove x
                idx = S_list.index(x)
                S_list.pop(idx)
                S_set.remove(x)
                counts[2*x] -= 1
                for z in S_list:
                    counts[x+z] -= 1
                
                # Try to add y
                valid = True
                if counts[2*y] >= 2:
                    valid = False
                else:
                    for z in S_list:
                        if counts[z+y] >= 2:
                            valid = False
                            break
                
                if valid:
                    # Accept swap
                    S_list.append(y)
                    S_set.add(y)
                    counts[2*y] += 1
                    for z in S_list:
                        if z != y:
                            counts[y+z] += 1
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S = S_list[:]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    # Reject swap, revert removal of x
                    S_list.append(x)
                    S_set.add(x)
                    counts[2*x] += 1
                    for z in S_list:
                        if z != x:
                            counts[x+z] += 1
                    no_improve_steps += 1
        
        T *= cooling_rate
        
        # Diversification
        if no_improve_steps > max_no_improve:
            T = max(T, 1.0)
            no_improve_steps = 0
            # Optionally, randomize a few elements
            # For now, just reset T

    # Final polish: try to add any remaining valid numbers
    for y in range(1, 1001):
        if y in S_set:
            continue
        valid = True
        if counts[2*y] >= 2:
            continue
        for z in S_list:
            if counts[z+y] >= 2:
                valid = False
                break
        if valid:
            S_list.append(y)
            S_set.add(y)
            counts[2*y] += 1
            for z in S_list:
                if z != y:
                    counts[y+z] += 1
            if len(S_list) > best_size:
                best_size = len(S_list)
                best_S = S_list[:]

    return best_S
