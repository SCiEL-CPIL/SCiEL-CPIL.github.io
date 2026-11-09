"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 192
    MAX_SIZE = 48
    
    def check_valid_addition(S_list, x):
        """Check if x can be added to S_list."""
        n = len(S_list)
        if n == 0:
            return True
        
        # Check 1: x doesn't divide any triple sum in S_list
        if n >= 3:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if (S_list[i] + S_list[j] + S_list[k]) % x == 0:
                            return False
        
        # Check 2: for each a in S_list, a doesn't divide (b + c + x) for any pair b,c in S_list\{a}
        for idx, a in enumerate(S_list):
            for i in range(n):
                if i == idx:
                    continue
                for j in range(i+1, n):
                    if j == idx:
                        continue
                    if (S_list[i] + S_list[j] + x) % a == 0:
                        return False
        
        return True
    
    # Greedy construction with multiple strategies
    best_set = []
    random.seed(42)
    
    strategies = []
    # Strategy 1: descending from 192
    strategies.append(list(range(N, 0, -1)))
    # Strategy 2: only large numbers > 96, descending
    strategies.append(list(range(N, 96, -1)))
    # Strategy 3: large numbers with random shuffle
    for _ in range(20):
        order = list(range(N, 96, -1))
        random.shuffle(order)
        strategies.append(order)
    # Strategy 4: mix of large and some medium
    for _ in range(10):
        order = list(range(N, 50, -1))
        random.shuffle(order)
        strategies.append(order)
    
    for order in strategies:
        if time.time() > deadline:
            break
        
        S_list = []
        for x in order:
            if len(S_list) >= MAX_SIZE:
                break
            if check_valid_addition(S_list, x):
                S_list.append(x)
        
        if len(S_list) > len(best_set):
            best_set = S_list[:]
        
        if len(best_set) >= MAX_SIZE:
            break
    
    # Local search to try to improve
    if time.time() < deadline and len(best_set) < MAX_SIZE:
        S_list = best_set[:]
        S_set = set(S_list)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding
            for x in range(N, 0, -1):
                if x in S_set:
                    continue
                if check_valid_addition(S_list, x):
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
                    break
            # Try swaps
            if not improved and len(S_list) > 0:
                for remove_idx in range(len(S_list)):
                    if time.time() > deadline:
                        break
                    remove_x = S_list[remove_idx]
                    S_list_temp = [v for v in S_list if v != remove_x]
                    S_set_temp = S_set - {remove_x}
                    for add_x in range(N, 0, -1):
                        if add_x in S_set_temp:
                            continue
                        if check_valid_addition(S_list_temp, add_x):
                            S_list = S_list_temp + [add_x]
                            S_set = S_set_temp | {add_x}
                            improved = True
                            break
                    if improved:
                        break
        
        if len(S_list) > len(best_set):
            best_set = S_list
    
    return best_set

# EVOLVE-BLOCK-END
