import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin for 90s limit
    
    n = 192
    max_val = n
    
    def is_valid_addition(S, x):
        """
        Check if adding x to valid set S keeps it valid.
        S is a list of distinct integers. x is not in S.
        Condition: No element in S U {x} divides the sum of any 3 distinct OTHER elements in S U {x}.
        """
        S_list = list(S)
        
        # 1. Check if x divides sum of any 3 distinct elements in S
        if len(S_list) >= 3:
            for a, b, c in combinations(S_list, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. Check for each a in S: does a divide sum of any 3 distinct from (S U {x}) \ {a}?
        for a in S_list:
            others = [v for v in S_list if v != a]
            if len(others) < 2:
                continue
                
            target_rem = (-x) % a
            
            for i in range(len(others)):
                b = others[i]
                for j in range(i + 1, len(others)):
                    c = others[j]
                    if (b + c) % a == target_rem:
                        return False
                        
        return True

    def greedy_build(order):
        S = []
        for x in order:
            if is_valid_addition(S, x):
                S.append(x)
        return S

    def local_search(S, time_limit):
        """
        Local search: try to add elements, and if stuck, do random swaps to escape local optima.
        Returns the best set found.
        """
        current = list(S)
        best = list(S)
        best_size = len(S)
        
        while time.time() < time_limit:
            improved = False
            
            # Try to add any element
            candidates = [x for x in range(1, n + 1) if x not in current]
            random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_addition(current, x):
                    current.append(x)
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best = list(current)
                    break
            
            if improved:
                continue
                
            # If no add possible, try random swaps to escape local optimum
            # Do a few random swaps
            swapped = False
            for _ in range(30):
                if time.time() > time_limit:
                    break
                if len(current) == 0:
                    break
                    
                # Pick random element to remove
                idx = random.randint(0, len(current) - 1)
                a = current[idx]
                
                # Pick random element to add
                cands_swap = [x for x in range(1, n + 1) if x not in current]
                if not cands_swap:
                    break
                x = random.choice(cands_swap)
                
                # Check if swap is valid
                temp_set = [v for i, v in enumerate(current) if i != idx]
                if is_valid_addition(temp_set, x):
                    current = temp_set + [x]
                    swapped = True
                    break
                    
            if not swapped:
                # If we can't even swap, we're truly stuck. Break.
                break

        return best

    # Strategy 1: Descending Greedy
    S1 = greedy_build(range(n, 0, -1))
    best_set = list(S1)
    best_size = len(S1)
    
    # Polish S1
    S1_polished = local_search(S1, deadline - 30)
    if len(S1_polished) > best_size:
        best_size = len(S1_polished)
        best_set = list(S1_polished)
    
    # Strategy 2: Ascending Greedy
    S2 = greedy_build(range(1, n + 1))
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = list(S2)
        
    S2_polished = local_search(S2, deadline - 30)
    if len(S2_polished) > best_size:
        best_size = len(S2_polished)
        best_set = list(S2_polished)

    # Strategy 3: Random Greedy with Local Search
    while time.time() < deadline - 10:
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_build(order)
        
        if len(S_r) > best_size:
            best_size = len(S_r)
            best_set = list(S_r)
            
        # Polish if time permits
        if time.time() < deadline - 5:
            S_r_polished = local_search(S_r, deadline - 5)
            if len(S_r_polished) > best_size:
                best_size = len(S_r_polished)
                best_set = list(S_r_polished)

    # Final polish on the best set found
    if time.time() < deadline - 1:
        S_final = local_search(best_set, deadline)
        if len(S_final) > best_size:
            best_set = S_final

    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))
