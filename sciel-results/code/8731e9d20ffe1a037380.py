import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 128
    all_nums = list(range(1, N + 1))
    
    # Helper: Check if adding x to S is valid
    def can_add(S, x):
        # S is a set
        # x is not in S
        # Check Case 1: x is an element
        # For all b, c in S, b < c
        # Check if x+b in S U {x}, x+c in S U {x}, b+c in S U {x}
        s_list = list(S)
        n = len(s_list)
        for i in range(n):
            b = s_list[i]
            for j in range(i + 1, n):
                c = s_list[j]
                # Case 1: x, b, c are the elements
                sum1 = x + b
                sum2 = x + c
                sum3 = b + c
                if (sum1 in S or sum1 == x) and \
                   (sum2 in S or sum2 == x) and \
                   (sum3 in S or sum3 == x):
                    return False
                
                # Case 2: x is a sum
                # If b + c == x, then x is the sum of b and c.
                # The elements are a, b, c. Sums are a+b, a+c, x.
                # We need a+b in S U {x} and a+c in S U {x}.
                if b + c == x:
                    for a in S:
                        if a == b or a == c:
                            continue
                        sum_a_b = a + b
                        sum_a_c = a + c
                        if (sum_a_b in S or sum_a_b == x) and \
                           (sum_a_c in S or sum_a_c == x):
                            return False
        return True

    # Greedy Initialization
    # Try different orders to find a good start
    # Order 1: Descending (Large numbers first)
    # Order 2: Ascending
    # Order 3: Random
    
    best_set = set()
    
    def run_greedy(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        return S

    # Try a few greedy orders
    candidates = []
    
    # Descending
    S_desc = run_greedy(range(N, 0, -1))
    candidates.append(S_desc)
    
    # Ascending
    S_asc = run_greedy(range(1, N + 1))
    candidates.append(S_asc)
    
    # Random (a few times)
    for _ in range(5):
        random_order = all_nums.copy()
        random.shuffle(random_order)
        S_rand = run_greedy(random_order)
        candidates.append(S_rand)
        
    # Pick the largest one
    S = max(candidates, key=len)
    
    # Local Search: Add any possible
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any number from 1..128 not in S
        for x in range(1, N + 1):
            if x not in S:
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    # Restart loop to check for more additions
                    break
        
        if not improved:
            # Try 2-for-1 swap
            # Remove u, try to add v1, v2
            found_swap = False
            s_list = list(S)
            for u in s_list:
                if time.time() > deadline:
                    break
                S_temp = S - {u}
                added = 0
                # Try to add numbers to S_temp
                # Sort candidates by some heuristic? Just iterate 1..128
                for x in range(1, N + 1):
                    if x in S_temp:
                        continue
                    if x == u:
                        continue
                    if can_add(S_temp, x):
                        S_temp.add(x)
                        added += 1
                        if added >= 2:
                            # We can add at least 2, so net gain +1
                            # Accept this move
                            S = S_temp
                            found_swap = True
                            improved = True
                            break
                if found_swap:
                    break
                    
    # Final check for any single additions (just in case)
    for x in range(1, N + 1):
        if x not in S:
            if can_add(S, x):
                S.add(x)
                
    # Ensure types are correct
    result = list(S)
    return [int(x) for x in result]
