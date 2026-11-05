import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # 1. Greedy initialization to get a strong lower bound
    greedy_seq = []
    greedy_prods = set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        # Check x
        tails = [x]
        prod = x
        for a in reversed(greedy_seq):
            prod *= a
            tails.append(prod)
            
        if not any(p in greedy_prods for p in tails):
            greedy_seq.append(x)
            greedy_prods.update(tails)
            
    best_seq = greedy_seq
    best_len = len(greedy_seq)
    
    # 2. DFS to improve
    def check_and_get_new_products(x, seq, products):
        new_prods = set()
        prod = x
        if prod in products:
            return None
        new_prods.add(prod)
        
        for a in reversed(seq):
            prod *= a
            if prod in products:
                return None
            new_prods.add(prod)
            
        return new_prods

    def dfs(start_val, current_seq, current_products):
        nonlocal best_seq, best_len
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        if not current_seq:
            max_additional = 127
        else:
            max_additional = 128 - current_seq[-1]
            
        if len(current_seq) + max_additional <= best_len:
            return
            
        if time.time() > deadline:
            return

        for x in range(start_val, 129):
            new_prods = check_and_get_new_products(x, current_seq, current_products)
            if new_prods is not None:
                next_products = current_products.union(new_prods)
                dfs(x + 1, current_seq + [x], next_products)

    # Start DFS from empty to find global optimum, or from greedy?
    # Starting from empty allows exploring branches that skip early numbers (like 3).
    # Starting from greedy would only extend it, missing better alternatives.
    # So start from empty.
    dfs(2, [], set())
    
    return best_seq
