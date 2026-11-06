import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin for 90s limit
    
    def greedy_init():
        values = []
        products = set()
        suffix_prods = []
        
        for x in range(2, 129):
            new_suffix = x
            new_suffixes = [x]
            valid = True
            
            if x in products:
                continue
                
            for sp in suffix_prods:
                new_suffix *= sp
                new_suffixes.append(new_suffix)
                if new_suffix in products:
                    valid = False
                    break
            
            if valid:
                values.append(x)
                products.update(new_suffixes)
                suffix_prods = new_suffixes
                
        return values

    # Get initial best from greedy
    init_seq = greedy_init()
    best_seq = init_seq
    best_len = len(init_seq)
    
    def dfs(last_val, current_seq, suffix_prods, products_set):
        nonlocal best_seq, best_len, deadline, start_time
        
        if time.time() - start_time > deadline:
            return
            
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq.copy()
            
        # Pruning: if we can't beat best_len, return
        if len(current_seq) + (128 - last_val) <= best_len:
            return
            
        for x in range(last_val + 1, 129):
            if time.time() - start_time > deadline:
                return
                
            # Quick check: if x is already in products, skip
            if x in products_set:
                continue
                
            new_suffix = x
            new_suffixes = [x]
            valid = True
            
            for sp in suffix_prods:
                new_suffix *= sp
                new_suffixes.append(new_suffix)
                if new_suffix in products_set:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Recurse
            new_products_set = products_set.union(new_suffixes)
            dfs(x, current_seq + [x], new_suffixes, new_products_set)

    # Start DFS from empty sequence
    dfs(1, [], [], set())
    
    return best_seq
