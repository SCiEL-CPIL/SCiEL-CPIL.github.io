<<<<<<< SEARCH
def construct():
    import sys
    
    def get_tails(x, current_values):
        """Get the new block products if we append x to current_values."""
        tails = [x]
        prod = x
        for a in reversed(current_values):
            prod *= a
            tails.append(prod)
        return tails

    def baseline_greedy():
        values = []
        products = set()
        remaining = set(range(2, 129))
        
        for _ in range(127):
            best_next = None
            best_score = -1
            best_tails = None
            
            for x in sorted(remaining):
                if values and x <= values[-1]:
                    continue
                
                tails = get_tails(x, values)
                if products.intersection(tails):
                    continue
                
                new_products = products.union(tails)
                future_valid = 0
                for y in sorted(remaining):
                    if y <= x:
                        continue
                    y_tails = [y]
                    prod_y = y
                    for a in reversed(values + [x]):
                        prod_y *= a
                        y_tails.append(prod_y)
                    if not new_products.intersection(y_tails):
                        future_valid += 1
                
                score = future_valid
                if score > best_score:
                    best_score = score
                    best_next = x
                    best_tails = tails
            
            if best_next is None:
                break
            
            values.append(best_next)
            products.update(best_tails)
            remaining.discard(best_next)
            
        return values

    best_seq = baseline_greedy()
    best_len = len(best_seq)
    
    sys.setrecursionlimit(10000)
    nodes_visited = 0
    MAX_NODES = 200000
    
    def dfs(last_val, prods_frozen, current_seq):
        nonlocal best_len, best_seq, nodes_visited
        
        nodes_visited += 1
        if nodes_visited > MAX_NODES:
            return False
        
        if len(current_seq) + (128 - last_val) <= best_len:
            return True
            
        for next_val in range(last_val + 1, 129):
            suff = [1]
            p = 1
            for v in reversed(current_seq):
                p *= v
                suff.append(p)
                
            new_prods_list = [next_val * s for s in suff]
            new_prods_set = frozenset(new_prods_list)
            
            if prods_frozen.intersection(new_prods_set):
                continue
                
            new_prods = prods_frozen.union(new_prods_set)
            new_seq = current_seq + [next_val]
            
            if len(new_seq) > best_len:
                best_len = len(new_seq)
                best_seq = new_seq
                
            if not dfs(next_val, new_prods, new_seq):
                return False
                
        return True
        
    dfs(1, frozenset(), [])
    
    return best_seq
=======
def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # Leave 5s buffer
    
    # 1. Initial Greedy
    def fast_greedy():
        values = []
        products = set()
        for x in range(2, 129):
            if values and x <= values[-1]:
                continue
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if products.isdisjoint(tails):
                values.append(x)
                products.update(tails)
        return values

    best_seq = fast_greedy()
    best_len = len(best_seq)
    
    # 2. DFS to find longer sequences
    def find_sequence(target_len):
        def dfs(current_seq, suffix_prods, products_set, depth):
            if time.time() - start_time > TIME_LIMIT:
                return None
            
            if depth == target_len:
                return current_seq[:]
                
            last_val = current_seq[-1] if current_seq else 1
            current_len = len(current_seq)
            
            # Pruning
            if current_len + (128 - last_val) < target_len:
                return None
            
            candidates = list(range(last_val + 1, 129))
            # Heuristic: Try numbers with larger smallest prime factor first
            candidates.sort(key=_spf, reverse=True)
            
            for next_val in candidates:
                new_suffixes = [next_val]
                prod = next_val
                for sp in suffix_prods:
                    prod *= sp
                    new_suffixes.append(prod)
                
                if products_set.isdisjoint(new_suffixes):
                    new_products = products_set.union(new_suffixes)
                    new_seq = current_seq + [next_val]
                    
                    res = dfs(new_seq, new_suffixes, new_products, depth + 1)
                    if res:
                        return res
                        
            return None
            
        return dfs([], [], set(), 0)
    
    while True:
        if time.time() - start_time > TIME_LIMIT:
            break
        
        next_target = best_len + 1
        if next_target > 127:
            break
            
        seq = find_sequence(next_target)
        if seq:
            best_seq = seq
            best_len = next_target
        else:
            break
            
    return best_seq
>>>>>>> REPLACE
