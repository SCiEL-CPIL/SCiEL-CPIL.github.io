import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    # 1. Generate Greedy Solution
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    best_sequence = values[:]
    best_len = len(values)
    
    # 2. DFS to find longer sequence
    # We search for sequences longer than best_len
    # To speed up, we can use the greedy solution as a guide?
    # Or just standard DFS.
    
    def dfs(current_seq, current_products, next_start_idx, current_len):
        nonlocal best_len, best_sequence, deadline
        
        if time.time() > deadline:
            return
        
        if current_len > best_len:
            best_len = current_len
            best_sequence = current_seq[:]
            
        # Pruning: Max possible additional elements
        # We have 128 - next_start_idx numbers left.
        if current_len + (128 - next_start_idx) <= best_len:
            return
            
        for i in range(next_start_idx, 129):
            if time.time() > deadline:
                return
                
            x = i + 2
            
            # Check validity
            tails, prod = [x], x
            valid = True
            for a in reversed(current_seq):
                prod *= a
                if prod in current_products:
                    valid = False
                    break
                tails.append(prod)
            
            if valid:
                for t in tails:
                    if t in current_products:
                        valid = False
                        break
                
                if valid:
                    new_products = current_products.union(tails)
                    new_seq = current_seq + [x]
                    dfs(new_seq, new_products, i + 1, current_len + 1)
            
            # Optional: If we find a valid x, we don't need to check larger x for this branch?
            # No, we want to maximize length, so we should try to add as many as possible.
            # But if we add x, we might block y > x.
            # So we must explore all i.
            
    # Start DFS
    # To make it faster, we can start from the greedy sequence?
    # No, that restricts the search.
    # But we can use the greedy length as a lower bound.
    
    dfs([], set(), 0, 0)
    
    return best_sequence
