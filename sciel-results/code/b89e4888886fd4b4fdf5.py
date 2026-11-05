import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    def is_valid_extension(current_products, new_tails):
        for p in new_tails:
            if p in current_products:
                return False
        return True

    def dfs(last_picked, current_seq, current_products):
        nonlocal best_seq, best_len
        
        if time.time() > deadline:
            return

        current_len = len(current_seq)
        
        # Update best if current is better
        if current_len > best_len:
            best_len = current_len
            best_seq = current_seq[:]

        # Pruning: if we can't beat best_len with remaining numbers, stop
        if current_len + (128 - last_picked) <= best_len:
            return

        # If we've picked up to 128, stop
        if last_picked >= 128:
            return

        # Try adding next numbers
        for x in range(last_picked + 1, 129):
            # Calculate new tails
            # tails[0] = x
            # tails[i] = x * current_seq[-1] * ... * current_seq[-i]
            tails = [x]
            prod = x
            for a in reversed(current_seq):
                prod *= a
                tails.append(prod)
            
            # Check for collisions
            if is_valid_extension(current_products, tails):
                # Add x
                current_seq.append(x)
                current_products.update(tails)
                
                dfs(x, current_seq, current_products)
                
                # Undo
                current_seq.pop()
                for p in tails:
                    current_products.remove(p)
                    
                if time.time() > deadline:
                    return

    # 1. Get a good initial lower bound using greedy
    greedy_values = []
    greedy_products = set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        tails = [x]
        prod = x
        for a in reversed(greedy_values):
            prod *= a
            tails.append(prod)
        if not greedy_products.intersection(tails):
            greedy_values.append(x)
            greedy_products.update(tails)
            
    best_seq = greedy_values
    best_len = len(greedy_values)
    
    # 2. Run DFS to find a better solution
    # Start with empty sequence
    dfs(1, [], set())
    
    return best_seq
