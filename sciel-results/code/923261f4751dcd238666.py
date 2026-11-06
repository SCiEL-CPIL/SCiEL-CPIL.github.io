import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    best_len = 0
    best_seq = []
    used_products = set()
    current_seq = []

    def dfs(last_val):
        nonlocal best_len, best_seq

        # Check time limit
        if time.time() > deadline:
            return False

        # Update best solution
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]

        # Pruning: If we can't beat the best length even by taking all remaining numbers, stop
        # Max additional numbers is (128 - last_val)
        if len(current_seq) + (128 - last_val) <= best_len:
            return True

        # Try adding the next number x
        for x in range(last_val + 1, 129):
            # Pruning: If starting with x, the max possible length is len(current_seq) + (128 - x + 1)
            if len(current_seq) + (128 - x + 1) <= best_len:
                break

            # Calculate products of all blocks ending at x
            # Blocks: [x], [v_k, x], [v_{k-1}, v_k, x], ...
            new_prods = []
            prod = x
            new_prods.append(prod)
            
            for a in reversed(current_seq):
                prod *= a
                new_prods.append(prod)

            # Check for conflicts with existing products
            # If any new product is already in used_products, we have a duplicate
            conflict = False
            for p in new_prods:
                if p in used_products:
                    conflict = True
                    break

            if conflict:
                continue

            # No conflict, add to state
            used_products.update(new_prods)
            current_seq.append(x)

            # Recurse
            if not dfs(x):
                # Time limit reached, cleanup and propagate stop signal
                used_products.difference_update(new_prods)
                current_seq.pop()
                return False

            # Backtrack
            used_products.difference_update(new_prods)
            current_seq.pop()

        return True

    dfs(1)
    return best_seq
