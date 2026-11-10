import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 90s limit

    def is_valid_full(seq):
        """Check if the sequence has distinct products for all non-empty consecutive blocks."""
        n = len(seq)
        if n == 0:
            return True
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def is_valid_insertion(S, x):
        """
        Check if inserting x into sorted S at its sorted position keeps validity.
        Only checks products that involve x (new blocks containing x).
        S must be a sorted list.
        Returns (valid, new_S) if valid, else (False, None).
        """
        n = len(S)
        idx = bisect.bisect_left(S, x)
        
        # Build the new sequence
        new_S = S[:idx] + [x] + S[idx:]
        new_n = n + 1
        
        # We need to check all products involving x
        # A product involves x if the block [i..j] in new_S contains index idx
        # i.e., i <= idx <= j
        
        # First, build the set of all existing products in S (not involving x)
        # Then check new products involving x against each other and existing products
        
        # Actually, we need to be careful. The products NOT involving x are the same as before.
        # The products involving x are new. We need:
        # 1. No two products involving x are equal
        # 2. No product involving x equals any product not involving x (which are in the original set)
        
        # Let's compute all products in the original S
        orig_products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= S[j]
                orig_products.add(prod)
        
        # Now compute all products involving x in new_S
        # x is at index idx in new_S
        # Blocks containing x: [i..j] where i <= idx <= j
        # In terms of original S indices:
        #   i in new_S corresponds to: if i <= idx, it's S[i] (for i < idx), or x (for i == idx)
        #   Actually let's just work with new_S directly
        
        new_products_involving_x = set()
        for i in range(idx + 1):  # i can be 0..idx
            prod = 1
            for j in range(idx, new_n):  # j can be idx..new_n-1
                prod *= new_S[j]
                if prod in orig_products:
                    return False, None
                if prod in new_products_involving_x:
                    return False, None
                new_products_involving_x.add(prod)
        
        return True, new_S

    def is_valid_swap(S, remove_idx, add_val):
        """
        Check if removing S[remove_idx] and adding add_val (maintaining sorted order) keeps validity.
        Returns (valid, new_S) if valid, else (False, None).
        """
        n = len(S)
        # Remove element at remove_idx
        S_without = S[:remove_idx] + S[remove_idx+1:]
        
        # Find where add_val goes in S_without
        idx = bisect.bisect_left(S_without, add_val)
        new_S = S_without[:idx] + [add_val] + S_without[idx:]
        
        # Full check since the structure changed significantly
        return is_valid_full(new_S), new_S

    # 1. Greedy Initialization
    S = []
    for x in range(2, 129):
        valid, new_S = is_valid_insertion(S, x)
        if valid:
            S = new_S
    
    best_S = S[:]
    best_len = len(S)
    
    # 2. Local Search / Hill Climbing
    while time.time() - start_time < deadline:
        improved = False
        
        # Step A: Try to insert any single missing number
        missing = [x for x in range(2, 129) if x not in S]
        random.shuffle(missing)
        
        for x in missing:
            if time.time() - start_time > deadline:
                break
            valid, new_S = is_valid_insertion(S, x)
            if valid:
                S = new_S
                improved = True
                if len(S) > best_len:
                    best_S = S[:]
                    best_len = len(S)
                break  # Restart the missing list scan with the new S
        
        if time.time() - start_time > deadline:
            break
        
        if not improved:
            # Step B: Try swap moves (1-out, 1-in) to escape local optima
            if len(S) < 127:
                moves_tried = 0
                max_moves = 200
                while moves_tried < max_moves and time.time() - start_time < deadline:
                    moves_tried += 1
                    # Pick a random element to remove
                    remove_idx = random.randint(0, len(S) - 1)
                    y = S[remove_idx]
                    # Pick a random element to add
                    candidates = [x for x in range(2, 129) if x not in S]
                    if not candidates:
                        continue
                    add_val = random.choice(candidates)
                    
                    valid, new_S = is_valid_swap(S, remove_idx, add_val)
                    if valid:
                        S = new_S
                        improved = True
                        if len(S) > best_len:
                            best_S = S[:]
                            best_len = len(S)
                        break  # Restart with new S
        
        if time.time() - start_time > deadline:
            break
        
        # Step C: Try 1-out, 2-in moves to increase length
        if not improved and len(S) < 127:
            moves_tried = 0
            max_moves = 100
            while moves_tried < max_moves and time.time() - start_time < deadline:
                moves_tried += 1
                # Pick a random element to remove
                remove_idx = random.randint(0, len(S) - 1)
                y = S[remove_idx]
                # Pick two random elements to add
                candidates = [x for x in range(2, 129) if x not in S]
                if len(candidates) < 2:
                    continue
                
                x1, x2 = random.sample(candidates, 2)
                
                # Build new sequence
                S_without = S[:remove_idx] + S[remove_idx+1:]
                new_S = S_without + [x1, x2]
                new_S.sort()
                
                if is_valid_full(new_S):
                    S = new_S
                    improved = True
                    if len(S) > best_len:
                        best_S = S[:]
                        best_len = len(S)
                    break  # Restart with new S
    
    return best_S
