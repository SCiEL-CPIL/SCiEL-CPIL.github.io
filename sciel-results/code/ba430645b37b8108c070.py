import time
import sys

def construct():
        start_time = time.time()
        deadline = start_time + 55.0 # 5s safety margin
        
        best_seq = []
        best_len = 0
        
        # State: current sequence, set of all subarray products
        # We pass these recursively or use global variables for speed
        
        # To speed up, we can pre-filter candidates?
        # No, conflict depends on context.
        
        # Heuristic for candidate ordering:
        # Prefer numbers that are "simple" (fewer prime factors) or "large"?
        # Large numbers are less likely to divide other products?
        # Actually, if we pick large numbers, the products grow very fast,
        # reducing the chance of collision with smaller products?
        # But we need strictly increasing.
        
        # Let's try DFS trying candidates in increasing order.
        # If that fails to find long sequences, we might need randomization.
        
        # However, the greedy solution is a lower bound.
        # Let's run the greedy first to get a good initial bound.
        
        def greedy():
            vals, prods = [], set()
            for x in range(2, 129):
                # Check if x can be added
                tails = [x]
                prod = x
                valid = True
                for a in reversed(vals):
                    prod *= a
                    tails.append(prod)
                    if prod in prods:
                        valid = False
                        break
                if not valid:
                    continue
                # Check if any tail is in prods
                if not prods.intersection(tails):
                    vals.append(x)
                    prods.update(tails)
            return vals
        
        initial = greedy()
        best_seq = initial
        best_len = len(initial)
        
        # Now, try to improve via local search or DFS from scratch?
        # DFS from scratch might be too slow if branching factor is high.
        # But the depth is small.
        
        # Let's try a randomized restart strategy with greedy + local swaps?
        # Or just a simple DFS with a node limit?
        
        # Given the 60s limit, a well-pruned DFS might work.
        # But worst case is huge.
        
        # Let's try a different heuristic:
        # Construct the sequence by picking the number that minimizes the "conflict score"
        # with the remaining numbers.
        
        # Conflict score for candidate x given current seq S:
        # 1. x must be valid with S.
        # 2. For each remaining number y > x, check if y would be blocked by adding x?
        #    This is expensive.
        
        # Let's stick to DFS with a strict node limit and time check.
        
        # To make DFS efficient, we can use the fact that if a number is skipped,
        # we might come back to it? No, strictly increasing means we pick a subsequence.
        
        # Let's try to find a sequence of length L+1 where L is current best.
        
        # Actually, a very effective heuristic for this specific problem (distinct subarray products)
        # is to use numbers that are powers of distinct primes, or similar.
        # E.g., 2, 3, 5, 7, 11...
        # But we can use composites if they don't collide.
        
        # Let's implement a DFS that explores candidates in increasing order.
        # We add a pruning rule: if current length + (128 - last_val) <= best_len, prune.
        # This is weak but helps.
        
        # Also, if we are at depth d, and we have already tried all candidates > last_val
        # and none worked, we backtrack.
        
        # To avoid TLE, we will limit the number of nodes visited.
        
        nodes_visited = 0
        MAX_NODES = 1000000 # Adjust based on speed
        
        def dfs(current_seq, current_prods, last_val):
            nonlocal best_seq, best_len, nodes_visited, deadline
            
            if time.time() > deadline:
                return
            
            nodes_visited += 1
            if nodes_visited > MAX_NODES:
                return
            
            # Update best
            if len(current_seq) > best_len:
                best_len = len(current_seq)
                best_seq = current_seq[:]
                
            # Pruning
            if len(current_seq) + (128 - last_val) <= best_len:
                return
                
            # Try next candidates
            for x in range(last_val + 1, 129):
                # Check validity
                # Compute new products
                new_prods = []
                prod = x
                new_prods.append(prod)
                valid = True
                for a in reversed(current_seq):
                    prod *= a
                    new_prods.append(prod)
                    if prod in current_prods:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check if any new_prod is in current_prods
                # We already checked in the loop above for the suffixes including x?
                # Wait, the loop checks prod (which includes x and suffix of current_seq).
                # Does it check x itself? Yes, prod starts as x.
                # Does it check x * a[-1]? Yes.
                # So if any is in current_prods, valid is False.
                
                # However, we also need to ensure new_prods don't collide with each other?
                # No, new_prods are products of x * suffix.
                # suffixes are distinct and increasing in length.
                # Since numbers are > 1, the products are strictly increasing.
                # So new_prods are distinct.
                
                # So we just need to check intersection with current_prods.
                
                if not current_prods.intersection(new_prods):
                    # Recurse
                    new_seq = current_seq + [x]
                    new_prods_set = current_prods.union(new_prods)
                    dfs(new_seq, new_prods_set, x)
                    
                    if time.time() > deadline:
                        return

        dfs([], set(), 1)
        
        return best_seq
