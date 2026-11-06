import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    def get_subarray_products(seq):
        """Returns a set of all non-empty consecutive block products."""
        n = len(seq)
        if n == 0:
            return set()
        prods = set()
        # Use a list to avoid re-computing prefix sums if we wanted, 
        # but N is small, so direct multiplication is fine.
        # Actually, prefix products are better for O(1) range product?
        # No, products can be huge. But Python handles big ints.
        # O(N^2) is fine.
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def is_valid(seq):
        """Checks if all subarray products are distinct."""
        n = len(seq)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def greedy_construct():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    # Initialize with greedy solution
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    all_numbers = list(range(2, 129))
    
    # Local Search
    # We will try to improve the current best sequence by random moves.
    # To avoid getting stuck, we run multiple restarts with perturbations.
    
    while time.time() < deadline:
        # Start from a perturbed version of the best sequence
        # Or just start from best_seq and try to improve
        current_seq = list(best_seq)
        current_set = set(current_seq)
        
        # Try local improvements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomly choose a move: Add or Remove
            if random.random() < 0.5:
                # Try to add a number
                candidates = [x for x in all_numbers if x not in current_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                # Insert x into sorted position
                # Find position
                lo, hi = 0, len(current_seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if current_seq[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                new_seq = current_seq[:lo] + [x] + current_seq[lo:]
                if is_valid(new_seq):
                    if len(new_seq) > len(current_seq):
                        current_seq = new_seq
                        current_set.add(x)
                        improved = True
                    else:
                        # Accept same length moves to explore? 
                        # Risky, might loop. 
                        # Let's only accept if it leads to better future potential? 
                        # Hard to know. 
                        # Let's stick to strict improvement for stability, 
                        # but use restarts for diversity.
                        pass
            else:
                # Try to remove a number
                if not current_seq:
                    continue
                idx = random.randint(0, len(current_seq) - 1)
                x = current_seq[idx]
                new_seq = current_seq[:idx] + current_seq[idx+1:]
                # Removing always maintains validity? 
                # Yes, removing elements reduces the set of products, 
                # so if it was valid, it remains valid.
                # But we only want to remove if it allows adding more later?
                # In pure hill climbing for max length, we don't remove unless forced.
                # But the knowledge says deletion moves help escape local optima.
                # This implies we should allow temporary decreases in length.
                # Let's implement a "Simulated Annealing" style or "Random Walk"
                # that allows decreases.
                
                # Let's try a different approach: 
                # Randomly pick a new sequence by:
                # 1. Taking current_seq
                # 2. Removing k random elements (k=1 or 2)
                # 3. Adding k random elements
                # 4. If valid and length > current, accept.
                # This is a "Jump" move.
                
                pass

        # If we found a longer sequence, update best
        if len(current_seq) > best_len:
            best_seq = current_seq
            best_len = len(best_seq)
            
        # If we didn't improve, we might be stuck. 
        # Let's try a random restart from scratch or a heavily perturbed version.
        # Given the time, let's just let the loop continue trying random adds/removes 
        # but we need to ensure we don't get stuck in an infinite loop of no-improvement.
        # The `improved` flag breaks the inner loop if no improvement is found in one pass.
        # The outer loop then restarts the local search from `best_seq` again.
        # This effectively does many independent hill climbs from the same peak.
        # To diversify, we should perturb `best_seq` before starting the next hill climb.
        
        # Perturb best_seq for the next iteration
        # Remove 1-2 random elements from best_seq to create a new start point
        if best_seq:
            start_seq = list(best_seq)
            # Remove 1 or 2 random elements
            for _ in range(random.randint(1, 2)):
                if not start_seq: break
                idx = random.randint(0, len(start_seq) - 1)
                start_seq.pop(idx)
            # Add 1 or 2 random elements from unused
            unused = [x for x in all_numbers if x not in set(start_seq)]
            if unused:
                for _ in range(random.randint(1, 2)):
                    if not unused: break
                    x = random.choice(unused)
                    unused.remove(x)
                    # Insert in sorted position
                    lo, hi = 0, len(start_seq)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if start_seq[mid] < x:
                            lo = mid + 1
                        else:
                            hi = mid
                    start_seq.insert(lo, x)
            
            # Try to hill climb from this perturbed start
            current_seq = start_seq
            current_set = set(current_seq)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try all possible single additions
                # This is more thorough than random picking
                # For each unused number, check if adding it makes it valid
                # If yes, and length increases, add it.
                # Repeat until no more additions possible.
                # This is a "Greedily Fill" strategy.
                
                # To make it stochastic, we shuffle the order of trying numbers
                unused = [x for x in all_numbers if x not in current_set]
                random.shuffle(unused)
                
                for x in unused:
                    if time.time() >= deadline: break
                    # Find insertion point
                    lo, hi = 0, len(current_seq)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if current_seq[mid] < x:
                            lo = mid + 1
                        else:
                            hi = mid
                    new_seq = current_seq[:lo] + [x] + current_seq[lo:]
                    if is_valid(new_seq):
                        current_seq = new_seq
                        current_set.add(x)
                        improved = True
                        # After adding, restart the loop to check other numbers
                        # because the set of valid additions might have changed
                        break 

                # If no addition was made, try a removal to escape?
                # If we are stuck (no additions possible), we might try removing one
                # and then trying to add a different one.
                if not improved and current_seq:
                    # Try removing a random element
                    idx = random.randint(0, len(current_seq) - 1)
                    removed_x = current_seq.pop(idx)
                    current_set.remove(removed_x)
                    # Now try to add something
                    improved = True # Force one more iteration
                    # Note: This might decrease length. 
                    # We only update best_seq if length is strictly greater.
                    # But we continue searching from this shorter sequence.
                    # This allows escaping local optima.
                    # We need to be careful not to loop forever.
                    # The `improved` flag logic above restarts the inner while.
                    # Let's structure it carefully.
            
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_len = len(current_seq)
                
    return best_seq
