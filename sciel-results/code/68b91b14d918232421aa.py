import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    def is_valid(seq):
        """Check if sequence has distinct consecutive block products."""
        n = len(seq)
        if n == 0:
            return True
        products = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in products:
                    return False
                products.add(p)
        return True

    def greedy_construct():
        """Greedy: include x if it doesn't create duplicate products."""
        values = []
        products = set()
        for x in range(2, 129):
            new_tails = [x]
            p = x
            for a in reversed(values):
                p *= a
                new_tails.append(p)
            if not products.intersection(new_tails):
                values.append(x)
                products.update(new_tails)
        return values

    def randomized_greedy():
        """Randomized greedy: shuffle candidates, add if valid."""
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        values = []
        products = set()
        for x in candidates:
            # Insert in sorted position for product calculation? 
            # The product set depends on the relative order.
            # Since the final sequence must be sorted, we should maintain sorted order in 'values'
            # and calculate new products based on insertion into sorted list.
            
            # Find position
            pos = 0
            for i, v in enumerate(values):
                if v < x:
                    pos = i + 1
                else:
                    break
            
            # Calculate new products involving x
            # New products are:
            # 1. x
            # 2. x * (suffix of values[pos:]?) No.
            # The sequence is values[:pos] + [x] + values[pos:]
            # New blocks ending at x: x, x*values[pos-1], x*values[pos-1]*values[pos-2]...
            # New blocks starting at x: x, x*values[pos], x*values[pos]*values[pos+1]...
            # New blocks containing x: x * (suffix of left) * (prefix of right)?
            # Actually, any block containing x is (suffix of left) * x * (prefix of right).
            # We need to check if any of these new products are already in 'products'.
            
            left = values[:pos]
            right = values[pos:]
            
            new_prods = set()
            
            # Blocks ending at x (from left)
            p = x
            new_prods.add(p)
            for a in reversed(left):
                p *= a
                new_prods.add(p)
                
            # Blocks starting at x (to right)
            p = x
            # x is already added
            for a in right:
                p *= a
                new_prods.add(p)
                
            # Blocks containing x (left + x + right)
            # We can generate these by taking suffixes of left and prefixes of right
            # But wait, the "ending at x" and "starting at x" cover the cases where one side is empty.
            # The mixed cases: left_suffix * x * right_prefix
            # We can compute this efficiently.
            
            # To be safe and simple, let's just recompute all products for the new candidate sequence.
            # Since n is small (< 40), O(n^2) is fast enough for initialization.
            test_seq = values[:pos] + [x] + values[pos:]
            if is_valid(test_seq):
                values = test_seq
                # Update products set? 
                # For efficiency in the loop, we should update the set.
                # But is_valid recomputes. 
                # Let's stick to recomputing for correctness in init, 
                # but for SA we need speed.
                pass
        
        return values

    # Get initial solution
    initial_seq = greedy_construct()
    best_seq = initial_seq[:]
    best_len = len(initial_seq)
    
    # Try a few randomized greedy starts
    for _ in range(20):
        r_seq = randomized_greedy()
        if len(r_seq) > best_len:
            best_seq = r_seq
            best_len = len(r_seq)
            
    # Simulated Annealing on Valid States
    # State: A valid sequence (subset of 2..128)
    # Moves:
    # 1. Remove a random element. (Always valid)
    # 2. Add a random missing element. (Check validity)
    
    current_seq = best_seq[:]
    current_len = len(current_seq)
    
    T = 2.0
    cool_rate = 0.999
    
    # Precompute set of current elements for fast lookup
    in_seq = set(current_seq)
    
    no_improve = 0
    
    while time.time() < deadline - 3:
        r = random.random()
        
        if r < 0.5:
            # Remove
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            x = current_seq[idx]
            new_seq = current_seq[:idx] + current_seq[idx+1:]
            new_len = len(new_seq)
            
            # Always valid
            # Accept if better (longer) or with prob exp(-delta/T)
            # Delta = new_len - current_len = -1
            # Accept if random < exp(-1/T)
            if new_len > current_len or random.random() < np.exp((new_len - current_len) / T):
                current_seq = new_seq
                current_len = new_len
                in_seq.discard(x)
                
                if current_len > best_len:
                    best_len = current_len
                    best_seq = current_seq[:]
                    no_improve = 0
                else:
                    no_improve += 1
                    
        else:
            # Add
            missing = [x for x in range(2, 129) if x not in in_seq]
            if not missing:
                continue
            x = random.choice(missing)
            
            # Insert in sorted position
            pos = 0
            for i, v in enumerate(current_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            
            new_seq = current_seq[:pos] + [x] + current_seq[pos:]
            
            if is_valid(new_seq):
                new_len = len(new_seq)
                # Delta = +1
                # Always accept if valid and longer? 
                # Or use SA? Since it's an improvement in length, we should accept it greedily 
                # or with high probability.
                # If we accept all valid adds, we might get stuck if we can't add more.
                # But we also have remove moves.
                if new_len > current_len:
                    current_seq = new_seq
                    current_len = new_len
                    in_seq.add(x)
                    
                    if current_len > best_len:
                        best_len = current_len
                        best_seq = current_seq[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # This shouldn't happen if we only add
                    no_improve += 1
            else:
                # Invalid add, reject
                no_improve += 1
                
        # Cooling
        T *= cool_rate
        if T < 0.1:
            T = 0.1
            
        # Restart logic
        if no_improve > 100:
            no_improve = 0
            T = 2.0
            # Perturb: remove 2-3 random elements to allow different adds
            for _ in range(random.randint(2, 4)):
                if current_seq:
                    idx = random.randint(0, len(current_seq) - 1)
                    x = current_seq[idx]
                    current_seq = current_seq[:idx] + current_seq[idx+1:]
                    in_seq.discard(x)
                    current_len = len(current_seq)

    # Final polish: try to add any missing elements
    in_best = set(best_seq)
    for x in range(2, 129):
        if x not in in_best:
            pos = 0
            for i, v in enumerate(best_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            test_seq = best_seq[:pos] + [x] + best_seq[pos:]
            if is_valid(test_seq):
                best_seq = test_seq
                in_best.add(x)

    return best_seq
