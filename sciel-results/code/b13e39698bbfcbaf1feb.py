import time
import random
import sys

# Set recursion limit just in case, though we use iterative/short recursion
sys.setrecursionlimit(2000)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Range of numbers
    candidates = list(range(2, 129))
    
    # Helper to check if adding x to the end of seq is valid
    # seq is sorted increasing
    # products is a set of all consecutive products of seq
    def is_valid_addition(x, seq, products):
        # Calculate new products formed by appending x
        # New products are: x, x*last, x*last*second_last, ...
        new_prods = []
        prod = x
        new_prods.append(prod)
        for a in reversed(seq):
            prod *= a
            new_prods.append(prod)
        
        # Check for conflicts with existing products
        # Also check for conflicts among the new products themselves?
        # No, new products are distinct because seq is strictly increasing and x > all in seq?
        # Wait, x is added at the end, so x > seq[-1].
        # The products are x, x*seq[-1], x*seq[-1]*seq[-2]...
        # Are they distinct?
        # x * (product of suffix of seq)
        # Since seq elements are > 1, the suffix products are strictly increasing.
        # So new_prods are distinct.
        # We only need to check against existing products.
        
        for p in new_prods:
            if p in products:
                return False, None
        
        return True, new_prods

    # Greedy construction
    def greedy_solve():
        seq = []
        products = set()
        for x in candidates:
            valid, new_prods = is_valid_addition(x, seq, products)
            if valid:
                seq.append(x)
                products.update(new_prods)
        return seq, products

    # Get initial solution
    best_seq, best_products = greedy_solve()
    best_len = len(best_seq)
    
    # Local Search / ILS
    current_seq, current_products = best_seq, best_products
    current_len = best_len
    
    # To speed up validity checks during repair, we might need to rebuild products.
    # Rebuilding products for a sequence of length L takes O(L^2).
    # L is ~40. 1600 ops. Very fast.
    
    def rebuild_products(seq):
        prods = set()
        for i in range(len(seq)):
            p = 1
            for j in range(i, len(seq)):
                p *= seq[j]
                prods.add(p)
        return prods

    # Repair function: Given a sequence (possibly with some elements removed),
    # try to fill it back up using unused candidates.
    def repair(seq):
        # seq is the base sequence (sorted)
        # We want to insert unused numbers into it.
        # Since the final sequence must be sorted, we iterate through unused numbers in increasing order.
        # If an unused number can be inserted at its sorted position without conflict, we insert it.
        
        used = set(seq)
        new_seq = list(seq)
        # We need to maintain the product set for the *current* new_seq to check validity.
        # But inserting in the middle is hard for the product set.
        # Strategy: 
        # 1. Start with the base sequence.
        # 2. Iterate through all candidates in increasing order.
        # 3. If candidate is not in base_seq, try to insert it.
        #    To check validity, we can temporarily build the new sequence and check all products?
        #    That's O(L^3) per insertion attempt. Too slow if done naively for many attempts.
        
        # Alternative Repair Strategy:
        # The "holes" are just missing numbers.
        # We can simply iterate through candidates 2..128.
        # If x is not in the current sequence, try to add it.
        # Since we are building the sequence from scratch in sorted order, we can just use the greedy logic
        # but start with the elements from `seq` already "accepted" at their positions?
        
        # Actually, the simplest repair is:
        # Take the set of numbers in `seq`.
        # Run a greedy pass over all candidates 2..128.
        # If a candidate is in `seq`, we MUST include it (to preserve the core).
        # If a candidate is not in `seq`, we include it if valid.
        # This ensures we keep the good parts and try to fill gaps.
        
        forced = set(seq)
        new_seq = []
        products = set()
        
        for x in candidates:
            if x in forced:
                # Must include x. Check if it's valid given current new_seq
                valid, new_prods = is_valid_addition(x, new_seq, products)
                if not valid:
                    # If a forced element causes conflict, the repair fails or we have to drop it?
                    # In ILS, we usually drop the forced element if it can't fit.
                    # But we want to preserve the structure.
                    # If it can't fit, we skip it (effectively removing it from the core).
                    continue
                new_seq.append(x)
                products.update(new_prods)
            else:
                # Optional element. Include if valid.
                valid, new_prods = is_valid_addition(x, new_seq, products)
                if valid:
                    new_seq.append(x)
                    products.update(new_prods)
        
        return new_seq, products

    # ILS Loop
    temperature = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Perturbation
        if random.random() < 0.1:
            # Strong kick: remove 5-10 elements
            k = random.randint(5, 10)
        else:
            # Weak kick: remove 1-3 elements
            k = random.randint(1, 3)
        
        if len(current_seq) <= k:
            # Can't remove more than we have
            k = max(1, len(current_seq) - 1) if len(current_seq) > 1 else 0
        
        if k == 0:
            continue
            
        # Randomly select k elements to remove
        to_remove = random.sample(current_seq, k)
        perturbed_seq = [x for x in current_seq if x not in to_remove]
        
        # Repair
        new_seq, new_products = repair(perturbed_seq)
        new_len = len(new_seq)
        
        # Acceptance
        if new_len > best_len:
            best_len = new_len
            best_seq = new_seq
            best_products = new_products
            current_seq = new_seq
            current_products = new_products
            current_len = new_len
            temperature = 1.0 # Reset temp on improvement
        elif new_len == current_len:
            # Accept with some probability to explore
            if random.random() < 0.5:
                current_seq = new_seq
                current_products = new_products
        else:
            # Worse. Accept with SA probability
            delta = current_len - new_len
            # Simple probability
            p = 1.0 / (1.0 + delta * delta)
            if random.random() < p:
                current_seq = new_seq
                current_products = new_products
                current_len = new_len
        
        temperature *= cooling

    return best_seq

# Run a quick sanity check if executed directly
if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(res)
