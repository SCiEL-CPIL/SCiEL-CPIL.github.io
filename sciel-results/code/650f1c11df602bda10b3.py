import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    best_seq = []
    best_len = 0
    
    # Helper to check if a sequence is valid and return its length
    # This is used for final verification and potentially during search if needed
    def is_valid(seq):
        if not seq:
            return True
        if not all(2 <= x <= 128 for x in seq):
            return False
        if not all(seq[i] < seq[i+1] for i in range(len(seq)-1)):
            return False
        
        products = set()
        # Check all consecutive blocks
        # For a sequence of length n, there are n*(n+1)/2 blocks.
        # For n~20-30, this is manageable.
        for i in range(len(seq)):
            prod = 1
            for j in range(i, len(seq)):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def greedy_construct(pool, seed_val=None):
        """
        Greedily construct a sequence from the given pool.
        If seed_val is provided, it must be the first element.
        Otherwise, we pick the first element from the pool that allows the longest greedy extension?
        No, standard greedy: iterate through pool, add if valid.
        To get diversity, we shuffle the pool.
        """
        seq = []
        prods = set()
        
        # If seed_val is provided, we must start with it
        # But the pool might not contain it? The pool is a subset of [2, 128].
        # Let's assume the pool is a list of candidates we can pick from, in some order.
        
        # For a true greedy that maximizes length, we usually just iterate through sorted candidates.
        # But to get different results, we shuffle the order of consideration?
        # If we shuffle, we might pick a small number late, which is bad for increasing order?
        # The sequence MUST be strictly increasing.
        # So the order of selection is constrained by value.
        # We cannot just shuffle and pick any. We must pick increasing values.
        
        # Strategy:
        # 1. Generate a random subset of [2, 128] to act as "available" candidates?
        #    No, we can use any.
        # 2. Randomize the *priority* of candidates?
        #    Since they must be increasing, the "order" is fixed by value.
        #    However, we can skip certain values.
        
        # Better Randomized Greedy Strategy:
        # Iterate through candidates x in 2..128.
        # With some probability, skip x even if it's valid?
        # Or, use a "lookahead" or different heuristic?
        
        # Simplest effective randomized greedy:
        # 1. Start with an empty sequence.
        # 2. Iterate x from 2 to 128.
        # 3. If x is valid to add, add it.
        # This is deterministic.
        
        # To make it randomized:
        # We can randomize the *start*? No, must be increasing.
        # We can randomize the *decision* to include a valid x?
        # If we skip a valid x, we might be able to include more later?
        # Or we might lose length.
        
        # Let's try:
        # For each x in 2..128:
        #   If x is valid to add:
        #     With probability p, add it.
        #     With probability 1-p, skip it.
        # This explores different subsets.
        
        # However, skipping a small number might allow a larger number to be added?
        # No, larger numbers are always "larger" in value, so they don't conflict with "increasing" constraint directly,
        # but they might create product conflicts.
        
        # Actually, a better randomized approach for increasing sequences:
        # Randomly permute the *weights* or use a random threshold?
        
        # Let's stick to a simpler robust method:
        # 1. Randomly select a starting point s from [2, 128].
        # 2. Greedily add the smallest valid x > last_val.
        # This is still deterministic for a given start.
        
        # Let's try: Randomly shuffle the candidate set [2..128].
        # Then iterate through the shuffled list.
        # Maintain current_max_val.
        # If candidate > current_max_val AND is valid: add it.
        # This ensures increasing order while allowing random skips.
        
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        
        seq = []
        prods = set()
        last_val = 1
        
        for x in candidates:
            if x <= last_val:
                continue
            
            # Check if x can be added
            # New suffix products
            new_prods = [x]
            prod = x
            # Check against existing products
            valid = True
            # We need to compute suffix products of (seq + [x])
            # These are: x, x*seq[-1], x*seq[-1]*seq[-2], ...
            
            # Optimization: Precompute suffix products of current seq?
            # No, just compute on the fly.
            
            if x in prods:
                valid = False
            
            if valid:
                # Check longer suffixes
                # We need to multiply x by suffixes of seq
                # Suffixes of seq are: seq[-1], seq[-2]*seq[-1], ...
                # Let's generate them
                suffix_prod = 1
                for i in range(len(seq) - 1, -1, -1):
                    suffix_prod *= seq[i]
                    new_prod = x * suffix_prod
                    if new_prod in prods:
                        valid = False
                        break
                    # We don't add to prods yet, just checking
            
            if valid:
                # Add x
                seq.append(x)
                # Add new products to prods
                prods.add(x)
                suffix_prod = x
                for i in range(len(seq) - 2, -1, -1):
                    suffix_prod *= seq[i]
                    prods.add(suffix_prod)
                last_val = x
            
        return seq

    # Multi-start loop
    # We want to run as many iterations as possible.
    # Each iteration is O(N^2) roughly, which is fast (128^2 ~ 16k ops).
    # We can run thousands of iterations.
    
    # To avoid duplicates in search, we can use a set of tuples of best sequences found?
    # Or just track best_len.
    
    # Let's also try a deterministic greedy as a baseline
    baseline_seq = greedy_construct(list(range(2, 129)))
    # Note: greedy_construct shuffles internally. 
    # For a deterministic baseline, we shouldn't shuffle.
    
    def deterministic_greedy():
        seq = []
        prods = set()
        last_val = 1
        for x in range(2, 129):
            if x <= last_val:
                continue
            valid = True
            if x in prods:
                valid = False
            else:
                suffix_prod = 1
                for i in range(len(seq) - 1, -1, -1):
                    suffix_prod *= seq[i]
                    new_prod = x * suffix_prod
                    if new_prod in prods:
                        valid = False
                        break
            if valid:
                seq.append(x)
                prods.add(x)
                suffix_prod = x
                for i in range(len(seq) - 2, -1, -1):
                    suffix_prod *= seq[i]
                    prods.add(suffix_prod)
                last_val = x
        return seq

    det_seq = deterministic_greedy()
    if len(det_seq) > best_len:
        best_len = len(det_seq)
        best_seq = det_seq.copy()
        
    # Now randomized multi-start
    # Use a fixed seed for reproducibility? No, let it be random.
    # But to ensure we don't get stuck in a local optimum of the random shuffle,
    # we just run many.
    
    num_restarts = 0
    while time.time() < deadline:
        # Randomized greedy
        seq = greedy_construct(None) # The function shuffles internally
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq.copy()
            # If we found a very good one, we might want to try to improve it?
            # For now, just keep searching for better.
            
        num_restarts += 1
        
        # Optional: If we hit a high score, we can slow down or change strategy?
        # For now, just keep running.
        
        # Check time more frequently if loops are fast
        if num_restarts % 100 == 0 and time.time() > deadline:
            break

    return best_seq

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Length: {len(res)}")
    print(res)
