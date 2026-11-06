"""Improved program using randomized restarts and multiple greedy heuristics."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_seq = []
    best_len = 0
    
    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
            
    def run_greedy(direction, p_accept=1.0):
        values = []
        sums = set()
        
        if direction == 'reverse':
            candidates = range(256, 0, -1)
        else:
            candidates = range(1, 257)
            
        for x in candidates:
            # Compute all new consecutive block sums
            # If adding to the end (reverse logic: we are building from large to small, so new x is at the 'front' of the final sorted list? 
            # No, the code appends to values.
            # If we iterate 256->1, values will be [256, 255, ...].
            # The consecutive blocks in the final sorted list [small, ..., large] 
            # are the same as in the reverse list [large, ..., small] just reversed?
            # Sum(a..b) is the same.
            # So checking tails in the current list `values` (which is descending) works.
            # Wait, if values = [256, 255], sorted is [255, 256].
            # Blocks in [255, 256]: 255, 256, 511.
            # Blocks in [256, 255] (if we treated it as a sequence): 256, 255, 511.
            # The set of sums is the same.
            # So we can build in any order and check sums of consecutive elements in that built order.
            
            tails = []
            total = x
            tails.append(total)
            for a in reversed(values):
                total += a
                tails.append(total)
                
            if not sums.intersection(tails):
                if p_accept >= 1.0 or random.random() < p_accept:
                    values.append(x)
                    sums.update(tails)
                    
        values.sort()
        return values

    # 1. Deterministic Reverse Greedy
    seq = run_greedy('reverse', 1.0)
    update_best(seq)
    
    # 2. Deterministic Forward Greedy
    seq = run_greedy('forward', 1.0)
    update_best(seq)
    
    # 3. Randomized Restarts
    # We vary the acceptance probability and direction
    while time.time() - start_time < 55:
        # Randomly choose direction and probability
        direction = random.choice(['reverse', 'forward'])
        p = random.uniform(0.6, 1.0)
        
        seq = run_greedy(direction, p)
        update_best(seq)
        
        # Occasionally try a very aggressive skip (low p) to find sparser but longer sequences?
        # Or just keep it simple.
        
    return best_seq
# EVOLVE-BLOCK-END
