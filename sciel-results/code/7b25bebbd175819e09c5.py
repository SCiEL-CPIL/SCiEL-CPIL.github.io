import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    # Parameters for this specific task
    # v=24, k=7, t=3
    
    # Precompute triple indices
    # Map each combination (a, b, c) with a < b < c to a unique ID
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_id = {triple: i for i, triple in enumerate(all_triples)}
    id_to_triple = {i: triple for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute a 3D lookup table for triple IDs to allow O(1) access
    # idx[u][v][w] gives the ID of the triple {u,v,w} if u,v,w are distinct, else -1
    # Using a flat array for better cache performance? Or just a list of lists.
    # 24x24x24 is small.
    idx_table = np.full((v, v, v), -1, dtype=np.int32)
    for a in range(v):
        for b in range(v):
            if a == b: continue
            for c in range(v):
                if a == c or b == c: continue
                # Ensure consistent ordering for the dictionary lookup
                p, q, r = sorted((a, b, c))
                idx_table[a, b, c] = triple_to_id[(p, q, r)]

    def run_greedy(seed_val):
        random.seed(seed_val)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        blocks = []
        
        # List of currently uncovered triple IDs
        # We can just check cover_count == 0 each time, or maintain a list.
        # Maintaining a list is better for finding "first" uncovered quickly if we shuffle.
        # But for greedy, we often pick the one with highest "pressure" or just first.
        # Let's pick a random uncovered triple to introduce diversity.
        
        # To speed up finding uncovered triples, we can maintain a list and filter it occasionally,
        # or just use np.where. np.where is fast for 2000 elements.
        
        while True:
            uncovered_ids = np.where(cover_count == 0)[0]
            if len(uncovered_ids) == 0:
                break
            
            # Pick a random uncovered triple to start the block
            # This helps in escaping local minima compared to always picking the first
            start_tid = random.choice(uncovered_ids)
            a, b, c = id_to_triple[start_tid]
            block = [a, b, c]
            
            # Add k - t more points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                
                # Calculate gain for each candidate point
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Gain is the number of NEW triples covered by adding x
                    # New triples are formed by x and any pair in block
                    gain = 0
                    for i in range(len(block)):
                        for j in range(i + 1, len(block)):
                            u = block[i]
                            w = block[j]
                            tid = idx_table[u, w, x]
                            if tid != -1 and cover_count[tid] == 0:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point != -1:
                        # Tie-breaking: random to diversify
                        if random.random() < 0.5:
                            best_point = x
                            
                if best_point == -1:
                    # Fallback, should not happen if v > k
                    break
                block.append(best_point)
            
            # Sort block for consistency (though not strictly required for validity)
            block.sort()
            blocks.append(block)
            
            # Update cover counts for all triples in this block
            for u, w, z in itertools.combinations(block, t):
                tid = triple_to_id[(u, w, z)]
                cover_count[tid] += 1
                
        return blocks

    # Main driver: run multiple restarts
    deadline = time.time() + 100.0  # 100s safety margin
    best_blocks = None
    best_len = float('inf')
    
    # First run is deterministic (seed 0)
    current_blocks = run_greedy(0)
    if len(current_blocks) < best_len:
        best_len = len(current_blocks)
        best_blocks = current_blocks
        
    seed_val = 1
    while time.time() < deadline:
        current_blocks = run_greedy(seed_val)
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks
            # If we hit the theoretical lower bound or known best, we can stop early?
            # Known best is 76. Lower bound 69.
            # If we find 75, that's a record.
        seed_val += 1
        
        # Optional: If we've been running for a while and haven't improved, 
        # we might just return. But more restarts can only help find a better solution.
        
    return best_blocks
