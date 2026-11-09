import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute base subsets
    base_t_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_len = float('inf')
    
    total_t_subsets = len(base_t_subsets)
    num_k_subsets = len(list(itertools.combinations(range(k), t)))
    lower_bound = -(-total_t_subsets // num_k_subsets)
    
    while time.time() < deadline:
        # Create a random permutation
        perm = list(range(v))
        random.shuffle(perm)
        inv_perm = [0] * v
        for i, p in enumerate(perm):
            inv_perm[p] = i
            
        # Generate permuted subsets
        uncovered = set()
        for s in base_t_subsets:
            permuted_s = tuple(perm[x] for x in s)
            uncovered.add(permuted_s)
            
        blocks = []
        
        while uncovered:
            start_subset = min(uncovered)
            block = list(start_subset)
            
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if not best_candidates:
                    break
                    
                block.append(random.choice(best_candidates))
                
            block.sort()
            blocks.append(block)
            
            for subset in itertools.combinations(block, t):
                uncovered.discard(subset)
        
        # Map back
        original_blocks = []
        for b in blocks:
            original_b = [inv_perm[x] for x in b]
            original_b.sort()
            original_blocks.append(original_b)
            
        if len(original_blocks) < best_len:
            best_len = len(original_blocks)
            best_blocks = original_blocks
            
        if best_len <= lower_bound:
            break
            
    return best_blocks
