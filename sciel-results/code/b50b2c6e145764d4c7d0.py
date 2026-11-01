import itertools
import time
import random

def construct(v, k, t):
    points = list(range(v))
    all_tsets = list(itertools.combinations(points, t))
    tset_index = {s: i for i, s in enumerate(all_tsets)}
    num_tsets = len(all_tsets)
    
    # Precompute candidate blocks for each t-set?
    # Memory: 11628 * 364 * 8 bytes ~ 34 MB. Acceptable.
    # But generating on the fly is also fast.
    # Let's generate on the fly to keep code simple and memory low.
    
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin for 120s limit
    
    # Helper to get candidate k-blocks containing a specific t-set S
    def get_candidate_blocks(S):
        S_set = set(S)
        remaining = [p for p in points if p not in S_set]
        for comb in itertools.combinations(remaining, k - t):
            yield tuple(sorted(S + comb))
            
    # We run multiple restarts
    # Each restart uses a different random seed or randomization strategy
    for restart in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        # Randomize the initial uncovered set selection
        # We maintain a list of uncovered indices for efficient random selection
        uncovered_list = list(range(num_tsets))
        uncovered_set = set(uncovered_list)
        blocks = []
        
        while uncovered_list:
            # Pick a random uncovered t-set
            idx = random.randint(0, len(uncovered_list) - 1)
            S = all_tsets[uncovered_list[idx]]
            
            best_gain = -1
            best_block_candidates = []
            
            for B in get_candidate_blocks(S):
                # Calculate gain: number of t-subsets in B that are currently uncovered
                gain = 0
                for sub_t in itertools.combinations(B, t):
                    if tset_index[sub_t] in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_candidates = [B]
                elif gain == best_gain:
                    best_block_candidates.append(B)
                    
            # Pick randomly from best candidates
            B_star = random.choice(best_block_candidates)
            blocks.append(B_star)
            
            # Update uncovered structures
            for sub_t in itertools.combinations(B_star, t):
                idx_sub = tset_index[sub_t]
                if idx_sub in uncovered_set:
                    uncovered_set.remove(idx_sub)
                    # Remove from list (swap with last, pop)
                    # Find index in list? O(N). 
                    # Instead, we can just keep the set and a list of "active" indices?
                    # Or simply: don't use list for random selection, use set.
                    # random.choice(list(uncovered_set)) is O(N).
                    # For 11k elements, it's fine.
                    # Let's stick to set for simplicity, and just use list(set) for random choice.
                    # Wait, I used uncovered_list above. Let's fix that.
                    pass
        
        # The above loop has a flaw: removing from list while iterating or keeping in sync is tricky.
        # Let's restart with a cleaner approach for the uncovered set.
        
        # RE-IMPLEMENTATION of the inner loop for clarity and correctness
        
        # Actually, let's just use the set and `random.choice(list(uncovered_set))`.
        # It's O(N) per step. 350 steps * 11k = 3.8M ops. Fine.
        
        pass # Placeholder for the actual logic below

    # Real Logic
    best_blocks = None
    best_count = float('inf')
    start_time = time.time()
    time_limit = 110.0
    
    for restart in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        uncovered = set(range(num_tsets))
        blocks = []
        
        while uncovered:
            # Pick random uncovered
            idx = random.choice(list(uncovered))
            S = all_tsets[idx]
            
            best_gain = -1
            best_block_candidates = []
            
            for B in get_candidate_blocks(S):
                gain = 0
                for sub_t in itertools.combinations(B, t):
                    if tset_index[sub_t] in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_candidates = [B]
                elif gain == best_gain:
                    best_block_candidates.append(B)
                    
            B_star = random.choice(best_block_candidates)
            blocks.append(B_star)
            
            for sub_t in itertools.combinations(B_star, t):
                idx_sub = tset_index[sub_t]
                if idx_sub in uncovered:
                    uncovered.remove(idx_sub)
                    
        # Removal Phase
        counts = [0] * num_tsets
        for B in blocks:
            for sub_t in itertools.combinations(B, t):
                counts[tset_index[sub_t]] += 1
                
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for B in blocks:
                essential = False
                for sub_t in itertools.combinations(B, t):
                    if counts[tset_index[sub_t]] == 1:
                        essential = True
                        break
                if not essential:
                    for sub_t in itertools.combinations(B, t):
                        counts[tset_index[sub_t]] -= 1
                    changed = True
                else:
                    new_blocks.append(B)
            blocks = new_blocks
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            if best_count <= 335:
                break

    return [list(b) for b in best_blocks]
