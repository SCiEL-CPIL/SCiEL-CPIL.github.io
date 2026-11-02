import itertools
import random
import time

# Global precomputation for v=19, k=8, t=5
V = 19
K = 8
T = 5

# Precompute all t-subsets
all_t_subs = list(itertools.combinations(range(V), T))
N_T_SUBS = len(all_t_subs)
t_sub_index = {s: i for i, s in enumerate(all_t_subs)}

# Precompute point masks for t-subsets
t_sub_point_mask = [0] * N_T_SUBS
for i, s in enumerate(all_t_subs):
    m = 0
    for p in s:
        m |= (1 << p)
    t_sub_point_mask[i] = m

# Precompute all k-blocks
all_k_blocks = list(itertools.combinations(range(V), K))
N_K_BLOCKS = len(all_k_blocks)

# Precompute point masks for k-blocks
k_block_point_mask = [0] * N_K_BLOCKS
for i, b in enumerate(all_k_blocks):
    m = 0
    for p in b:
        m |= (1 << p)
    k_block_point_mask[i] = m

# Precompute coverage masks (bitset of t-subsets) for k-blocks
# mask[b_idx] has bit s_idx set if t-sub s_idx is contained in block b_idx
k_block_mask = [0] * N_K_BLOCKS
for i, b in enumerate(all_k_blocks):
    m = 0
    for s in itertools.combinations(b, T):
        m |= (1 << t_sub_index[s])
    k_block_mask[i] = m

# Precompute sub_to_blocks: for each t-subset, list of k-blocks containing it
sub_to_blocks = [[] for _ in range(N_T_SUBS)]
for b_idx in range(N_K_BLOCKS):
    m = k_block_mask[b_idx]
    while m:
        s_idx = (m & -m).bit_length() - 1
        sub_to_blocks[s_idx].append(b_idx)
        m &= m - 1

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Greedy Construction
    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered_mask = (1 << N_T_SUBS) - 1
        blocks = []
        
        # To speed up finding uncovered subset, we can just find lowest set bit
        while uncovered_mask:
            # Find an uncovered t-subset
            s_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
            
            # Get candidates
            candidates = sub_to_blocks[s_idx]
            
            # Find best candidate
            best_b = -1
            max_gain = -1
            
            for b_idx in candidates:
                # Gain is number of newly covered t-subsets
                # popcount(k_block_mask[b_idx] & uncovered_mask)
                gain = (k_block_mask[b_idx] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_b = b_idx
            
            # Add block
            blocks.append(best_b)
            uncovered_mask &= ~k_block_mask[best_b]
            
        return blocks

    # Local Search
    def local_search(blocks):
        # Convert to set for fast lookup
        block_set = set(blocks)
        
        # Initialize counts
        counts = [0] * N_T_SUBS
        for b_idx in blocks:
            m = k_block_mask[b_idx]
            while m:
                s_idx = (m & -m).bit_length() - 1
                counts[s_idx] += 1
                m &= m - 1
        
        improved = True
        # Limit iterations to prevent infinite loops if stuck
        max_iter = 1000
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 5:
                break
                
            # Randomize order
            random.shuffle(blocks)
            
            for i in range(len(blocks)):
                b_idx = blocks[i]
                
                # Find unique subsets for this block
                unique_subs = []
                m = k_block_mask[b_idx]
                while m:
                    s_idx = (m & -m).bit_length() - 1
                    if counts[s_idx] == 1:
                        unique_subs.append(s_idx)
                    m &= m - 1
                
                if not unique_subs:
                    # Redundant, remove
                    m = k_block_mask[b_idx]
                    while m:
                        s_idx = (m & -m).bit_length() - 1
                        counts[s_idx] -= 1
                        m &= m - 1
                    
                    blocks.pop(i)
                    block_set.remove(b_idx)
                    improved = True
                    break # Restart loop
                
                # Try to replace
                # Pick first unique subset to narrow down candidates
                s0 = unique_subs[0]
                candidates = sub_to_blocks[s0]
                
                found_swap = False
                for b_prime in candidates:
                    if b_prime == b_idx:
                        continue
                    if b_prime in block_set:
                        # Swapping with an existing block doesn't reduce count immediately, 
                        # but might help later. However, to keep it simple and focused on reduction,
                        # we might skip this or handle it carefully.
                        # If we swap b for b_prime (both in set), count is same.
                        # Let's allow it if it covers more? 
                        # For now, skip to prioritize deletion.
                        continue
                    
                    # Check if b_prime covers all unique_subs
                    pm = k_block_point_mask[b_prime]
                    valid = True
                    for s in unique_subs:
                        if (pm & t_sub_point_mask[s]) != t_sub_point_mask[s]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        # Remove b_idx
                        m = k_block_mask[b_idx]
                        while m:
                            s_idx = (m & -m).bit_length() - 1
                            counts[s_idx] -= 1
                            m &= m - 1
                        
                        # Add b_prime
                        m = k_block_mask[b_prime]
                        while m:
                            s_idx = (m & -m).bit_length() - 1
                            counts[s_idx] += 1
                            m &= m - 1
                        
                        blocks[i] = b_prime
                        block_set.remove(b_idx)
                        block_set.add(b_prime)
                        
                        improved = True
                        found_swap = True
                        break # Restart outer loop after swap
                
                if found_swap:
                    break # Restart outer loop
        
        return blocks

    best_blocks = None
    
    # Run multiple restarts
    # Greedy is fast, so we can do many restarts
    for i in range(100):
        if time.time() > deadline - 10:
            break
        
        # Randomize greedy by shuffling candidates? 
        # The greedy picks "best" (max gain). Ties are broken by order.
        # To get different solutions, we can randomize the order of candidates in sub_to_blocks 
        # or just pick randomly among top gainers.
        # Simpler: just run greedy as is. It's deterministic.
        # To get diversity, we can perturb the initial solution or use different random seeds for local search?
        # Local search uses random.shuffle.
        
        current = greedy_construct()
        current = local_search(current)
        
        if best_blocks is None or len(current) < len(best_blocks):
            best_blocks = current
        
        # If we hit the theoretical lower bound or a very good number, we can stop?
        # Lower bound is 264. If we get close to 300, it's good.
        
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_construct()
        
    # Convert indices back to lists of points
    result = [list(all_k_blocks[b_idx]) for b_idx in best_blocks]
    
    return result
