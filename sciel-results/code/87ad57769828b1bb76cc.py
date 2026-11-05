import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # 1. Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_to_index = {b: i for i, b in enumerate(all_blocks)}
    
    triplet_to_blocks = [[] for _ in range(n_triplets)]
    block_masks = [0] * n_blocks
    
    for bi, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
            triplet_to_blocks[tid].append(bi)
        block_masks[bi] = mask
        
    def update_counts(blocks, counts, delta):
        for b in blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += delta

    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        selected_indices = []
        
        while uncovered:
            lsb = uncovered & -uncovered
            tid = lsb.bit_length() - 1
            
            candidate_blocks = triplet_to_blocks[tid]
            
            best_block_idx = -1
            max_gain = -1
            
            for bi in candidate_blocks:
                gain = (block_masks[bi] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = bi
                elif gain == max_gain:
                    if random.random() < 0.5:
                        best_block_idx = bi

            if best_block_idx != -1:
                selected_indices.append(best_block_idx)
                uncovered &= ~block_masks[best_block_idx]
            else:
                break
                
        return [all_blocks[i] for i in selected_indices]

    def optimize(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        update_counts(blocks, counts, 1)
        
        max_iters = 3000
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            n = len(blocks)
            
            # 1. Remove Redundant
            for i in range(n - 1, -1, -1):
                b = blocks[i]
                redundant = True
                for triplet in itertools.combinations(b, t):
                    if counts[triplet_to_id[triplet]] == 1:
                        redundant = False
                        break
                if redundant:
                    for triplet in itertools.combinations(b, t):
                        counts[triplet_to_id[triplet]] -= 1
                    blocks.pop(i)
                    improved = True
                    break
                    
            if improved:
                continue
                
            # 2. 2-for-1
            found_swap = False
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > deadline - 10:
                    break
                    
                b1 = blocks[i]
                b2 = blocks[j]
                
                unique_triplets = []
                for triplet in itertools.combinations(b1, t):
                    tid = triplet_to_id[triplet]
                    if counts[tid] == 1:
                        unique_triplets.append(tid)
                        
                for triplet in itertools.combinations(b2, t):
                    tid = triplet_to_id[triplet]
                    if counts[tid] == 1 and tid not in unique_triplets:
                        unique_triplets.append(tid)
                        
                if not unique_triplets:
                    continue
                
                candidate_blocks = set(triplet_to_blocks[unique_triplets[0]])
                for tid in unique_triplets[1:]:
                    candidate_blocks &= set(triplet_to_blocks[tid])
                    if not candidate_blocks:
                        break
                        
                if not candidate_blocks:
                    continue
                    
                valid_candidates = [c for c in candidate_blocks if all_blocks[c] != b1 and all_blocks[c] != b2]
                if not valid_candidates:
                    valid_candidates = list(candidate_blocks)
                    
                if not valid_candidates:
                    continue
                    
                new_bi = random.choice(valid_candidates)
                new_block = all_blocks[new_bi]
                
                for triplet in itertools.combinations(b1, t):
                    counts[triplet_to_id[triplet]] -= 1
                for triplet in itertools.combinations(b2, t):
                    counts[triplet_to_id[triplet]] -= 1
                for triplet in itertools.combinations(new_block, t):
                    counts[triplet_to_id[triplet]] += 1
                    
                blocks.pop(j)
                blocks.pop(i)
                blocks.append(new_block)
                
                improved = True
                found_swap = True
                break
                
            if found_swap:
                continue
                
            # 3. 1-for-1
            if n > 0:
                i = random.randint(0, n - 1)
                b1 = blocks[i]
                
                unique_triplets = []
                for triplet in itertools.combinations(b1, t):
                    tid = triplet_to_id[triplet]
                    if counts[tid] == 1:
                        unique_triplets.append(tid)
                        
                if unique_triplets:
                    candidate_blocks = set(triplet_to_blocks[unique_triplets[0]])
                    for tid in unique_triplets[1:]:
                        candidate_blocks &= set(triplet_to_blocks[tid])
                        if not candidate_blocks:
                            break
                    
                    b1_idx = block_to_index[b1]
                    candidate_blocks.discard(b1_idx)
                    
                    valid_candidates = list(candidate_blocks)
                    
                    if valid_candidates:
                        new_bi = random.choice(valid_candidates)
                        new_block = all_blocks[new_bi]
                        
                        for triplet in itertools.combinations(b1, t):
                            counts[triplet_to_id[triplet]] -= 1
                        for triplet in itertools.combinations(new_block, t):
                            counts[triplet_to_id[triplet]] += 1
                            
                        blocks[i] = new_block
                        improved = True
                        continue

            if not improved:
                break
                
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    n_restarts = 30
    
    for seed in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        blocks = greedy_construct(seed)
        opt_blocks = optimize(blocks)
        
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1
                
        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks
                
    if best_blocks is None:
        best_blocks = greedy_construct(0)
        
    return best_blocks

if __name__ == "__main__":
    pass
