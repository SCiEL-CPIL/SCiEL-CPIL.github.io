import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the list of triplet IDs it covers
    block_triplets = []
    for block in all_blocks:
        tids = []
        for triplet in itertools.combinations(block, t):
            tids.append(triplet_to_id[triplet])
        block_triplets.append(tids)
        
    # Map each triplet to the list of block IDs that cover it
    blocks_covering = [[] for _ in range(n_triplets)]
    for b_idx, tids in enumerate(block_triplets):
        for tid in tids:
            blocks_covering[tid].append(b_idx)

    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        # Initial score: number of triplets in the block (which is C(k,t))
        score = np.array([len(tids) for tids in block_triplets], dtype=np.int32)
        used_blocks = set()
        solution = []
        
        # We need to find the max score block efficiently.
        # Since scores only decrease, we can just scan.
        # To add randomness to tie-breaking, we can add a small random noise to scores initially?
        # Or just pick random among max.
        
        while True:
            if time.time() > deadline - 20:
                break
                
            # Find max score block not in used_blocks
            max_score = -1
            best_block_idx = -1
            candidates = []
            
            # Optimization: if many blocks are used, scanning all is still O(N_blocks)
            # 346k is small enough.
            for b_idx in range(n_blocks):
                if b_idx not in used_blocks:
                    s = score[b_idx]
                    if s > max_score:
                        max_score = s
                        best_block_idx = b_idx
                        candidates = [b_idx]
                    elif s == max_score:
                        candidates.append(b_idx)
                        
            if best_block_idx == -1:
                break
                
            # Randomly pick from candidates to diversify
            chosen_block = rng.choice(candidates)
            solution.append(all_blocks[chosen_block])
            used_blocks.add(chosen_block)
            
            # Update coverage and scores
            newly_covered = []
            for tid in block_triplets[chosen_block]:
                if covered_count[tid] == 0:
                    newly_covered.append(tid)
                covered_count[tid] += 1
                
            for tid in newly_covered:
                for b_idx in blocks_covering[tid]:
                    score[b_idx] -= 1
                    
            if np.all(covered_count > 0):
                break
                
        return solution

    def local_search_remove(blocks):
        # Re-compute covered_count from scratch for safety? Or maintain?
        # Maintaining is tricky if we restart. Let's re-compute.
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                covered_count[tid] += 1
                
        new_blocks = []
        changed = True
        while changed:
            changed = False
            next_blocks = []
            for block in blocks:
                # Check if block is redundant
                redundant = True
                for triplet in itertools.combinations(block, t):
                    tid = triplet_to_id[triplet]
                    if covered_count[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for triplet in itertools.combinations(block, t):
                        tid = triplet_to_id[triplet]
                        covered_count[tid] -= 1
                    changed = True
                else:
                    next_blocks.append(block)
            blocks = next_blocks
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds
    num_seeds = 50
    for seed in range(num_seeds):
        if time.time() > deadline - 30:
            break
        blocks = greedy_set_cover(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = greedy_set_cover(0)
        
    # Local search polish
    best_blocks = local_search_remove(best_blocks)
    
    return best_blocks

if __name__ == "__main__":
    pass
