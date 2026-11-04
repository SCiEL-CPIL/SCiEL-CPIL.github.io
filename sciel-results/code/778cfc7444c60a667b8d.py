import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Precompute all t-subsets and map to IDs
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_masks[i] is an integer bitmask of triplets covered by block i
    block_masks = [0] * n_blocks
    # blocks_covering_triplet[tid] is a list of block indices that cover triplet tid
    blocks_covering_triplet = [[] for _ in range(n_triplets)]
    
    for idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
            blocks_covering_triplet[tid].append(idx)
        block_masks[idx] = mask
        
    # Greedy Construction
    def greedy_construct():
        covered = 0
        all_mask = (1 << n_triplets) - 1
        uncovered = all_mask
        blocks = []
        block_indices = []
        
        while uncovered:
            if time.time() > deadline - 30:
                break
                
            best_idx = -1
            best_gain = 0
            
            for i in range(n_blocks):
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if gain == n_triplets:
                        break
            
            if best_idx == -1:
                break
                
            blocks.append(all_blocks[best_idx])
            block_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return blocks, block_indices

    blocks, block_indices = greedy_construct()
    
    # Local Search
    def calculate_counts(block_indices_list):
        counts = np.zeros(n_triplets, dtype=np.int32)
        for idx in block_indices_list:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                tid = lsb.bit_length() - 1
                counts[tid] += 1
                m ^= lsb
        return counts

    def find_redundant_and_swaps(block_indices_list):
        counts = calculate_counts(block_indices_list)
        unique_masks = [0] * len(block_indices_list)
        
        for i, b_idx in enumerate(block_indices_list):
            mask = block_masks[b_idx]
            unique_mask = 0
            m = mask
            while m:
                lsb = m & -m
                tid = lsb.bit_length() - 1
                if counts[tid] == 1:
                    unique_mask |= lsb
                m ^= lsb
            unique_masks[i] = unique_mask
            
        moves = []
        for i in range(len(block_indices_list)):
            if unique_masks[i] == 0:
                moves.append(('remove', i))
                
        in_solution_set = set(block_indices_list)
        
        for i in range(len(block_indices_list)):
            u_mask = unique_masks[i]
            if u_mask == 0:
                continue
                
            candidates = None
            m = u_mask
            while m:
                lsb = m & -m
                tid = lsb.bit_length() - 1
                cand_list = blocks_covering_triplet[tid]
                if candidates is None:
                    candidates = set(cand_list)
                else:
                    candidates.intersection_update(cand_list)
                m ^= lsb
                
            if candidates:
                current_idx = block_indices_list[i]
                candidates.discard(current_idx)
                candidates = [c for c in candidates if c not in in_solution_set]
                
                if candidates:
                    new_idx = candidates[0]
                    moves.append(('swap', i, new_idx))
                    
        return moves

    best_blocks = blocks[:]
    best_count = len(blocks)
    current_indices = block_indices[:]
    
    iterations = 0
    max_iterations = 5000
    
    while time.time() < deadline - 5 and iterations < max_iterations:
        moves = find_redundant_and_swaps(current_indices)
        
        if not moves:
            if len(current_indices) > 2:
                remove_count = min(3, len(current_indices) // 2)
                to_remove = random.sample(range(len(current_indices)), remove_count)
                to_remove.sort(reverse=True)
                for i in to_remove:
                    del current_indices[i]
                    
                counts = calculate_counts(current_indices)
                uncovered_mask = 0
                for tid in range(n_triplets):
                    if counts[tid] == 0:
                        uncovered_mask |= (1 << tid)
                        
                while uncovered_mask:
                    best_idx = -1
                    best_gain = 0
                    for i in range(n_blocks):
                        gain = (block_masks[i] & uncovered_mask).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = i
                    if best_idx == -1: break
                    current_indices.append(best_idx)
                    uncovered_mask &= ~block_masks[best_idx]
                    
            iterations += 1
            continue
            
        removals = [m for m in moves if m[0] == 'remove']
        if removals:
            move = random.choice(removals)
        else:
            move = random.choice(moves)
            
        if move[0] == 'remove':
            i = move[1]
            current_indices = current_indices[:i] + current_indices[i+1:]
        else:
            i, new_idx = move[1], move[2]
            current_indices = current_indices[:i] + [new_idx] + current_indices[i+1:]
            
        iterations += 1
        
        if len(current_indices) < best_count:
            best_count = len(current_indices)
            best_blocks = [all_blocks[i] for i in current_indices]
            
    return best_blocks

if __name__ == "__main__":
    pass
