import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute all t-subsets and map them to indices
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the list of subset indices it covers
    block_subsets = []
    for block in all_blocks:
        indices = [subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_subsets.append(indices)
    
    # Map each subset to the list of blocks that cover it
    subset_blocks = [[] for _ in range(n_subsets)]
    for b_idx, indices in enumerate(block_subsets):
        for idx in indices:
            subset_blocks[idx].append(b_idx)
    
    # Greedy construction
    cover_count = [0] * n_subsets
    blocks = []
    block_set = set()
    uncovered = set(range(n_subsets))
    
    while uncovered and time.time() < deadline:
        best_block = -1
        best_count = -1
        
        for b_idx in range(n_blocks):
            if b_idx in block_set:
                continue
            b_subs = block_subsets[b_idx]
            cnt = 0
            for idx in b_subs:
                if cover_count[idx] == 0:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_block = b_idx
                if best_count == len(uncovered):
                    break
        
        if best_block == -1 or best_count == 0:
            break
            
        blocks.append(best_block)
        block_set.add(best_block)
        for idx in block_subsets[best_block]:
            cover_count[idx] += 1
            if cover_count[idx] == 1:
                uncovered.discard(idx)
    
    # Local Search
    def is_redundant(b_idx):
        for idx in block_subsets[b_idx]:
            if cover_count[idx] <= 1:
                return False
        return True

    def remove_block(b_idx):
        nonlocal blocks, block_set, cover_count
        for idx in block_subsets[b_idx]:
            cover_count[idx] -= 1
        blocks.remove(b_idx)
        block_set.discard(b_idx)

    def add_block(b_idx):
        nonlocal blocks, block_set, cover_count
        for idx in block_subsets[b_idx]:
            cover_count[idx] += 1
        blocks.append(b_idx)
        block_set.add(b_idx)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Reduction Phase
        i = 0
        while i < len(blocks):
            if is_redundant(blocks[i]):
                remove_block(blocks[i])
                improved = True
            else:
                i += 1
                
        # 2. Swap Phase: try replacing each block with a better one
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
                
            B = blocks[i]
            b_subs = block_subsets[B]
            
            # Find critical subsets (covered only by B)
            critical = [idx for idx in b_subs if cover_count[idx] == 1]
            
            if not critical:
                continue
                
            # Union of elements in critical subsets
            U_B = set()
            for idx in critical:
                U_B.update(all_subsets[idx])
                
            if len(U_B) > k:
                continue
                
            # Find best replacement block B' containing U_B
            remaining_points = [p for p in range(v) if p not in U_B]
            num_to_choose = k - len(U_B)
            
            best_B_prime = -1
            best_score = -1
            
            for candidate_extra in itertools.combinations(remaining_points, num_to_choose):
                B_prime = tuple(sorted(list(U_B) + list(candidate_extra)))
                
                if B_prime == B:
                    continue
                
                if B_prime in block_set:
                    continue
                    
                # Calculate score: number of subsets in B' that are already well-covered
                b_prime_subs = block_subsets[list(all_blocks).index(B_prime)] if B_prime in all_blocks else None
                if b_prime_subs is None:
                    continue
                    
                score = 0
                for idx in b_prime_subs:
                    if cover_count[idx] > 1:
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_B_prime = B_prime
                    
            if best_B_prime != -1:
                # Simulate replacement
                b_subs_B = block_subsets[B]
                b_subs_Bp = block_subsets[list(all_blocks).index(best_B_prime)]
                
                for idx in b_subs_B:
                    cover_count[idx] -= 1
                for idx in b_subs_Bp:
                    cover_count[idx] += 1
                    
                blocks[i] = best_B_prime
                block_set.discard(B)
                block_set.add(best_B_prime)
                
                # Check if any block is now redundant
                redundant_found = False
                for j in range(len(blocks)):
                    if is_redundant(blocks[j]):
                        redundant_found = True
                        break
                        
                if redundant_found:
                    improved = True
                    break
                else:
                    # Revert
                    for idx in b_subs_Bp:
                        cover_count[idx] -= 1
                    for idx in b_subs_B:
                        cover_count[idx] += 1
                    blocks[i] = B
                    block_set.discard(best_B_prime)
                    block_set.add(B)

    # Final polish: try to remove blocks in random order
    for _ in range(10):
        if time.time() > deadline:
            break
        random.shuffle(blocks)
        i = 0
        while i < len(blocks):
            if is_redundant(blocks[i]):
                remove_block(blocks[i])
            else:
                i += 1

    return blocks
