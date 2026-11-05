import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all blocks and their triples
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # block_triples[b] is a list of triple indices
    block_triples = []
    for b in all_blocks:
        trs = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(trs)
        
    triples_per_block = len(block_triples[0])
    block_triples_np = np.array(block_triples, dtype=np.int32)
    
    # Inverse index: for each triple, which blocks contain it?
    triple_to_blocks = [[] for _ in range(num_triples)]
    for b_idx in range(num_blocks):
        for t_idx in block_triples_np[b_idx]:
            triple_to_blocks[t_idx].append(b_idx)
    
    # Convert triple_to_blocks to numpy arrays for faster intersection
    triple_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in triple_to_blocks]
    
    def greedy_construct_idx(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = set(range(num_triples))
        selected_indices = []
        
        while uncovered:
            # Find block covering most uncovered triples
            best_block = -1
            best_count = -1
            
            # Sample a subset of blocks to speed up
            if num_blocks > 1000:
                sample_size = min(num_blocks, 200)
                candidates = random.sample(range(num_blocks), sample_size)
            else:
                candidates = range(num_blocks)
            
            for b_idx in candidates:
                count = 0
                for t_idx in block_triples_np[b_idx]:
                    if t_idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected_indices.append(best_block)
            for t_idx in block_triples_np[best_block]:
                uncovered.discard(t_idx)
                
        return selected_indices

    def prune_indices(indices):
        if not indices:
            return [], np.zeros(num_triples, dtype=np.int32)
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        
        for b_idx in indices:
            cover_count[block_triples_np[b_idx]] += 1
            
        new_indices = []
        new_cover_count = np.zeros(num_triples, dtype=np.int32)
        
        for b_idx in indices:
            trs = block_triples_np[b_idx]
            # Check if any triple in this block is exclusively covered by it
            is_redundant = True
            for t_idx in trs:
                if cover_count[t_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                cover_count[trs] -= 1
            else:
                new_indices.append(b_idx)
                new_cover_count[trs] += 1
                
        return new_indices, new_cover_count

    def calculate_energy(indices, cover):
        # Energy is primarily the number of uncovered triples, secondarily the number of blocks
        # Uncovered triples are those with cover == 0
        uncovered_count = np.sum(cover == 0)
        block_count = len(indices)
        # We weight uncovered triples heavily to ensure validity is prioritized
        return uncovered_count * 1000 + block_count

    # Initial solution: try multiple greedy runs with different seeds
    best_indices = None
    best_cover = None
    best_size = float('inf')
    
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline - 30:
            break
        indices = greedy_construct_idx(seed)
        indices, cover = prune_indices(indices)
        if len(indices) < best_size:
            best_size = len(indices)
            best_indices = indices
            best_cover = cover
    
    if best_indices is None:
        best_indices = greedy_construct_idx(42)
        best_indices, best_cover = prune_indices(best_indices)
        best_size = len(best_indices)
    
    current_indices = best_indices.copy()
    current_cover = best_cover.copy()
    current_energy = calculate_energy(current_indices, current_cover)
    
    # Simulated Annealing
    # T0: initial temperature, Tf: final temperature
    # Alpha: cooling rate
    T = 50.0
    T_min = 0.01
    alpha = 0.999
    
    # To make SA effective, we need to allow moves that might temporarily increase energy
    # but hopefully lead to better solutions later.
    
    # We will run SA for the remaining time
    # We need to track the best solution found during SA
    
    # Optimization: Pre-compute a random permutation of blocks for sampling?
    # Or just use np.random.randint
    
    while time.time() < deadline and T > T_min:
        T *= alpha
        
        # Choose a move
        # 1. Add a random block
        # 2. Remove a random block
        # 3. Swap a random selected block with a random unselected block
        
        move_type = np.random.choice(3, p=[0.2, 0.3, 0.5])
        
        new_indices = current_indices.copy()
        new_cover = current_cover.copy()
        
        if move_type == 0: # Add
            # Pick a random block not in current_indices
            # To be efficient, we can pick a random block and check if it's in the set
            # Since num_blocks is ~300k and current size is ~70, collision is low
            # But checking membership in a list is O(N). Better to use a set for membership.
            # However, creating a set every step is O(N).
            # Let's just pick a random block. If it's already there, skip move (energy delta 0).
            b_idx = np.random.randint(num_blocks)
            if b_idx in current_indices:
                continue
            
            trs = block_triples_np[b_idx]
            new_cover[trs] += 1
            new_indices.append(b_idx)
            
        elif move_type == 1: # Remove
            if not current_indices:
                continue
            pos = np.random.randint(len(current_indices))
            b_idx = current_indices[pos]
            trs = block_triples_np[b_idx]
            new_cover[trs] -= 1
            new_indices.pop(pos)
            
        else: # Swap
            if not current_indices:
                continue
            pos = np.random.randint(len(current_indices))
            b_idx_old = current_indices[pos]
            b_idx_new = np.random.randint(num_blocks)
            
            if b_idx_new == b_idx_old or b_idx_new in current_indices:
                # If it's already in the list (and not the one we're swapping out), it's not a valid swap in terms of distinct blocks
                # Actually, if b_idx_new is in current_indices at another position, swapping just moves it?
                # No, blocks are distinct. If we swap b_old with b_new, and b_new is already there, we end up with duplicate b_new and missing b_old.
                # The problem requires distinct blocks.
                # So if b_new is in current_indices and b_new != b_old, we cannot simply swap.
                # We would have to remove b_old and add b_new, but b_new is already there.
                # This effectively removes b_old and keeps b_new (which was already there).
                # This is equivalent to a Remove move.
                # So we can treat this as a Remove move.
                trs = block_triples_np[b_idx_old]
                new_cover[trs] -= 1
                new_indices.pop(pos)
            else:
                trs_old = block_triples_np[b_idx_old]
                trs_new = block_triples_np[b_idx_new]
                new_cover[trs_old] -= 1
                new_cover[trs_new] += 1
                new_indices[pos] = b_idx_new
        
        new_energy = calculate_energy(new_indices, new_cover)
        delta_energy = new_energy - current_energy
        
        # Acceptance probability
        if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
            current_indices = new_indices
            current_cover = new_cover
            current_energy = new_energy
            
            # Update best if improved
            # Note: calculate_energy prioritizes uncovered. 
            # We want to track best *valid* solution (uncovered=0) with min blocks.
            # If uncovered > 0, it's not a valid solution yet.
            # We can still track best energy, but for final return we need valid.
            
            # If we find a valid solution, update best_size
            if current_energy < 1000: # Means 0 uncovered triples
                if len(current_indices) < best_size:
                    best_size = len(current_indices)
                    best_indices = current_indices.copy()
                    best_cover = current_cover.copy()
                    
            # Prune occasionally to remove redundancy
            # Pruning is deterministic and reduces size if possible
            # We can prune every 100 steps or if energy is low
            # Let's prune if we are valid and haven't pruned recently
            # To save time, only prune if we think we are close to optimal
            # Actually, pruning is cheap (O(N * C(k,t))).
            # Let's prune every 50 iterations if valid.
            # We need a counter.
            
    # Final prune on best found
    if best_indices is not None:
        best_indices, best_cover = prune_indices(best_indices)
    
    return [list(all_blocks[b]) for b in best_indices]
