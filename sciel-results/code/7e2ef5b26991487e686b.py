import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Number of triples in a block
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    # Using a 2D numpy array for vectorized operations
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Precompute triple frequencies for weighted greedy
    triple_freq = np.zeros(n_triples, dtype=np.float64)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
    triple_weight = 1.0 / np.maximum(triple_freq, 1.0)

    def greedy_cover(use_weighted=False, random_tiebreak=False, rng=None, top_k=1):
        """Run greedy set cover."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Calculate gains
            if use_weighted:
                gains = np.sum(uncovered[block_triples] * triple_weight[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            if random_tiebreak and rng is not None:
                # Pick from top K candidates
                threshold = max_gain - 1e-9 # Allow small floating point errors
                candidates = np.where(gains >= threshold)[0]
                if len(candidates) > top_k:
                    # Weighted random choice? Or just uniform?
                    # Uniform among top K is a good diversity strategy
                    best_idx = rng.choice(candidates)
                else:
                    best_idx = rng.choice(candidates)
            else:
                best_idx = np.argmax(gains)
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected

    def peel_redundant(block_indices):
        """Remove redundant blocks from a solution."""
        if not block_indices:
            return list(block_indices)
        
        n_sel = len(block_indices)
        # coverage_count[triple] = number of blocks covering it
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        keep = []
        # We need to be careful: removing one block might make another redundant, 
        # but also removing one might make a previously redundant block critical again?
        # No, if a block was redundant (all its triples covered by others), removing it 
        # only decreases coverage counts. It cannot make a triple uncovered if it wasn't 
        # the only cover.
        # However, the order matters for *which* blocks are kept if we want to maximize 
        # removals? Actually, any order of removing redundant blocks leads to a minimal 
        # (by inclusion) solution.
        # But to find a smaller *size* solution, we just remove all possible.
        
        # Iterative removal
        changed = True
        current_indices = list(block_indices)
        while changed:
            changed = False
            new_indices = []
            # Recalculate coverage for current_indices? 
            # Or maintain it?
            # For simplicity and correctness in the peel phase (which is fast), 
            # let's just check redundancy against the *current* set.
            
            # Efficient check:
            # A block is redundant if for all t in block, coverage_count[t] > 1.
            # We can check this quickly.
            
            # Let's just do a single pass first.
            pass
            
        # Single pass is usually sufficient for greedy outputs, but let's do a loop.
        # To be safe and fast, we'll just do a few passes.
        
        final_indices = list(block_indices)
        for _ in range(10):
            coverage_count = np.zeros(n_triples, dtype=np.int32)
            for bidx in final_indices:
                coverage_count[block_triples[bidx]] += 1
            
            new_indices = []
            for bidx in final_indices:
                my_triples = block_triples[bidx]
                # If any triple in this block is covered only by this block, keep it
                # i.e., if min(coverage_count[my_triples]) > 1, it is redundant
                if np.min(coverage_count[my_triples]) > 1:
                    # Redundant, skip
                    continue
                else:
                    new_indices.append(bidx)
            
            if len(new_indices) == len(final_indices):
                break
            final_indices = new_indices
            
        return final_indices

    def simulated_annealing(block_indices, rng, time_limit):
        """Simulated annealing to minimize block count."""
        if not block_indices:
            return list(block_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        current_blocks = list(block_indices)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        # Precompute set of blocks for fast lookup? 
        # We need to check if a candidate block is already in current_blocks.
        # Using a set of indices is fast.
        
        while time.time() < time_limit and T > min_T:
            T *= cooling
            
            if len(current_blocks) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to remove a redundant block
                # Find all redundant blocks
                redundant_blocks = []
                for i, bidx in enumerate(current_blocks):
                    my_triples = block_triples[bidx]
                    # Check if any triple is covered only by this block
                    # i.e., if min(coverage_count[my_triples]) == 1, it's critical
                    # If min > 1, it's redundant
                    if np.min(coverage_count[my_triples]) > 1:
                        redundant_blocks.append(i)
                
                if redundant_blocks:
                    i = rng.choice(redundant_blocks)
                    bidx = current_blocks[i]
                    my_triples = block_triples[bidx]
                    coverage_count[my_triples] -= 1
                    current_blocks.pop(i)
                    
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                    continue
                else:
                    # No redundant blocks, skip
                    continue

            elif move_type < 0.8:
                # Swap: replace one block with another
                i = rng.randrange(len(current_blocks))
                old_bidx = current_blocks[i]
                old_triples = block_triples[old_bidx]
                
                # Find critical triples (covered only by this block)
                critical_mask = (coverage_count == 1)
                critical_triples = old_triples[critical_mask[old_triples]]
                
                if len(critical_triples) == 0:
                    # Block is redundant, should have been caught above, but just in case
                    coverage_count[old_triples] -= 1
                    current_blocks.pop(i)
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                    continue
                
                # Find candidate blocks that cover all critical triples
                # To speed up, we can sample candidates or filter.
                # n_blocks is ~200k. Checking all is slow.
                # We can precompute for each triple, the list of blocks covering it.
                # But that's memory intensive.
                # Instead, we can just sample a few candidates.
                
                best_candidate = -1
                best_score = -1
                
                # Sample 50 random blocks
                for _ in range(50):
                    cand_idx = rng.randrange(n_blocks)
                    if cand_idx == old_bidx:
                        continue
                    
                    cand_triples = block_triples[cand_idx]
                    
                    # Check if candidate covers all critical triples
                    covers_all = True
                    for ct in critical_triples:
                        if not np.isin(ct, cand_triples):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Calculate score: how many "useful" triples does it cover?
                    # Useful = triples currently covered by 1 or 2 blocks?
                    # Or just maximize total coverage of low-count triples.
                    # Let's say score = sum(1 / (coverage_count[ct] + 1) for ct in cand_triples)
                    # This favors blocks that cover rare triples.
                    scores = 1.0 / (coverage_count[cand_triples] + 1.0)
                    score = np.sum(scores)
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = cand_idx
                
                if best_candidate >= 0:
                    # Perform swap
                    # Update coverage counts
                    coverage_count[old_triples] -= 1
                    coverage_count[block_triples[best_candidate]] += 1
                    current_blocks[i] = best_candidate
                    
                    # Note: Swapping doesn't change size, but might enable future removals.
                    # We don't update best_size here unless we immediately remove.
                    # But we can check if the new block allows immediate removal of another?
                    # That's complex. Just accept the swap if it's "good".
                    # Since size is same, we accept it to explore.
                    # To avoid cycling, we rely on T.
                    # Actually, if size is same, we should only accept if it improves some heuristic 
                    # or with probability exp(-delta/T). Here delta=0 for size.
                    # So we always accept? No, that might lead to random walk.
                    # Let's accept if the "score" of the new block is better than old?
                    # Or just accept to explore.
                    
            else:
                # Perturbation: remove a random block and repair
                if len(current_blocks) > 1:
                    j = rng.randrange(len(current_blocks))
                    removed_bidx = current_blocks[j]
                    coverage_count[block_triples[removed_bidx]] -= 1
                    current_blocks.pop(j)
                    
                    # Rebuild to cover uncovered triples
                    uncovered_mask = (coverage_count == 0)
                    # Greedy repair
                    while np.any(uncovered_mask):
                        if time.time() > time_limit:
                            break
                        # Find best block to add
                        gains = np.sum(uncovered_mask[block_triples], axis=1)
                        best_add = np.argmax(gains)
                        if gains[best_add] == 0:
                            break
                        coverage_count[block_triples[best_add]] += 1
                        current_blocks.append(best_add)
                        uncovered_mask = (coverage_count == 0)
                    
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
        
        return best_blocks

    # Run multiple strategies
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Strategy 1: Deterministic greedy
    sel = greedy_cover(use_weighted=False, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = peeled
    
    # Strategy 2: Weighted greedy
    sel = greedy_cover(use_weighted=True, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = peeled
            
    # Strategy 3+: Random restarts with weighted greedy + simulated annealing
    # Allocate remaining time to SA
    time_elapsed = time.time() - start_time
    time_remaining = deadline - time_elapsed
    
    # Run SA for a few restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Randomized greedy
        sel = greedy_cover(use_weighted=True, random_tiebreak=True, rng=rng, top_k=5)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            
            # Apply simulated annealing
            # Give SA a portion of the remaining time
            sa_time_limit = deadline - 5.0
            sa_result = simulated_annealing(peeled, rng, sa_time_limit)
            sa_size = len(sa_result)
            
            if sa_size < size:
                size = sa_size
                peeled = sa_result
            
            if size < best_size:
                best_size = size
                best_solution = peeled
                
    # Final polish
    if best_solution is not None:
        final_peeled = peel_redundant(best_solution)
        best_solution = final_peeled
        best_size = len(final_peeled)
        
    if best_solution is None:
        best_solution = []
        
    # Convert indices to lists of points
    return [list(all_blocks[i]) for i in best_solution]
