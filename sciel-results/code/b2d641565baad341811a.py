import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_id = {tr: i for i, tr in enumerate(triples)}
    num_triples = len(triples)
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_triples: for each block, list of triple IDs it covers
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_id[tup] for tup in itertools.combinations(b, t)]
        block_triples.append(ts)
    
    # triple_blocks: for each triple, list of block indices covering it
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)
    
    # Convert to numpy arrays for fast operations
    block_triples_np = [np.array(ts, dtype=np.int32) for ts in block_triples]
    triple_blocks_np = [np.array(lst, dtype=np.int32) for lst in triple_blocks]
    
    def greedy_construction(seed=0, time_budget=deadline):
        """Greedy set cover with random tie-breaking."""
        rng = random.Random(seed)
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            if time.time() > time_budget:
                break
            max_gain = int(np.max(gain))
            if max_gain == 0:
                break
            
            # Find candidates with max gain
            candidates = np.where(gain == max_gain)[0]
            b_idx = int(rng.choice(candidates.tolist()))
            
            selected_blocks.append(b_idx)
            
            # Update gain and is_uncovered
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    gain[triple_blocks_np[t_idx]] -= 1
                    
        return selected_blocks

    def complete_cover(selected, deadline):
        """Ensure all triples are covered by adding blocks as needed."""
        if not selected:
            return selected
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        selected_set = set(selected)
        for b_idx in selected:
            cover_count[block_triples_np[b_idx]] += 1
            
        while True:
            if time.time() > deadline:
                break
            uncovered_mask = (cover_count == 0)
            if not np.any(uncovered_mask):
                break
                
            # Find uncovered triples
            unc_idx = np.where(uncovered_mask)[0]
            if len(unc_idx) == 0:
                break
                
            # Pick a random uncovered triple as anchor
            target_tid = int(unc_idx[random.randint(0, len(unc_idx)-1)])
            candidates = triple_blocks_np[target_tid]
            
            best_b = -1
            best_gain = -1
            for b_idx in candidates:
                if b_idx in selected_set:
                    continue
                g = int(np.sum(uncovered_mask[block_triples_np[b_idx]]))
                if g > best_gain:
                    best_gain = g
                    best_b = int(b_idx)
                    
            if best_b == -1:
                break
                
            selected.append(best_b)
            selected_set.add(best_b)
            cover_count[block_triples_np[best_b]] += 1
            
        return selected

    def reduce_cover(selected, deadline):
        """Remove redundant blocks iteratively."""
        if not selected:
            return selected
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples_np[b_idx]] += 1
            
        active = [True] * len(selected)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for i in indices:
                if not active[i]:
                    continue
                b_idx = selected[i]
                # Check if redundant: all its triples are covered by other blocks
                if int(np.min(cover_count[block_triples_np[b_idx]])) > 1:
                    active[i] = False
                    cover_count[block_triples_np[b_idx]] -= 1
                    changed = True
                    
        return [selected[i] for i in range(len(selected)) if active[i]]

    def local_search_2opt(selected, deadline, rng):
        """
        Local search: try 1-for-1 and 2-for-1 swaps to reduce block count.
        A 2-for-1 swap removes one block and adds two blocks, netting -1 if the 
        two new blocks together cover everything the removed block covered 
        plus some previously uncovered triples, while not increasing total.
        Actually, for minimization, we want to remove blocks. 
        Better approach: try removing a block and see if we can add one block 
        to restore coverage (1-for-1). Or remove two and add one (2-for-1).
        
        Let's do: 
        1. Try to remove a single block (if redundant, already handled by reduce_cover).
        2. Try 2-for-1: remove two blocks, add one block that covers all triples 
           that were only covered by those two blocks.
        """
        selected = list(selected)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        selected_set = set(selected)
        for b_idx in selected:
            cover_count[block_triples_np[b_idx]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try 2-for-1 swaps: remove blocks i and j, add block b
            # Only consider if removing i and j leaves some triples uncovered,
            # and there exists a block b that covers all those uncovered triples.
            n = len(selected)
            if n < 3:
                break
                
            # To keep it fast, only try a limited number of pairs per iteration
            max_pairs = min(500, n * (n-1) // 2)
            pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append((i, j))
            if len(pairs) > max_pairs:
                rng.shuffle(pairs)
                pairs = pairs[:max_pairs]
            else:
                rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > deadline:
                    break
                if i >= len(selected) or j >= len(selected):
                    continue
                if i == j:
                    continue
                    
                b_i = selected[i]
                b_j = selected[j]
                
                # Simulate removal
                cover_count[block_triples_np[b_i]] -= 1
                cover_count[block_triples_np[b_j]] -= 1
                
                uncovered = (cover_count == 0)
                if not np.any(uncovered):
                    # Both blocks were redundant! Remove both.
                    # This shouldn't happen if reduce_cover was run, but just in case.
                    selected.pop(max(i,j))
                    selected.pop(min(i,j))
                    selected_set.discard(b_i)
                    selected_set.discard(b_j)
                    improved = True
                    # Recalculate cover_count
                    cover_count = np.zeros(num_triples, dtype=np.int32)
                    for b_idx in selected:
                        cover_count[block_triples_np[b_idx]] += 1
                    break
                else:
                    # Find a block that covers all uncovered triples
                    unc_triples = np.where(uncovered)[0]
                    # The candidate block must cover all of unc_triples
                    # Find intersection of triple_blocks for each uncovered triple
                    # Start with blocks covering the first uncovered triple
                    candidates = set(triple_blocks_np[unc_triples[0]].tolist())
                    for tid in unc_triples[1:]:
                        candidates &= set(triple_blocks_np[tid].tolist())
                        if not candidates:
                            break
                    
                    if candidates:
                        # Check if any candidate is not already in selected
                        for b_idx in candidates:
                            if b_idx in selected_set:
                                continue
                            # Check if adding b_idx covers all uncovered
                            # It should by construction, but verify
                            if int(np.sum(uncovered[block_triples_np[b_idx]])) == len(unc_triples):
                                # Perform the swap
                                selected.pop(max(i,j))
                                selected.pop(min(i,j))
                                selected_set.discard(b_i)
                                selected_set.discard(b_j)
                                selected.append(b_idx)
                                selected_set.add(b_idx)
                                cover_count[block_triples_np[b_idx]] += 1
                                improved = True
                                # Recalculate cover_count for consistency
                                cover_count = np.zeros(num_triples, dtype=np.int32)
                                for b_idx2 in selected:
                                    cover_count[block_triples_np[b_idx2]] += 1
                                break
                    # Restore cover_count if no swap was made
                    if not improved:
                        cover_count[block_triples_np[b_i]] += 1
                        cover_count[block_triples_np[b_j]] += 1
                    else:
                        break
                if improved:
                    break
                    
        return selected

    def entropy_guided_greedy(seed=0, time_budget=deadline):
        """
        Greedy with entropy-based scoring.
        Score = gain / (redundancy_penalty + 1)
        where redundancy_penalty measures how much the block overlaps with existing coverage.
        """
        rng = random.Random(seed)
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            if time.time() > time_budget:
                break
            max_gain = int(np.max(gain))
            if max_gain == 0:
                break
            
            candidates = np.where(gain == max_gain)[0]
            
            # Among candidates, pick one with lowest "redundancy"
            # Redundancy = number of triples in the block that are already covered
            best_b = -1
            best_score = -1
            for b_idx in candidates.tolist():
                redundancy = int(np.sum(~is_uncovered[block_triples_np[b_idx]]))
                score = max_gain / (redundancy + 1)
                if score > best_score:
                    best_score = score
                    best_b = b_idx
            
            if best_b == -1:
                best_b = int(rng.choice(candidates.tolist()))
            
            selected_blocks.append(best_b)
            
            for t_idx in block_triples[best_b]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    gain[triple_blocks_np[t_idx]] -= 1
                    
        return selected_blocks

    # Main search loop
    best_cover = None
    best_size = float('inf')
    
    rng_main = random.Random(12345)
    
    # Phase 1: Multiple greedy restarts with different strategies
    phase1_deadline = deadline - 30.0
    seed = 0
    
    while time.time() < phase1_deadline:
        # Alternate between standard greedy and entropy-guided
        if seed % 2 == 0:
            blocks = greedy_construction(seed, phase1_deadline - 2.0)
        else:
            blocks = entropy_guided_greedy(seed, phase1_deadline - 2.0)
            
        blocks = complete_cover(blocks, phase1_deadline - 2.0)
        blocks = reduce_cover(blocks, phase1_deadline - 2.0)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_cover = blocks[:]
            
        if best_size <= 70:
            break
            
        seed += 1
        
        # Occasionally run local search on the best
        if seed % 3 == 0 and best_cover:
            ls_deadline = min(deadline - 10.0, time.time() + 8.0)
            improved = local_search_2opt(best_cover[:], ls_deadline, rng_main)
            improved = reduce_cover(improved, ls_deadline)
            if len(improved) < best_size:
                best_size = len(improved)
                best_cover = improved[:]
                
    # Phase 2: Intensive local search on the best solution
    if best_cover:
        final_ls_deadline = deadline - 5.0
        current = best_cover[:]
        
        # Multiple rounds of local search
        for _ in range(3):
            if time.time() > final_ls_deadline:
                break
            current = local_search_2opt(current, final_ls_deadline, rng_main)
            current = reduce_cover(current, final_ls_deadline)
            if len(current) < best_size:
                best_size = len(current)
                best_cover = current[:]
                
        # Final reduction
        best_cover = reduce_cover(best_cover, deadline - 2.0)
        best_size = len(best_cover)

    # Fallback: if we didn't get a good solution, do a final greedy
    if not best_cover or best_size > 80:
        blocks = greedy_construction(0, deadline - 5.0)
        blocks = complete_cover(blocks, deadline - 2.0)
        blocks = reduce_cover(blocks, deadline - 1.0)
        best_cover = blocks

    return [list(all_blocks[b_idx]) for b_idx in best_cover]
