"""WalkSAT-style stochastic local search for covering design C(v,k,t)."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Precompute all t-subsets and their indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks and which t-subsets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(tsubs)
    
    # For each t-subset, list of block indices that cover it
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
    
    def greedy_initial(seed=0):
        """Fast greedy to get a starting solution."""
        rng = random.Random(seed)
        uncovered = set(range(n_tsubsets))
        blocks_used = []
        
        # Precompute gain: for each block, how many currently uncovered t-subsets it covers
        # Start with all uncovered
        gain = [len(block_tsubs[i]) for i in range(n_blocks)]
        
        while uncovered:
            if time.time() > deadline - 2:
                break
            
            # Find block with max gain
            max_gain = max(gain)
            if max_gain == 0:
                break
            
            # Pick randomly among max-gain blocks
            candidates = [i for i in range(n_blocks) if gain[i] == max_gain]
            chosen = rng.choice(candidates)
            blocks_used.append(chosen)
            
            # Update uncovered and gains
            for ts_idx in block_tsubs[chosen]:
                if ts_idx in uncovered:
                    uncovered.discard(ts_idx)
                    # Decrement gain for all blocks covering this t-subset
                    for bi in tsub_blocks[ts_idx]:
                        if gain[bi] > 0:
                            gain[bi] -= 1
            
            gain[chosen] = 0
        
        return blocks_used
    
    def walksat_refine(block_indices, seed=0):
        """WalkSAT-style stochastic local search to minimize block count."""
        rng = random.Random(seed)
        
        # Current state: set of selected block indices
        selected = set(block_indices)
        
        # Compute coverage: for each t-subset, count how many selected blocks cover it
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in selected:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
        
        # Track best solution found
        best_selected = set(selected)
        best_count = len(selected)
        
        # Parameters
        p_remove = 0.6  # probability of trying to remove a block
        max_steps = 50000
        no_improve_limit = 5000
        
        step = 0
        no_improve = 0
        
        while step < max_steps and time.time() < deadline - 1:
            step += 1
            
            if len(selected) < best_count:
                best_selected = set(selected)
                best_count = len(selected)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > no_improve_limit:
                    break
            
            if not selected:
                break
            
            if rng.random() < p_remove:
                # Try to remove a random block
                bi = rng.choice(list(selected))
                
                # Check if removing it leaves any t-subset uncovered
                can_remove = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove the block
                    selected.discard(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                else:
                    # WalkSAT move: pick a random block to remove and accept
                    # the resulting unsatisfied clauses with some probability
                    # Here we just skip this move
                    pass
            else:
                # Try to swap: remove one block, add another
                # Pick a block that has unique coverage (hard to remove)
                bi_to_remove = rng.choice(list(selected))
                
                # Find t-subsets uniquely covered by this block
                unique_tsubs = [ts_idx for ts_idx in block_tsubs[bi_to_remove] 
                               if cover_count[ts_idx] == 1]
                
                if not unique_tsubs:
                    # Block is redundant, remove it
                    selected.discard(bi_to_remove)
                    for ts_idx in block_tsubs[bi_to_remove]:
                        cover_count[ts_idx] -= 1
                    continue
                
                # Try to find a replacement block that covers all unique t-subsets
                # Sample candidate blocks
                n_samples = min(50, n_blocks)
                candidates = rng.sample(range(n_blocks), n_samples)
                
                found_swap = False
                for ci in candidates:
                    if ci in selected:
                        continue
                    
                    # Check if ci covers all unique t-subsets
                    cand_set = set(block_tsubs[ci])
                    if all(ts in cand_set for ts in unique_tsubs):
                        # Verify no t-subset becomes uncovered after swap
                        ok = True
                        for ts_idx in block_tsubs[bi_to_remove]:
                            if cover_count[ts_idx] == 1 and ts_idx not in cand_set:
                                ok = False
                                break
                        if ok:
                            # Perform swap
                            selected.discard(bi_to_remove)
                            for ts_idx in block_tsubs[bi_to_remove]:
                                cover_count[ts_idx] -= 1
                            selected.add(ci)
                            for ts_idx in block_tsubs[ci]:
                                cover_count[ts_idx] += 1
                            found_swap = True
                            break
                
                if not found_swap:
                    # WalkSAT: with some probability, make a random move
                    # that may temporarily worsen the solution
                    if rng.random() < 0.1:
                        # Randomly add a block (to escape local minimum)
                        ci = rng.randint(0, n_blocks - 1)
                        if ci not in selected:
                            selected.add(ci)
                            for ts_idx in block_tsubs[ci]:
                                cover_count[ts_idx] += 1
        
        # Convert back to block tuples
        return [all_blocks[bi] for bi in best_selected]
    
    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts with different seeds
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Get initial solution from greedy
        init_blocks = greedy_initial(seed)
        if not init_blocks:
            continue
        
        # Refine with WalkSAT
        refined = walksat_refine(init_blocks, seed)
        
        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = refined
        
        # Early termination
        if best_len <= 135:
            break
    
    if best_blocks is None:
        # Fallback
        init_blocks = greedy_initial(0)
        best_blocks = walksat_refine(init_blocks, 0)
    
    return best_blocks
