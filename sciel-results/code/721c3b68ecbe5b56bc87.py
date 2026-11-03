import time
import random
import itertools
import math
import numpy as np
from collections import defaultdict, deque

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    # 1. Precomputation
    # All t-subsets (items to cover)
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx_t = {s: i for i, s in enumerate(all_t)}
    
    # All k-subsets (potential blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    M = len(all_blocks)
    
    # block_cover[b] = list of indices of t-subsets covered by block b
    block_cover = []
    for b in all_blocks:
        covered = [idx_t[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # t_to_blocks[i] = list of block indices that cover t-subset i
    t_to_blocks = [[] for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Helper: Check if a list of block indices forms a valid cover
    def is_valid(blocks):
        if not blocks:
            return N == 0
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        return np.all(cnt > 0)

    # Helper: Compute coverage counts
    def get_counts(blocks):
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        return cnt

    # 2. Initial Construction: Greedy Set Cover
    # We use a standard greedy approach to get a valid starting point.
    # To improve quality, we use a randomized greedy with multiple restarts 
    # and keep the best one.
    
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = set(range(N))
        blocks = []
        # Precompute list of blocks for faster access? No, we use t_to_blocks.
        
        while uncovered:
            # Pick a random uncovered item to break symmetry and allow diversity
            target = next(iter(uncovered))
            candidates = t_to_blocks[target]
            if not candidates:
                return None # Should not happen
            
            best_b = -1
            max_gain = -1
            
            # Evaluate all candidates covering the target
            # Optimization: We can limit candidates if the list is huge, but C(14,3)=364 is small.
            for bi in candidates:
                gain = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_b = bi
                    # Early termination if we cover all? Unlikely in one step.
                    
            if best_b == -1:
                return None
                
            blocks.append(best_b)
            # Update uncovered
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
                
        return blocks

    best_init = None
    best_init_len = float('inf')
    
    # Run a few quick greedy restarts to find a good starting point
    for s in range(10):
        if time.time() > deadline - 5.0:
            break
        cand = greedy_init(s)
        if cand and len(cand) < best_init_len:
            best_init_len = len(cand)
            best_init = cand
            
    if not best_init:
        # Fallback: just return a valid cover, even if large
        # This shouldn't happen
        return [list(b) for b in all_blocks[:100]] 

    best_blocks = best_init
    best_score = len(best_blocks)

    # 3. Structural Refinement: Max-Flow Min-Cut for Minimal Subcover
    # Given a valid cover S, find the minimal subset S' of S that is still a valid cover.
    # This is equivalent to finding a minimum hitting set or, in this context, 
    # identifying "essential" blocks.
    # However, finding the *minimum* subcover is NP-hard.
    # But we can find a *minimal* (inclusion-wise) subcover by greedily removing blocks
    # if they are redundant. This is fast.
    # The "Max-Flow" inspiration suggests a more global optimization.
    # Let's interpret "Max-Flow" here as using a flow network to find a set of blocks
    # that covers all items with minimum weight (where weight is 1 for all).
    # Actually, for a fixed set S, checking if a subset is a cover is easy.
    # A better structural move: 
    # Identify "critical" blocks (blocks that cover some t-subset exclusively).
    # These MUST stay.
    # For the rest, we can try to swap them out for better blocks.
    
    def prune_to_minimal(blocks):
        """Remove redundant blocks from a valid cover."""
        cnt = get_counts(blocks)
        current_blocks = list(blocks)
        i = 0
        while i < len(current_blocks):
            bi = current_blocks[i]
            # Check if bi is redundant
            redundant = True
            for ti in block_cover[bi]:
                if cnt[ti] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it
                current_blocks.pop(i)
                for ti in block_cover[bi]:
                    cnt[ti] -= 1
                # Do not increment i, check next block at same index
            else:
                i += 1
        return current_blocks

    # Apply pruning to initial solution
    best_blocks = prune_to_minimal(best_blocks)
    best_score = len(best_blocks)

    # 4. Iterative Improvement Loop
    # We will alternate between:
    # A. Greedy Fill: Identify uncovered areas (if we allow temporary invalidity? No, we stay valid).
    #    Instead, we look for "swaps" that reduce the "potential" for future reductions.
    # B. Deterministic 1-Exchange: Try replacing one block with another to enable more drops.
    
    # A more powerful structural approach:
    # Use a "Simulated Annealing" variant but with a specific focus on 
    # "Diversity" of the cover. 
    # Actually, let's stick to the directive: "Different design family".
    # The current program uses SA. The inspiration 3 uses Multi-Start Greedy.
    # My proposal: **Greedy Set Cover + Local Search with "Essential Core" preservation**.
    
    # Let's implement a robust local search that focuses on:
    # 1. Identifying the "Core" (blocks that cannot be removed).
    # 2. Trying to replace non-core blocks with others that cover more "weak" spots.
    
    def local_search(blocks, max_time=10.0):
        blocks = list(blocks)
        cnt = get_counts(blocks)
        current_cost = len(blocks)
        best_local = list(blocks)
        best_local_cost = current_cost
        
        start_time = time.time()
        no_improve_steps = 0
        max_no_improve = 2000
        
        # Precompute for each block, the list of t-subsets it covers
        # We already have block_cover.
        
        # To speed up swap evaluation, we can precompute for each t-subset 
        # the list of blocks covering it. We have t_to_blocks.
        
        while time.time() - start_time < max_time and no_improve_steps < max_no_improve:
            # 1. Try to drop a redundant block
            dropped = False
            for i in range(len(blocks)):
                bi = blocks[i]
                # Check redundancy
                can_drop = True
                for ti in block_cover[bi]:
                    if cnt[ti] <= 1:
                        can_drop = False
                        break
                if can_drop:
                    # Drop it
                    blocks.pop(i)
                    for ti in block_cover[bi]:
                        cnt[ti] -= 1
                    current_cost -= 1
                    if current_cost < best_local_cost:
                        best_local_cost = current_cost
                        best_local = list(blocks)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                    dropped = True
                    break # Restart scan from beginning as indices shifted
            
            if dropped:
                continue
                
            # 2. Try a swap
            # Pick a random block to replace
            if len(blocks) == 0:
                break
                
            idx_old = random.randrange(len(blocks))
            bi_old = blocks[idx_old]
            
            # Find "weak" t-subsets covered by bi_old (those with cnt == 1)
            # If we remove bi_old, these become uncovered.
            # We need a new block that covers all of them.
            weaks = []
            for ti in block_cover[bi_old]:
                if cnt[ti] == 1:
                    weaks.append(ti)
            
            if not weaks:
                # This block is redundant? 
                # If weaks is empty, all its t-subsets have cnt > 1.
                # Then we should have dropped it in step 1.
                # Unless there's a bug or we missed it.
                # Force drop?
                blocks.pop(idx_old)
                for ti in block_cover[bi_old]:
                    cnt[ti] -= 1
                current_cost -= 1
                if current_cost < best_local_cost:
                    best_local_cost = current_cost
                    best_local = list(blocks)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                continue
                
            # Find candidates that cover ALL weaks
            # Intersection of t_to_blocks[w] for w in weaks
            # To be fast, sort weaks by len(t_to_blocks[w]) and intersect smallest first
            weaks_sorted = sorted(weaks, key=lambda w: len(t_to_blocks[w]))
            
            candidates = set(t_to_blocks[weaks_sorted[0]])
            for w in weaks_sorted[1:]:
                candidates &= set(t_to_blocks[w])
                if not candidates:
                    break
                    
            if not candidates:
                no_improve_steps += 1
                continue
                
            candidates.discard(bi_old)
            if not candidates:
                no_improve_steps += 1
                continue
                
            # Pick a random candidate
            bi_new = random.choice(list(candidates))
            
            # Evaluate the swap
            # Remove bi_old, add bi_new
            # Check if this creates any new redundancy opportunities?
            # Or just accept if it doesn't break validity (it doesn't, by construction of weaks).
            # But does it help future drops?
            # Heuristic: Prefer swaps that increase the count of "weak" spots elsewhere?
            # Or simply accept random swaps to explore.
            
            # Apply swap
            blocks[idx_old] = bi_new
            
            # Update counts
            for ti in block_cover[bi_old]:
                cnt[ti] -= 1
            for ti in block_cover[bi_new]:
                cnt[ti] += 1
                
            # Cost is unchanged (swap).
            # We accept this move always to explore the neighborhood.
            # But we only reset no_improve if we eventually get a drop.
            # To prevent infinite loops of useless swaps, we limit consecutive swaps.
            # For now, just count it as a step.
            no_improve_steps += 1
            
            # Optional: If the new block covers many *other* weak spots, it's good.
            # Let's add a slight bias: If the swap results in more "redundant" blocks, it's good.
            # But calculating that is expensive.
            
        return best_local

    # Run local search for the remaining time
    while time.time() < deadline - 2.0:
        # Run a local search session
        candidate = local_search(best_blocks, max_time=5.0)
        
        if len(candidate) < best_score:
            best_score = len(candidate)
            best_blocks = candidate
            
        # If we hit the lower bound or are close, we might stop early?
        # Lower bound is 264.
        if best_score <= 264:
            break

    # Final validation
    if not is_valid(best_blocks):
        # Fallback to the initial valid cover
        best_blocks = best_init
        best_score = len(best_blocks)

    return [list(all_blocks[bi]) for bi in best_blocks]
