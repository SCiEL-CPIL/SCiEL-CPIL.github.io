import itertools
import time
import numpy as np
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)
        
    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)
            
    # --- Phase 1: Greedy Construction ---
    # uncovered_count_per_block[b] = number of t-subsets in block b that are currently uncovered
    uncovered_count_per_block = [len(block_to_tsubs[b]) for b in range(n_blocks)]
    uncovered = [1] * n_t_subs
    total_uncovered = n_t_subs
    solution_blocks = []
    solution_set = set()
    
    while total_uncovered > 0:
        if time.time() > deadline:
            break
            
        # Find block with max uncovered coverage
        max_count = -1
        best_b_idx = -1
        # Optimization: If we find a block covering all its subsets, it's optimal for this step
        for i in range(n_blocks):
            c = uncovered_count_per_block[i]
            if c > max_count:
                max_count = c
                best_b_idx = i
                if c == len(block_to_tsubs[i]):
                    break
        if best_b_idx == -1 or max_count == 0:
            break
            
        solution_blocks.append(best_b_idx)
        solution_set.add(best_b_idx)
        
        # Update counts
        for s in block_to_tsubs[best_b_idx]:
            if uncovered[s]:
                uncovered[s] = 0
                total_uncovered -= 1
                # Update uncovered count for all blocks covering s
                for b_idx in tsub_to_blocks[s]:
                    uncovered_count_per_block[b_idx] -= 1
                    
    # --- Phase 2: Local Search (Redundancy Removal and Swaps) ---
    # Initialize cover_count: number of blocks in solution covering each t-subset
    cover_count = [0] * n_t_subs
    for b_idx in solution_blocks:
        for s in block_to_tsubs[b_idx]:
            cover_count[s] += 1
            
    # Helper: score a candidate block by how many new t-subsets it would cover
    # (i.e., subsets currently covered by the block being removed but not by the candidate)
    def swap_score(old_b, new_b):
        """Score the swap: higher is better.
        We want new_b to cover all critical subsets of old_b (guaranteed by construction),
        and to maximize the number of t-subsets that new_b covers which are currently
        covered only by old_b or by other blocks (to create future redundancy).
        Actually, the key is: after swap, can we remove more blocks?
        Simple heuristic: maximize overlap of new_b with existing solution's coverage
        of subsets NOT in old_b, to create redundancy.
        """
        # Count how many t-subsets in new_b are already covered by >= 2 blocks
        # (after removing old_b and adding new_b)
        # This is complex; use a simpler proxy:
        # Count subsets in new_b that are covered by some block in solution_set \ {old_b}
        cnt = 0
        for s in block_to_tsubs[new_b]:
            if cover_count[s] >= 2:  # currently covered by >=2, so even after removing old_b, still covered
                cnt += 1
            elif cover_count[s] == 1:
                # Check if old_b covers s
                # We'll precompute this differently
                pass
        return cnt
    
    improved = True
    max_passes = 50
    pass_count = 0
    while improved and time.time() < deadline - 10 and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # Pass 1: Try to remove any redundant block
        for i in range(len(solution_blocks)):
            b_idx = solution_blocks[i]
            redundant = True
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for s in block_to_tsubs[b_idx]:
                    cover_count[s] -= 1
                solution_blocks.pop(i)
                solution_set.remove(b_idx)
                improved = True
                break # Restart loop as indices changed
                
        # Pass 2: If no removal, try to swap a block to create redundancy elsewhere
        if not improved and time.time() < deadline - 10:
            best_swap = None  # (i, old_b, new_b, score)
            best_score = -1
            
            for i in range(len(solution_blocks)):
                b_idx = solution_blocks[i]
                # Find critical subsets (covered only by this block)
                critical_subsets = []
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        critical_subsets.append(s)
                        
                if not critical_subsets:
                    # Should have been removed in Pass 1, but just in case
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_blocks.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break
                    
                # Find a candidate block that covers all critical subsets
                # Intersect tsub_to_blocks for all critical subsets
                candidate_set = set(tsub_to_blocks[critical_subsets[0]])
                for s in critical_subsets[1:]:
                    candidate_set.intersection_update(tsub_to_blocks[s])
                    if not candidate_set:
                        break
                        
                if candidate_set:
                    # Score each candidate
                    for c_idx in candidate_set:
                        if c_idx not in solution_set:
                            # Score: number of t-subsets in c_idx that are currently covered by >= 2 blocks
                            # (these will remain covered even after removing b_idx)
                            # Plus: number of t-subsets in c_idx that are NOT in b_idx but covered by >= 1
                            # (to create new redundancy)
                            score = 0
                            for s in block_to_tsubs[c_idx]:
                                if cover_count[s] >= 2:
                                    score += 2  # already redundant, good
                                elif cover_count[s] == 1:
                                    # Check if b_idx covers s
                                    # If b_idx doesn't cover s, then this is a new coverage that might help
                                    # If b_idx covers s, then after removing b_idx, s would be uncovered unless c_idx covers it
                                    # But c_idx does cover s (by definition), so it's fine
                                    # The value is in creating redundancy for OTHER subsets
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_swap = (i, b_idx, c_idx)
            
            if best_swap is not None:
                i, old_b, new_b = best_swap
                # Swap: Remove old_b, Add new_b
                # Update cover counts
                for s in block_to_tsubs[old_b]:
                    cover_count[s] -= 1
                for s in block_to_tsubs[new_b]:
                    cover_count[s] += 1
                    
                # Replace in solution
                solution_blocks[i] = new_b
                solution_set.remove(old_b)
                solution_set.add(new_b)
                improved = True
                
    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[b_idx]) for b_idx in solution_blocks]
    return final_blocks
