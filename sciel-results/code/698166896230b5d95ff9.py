import itertools
import time
import random
import heapq
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

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

    # --- Phase 1: Initial Greedy Construction ---
    # uncovered[s] = 1 if t-subset s is not covered
    uncovered = [1] * n_t_subs
    total_uncovered = n_t_subs
    solution_blocks = []
    solution_set = set()
    
    # uncovered_count_per_block[b] = number of t-subsets in block b that are currently uncovered
    uncovered_count_per_block = [len(block_to_tsubs[b]) for b in range(n_blocks)]
    
    # Use a heap for max-heap behavior (negate values)
    # ( -count, block_index )
    heap = [(-uncovered_count_per_block[b], b) for b in range(n_blocks)]
    heapq.heapify(heap)
    
    while total_uncovered > 0 and time.time() < deadline:
        valid_block_idx = -1
        valid_count = 0
        while heap:
            neg_c, b_idx = heapq.heappop(heap)
            c = -neg_c
            if c == uncovered_count_per_block[b_idx] and c > 0:
                valid_block_idx = b_idx
                valid_count = c
                break
        
        if valid_block_idx == -1:
            break
            
        solution_blocks.append(valid_block_idx)
        solution_set.add(valid_block_idx)
        
        for s in block_to_tsubs[valid_block_idx]:
            if uncovered[s]:
                uncovered[s] = 0
                total_uncovered -= 1
                for b_idx in tsub_to_blocks[s]:
                    old_count = uncovered_count_per_block[b_idx]
                    new_count = old_count - 1
                    uncovered_count_per_block[b_idx] = new_count
                    if new_count > 0:
                        heapq.heappush(heap, (-new_count, b_idx))

    # --- Phase 2: Local Search (ILS) ---
    # cover_count[s] = number of blocks in solution covering t-subset s
    cover_count = np.zeros(n_t_subs, dtype=np.int32)
    for b_idx in solution_blocks:
        cover_count[block_to_tsubs[b_idx]] += 1
            
    best_solution = list(solution_blocks)
    best_len = len(solution_blocks)
    
    # Helper to check if a solution is valid (for safety, though we maintain state)
    def is_valid(sol):
        if not sol: return n_t_subs == 0
        cc = np.zeros(n_t_subs, dtype=np.int32)
        for b in sol:
            cc[block_to_tsubs[b]] += 1
        return np.all(cc > 0)

    # Ensure initial solution is valid
    if not is_valid(best_solution):
         # Fallback: if greedy failed, just return a valid but large cover
         # For v=19, k=7, t=4, a simple greedy should work. 
         # If it doesn't, we force repair.
         print("Initial solution invalid, repairing...")
         for s in range(n_t_subs):
             if cover_count[s] == 0:
                 # Pick any block covering s
                 b = tsub_to_blocks[s][0]
                 if b not in solution_set:
                     solution_blocks.append(b)
                     solution_set.add(b)
                     cover_count[block_to_tsubs[b]] += 1
         best_solution = list(solution_blocks)
         best_len = len(solution_blocks)

    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        improved = False
        current_len = len(solution_blocks)
        
        # Pass 1: Try to remove redundant blocks
        # A block is redundant if all its t-subsets have cover_count > 1
        n_sol = len(solution_blocks)
        if n_sol > 0:
            # Find a redundant block
            redundant_idx = -1
            for i in range(n_sol):
                b_idx = solution_blocks[i]
                if np.all(cover_count[block_to_tsubs[b_idx]] > 1):
                    redundant_idx = i
                    break
            
            if redundant_idx != -1:
                b_idx = solution_blocks[redundant_idx]
                cover_count[block_to_tsubs[b_idx]] -= 1
                solution_blocks.pop(redundant_idx)
                solution_set.remove(b_idx)
                improved = True
                no_improve_count = 0
                if len(solution_blocks) < best_len:
                    best_len = len(solution_blocks)
                    best_solution = list(solution_blocks)
                continue # Restart loop as indices changed

        # Pass 2: Try to swap
        if not improved and time.time() < deadline - 5:
            n_sol = len(solution_blocks)
            if n_sol > 0:
                # Find a block with the fewest critical subsets
                min_crit_count = float('inf')
                target_idx = -1
                target_crit_subs = []
                
                for i in range(n_sol):
                    b_idx = solution_blocks[i]
                    crit_subs = [s for s in block_to_tsubs[b_idx] if cover_count[s] == 1]
                    if len(crit_subs) < min_crit_count:
                        min_crit_count = len(crit_subs)
                        target_idx = i
                        target_crit_subs = crit_subs
                        if min_crit_count == 0:
                            # This shouldn't happen if Pass 1 ran correctly, but handle it
                            break

                if target_idx != -1 and min_crit_count > 0:
                    b_idx = solution_blocks[target_idx]
                    # Find a replacement block that covers all critical subsets
                    # Use intersection of candidate sets for each critical subset
                    candidate_set = set(tsub_to_blocks[target_crit_subs[0]])
                    for s in target_crit_subs[1:]:
                        candidate_set.intersection_update(tsub_to_blocks[s])
                        if not candidate_set:
                            break
                    
                    if candidate_set:
                        # Pick a candidate not in current solution
                        candidates = [c for c in candidate_set if c not in solution_set]
                        if candidates:
                            # Pick the one that maximizes overlap with current block (to minimize disruption)
                            best_cand = None
                            max_overlap = -1
                            b_subs_set = set(block_to_tsubs[b_idx])
                            for c in candidates:
                                overlap = len(b_subs_set & set(block_to_tsubs[c]))
                                if overlap > max_overlap:
                                    max_overlap = overlap
                                    best_cand = c
                            
                            if best_cand is not None:
                                # Perform swap
                                cover_count[block_to_tsubs[b_idx]] -= 1
                                cover_count[block_to_tsubs[best_cand]] += 1
                                
                                solution_blocks[target_idx] = best_cand
                                solution_set.remove(b_idx)
                                solution_set.add(best_cand)
                                
                                improved = True
                                no_improve_count = 0
                                if len(solution_blocks) < best_len:
                                    best_len = len(solution_blocks)
                                    best_solution = list(solution_blocks)
                                continue # Restart loop

        if not improved:
            no_improve_count += 1
            # Perturbation: Randomly remove some blocks and re-greedy
            if no_improve_count > max_no_improve:
                no_improve_count = 0
                if len(solution_blocks) > 10:
                    # Remove 2-4 random blocks
                    num_remove = random.randint(2, 4)
                    for _ in range(num_remove):
                        if not solution_blocks: break
                        idx = random.randint(0, len(solution_blocks)-1)
                        b_idx = solution_blocks[idx]
                        cover_count[block_to_tsubs[b_idx]] -= 1
                        solution_blocks.pop(idx)
                        solution_set.remove(b_idx)
                    
                    # Re-greedy fill
                    # Identify uncovered
                    uncovered_list = np.where(cover_count == 0)[0]
                    while len(uncovered_list) > 0 and time.time() < deadline - 5:
                        # Pick a random uncovered subset
                        s = uncovered_list[random.randint(0, len(uncovered_list)-1)]
                        # Pick a random block covering it (prefer ones not in solution)
                        candidates = [b for b in tsub_to_blocks[s] if b not in solution_set]
                        if not candidates:
                            candidates = tsub_to_blocks[s]
                        b = random.choice(candidates)
                        
                        solution_blocks.append(b)
                        solution_set.add(b)
                        cover_count[block_to_tsubs[b]] += 1
                        
                        # Update uncovered list
                        # This is slow if done naively, but we break early if time is up
                        # We'll just recompute at the end of perturbation or check periodically
                        # For efficiency in this loop, we just add blocks. 
                        # We need to know when we are done.
                        # Let's just add blocks until we cover all? 
                        # Better: add blocks for each uncovered s found.
                        # Since we pick random s from uncovered_list, we need to update uncovered_list.
                        # Simple approach: just add 1 block per loop iteration, check time.
                        # To ensure termination, we limit iterations.
                        
                        # Actually, let's just do a quick greedy fill for uncovered
                        # Re-evaluate uncovered
                        new_uncovered = np.where(cover_count == 0)[0]
                        if len(new_uncovered) == 0:
                            break
                        uncovered_list = new_uncovered
                        
                        # If we are taking too long, break
                        if time.time() > deadline - 5:
                            break

                    # After perturbation, check if we improved
                    if len(solution_blocks) < best_len:
                        best_len = len(solution_blocks)
                        best_solution = list(solution_blocks)
                    # Even if not improved, we might be in a better basin, so continue

    # Final Polish: Remove redundant blocks from best_solution
    # We need to recompute cover_count for best_solution to be safe
    if best_solution:
        final_cc = np.zeros(n_t_subs, dtype=np.int32)
        for b in best_solution:
            final_cc[block_to_tsubs[b]] += 1
        
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for i in range(len(best_solution)):
                b = best_solution[i]
                if np.all(final_cc[block_to_tsubs[b]] > 1):
                    # Remove
                    final_cc[block_to_tsubs[b]] -= 1
                    best_solution.pop(i)
                    changed = True
                    break

    # Final Validation
    if not is_valid(best_solution):
        # Emergency repair
        final_cc = np.zeros(n_t_subs, dtype=np.int32)
        for b in best_solution:
            final_cc[block_to_tsubs[b]] += 1
        for s in range(n_t_subs):
            if final_cc[s] == 0:
                b = tsub_to_blocks[s][0]
                if b not in set(best_solution):
                    best_solution.append(b)
                    final_cc[block_to_tsubs[b]] += 1

    return [list(all_blocks[b_idx]) for b_idx in best_solution]
