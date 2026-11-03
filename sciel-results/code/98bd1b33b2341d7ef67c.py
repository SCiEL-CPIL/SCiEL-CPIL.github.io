import itertools
import time
import random
import sys
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin

    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets as bitsets
    # A block covers a set of t-subsets. We represent this set as a large integer bitset.
    # bit j is 1 if t-subset j is covered by the block.
    
    # n_t is 11628. Python integers can handle this size efficiently.
    # We also need to know which t-subsets a block covers for quick lookup.
    
    blocks = []
    block_bitsets = []
    block_covers_list = [] # List of lists of indices, for initialization
    
    for block in itertools.combinations(range(v), k):
        covers = []
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            covers.append(idx)
            mask |= (1 << idx)
        blocks.append(block)
        block_bitsets.append(mask)
        block_covers_list.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Map from t-subset index to list of block indices that cover it
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers_list):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    def greedy_mcv():
        """Greedy set cover using bitsets for speed."""
        uncovered_mask = (1 << n_t) - 1
        selected_indices = []
        used_blocks = [False] * n_blocks
        
        while uncovered_mask:
            if not _deadline_ok(deadline):
                break
                
            # Find the uncovered subset with the fewest candidate blocks
            # Iterate through set bits of uncovered_mask
            # This is O(n_t) per step in worst case, but we can optimize by picking a random uncovered bit?
            # Standard greedy picks min candidates.
            # To find min candidates, we need to iterate all uncovered subsets.
            # 11628 subsets. ~300 steps. 3.5M ops. This is fine.
            
            min_count = float('inf')
            target_sub = -1
            m = uncovered_mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                idx = lsb.bit_length() - 1
                m ^= lsb
                
                c = len(sub_to_blocks[idx])
                if c < min_count:
                    min_count = c
                    target_sub = idx
                    if c == 1:
                        break
                        
            if target_sub == -1:
                break
                
            candidates = sub_to_blocks[target_sub]
            if not candidates:
                break

            # Calculate gain for each candidate
            # Gain = number of bits in (block_bitset & uncovered_mask)
            
            best_block = -1
            max_gain = -1
            
            # To speed up, if candidates are many, we can sample?
            # But for greedy, we should pick the best.
            # 11628 subsets, each covered by C(19,8)/C(19,5) ~ 75582/11628 ~ 6.5 blocks on average?
            # No, each block covers C(8,5)=56 subsets.
            # Total pairs = 75582 * 56 = 4.2M.
            # Average candidates per subset = 4.2M / 11628 ~ 360.
            # So we check ~360 blocks per step.
            
            for b in candidates:
                if used_blocks[b]:
                    continue
                gain = (block_bitsets[b] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
                    
            if best_block == -1:
                break
                
            selected_indices.append(best_block)
            used_blocks[best_block] = True
            uncovered_mask &= ~block_bitsets[best_block]
            
        return selected_indices

    def prune(selected_indices):
        """Remove redundant blocks."""
        selected_set = set(selected_indices)
        # Count coverage for each subset
        counts = np.zeros(n_t, dtype=np.int32)
        for i in selected_set:
            for j in block_covers_list[i]:
                counts[j] += 1
                
        changed = True
        while changed and _deadline_ok(deadline):
            changed = False
            for i in list(selected_set):
                # Check if all subsets covered by i have count > 1
                # We can check min count on its covers
                redundant = True
                for j in block_covers_list[i]:
                    if counts[j] == 1:
                        redundant = False
                        break
                if redundant:
                    selected_set.remove(i)
                    for j in block_covers_list[i]:
                        counts[j] -= 1
                    changed = True
                    if not _deadline_ok(deadline):
                        break
        return selected_set, counts

    def simulated_annealing(initial_solution):
        """
        Simulated Annealing to minimize the number of blocks.
        State: set of selected block indices.
        Objective: size of the set.
        Moves:
          1. Remove a block (if valid, i.e., it was redundant? No, we can try removing any block and check validity).
             Actually, removing a block makes the cover invalid if it was critical.
             So we need to track validity.
          2. Swap a block with another.
          3. Add a block.
        
        Let's use a state that tracks:
        - selected_set: set of block indices
        - counts: array of coverage counts for each t-subset
        - valid: boolean, is the current cover valid?
        
        If we remove a block and validity is lost, we must add a block to fix it, or reject the move.
        A simpler approach for SA:
        - State is just the set of blocks.
        - Cost is len(set) + penalty * (number of uncovered subsets).
        - We want to minimize cost.
        - If we move to a state with uncovered subsets, the penalty makes the cost high.
        
        Let's implement this.
        """
        selected_set = set(initial_solution)
        counts = np.zeros(n_t, dtype=np.int32)
        for i in selected_set:
            for j in block_covers_list[i]:
                counts[j] += 1
        
        # Initial cost
        uncovered_count = np.sum(counts == 0)
        cost = len(selected_set) + 100.0 * uncovered_count
        
        best_cost = cost
        best_set = set(selected_set)
        
        T = 10.0
        cooling = 0.995
        min_T = 0.01
        
        while T > min_T and _deadline_ok(deadline):
            # Choose a move
            # 1. Remove a block
            # 2. Swap a block
            # 3. Add a block
            
            move_type = random.random()
            
            new_set = set(selected_set)
            new_counts = counts.copy()
            
            if move_type < 0.4 and selected_set:
                # Remove
                b = random.choice(list(selected_set))
                new_set.remove(b)
                for j in block_covers_list[b]:
                    new_counts[j] -= 1
                    if new_counts[j] < 0:
                        new_counts[j] = 0 # Should not happen if state is valid, but safety
                        
            elif move_type < 0.8:
                # Swap
                if not selected_set:
                    # Must add
                    b = random.randint(0, n_blocks - 1)
                    new_set.add(b)
                    for j in block_covers_list[b]:
                        new_counts[j] += 1
                else:
                    b_old = random.choice(list(selected_set))
                    b_new = random.randint(0, n_blocks - 1)
                    if b_new in new_set:
                        continue
                    new_set.remove(b_old)
                    new_set.add(b_new)
                    for j in block_covers_list[b_old]:
                        new_counts[j] -= 1
                    for j in block_covers_list[b_new]:
                        new_counts[j] += 1
                        
            else:
                # Add
                b = random.randint(0, n_blocks - 1)
                if b not in new_set:
                    new_set.add(b)
                    for j in block_covers_list[b]:
                        new_counts[j] += 1
                else:
                    continue

            # Calculate new cost
            new_uncovered = np.sum(new_counts == 0)
            new_cost = len(new_set) + 100.0 * new_uncovered
            
            # Acceptance
            delta = new_cost - cost
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                selected_set = new_set
                counts = new_counts
                cost = new_cost
                
                if cost < best_cost:
                    best_cost = cost
                    best_set = set(selected_set)
            
            T *= cooling
            
        return best_set

    # 1. Get initial greedy solution
    greedy_sol = greedy_mcv()
    if not _deadline_ok(deadline):
        return [list(blocks[i]) for i in greedy_sol]
        
    pruned_sol, _ = prune(greedy_sol)
    
    best_result = pruned_sol
    best_len = len(pruned_sol)
    
    # 2. Run Simulated Annealing with multiple restarts
    # The SA is slow because of the copy and numpy ops.
    # We need to optimize the SA loop or reduce restarts.
    # Let's try 5 restarts.
    
    num_restarts = 5
    for i in range(num_restarts):
        if not _deadline_ok(deadline):
            break
            
        # Perturb the best solution found so far to escape local optima?
        # Or just run SA from the greedy solution?
        # Running from greedy is safe.
        # Let's run SA from the current best solution to refine it.
        
        sa_sol = simulated_annealing(list(best_result))
        
        # Prune the SA result
        pruned_sa, _ = prune(sa_sol)
        
        if len(pruned_sa) < best_len:
            best_len = len(pruned_sa)
            best_result = pruned_sa

    result = [list(blocks[i]) for i in best_result]
    
    # Verify validity
    if _deadline_ok(deadline - 2.0):
        check_counts = np.zeros(n_t, dtype=np.int32)
        for i in best_result:
            for j in block_covers_list[i]:
                check_counts[j] += 1
        if np.any(check_counts == 0):
             # Fallback: return a valid (but maybe larger) cover
             # Re-run greedy to ensure validity
             greedy_sol = greedy_mcv()
             pruned_sol, _ = prune(greedy_sol)
             result = [list(blocks[i]) for i in pruned_sol]
             
    return result

def _deadline_ok(deadline):
    return time.time() < deadline
