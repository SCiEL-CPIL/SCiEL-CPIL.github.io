import time
import random
import itertools
import numpy as np

def C(n, r):
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    r = min(r, n - r)
    result = 1
    for i in range(1, r + 1):
        result = result * (n - i + 1) // i
    return result

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Precomputation
    # We need to map t-subsets to indices and blocks to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures
    # block_to_ts[j]: list of t-subset indices contained in block j
    # ts_to_blocks[i]: list of block indices containing t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
        
    # 2. Initial Solution via Greedy (Fast)
    # This provides a valid starting point for the Max-SAT local search.
    # We use a simple greedy to get a valid cover quickly.
    
    def run_greedy(seed_val):
        random.seed(seed_val)
        # gain[j] = number of currently uncovered t-subsets in block j
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 10.0:
                return None
            
            # Find max gain among unselected
            # To speed up, we can just check a random subset or use argmax with masking
            # But for initial seed, simple argmax is fine if we mask selected.
            # However, argmax over 75k is fast.
            
            # Mask selected blocks with -1 for max finding
            # We don't want to modify the actual gain array permanently for the max find 
            # in a way that breaks updates, but since we set selected to -1 in gain 
            # during the update phase in the previous implementation, let's stick to that.
            
            max_idx = np.argmax(gain)
            if gain[max_idx] <= 0:
                break
                
            # In a randomized greedy, we might pick randomly from top candidates.
            # For the initial seed, deterministic greedy is robust.
            # Let's stick to deterministic for the first seed to ensure a good start.
            
            sol.append(list(all_k_blocks[max_idx]))
            is_selected[max_idx] = True
            gain[max_idx] = -1
            
            for ts_idx in block_to_ts[max_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            
        return sol

    initial_sol = run_greedy(42)
    if initial_sol is None:
        # Fallback: just return empty or something, but greedy should work
        return []
        
    best_sol = initial_sol
    best_len = len(best_sol)
    
    # 3. Max-SAT Local Search (WalkSAT style)
    # Variables: x_j (1 if block j is in solution)
    # Hard Constraints: For every t-subset s, sum(x_j for j in ts_to_blocks[s]) >= 1
    # Objective: Minimize sum(x_j)
    
    # We represent the current state as a boolean array `in_sol` of size num_k_blocks.
    # We also maintain `cover_count[s]` = number of blocks in solution covering s.
    # A solution is valid if min(cover_count) >= 1.
    
    def evaluate_state(in_sol):
        # Returns (num_blocks, is_valid, cover_count)
        # We assume in_sol is a boolean numpy array
        num_blocks = np.sum(in_sol)
        
        # Calculate cover counts efficiently?
        # cover_count = np.zeros(num_t_subsets)
        # for j in np.where(in_sol)[0]:
        #     for ts in block_to_ts[j]:
        #         cover_count[ts] += 1
        # This is O(num_blocks * 56). If num_blocks ~ 300, that's 16k ops. Very fast.
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        indices = np.where(in_sol)[0]
        for j in indices:
            for ts in block_to_ts[j]:
                cover_count[ts] += 1
                
        is_valid = np.all(cover_count >= 1)
        return num_blocks, is_valid, cover_count

    # Current best state
    best_in_sol = np.zeros(num_k_blocks, dtype=bool)
    for b in best_sol:
        # Need map from block tuple to index
        # We don't have block_index_map. Let's create one or use the list index.
        # Since all_k_blocks is a list of tuples, and b is a list, we need to convert.
        # Actually, in run_greedy we appended list(all_k_blocks[max_idx]).
        # So we can just find the index? No, that's slow.
        # Let's modify run_greedy to return indices.
        
        # Wait, I should refactor run_greedy to return indices to make this easier.
        pass

    # Refactor: Let's work with indices internally.
    # Re-run greedy to get indices.
    def run_greedy_indices(seed_val):
        random.seed(seed_val)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol_indices = []
        
        while True:
            if time.time() > deadline - 10.0:
                return None
            
            max_idx = np.argmax(gain)
            if gain[max_idx] <= 0:
                break
            
            sol_indices.append(max_idx)
            is_selected[max_idx] = True
            gain[max_idx] = -1
            
            for ts_idx in block_to_ts[max_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            
        return sol_indices

    best_indices = run_greedy_indices(42)
    if best_indices is None:
        return []
        
    best_in_sol = np.zeros(num_k_blocks, dtype=bool)
    for idx in best_indices:
        best_in_sol[idx] = True
        
    best_len = len(best_indices)
    
    # Local Search Function
    # State: in_sol (bool array)
    # cover_count (int array)
    
    def local_search(in_sol, max_steps=20000):
        # Initialize cover counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        indices = np.where(in_sol)[0]
        for j in indices:
            for ts in block_to_ts[j]:
                cover_count[ts] += 1
                
        current_len = len(indices)
        best_local_len = current_len
        best_local_sol = in_sol.copy()
        
        # We want to minimize length.
        # Move types:
        # 1. Remove a block j if it doesn't break coverage (all ts in j have count > 1).
        # 2. Swap block j with block k? 
        #    - Remove j, add k.
        #    - Condition: After removal, no ts has count 0. After addition, no ts has count < 0 (impossible).
        #    - Actually, we can just try to remove first. If we can't remove, try swaps.
        #    - A swap is valid if:
        #       a) Removing j leaves all ts covered (min count among ts in j > 1) OR 
        #          the ts that become 0 are covered by k.
        #       b) Adding k doesn't cause issues (it only increases counts).
        #    - So a swap is valid if: for all ts in j, (cover_count[ts] > 1) OR (ts in k).
        #      Wait, if ts in j has count 1, and ts is NOT in k, then removing j makes it 0. Invalid.
        #      If ts in j has count 1, and ts IS in k, then removing j makes it 0, but adding k makes it 1. Valid.
        
        # Let's implement a simple 1-flip (add/remove) and 2-flip (swap) search.
        
        # Precompute: for each block j, list of ts.
        # We already have block_to_ts.
        
        # To make this fast, we can precompute for each block j, the set of ts.
        # But we need to check conditions dynamically.
        
        # Heuristic:
        # 1. Try to remove any block that is redundant.
        # 2. If no removal, try to swap a "bad" block with a "good" one?
        #    Or just random swaps.
        
        # Let's use a simulated annealing / WalkSAT style.
        # Score = current_len + penalty * (number of uncovered ts)
        # We want to minimize score.
        
        # Actually, simpler:
        # Step 1: Greedy removal of redundant blocks.
        # Step 2: Randomized swaps to escape local minima.
        
        # Let's stick to the "Minimize Size" objective with hard constraints.
        # If a move violates constraints, we don't take it (unless we use penalty, but let's keep it hard).
        
        # Optimization: 
        # Check if block j is removable:
        #   min(cover_count[ts] for ts in block_to_ts[j]) > 1
        
        # Check if swap (j -> k) is valid:
        #   For all ts in block_to_ts[j]:
        #       if cover_count[ts] == 1:
        #           ts must be in block_to_ts[k]
        
        # To check "ts in k" fast, we can use a set for block k's ts?
        # Or precompute a bitset?
        # Given 56 ts per block, a simple loop is fine.
        
        # Let's implement the search.
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
                
            # 1. Try to find a removable block
            # Randomly shuffle blocks to avoid bias
            # But checking all 75k blocks every step is 75k * 56 = 4.2M ops. 
            # 20000 steps * 4.2M = 84B ops. Too slow.
            
            # We need to be smarter.
            # Only blocks that are in the solution can be removed.
            # So we iterate over current indices.
            
            current_indices = list(np.where(in_sol)[0])
            if not current_indices:
                break
                
            random.shuffle(current_indices)
            
            removed = False
            for j in current_indices:
                # Check if j is removable
                removable = True
                for ts in block_to_ts[j]:
                    if cover_count[ts] <= 1:
                        removable = False
                        break
                        
                if removable:
                    # Remove j
                    in_sol[j] = False
                    for ts in block_to_ts[j]:
                        cover_count[ts] -= 1
                    current_len -= 1
                    if current_len < best_local_len:
                        best_local_len = current_len
                        best_local_sol = in_sol.copy()
                    removed = True
                    break # Restart search with new state
                    
            if removed:
                continue
                
            # 2. Try a swap
            # Pick a random block j from solution
            j = random.choice(current_indices)
            
            # Pick a random block k NOT in solution
            # We need a list of non-selected blocks.
            # Maintaining a list of non-selected blocks is hard as it changes.
            # We can just pick a random index from 0..num_k_blocks-1 and check if it's selected.
            # If selected, pick another.
            
            k = -1
            for _ in range(10): # Try a few times
                k = random.randint(0, num_k_blocks - 1)
                if not in_sol[k]:
                    break
            if k == -1:
                continue
                
            # Check if swap j -> k is valid
            # Condition: For all ts in block_to_ts[j], if cover_count[ts] == 1, then ts must be in block_to_ts[k].
            
            # To check "ts in k", we can create a set of block_to_ts[k]?
            # Or just loop through block_to_ts[k] and mark?
            # 56 elements.
            
            # Let's precompute a set for k? No, creating a set every time is slow.
            # Let's use a boolean array? No.
            # Let's just do:
            # ts_k_set = set(block_to_ts[k])
            # This is 56 inserts. 20000 steps * 56 = 1.1M inserts. Fast enough.
            
            ts_k_set = set(block_to_ts[k])
            
            valid_swap = True
            for ts in block_to_ts[j]:
                if cover_count[ts] == 1 and ts not in ts_k_set:
                    valid_swap = False
                    break
                    
            if valid_swap:
                # Perform swap
                # Remove j
                for ts in block_to_ts[j]:
                    cover_count[ts] -= 1
                in_sol[j] = False
                
                # Add k
                for ts in block_to_ts[k]:
                    cover_count[ts] += 1
                in_sol[k] = True
                
                # Length stays same, but we might enable further removals later.
                # So we don't update best_local_len here unless we removed something else.
                # But the swap itself doesn't reduce length.
                # However, it might lead to a state where removal is possible.
                
        return best_local_sol, best_local_len

    # Run local search with restarts
    # We start from the greedy solution.
    # We can also try starting from slightly perturbed greedy solutions.
    
    current_best_sol = best_in_sol
    current_best_len = best_len
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Start from current best
        in_sol = current_best_sol.copy()
        
        # Run local search
        improved_sol, improved_len = local_search(in_sol, max_steps=50000)
        
        if improved_len < current_best_len:
            current_best_sol = improved_sol
            current_best_len = improved_len
            
    # Final conversion to list of blocks
    final_indices = np.where(current_best_sol)[0]
    final_blocks = [list(all_k_blocks[i]) for i in final_indices]
    
    return final_blocks

# Helper for combinations count
def C(n, r):
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    r = min(r, n - r)
    result = 1
    for i in range(1, r + 1):
        result = result * (n - i + 1) // i
    return result
