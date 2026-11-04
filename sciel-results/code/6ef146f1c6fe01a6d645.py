import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage
    # block_to_subsets: list of lists of subset indices
    block_to_subsets = []
    # subset_to_blocks: list of lists of block indices
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        covered_subsets = []
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            covered_subsets.append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
        block_to_subsets.append(covered_subsets)
        
    def run_greedy_init():
        """
        Fast greedy to get a valid initial solution.
        """
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(bs) for bs in block_to_subsets], dtype=np.int32)
        selected = []
        
        while True:
            if np.all(~uncovered):
                break
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            
            b_subs = block_to_subsets[best_idx]
            newly_covered = [s for s in b_subs if uncovered[s]]
            for s in newly_covered:
                uncovered[s] = False
            
            # Update gains
            # We need to decrease gains for all blocks that cover the newly covered subsets
            # This is O(num_blocks * t) roughly per step if done naively, but we can optimize.
            # For now, keep it simple but efficient enough for initialization.
            for s in newly_covered:
                for b in subset_to_blocks[s]:
                    if b != best_idx:
                        gains[b] -= 1
            gains[best_idx] = 0
            
        return set(selected)

    def local_search(initial_selected, time_limit=100):
        """
        Local search to minimize the number of blocks.
        Moves:
        1. Remove a block if it is redundant.
        2. Swap a block with another if it maintains coverage and allows further reduction.
        """
        selected = set(initial_selected)
        # count[s] = number of selected blocks covering subset s
        count = np.zeros(num_subsets, dtype=np.int32)
        for b in selected:
            for s in block_to_subsets[b]:
                count[s] += 1
        
        start_time = time.time()
        rng = random.Random(42)
        
        while time.time() - start_time < time_limit:
            progress = False
            # Try to remove redundant blocks
            # A block is redundant if for all s in block_to_subsets[b], count[s] > 1
            # We can check this efficiently.
            
            # Shuffle selected to avoid bias
            selected_list = list(selected)
            rng.shuffle(selected_list)
            
            for b in selected_list:
                if b not in selected:
                    continue
                is_redundant = True
                for s in block_to_subsets[b]:
                    if count[s] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove b
                    selected.remove(b)
                    for s in block_to_subsets[b]:
                        count[s] -= 1
                    progress = True
                    break # Restart loop after removal to re-evaluate others
            
            if progress:
                continue
                
            # If no removal, try a swap
            # Pick a random selected block b_in
            # Try to find an unselected block b_out such that:
            # 1. b_out covers all "critical" subsets of b_in (where count[s]==1 for b_in)
            # 2. Ideally b_out doesn't break coverage for others (it adds to counts, so it won't break)
            # 3. The swap might allow future removals.
            
            if selected:
                b_in = rng.choice(list(selected))
                critical_subs = [s for s in block_to_subsets[b_in] if count[s] == 1]
                
                if not critical_subs:
                    # b_in is redundant? We should have caught it above.
                    # Unless count changed? No, we re-evaluated.
                    # If critical_subs is empty, b_in is redundant.
                    selected.remove(b_in)
                    for s in block_to_subsets[b_in]:
                        count[s] -= 1
                    progress = True
                    continue

                # Find candidates for b_out
                # b_out must cover all critical_subs
                # We can precompute or just iterate.
                # To make it faster, we can check a random sample of unselected blocks.
                # Or iterate through all? num_blocks is ~ 75582. 
                # Checking each candidate against critical_subs (size <= C(8,5)=56) is cheap.
                
                # Let's try a few random unselected blocks
                unselected = [b for b in range(num_blocks) if b not in selected]
                if not unselected:
                    break
                    
                # Sample up to 100 random unselected blocks
                candidates = rng.sample(unselected, min(100, len(unselected)))
                
                found_swap = False
                for b_out in candidates:
                    # Check if b_out covers all critical_subs
                    valid = True
                    for s in critical_subs:
                        if s not in block_to_subsets[b_out]:
                            valid = False
                            break
                    if valid:
                        # Perform swap
                        selected.remove(b_in)
                        selected.add(b_out)
                        
                        # Update counts
                        for s in block_to_subsets[b_in]:
                            count[s] -= 1
                        for s in block_to_subsets[b_out]:
                            count[s] += 1
                            
                        found_swap = True
                        progress = True
                        break
                        
                if found_swap:
                    continue # Restart loop to check for new redundancies
            
            # If no progress in this pass, we might be stuck in a local minimum.
            # For a more robust search, we could accept a "worse" swap (increasing size) 
            # but that's complex. For now, if stuck, break or restart with different init.
            # Given time limits, breaking and relying on multiple restarts is safer.
            break
            
        return list(selected)

    # 1. Get initial solution via greedy
    initial_sel = run_greedy_init()
    
    # 2. Run local search
    best_blocks = local_search(initial_sel, time_limit=10)
    
    # 3. Try a few more restarts if time permits
    # The local search above is deterministic in its randomness seed if not reset.
    # Let's run it a couple of times with different random seeds for the local search part.
    # We need to modify local_search to accept a seed or use global random.
    
    # Actually, the above local_search uses random.Random(42). 
    # Let's just run the whole process a few times if we have time.
    # But the greedy init is deterministic.
    # Let's add a bit of randomness to the greedy init to get different starting points.
    
    def run_greedy_randomized(seed):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(bs) for bs in block_to_subsets], dtype=np.int32)
        selected = []
        
        while True:
            if np.all(~uncovered):
                break
            # Add noise to break ties
            noise = rng.uniform(0, 1.0, size=num_blocks)
            # Only add noise to blocks with gain > 0 to avoid selecting 0-gain blocks if possible?
            # Actually, if all gains are 0, we break.
            current_gains = gains + noise
            best_idx = np.argmax(current_gains)
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            
            b_subs = block_to_subsets[best_idx]
            newly_covered = [s for s in b_subs if uncovered[s]]
            for s in newly_covered:
                uncovered[s] = False
            
            for s in newly_covered:
                for b in subset_to_blocks[s]:
                    if b != best_idx:
                        gains[b] -= 1
            gains[best_idx] = 0
            
        return set(selected)

    best_len = float('inf')
    best_blocks = None
    
    # Try a few different greedy starts
    for seed in [0, 1, 2]:
        start_time = time.time()
        init_sel = run_greedy_randomized(seed)
        
        # Run local search on this init
        # We need to pass time limit carefully
        remaining = 110 - (time.time() - start_time)
        if remaining < 5:
            continue
            
        # Modify local_search to accept a seed? 
        # The current local_search uses random.Random(42). 
        # Let's just call it. The randomness inside is fixed, but the input is different.
        # To get different local search behaviors, we should randomize the local search too.
        # Let's refactor local_search to accept a seed.
        
        # For simplicity and speed, let's just use the deterministic local search on different inits.
        # The different inits will likely lead to different local minima.
        
        blocks = local_search(init_sel, time_limit=min(10, remaining))
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # Ensure we return a valid list of blocks
    if best_blocks is None:
        # Fallback to greedy
        best_blocks = [all_blocks[i] for i in run_greedy_init()]
    else:
        best_blocks = [all_blocks[i] for i in best_blocks]
        
    return best_blocks
