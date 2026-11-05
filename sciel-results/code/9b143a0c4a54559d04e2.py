import itertools
import random
import time
import math

def construct(v, k, t):
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block to list of subset IDs it covers
    block_to_subsets = []
    for b in all_blocks:
        b_set = set(b)
        subs = []
        for i, s in enumerate(all_subsets):
            if all(elem in b_set for elem in s):
                subs.append(i)
        block_to_subsets.append(subs)
        
    # Map subset to list of block IDs that cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_id, subs in enumerate(block_to_subsets):
        for s_id in subs:
            subset_to_blocks[s_id].append(b_id)

    def greedy_init():
        """Standard greedy to get a valid initial cover."""
        uncovered = set(range(n_subsets))
        blocks = []
        while uncovered:
            best_block = -1
            best_gain = -1
            # Pick a random uncovered subset to anchor the search
            # Converting set to list is O(N), but N is small enough
            target_s = random.choice(list(uncovered))
            candidates = subset_to_blocks[target_s]
            
            for b_id in candidates:
                # Calculate gain: how many uncovered subsets does b_id cover?
                gain = 0
                for s_id in block_to_subsets[b_id]:
                    if s_id in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_id
            
            if best_block != -1:
                blocks.append(best_block)
                # Update uncovered
                for s_id in block_to_subsets[best_block]:
                    uncovered.discard(s_id)
            
            if not uncovered:
                break
            
        return blocks

    def sa_optimize(initial_blocks, time_limit=90.0):
        """Simulated Annealing to minimize block count."""
        current_blocks = set(initial_blocks)
        cover_count = [0] * n_subsets
        for b_id in current_blocks:
            for s_id in block_to_subsets[b_id]:
                cover_count[s_id] += 1
        
        BIG_WEIGHT = 1000
        
        def get_energy(counts):
            uncov = 0
            for c in counts:
                if c == 0:
                    uncov += 1
            return len(current_blocks) * BIG_WEIGHT + uncov

        current_energy = get_energy(cover_count)
        
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        start_time = time.time()
        
        while T > T_min and (time.time() - start_time) < time_limit:
            move_type = random.random()
            
            if move_type < 0.4:
                # Remove a block
                if not current_blocks: continue
                b_remove = random.choice(list(current_blocks))
                
                # Calculate delta
                delta_energy = -BIG_WEIGHT
                for s_id in block_to_subsets[b_remove]:
                    if cover_count[s_id] == 1:
                        delta_energy += 1
                        
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    current_blocks.discard(b_remove)
                    for s_id in block_to_subsets[b_remove]:
                        cover_count[s_id] -= 1
                    current_energy += delta_energy
                    
            elif move_type < 0.7:
                # Add a block
                b_add = random.choice(range(n_blocks))
                if b_add in current_blocks:
                    continue
                
                delta_energy = BIG_WEIGHT
                for s_id in block_to_subsets[b_add]:
                    if cover_count[s_id] == 0:
                        delta_energy -= 1
                        
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    current_blocks.add(b_add)
                    for s_id in block_to_subsets[b_add]:
                        cover_count[s_id] += 1
                    current_energy += delta_energy
            else:
                # Swap
                if not current_blocks: continue
                b_remove = random.choice(list(current_blocks))
                b_add = random.choice(range(n_blocks))
                if b_add in current_blocks:
                    continue
                    
                # Calculate new energy
                temp_counts = cover_count[:]
                for s_id in block_to_subsets[b_remove]:
                    temp_counts[s_id] -= 1
                for s_id in block_to_subsets[b_add]:
                    temp_counts[s_id] += 1
                
                uncov_new = 0
                for c in temp_counts:
                    if c == 0:
                        uncov_new += 1
                        
                new_energy = len(current_blocks) * BIG_WEIGHT + uncov_new
                delta_energy = new_energy - current_energy
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    # Accept
                    current_blocks.discard(b_remove)
                    current_blocks.add(b_add)
                    for s_id in block_to_subsets[b_remove]:
                        cover_count[s_id] -= 1
                    for s_id in block_to_subsets[b_add]:
                        cover_count[s_id] += 1
                    current_energy = new_energy

            T *= cooling
            
        # Final polish: Remove redundant blocks
        final_blocks = list(current_blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(final_blocks)):
                b_id = final_blocks[i]
                redundant = True
                for s_id in block_to_subsets[b_id]:
                    if cover_count[s_id] == 1:
                        redundant = False
                        break
                if redundant:
                    for s_id in block_to_subsets[b_id]:
                        cover_count[s_id] -= 1
                    final_blocks.pop(i)
                    changed = True
                    break
        
        return final_blocks

    # Main execution
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    
    # Run a few greedy inits to find a good starting point
    num_greedy_runs = 5
    for i in range(num_greedy_runs):
        init_blocks = greedy_init()
        if len(init_blocks) < best_count:
            best_count = len(init_blocks)
            best_blocks = init_blocks
            
    elapsed = time.time() - start_time
    if elapsed < 110:
        # Apply SA to the best block set
        sa_blocks = sa_optimize(best_blocks, time_limit=110 - elapsed)
        
        if len(sa_blocks) < best_count:
            best_count = len(sa_blocks)
            best_blocks = sa_blocks
            
    # Convert block IDs to lists
    if best_blocks is not None:
        final_result = [list(all_blocks[b_id]) for b_id in best_blocks]
        return final_result
    else:
        return []
