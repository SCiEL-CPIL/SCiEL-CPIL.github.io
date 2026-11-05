import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

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

    # Coverage bitmask: use numpy uint64 array for speed
    # Each t-subset has a coverage count
    # For fast "is this t-subset uncovered?" check, we track counts

    def greedy_construct():
        """Fast greedy to get initial solution."""
        uncovered = np.ones(n_t_subs, dtype=np.int32)
        total_uncovered = n_t_subs
        solution = []
        sol_set = set()
        
        # Precompute coverage size for each block
        block_cov_sizes = np.array([len(block_to_tsubs[b]) for b in range(n_blocks)], dtype=np.int32)
        
        while total_uncovered > 0 and time.time() < deadline - 10:
            # Find block covering most uncovered subsets
            best_b = -1
            best_count = 0
            # Sample-based: check a random subset of blocks for speed
            sample_size = min(n_blocks, 2000)
            candidates = random.sample(range(n_blocks), sample_size)
            for b in candidates:
                if b in sol_set:
                    continue
                count = 0
                for s in block_to_tsubs[b]:
                    if uncovered[s]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_b = b
            
            if best_b == -1 or best_count == 0:
                break
                
            solution.append(best_b)
            sol_set.add(best_b)
            for s in block_to_tsubs[best_b]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
        
        return solution, sol_set, uncovered, total_uncovered

    def simulate_annealing(initial_solution, time_budget):
        """Simulated annealing to minimize block count."""
        solution = list(initial_solution)
        sol_set = set(solution)
        n_sol = len(solution)
        
        # Build coverage count
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b in solution:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        # Temperature schedule
        T0 = 2.0
        T_min = 0.01
        end_time = time.time() + time_budget
        
        best_solution = list(solution)
        best_len = n_sol
        
        # Precompute: for each block, the t-subsets it covers
        # For fast evaluation of adding/removing a block
        
        # Track number of blocks covering each t-subset for O(1) criticality checks
        # cover_count already does this
        
        # Annealing loop
        while time.time() < end_time:
            # Cooldown: adjust temperature based on remaining time
            remaining_frac = max(0.0, (end_time - time.time()) / time_budget)
            T = max(T_min, T0 * remaining_frac)
            
            # Choose a random move
            move_type = random.random()
            if move_type < 0.4 and len(solution) > 0:
                # Remove a block
                idx = random.randint(0, len(solution) - 1)
                b = solution[idx]
                # Check if removing b would leave any t-subset uncovered
                would_uncover = 0
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        would_uncover += 1
                
                if would_uncover == 0:
                    # Safe removal
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                    solution.pop(idx)
                    sol_set.remove(b)
                    new_len = len(solution)
                    delta = 1  # cost increases by 1 (we want to minimize, so removing is good)
                    # Accept (it's an improvement)
                    if new_len < best_len:
                        best_len = new_len
                        best_solution = list(solution)
                else:
                    # Would break coverage: cost penalty = would_uncover
                    delta = -would_uncover  # negative = worse
                    if random.random() < np.exp(delta / T):
                        # Accept worsening
                        for s in block_to_tsubs[b]:
                            cover_count[s] -= 1
                        solution.pop(idx)
                        sol_set.remove(b)
                        
            elif move_type < 0.7:
                # Add a random block
                b = random.randint(0, n_blocks - 1)
                if b not in sol_set:
                    # Count how many new t-subsets this covers
                    new_coverage = 0
                    for s in block_to_tsubs[b]:
                        if cover_count[s] == 0:
                            new_coverage += 1
                    
                    delta = -1 + new_coverage  # -1 for adding block, +new_coverage for fixing uncovered
                    if new_coverage > 0 or random.random() < np.exp(delta / T):
                        for s in block_to_tsubs[b]:
                            cover_count[s] += 1
                        solution.append(b)
                        sol_set.add(b)
                        new_len = len(solution)
                        if new_len < best_len:
                            best_len = new_len
                            best_solution = list(solution)
                        
            else:
                # Swap: remove one block, add another
                if len(solution) > 0:
                    idx = random.randint(0, len(solution) - 1)
                    b_old = solution[idx]
                    b_new = random.randint(0, n_blocks - 1)
                    if b_new != b_old and b_new not in sol_set:
                        # Evaluate swap
                        # Remove b_old, add b_new
                        # Net effect on coverage
                        # t-subsets only covered by b_old: must be covered by b_new
                        # t-subsets only covered by b_new: become covered (good)
                        
                        # Compute delta
                        would_uncover = 0
                        for s in block_to_tsubs[b_old]:
                            if cover_count[s] == 1 and s not in block_to_tsubs[b_new]:
                                would_uncover += 1
                        
                        new_coverage = 0
                        for s in block_to_tsubs[b_new]:
                            if cover_count[s] == 0:
                                new_coverage += 1
                        
                        delta = -would_uncover + new_coverage
                        # Size doesn't change, so no size delta
                        if would_uncover == 0 or random.random() < np.exp(delta / T):
                            # Perform swap
                            for s in block_to_tsubs[b_old]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[b_new]:
                                cover_count[s] += 1
                            solution[idx] = b_new
                            sol_set.remove(b_old)
                            sol_set.add(b_new)
        
        return best_solution, best_len

    # Main loop: multiple rounds of greedy + simulated annealing
    best_blocks = []
    best_len = float('inf')
    
    # First, get a quick greedy solution as baseline
    greedy_sol, greedy_set, _, total_unc = greedy_construct()
    if total_unc == 0 and len(greedy_sol) < best_len:
        best_len = len(greedy_sol)
        best_blocks = list(greedy_sol)
    
    # Now run multiple SA rounds with different initial solutions
    round_budget = 25.0  # seconds per round
    while time.time() < deadline - 5:
        # Generate a new greedy solution (with randomness)
        greedy_sol, greedy_set, _, total_unc = greedy_construct()
        
        if total_unc > 0:
            # Try to fix by adding blocks for uncovered subsets
            cover_count = np.zeros(n_t_subs, dtype=np.int32)
            for b in greedy_sol:
                for s in block_to_tsubs[b]:
                    cover_count[s] += 1
            
            for s_idx in range(n_t_subs):
                if cover_count[s_idx] == 0:
                    # Find a block covering this subset
                    for b in tsub_to_blocks[s_idx]:
                        if b not in greedy_set:
                            greedy_sol.append(b)
                            greedy_set.add(b)
                            for s2 in block_to_tsubs[b]:
                                cover_count[s2] += 1
                            break
            
            if time.time() > deadline - 5:
                break
        
        # Run simulated annealing
        sa_solution, sa_len = simulate_annealing(greedy_sol, round_budget)
        
        if sa_len < best_len:
            best_len = sa_len
            best_blocks = list(sa_solution)
        
        # If we're close to lower bound, reduce round budget for more rounds
        if best_len <= 140:
            round_budget = 15.0
        elif best_len <= 145:
            round_budget = 20.0
    
    # Final pass: try to remove redundant blocks from best solution
    if best_blocks:
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b in best_blocks:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(best_blocks)):
                b = best_blocks[i]
                redundant = True
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                    best_blocks.pop(i)
                    changed = True
                    break
    
    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return final_blocks
