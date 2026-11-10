import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # 1. Precomputation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # block_subsets[i] = list of indices of t-subsets covered by block i
    block_subsets = [[] for _ in range(n_blocks)]
    # block_subsets_set[i] = set of indices for O(1) lookup
    block_subsets_set = [set() for _ in range(n_blocks)]
    # subset_to_blocks[j] = list of block indices covering t-subset j
    subset_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            block_subsets[i].append(idx)
            block_subsets_set[i].add(idx)
            subset_to_blocks[idx].append(i)
    
    # 2. Greedy Construction
    in_block = [False] * n_blocks
    cover_count = [0] * n_t_subsets
    uncovered_count = n_t_subsets
    solution = []
    
    while uncovered_count > 0:
        best_block = -1
        max_gain = 0
        for i in range(n_blocks):
            if in_block[i]:
                continue
            gain = 0
            for s_idx in block_subsets[i]:
                if cover_count[s_idx] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = i
                if gain == uncovered_count:
                    break
        
        if best_block == -1 or max_gain == 0:
            break
            
        in_block[best_block] = True
        solution.append(best_block)
        for s_idx in block_subsets[best_block]:
            if cover_count[s_idx] == 0:
                uncovered_count -= 1
            cover_count[s_idx] += 1
    
    # 3. Remove redundant blocks
    def remove_redundants():
        changed = True
        while changed:
            changed = False
            for b in list(solution):
                if not in_block[b]:
                    continue
                redundant = True
                for s_idx in block_subsets[b]:
                    if cover_count[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    in_block[b] = False
                    solution.remove(b)
                    for s_idx in block_subsets[b]:
                        cover_count[s_idx] -= 1
                    changed = True
                    break
    
    remove_redundants()
    
    # 4. Simulated Annealing
    # State: which blocks are in the solution
    # Energy = number of blocks + lambda * number of uncovered t-subsets
    # We want to minimize energy while maintaining full coverage
    
    best_solution = list(solution)
    best_count = len(solution)
    
    # SA parameters
    T_start = 5.0
    T_end = 0.01
    n_iterations = 0
    
    # Helper to compute energy
    def get_energy():
        n_uncovered = sum(1 for c in cover_count if c == 0)
        return len(solution) + 10.0 * n_uncovered
    
    current_energy = get_energy()
    
    # SA loop
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / 10000.0)
    
    while time.time() < deadline - 2.0:  # Leave 2s for cleanup
        if time.time() - start_time > 115.0:
            break
            
        n_iterations += 1
        
        # Choose a move: add, remove, or swap a block
        move_type = random.random()
        
        if move_type < 0.3 and solution:
            # Remove a block
            b_out = random.choice(solution)
            # Check if removal keeps coverage valid
            new_uncovered = 0
            for s_idx in block_subsets[b_out]:
                if cover_count[s_idx] == 1:
                    new_uncovered += 1
            
            # Energy change: -1 (fewer blocks) + 10 * new_uncovered
            delta_e = -1.0 + 10.0 * new_uncovered
            
            if delta_e <= 0 or random.random() < math.exp(-delta_e / max(T, 1e-10)):
                # Accept move
                in_block[b_out] = False
                solution.remove(b_out)
                for s_idx in block_subsets[b_out]:
                    cover_count[s_idx] -= 1
                current_energy += delta_e
                
                # Remove redundants after removal
                remove_redundants()
                current_energy = get_energy()
                
                if len(solution) < best_count:
                    best_count = len(solution)
                    best_solution = list(solution)
                    
        elif move_type < 0.6:
            # Add a block
            b_in = random.randint(0, n_blocks - 1)
            if in_block[b_in]:
                T *= cooling_rate
                continue
                
            # Energy change: +1 (more blocks) - 10 * newly_covered
            newly_covered = 0
            for s_idx in block_subsets[b_in]:
                if cover_count[s_idx] == 0:
                    newly_covered += 1
                    
            delta_e = 1.0 - 10.0 * newly_covered
            
            if delta_e <= 0 or random.random() < math.exp(-delta_e / max(T, 1e-10)):
                # Accept move
                in_block[b_in] = True
                solution.append(b_in)
                for s_idx in block_subsets[b_in]:
                    cover_count[s_idx] += 1
                current_energy += delta_e
                
                # Remove redundants after addition
                remove_redundants()
                current_energy = get_energy()
                
                if len(solution) < best_count:
                    best_count = len(solution)
                    best_solution = list(solution)
                    
        else:
            # Swap: remove one, add another
            if not solution:
                T *= cooling_rate
                continue
                
            b_out = random.choice(solution)
            b_in = random.randint(0, n_blocks - 1)
            if in_block[b_in] and b_in != b_out:
                T *= cooling_rate
                continue
                
            # Compute energy change for swap
            # First compute what would happen if we removed b_out
            newly_uncovered_out = 0
            for s_idx in block_subsets[b_out]:
                if cover_count[s_idx] == 1:
                    newly_uncovered_out += 1
                    
            # Then compute what would be covered by b_in (considering b_out is removed)
            newly_covered_in = 0
            for s_idx in block_subsets[b_in]:
                # After removing b_out, cover_count would be cover_count[s_idx] - (1 if s_idx in b_out else 0)
                cc = cover_count[s_idx]
                if s_idx in block_subsets_set[b_out]:
                    cc -= 1
                if cc == 0:
                    newly_covered_in += 1
                    
            # Net change: blocks stays same (remove 1, add 1), uncovered changes
            net_uncovered_change = newly_uncovered_out - newly_covered_in
            delta_e = 10.0 * net_uncovered_change
            
            if delta_e <= 0 or random.random() < math.exp(-delta_e / max(T, 1e-10)):
                # Accept swap
                # Remove b_out
                in_block[b_out] = False
                solution.remove(b_out)
                for s_idx in block_subsets[b_out]:
                    cover_count[s_idx] -= 1
                    
                # Add b_in
                in_block[b_in] = True
                solution.append(b_in)
                for s_idx in block_subsets[b_in]:
                    cover_count[s_idx] += 1
                    
                current_energy = get_energy()
                
                # Remove redundants
                remove_redundants()
                current_energy = get_energy()
                
                if len(solution) < best_count:
                    best_count = len(solution)
                    best_solution = list(solution)
                    
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        # Periodic check: if we've been stuck, restart from best
        if n_iterations % 5000 == 0 and time.time() < deadline - 10.0:
            # Try to recover from best solution if current is worse
            if len(solution) > best_count + 5:
                # Reset to best solution
                for b in solution:
                    in_block[b] = False
                solution.clear()
                for s_idx in range(n_t_subsets):
                    cover_count[s_idx] = 0
                    
                for b in best_solution:
                    in_block[b] = True
                    solution.append(b)
                    for s_idx in block_subsets[b]:
                        cover_count[s_idx] += 1
                        
                current_energy = get_energy()
    
    # 5. Final cleanup: ensure we return a valid covering
    # If best_solution is valid, use it; otherwise use current solution
    # Verify best_solution is valid
    best_cover_count = [0] * n_t_subsets
    for b in best_solution:
        for s_idx in block_subsets[b]:
            best_cover_count[s_idx] += 1
            
    if all(c > 0 for c in best_cover_count):
        return [list(all_blocks[i]) for i in best_solution]
    else:
        # Fallback to current solution if best wasn't valid
        return [list(all_blocks[i]) for i in solution]
