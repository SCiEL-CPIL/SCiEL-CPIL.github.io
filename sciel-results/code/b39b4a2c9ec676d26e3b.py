import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit
    
    points = list(range(v))
    
    # 1. Precomputation
    # Generate all k-subsets (potential blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Map block -> list of t-subset indices
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
        
    # Map t-subset -> list of block indices
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
            
    # Convert to numpy arrays for faster access if needed, 
    # but lists are fine for iteration in reduction if we use indices.
    
    def run_greedy(seed):
        random.seed(seed)
        # Initial gains: number of t-subsets covered by each block
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        # For speed, we can use a heap, but with vectorized updates, 
        # np.argmax is O(N) which is fine for ~75k blocks.
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            # Find all blocks with max gain
            candidates = np.where(block_gain == max_gain)[0]
            best_block_idx = random.choice(candidates.tolist())
            
            selected_indices.append(best_block_idx)
            
            # Update coverage and gains
            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    # Update gains for all blocks covering this t-subset
                    block_gain[tsub_blocks[si]] -= 1
                        
        return selected_indices

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []
        
        n_sel = len(selected_indices)
        global_to_local = {g: i for i, g in enumerate(selected_indices)}
        
        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)
        
        # Compute total_count
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                total_count[si] += 1
                
        # Compute crit_count
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1
                    
        queue = [i for i in range(n_sel) if crit_count[i] == 0]
        removed = [False] * n_sel
        
        while queue:
            i = queue.pop()
            if removed[i] or crit_count[i] > 0:
                continue
                
            removed[i] = True
            g = selected_indices[i]
            
            for si in block_tsubs[g]:
                old_c = total_count[si]
                total_count[si] -= 1
                new_c = total_count[si]
                
                if old_c == 2 and new_c == 1:
                    # si is now critical. Update crit_count for all other blocks containing si.
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        if g_other in global_to_local:
                            i_other = global_to_local[g_other]
                            if not removed[i_other]:
                                crit_count[i_other] += 1
                
                elif old_c == 1:
                    pass
        
        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def is_valid_cover(selected_indices):
        """Check if selected_indices form a valid cover."""
        covered = np.zeros(N_t, dtype=np.int8)
        for g in selected_indices:
            for si in block_tsubs[g]:
                covered[si] = 1
        return np.all(covered == 1)

    def local_search(selected_indices, max_iter=100):
        """
        Local search: try to remove blocks or swap blocks to improve the solution.
        Returns an improved list of global block indices.
        """
        if not selected_indices:
            return []
        
        current = list(selected_indices)
        current_set = set(current)
        n_sel = len(current)
        
        # Compute coverage counts
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in current:
            for si in block_tsubs[g]:
                total_count[si] += 1
        
        # Try removals first
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            for i in range(len(current)):
                g = current[i]
                # Check if removing g keeps the cover valid
                removable = True
                for si in block_tsubs[g]:
                    if total_count[si] == 1:
                        removable = False
                        break
                if removable:
                    # Remove g
                    current.pop(i)
                    current_set.discard(g)
                    for si in block_tsubs[g]:
                        total_count[si] -= 1
                    improved = True
                    break  # Restart loop after removal
        
        # Try swaps: replace one block with another that covers more uncovered t-subsets
        # For simplicity, try random swaps
        random.shuffle(current)
        for i in range(len(current)):
            g_old = current[i]
            # Find blocks that could replace g_old
            # A good replacement covers t-subsets that g_old covers, plus maybe more
            # For now, try random blocks from a candidate set
            # This is expensive, so limit candidates
            candidates = []
            # Sample some random blocks
            for _ in range(50):
                g_new = random.randint(0, N_blocks - 1)
                if g_new not in current_set:
                    candidates.append(g_new)
            
            for g_new in candidates:
                # Check if replacing g_old with g_new keeps the cover valid
                # Remove g_old, add g_new, check coverage
                temp_count = total_count.copy()
                for si in block_tsubs[g_old]:
                    temp_count[si] -= 1
                for si in block_tsubs[g_new]:
                    temp_count[si] += 1
                
                if np.all(temp_count >= 1):
                    # Valid replacement
                    current[i] = g_new
                    current_set.discard(g_old)
                    current_set.add(g_new)
                    total_count = temp_count
                    improved = True
                    break
            
            if improved:
                break  # Restart after swap
        
        return current

    def crossover(sol1, sol2):
        """
        Crossover: combine two solutions by taking blocks from both.
        Returns a new solution that is a valid cover (hopefully smaller).
        """
        if not sol1 or not sol2:
            return sol1 if sol1 else sol2
        
        # Union of blocks
        union = list(set(sol1 + sol2))
        
        # Reduce the union
        reduced = reduce_solution_fast(union)
        
        return reduced

    def mutation(sol, rate=0.1):
        """
        Mutation: randomly add or remove blocks, then reduce.
        """
        if not sol:
            return sol
        
        mutated = list(sol)
        n_mut = max(1, int(len(mutated) * rate))
        
        for _ in range(n_mut):
            op = random.choice(['add', 'remove'])
            if op == 'add' and len(mutated) < N_blocks:
                g = random.randint(0, N_blocks - 1)
                if g not in mutated:
                    mutated.append(g)
            elif op == 'remove' and len(mutated) > 1:
                # Only remove if it doesn't break the cover
                g = random.choice(mutated)
                # Check if removable
                removable = True
                # Quick check: count how many blocks cover each t-subset in g
                # This is expensive, so do a quick heuristic
                # For now, just try and validate later
                mutated.remove(g)
        
        # Reduce to ensure validity
        reduced = reduce_solution_fast(mutated)
        
        # Validate
        if is_valid_cover(reduced):
            return reduced
        else:
            return sol  # Revert if invalid

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol = [all_blocks[i] for i in sol_idx]
    best_len = len(best_sol)
    
    # Population for genetic algorithm
    population = [sol_idx]
    
    # Randomized restarts with genetic improvement
    while time.time() < DEADLINE - 10.0:
        # Generate new candidate via greedy restart
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)
        
        # Apply local search
        cand_idx = local_search(cand_idx, max_iter=50)
        
        # If better, update best
        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol = [all_blocks[i] for i in cand_idx]
            population = [cand_idx]  # Reset population with best
        else:
            # Add to population if not duplicate
            if cand_idx not in population:
                population.append(cand_idx)
                # Keep population size manageable
                if len(population) > 10:
                    population = population[:10]
        
        # Genetic operations: crossover and mutation
        if len(population) >= 2 and time.time() < DEADLINE - 15.0:
            # Crossover
            p1, p2 = random.sample(population, 2)
            child = crossover(p1, p2)
            if child and is_valid_cover(child):
                child = local_search(child, max_iter=30)
                if len(child) < best_len:
                    best_len = len(child)
                    best_sol = [all_blocks[i] for i in child]
                    population = [child]
                elif child not in population:
                    population.append(child)
                    if len(population) > 10:
                        population = population[:10]
            
            # Mutation
            parent = random.choice(population)
            child = mutation(parent, rate=0.1)
            if child and is_valid_cover(child):
                child = local_search(child, max_iter=30)
                if len(child) < best_len:
                    best_len = len(child)
                    best_sol = [all_blocks[i] for i in child]
                    population = [child]
                elif child not in population:
                    population.append(child)
                    if len(population) > 10:
                        population = population[:10]
        
        # Final local search on best solution
        if time.time() < DEADLINE - 20.0:
            best_idx = [all_blocks.index(b) for b in best_sol] if best_sol else []
            # Actually, we need to map back to indices
            # Let's track indices separately
            pass
    
    # Final polish: apply local search to the best solution
    if best_sol and time.time() < DEADLINE - 5.0:
        # Convert back to indices
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        best_idx = [block_to_idx[b] for b in best_sol]
        improved = local_search(best_idx, max_iter=200)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol = [all_blocks[i] for i in improved]
    
    return best_sol
