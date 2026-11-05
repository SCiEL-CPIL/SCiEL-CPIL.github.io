import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin (150s limit)
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Precompute all blocks and their triple memberships
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_triples[i] = list of triple indices covered by block i
    block_triples = []
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            tris.append(triple_to_idx[tri])
        block_triples.append(tris)
    
    # 1. Fast Greedy Initial Solution
    def greedy_initial():
        uncovered = set(range(num_triples))
        selected = []
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            for i, tris in enumerate(block_triples):
                gain = len(set(tris) & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            for tri in block_triples[best_idx]:
                uncovered.discard(tri)
        
        return selected
    
    # 2. Local Search: Remove redundant blocks, then try swaps
    def local_search(sol, time_limit=20):
        # Convert to set for fast lookup
        selected = set(sol)
        
        # Phase 1: Remove redundant blocks (those whose triples are all covered by others)
        changed = True
        while changed:
            changed = False
            # Count how many selected blocks cover each triple
            counts = [0] * num_triples
            for idx in selected:
                for tri in block_triples[idx]:
                    counts[tri] += 1
            
            to_remove = []
            for idx in list(selected):
                redundant = True
                for tri in block_triples[idx]:
                    if counts[tri] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(idx)
            
            if to_remove:
                for idx in to_remove:
                    selected.remove(idx)
                    for tri in block_triples[idx]:
                        counts[tri] -= 1
                changed = True
        
        # Phase 2: Try to swap blocks to reduce size or improve
        # Try removing a block and adding a better one
        for _ in range(3):  # A few iterations
            # Count coverage
            counts = [0] * num_triples
            for idx in selected:
                for tri in block_triples[idx]:
                    counts[tri] += 1
            
            # Find blocks that are "critical" (cover at least one uniquely-covered triple)
            critical = []
            non_critical = []
            for idx in selected:
                is_crit = False
                for tri in block_triples[idx]:
                    if counts[tri] == 1:
                        is_crit = True
                        break
                if is_crit:
                    critical.append(idx)
                else:
                    non_critical.append(idx)
            
            # Try to replace a non-critical block with one that covers more uncovered (if any) 
            # or just keep size same but maybe allow future removals
            # For now, just try removing non-critical and re-covering with one block
            if non_critical and time.time() < time_limit:
                for idx in list(non_critical):
                    # Try removing idx
                    selected.remove(idx)
                    for tri in block_triples[idx]:
                        counts[tri] -= 1
                    
                    # Check if we can cover the now-uncovered triples with one block
                    uncovered = [tri for tri in block_triples[idx] if counts[tri] == 0]
                    if uncovered:
                        # Find a block that covers all uncovered triples
                        found = False
                        for j in range(num_blocks):
                            if j in selected:
                                continue
                            covers_all = True
                            for tri in uncovered:
                                if tri not in block_triples[j]:
                                    covers_all = False
                                    break
                            if covers_all:
                                selected.add(j)
                                for tri in block_triples[j]:
                                    counts[tri] += 1
                                found = True
                                break
                        if not found:
                            # Revert
                            selected.add(idx)
                            for tri in block_triples[idx]:
                                counts[tri] += 1
            
            # Also try random swaps of critical blocks with non-critical or random blocks
            # This is expensive, so limit it
            if time.time() > time_limit - 5:
                break
            
        return list(selected)
    
    # 3. MILP-based improvement
    # This is a hard problem, but we can try to solve a relaxation or a smaller sub-problem.
    # For C(24,7,3), the number of blocks is ~202598. Too large for MILP directly.
    # Instead, we can use the greedy solution and try to find a better set of blocks 
    # that covers all triples using a constraint solver on a reduced set of candidates.
    
    # A better approach: use the greedy solution as a starting point and use 
    # a stochastic local search with simulated annealing or tabu search.
    
    def simulated_annealing(sol, max_iter=5000):
        current = set(sol)
        # Initialize temperature
        T = 50.0
        T_min = 0.1
        cooling = 0.999
        
        # Precompute block_triples as a set for faster intersection
        block_triples_set = [set(tris) for tris in block_triples]
        
        best = list(current)
        best_size = len(current)
        
        # Helper to get uncovered triples
        def get_uncovered(selected):
            covered = set()
            for idx in selected:
                covered.update(block_triples_set[idx])
            return set(range(num_triples)) - covered
        
        current_uncovered = get_uncovered(current)
        current_cost = len(current) + 10 * len(current_uncovered)
        
        for _ in range(max_iter):
            if T < T_min:
                break
            # Random move: add, remove, or swap
            move = random.random()
            if move < 0.3:  # Remove
                if not current:
                    continue
                idx = random.choice(list(current))
                # Tentative removal
                current.discard(idx)
                new_uncovered = get_uncovered(current)
                new_cost = len(current) + 10 * len(new_uncovered)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_uncovered = new_uncovered
                    current_cost = new_cost
                    if len(current) < best_size and len(current_uncovered) == 0:
                        best = list(current)
                        best_size = len(current)
                else:
                    current.add(idx)
            elif move < 0.6:  # Add
                idx = random.randint(0, num_blocks - 1)
                if idx in current:
                    continue
                current.add(idx)
                new_uncovered = get_uncovered(current)
                new_cost = len(current) + 10 * len(new_uncovered)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_uncovered = new_uncovered
                    current_cost = new_cost
                    if len(current) < best_size and len(current_uncovered) == 0:
                        best = list(current)
                        best_size = len(current)
                else:
                    current.discard(idx)
            else:  # Swap
                if not current:
                    continue
                idx_out = random.choice(list(current))
                idx_in = random.randint(0, num_blocks - 1)
                if idx_in in current or idx_in == idx_out:
                    continue
                current.remove(idx_out)
                current.add(idx_in)
                new_uncovered = get_uncovered(current)
                new_cost = len(current) + 10 * len(new_uncovered)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_uncovered = new_uncovered
                    current_cost = new_cost
                    if len(current) < best_size and len(current_uncovered) == 0:
                        best = list(current)
                        best_size = len(current)
                else:
                    current.remove(idx_in)
                    current.add(idx_out)
            
            T *= cooling
        
        return best
    
    # Run greedy
    greedy_sol = greedy_initial()
    greedy_sol = local_search(greedy_sol, time_limit=10)
    
    # Run SA from greedy solution
    sa_sol = simulated_annealing(greedy_sol, max_iter=3000)
    sa_sol = local_search(sa_sol, time_limit=5)
    
    # Try a few more random restarts if time permits
    best_sol = min([greedy_sol, sa_sol], key=len)
    
    if time.time() < deadline - 30:
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            # Randomized greedy: randomly perturb the order of blocks
            random.shuffle(all_blocks)  # This is expensive, so avoid
            # Instead, just run SA with different parameters
            sa_sol2 = simulated_annealing(best_sol, max_iter=2000)
            sa_sol2 = local_search(sa_sol2, time_limit=3)
            if len(sa_sol2) < len(best_sol):
                best_sol = sa_sol2
    
    # Final local search
    best_sol = local_search(best_sol, time_limit=10)
    
    # Convert to final blocks
    final_blocks = [all_blocks[idx] for idx in best_sol]
    
    return final_blocks
