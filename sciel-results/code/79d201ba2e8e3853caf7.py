"""Covering design C(24,7,3) - MILP-based iterative compression."""
import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    deadline = time.time() + 138.0
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute coverage for a block: list of t-subset indices covered
    def get_block_covered_indices(block):
        return [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]

    # Greedy construction to get an initial solution and candidate pool
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered and time.time() < deadline - 10.0:
            # Pick a random uncovered t-subset to start a block
            start_idx = rng.choice(list(uncovered))
            block = list(all_t_subsets[start_idx])
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                candidates = [x for x in range(v) if x not in block]
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(tuple(block))
            
            for ts in itertools.combinations(block, t):
                uncovered.discard(t_subset_to_idx[ts])
        
        return blocks

    # MILP solver to minimize block count given a candidate pool
    def solve_milp(candidate_blocks, time_limit=10):
        n_blocks = len(candidate_blocks)
        
        # Coverage matrix: A[i, j] = 1 if block j covers t-subset i
        # To save memory/time, we can build it sparsely or just dense since n_t=2024, n_blocks ~ 80
        A = np.zeros((n_t, n_blocks), dtype=np.int8)
        for j, block in enumerate(candidate_blocks):
            covered = get_block_covered_indices(block)
            for idx in covered:
                A[idx, j] = 1
        
        # Variables: x_j = 1 if block j is selected
        # Minimize sum(x_j)
        c = np.ones(n_blocks)
        
        # Constraints: A @ x >= 1
        lb = np.ones(n_t)
        ub = np.full(n_t, np.inf)
        
        # Bounds for x: 0 <= x <= 1
        x_bounds = Bounds(lb=np.zeros(n_blocks), ub=np.ones(n_blocks))
        
        # Integrality: all x must be integer (0 or 1)
        integrality = np.ones(n_blocks)
        
        constraints = LinearConstraint(A, lb=lb, ub=ub)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=x_bounds,
                       options={'time_limit': time_limit, 'presolve': 'default'})
            
            if res.status == 0:  # Optimal
                selected = [j for j in range(n_blocks) if res.x[j] > 0.5]
                return [list(candidate_blocks[j]) for j in selected]
            elif res.status == 1:  # Iteration limit
                # Extract best incumbent
                if res.x is not None:
                    selected = [j for j in range(n_blocks) if res.x[j] > 0.5]
                    return [list(candidate_blocks[j]) for j in selected]
            elif res.status == 2:  # Infeasible (should not happen)
                return None
            else:
                return None
        except Exception:
            return None

    # Iterative compression
    def iterative_compression(initial_blocks, seed, max_iters=5):
        rng = random.Random(seed)
        current_blocks = [tuple(b) for b in initial_blocks]
        
        for it in range(max_iters):
            if time.time() > deadline - 15.0:
                break
                
            # Generate new candidates
            new_candidates = set()
            
            # 1. Current blocks
            for b in current_blocks:
                new_candidates.add(b)
            
            # 2. Blocks covering random uncovered triples (relative to current solution)
            # First, determine what is covered by current solution
            covered_by_current = np.zeros(n_t, dtype=bool)
            for b in current_blocks:
                for idx in get_block_covered_indices(b):
                    covered_by_current[idx] = True
            
            uncovered_indices = np.where(~covered_by_current)[0]
            if len(uncovered_indices) == 0:
                break # Already perfect? (Unlikely for covering number, but good check)
                
            # Add blocks that cover some uncovered triples
            # Sample some uncovered triples and try to find blocks in a larger pool?
            # Since we don't have a pre-computed huge pool, we generate random blocks that cover specific triples.
            
            # For each of a few random uncovered triples, generate a few random blocks containing them
            num_new_to_gen = 50
            for _ in range(num_new_to_gen):
                if time.time() > deadline - 10.0:
                    break
                if not uncovered_indices:
                    break
                # Pick a random uncovered triple
                target_idx = rng.choice(uncovered_indices)
                target_triple = all_t_subsets[target_idx]
                
                # Generate a random block containing this triple
                remaining_points = [x for x in range(v) if x not in target_triple]
                rng.shuffle(remaining_points)
                new_block = list(target_triple) + remaining_points[:k - t]
                new_block.sort()
                new_candidates.add(tuple(new_block))
            
            # 3. Perturb current blocks (swap one point)
            for b in current_blocks:
                if time.time() > deadline - 10.0:
                    break
                for _ in range(2): # 2 perturbations per block
                    b_list = list(b)
                    idx_to_remove = rng.randint(0, k - 1)
                    removed_point = b_list[idx_to_remove]
                    b_list.pop(idx_to_remove)
                    
                    candidates_add = [x for x in range(v) if x not in b_list]
                    if candidates_add:
                        add_point = rng.choice(candidates_add)
                        b_list.append(add_point)
                        b_list.sort()
                        new_candidates.add(tuple(b_list))
            
            candidate_list = list(new_candidates)
            
            if not candidate_list:
                break
                
            # Solve MILP
            # Limit pool size to avoid memory issues, though 2024 x 200 is small
            if len(candidate_list) > 200:
                # Prioritize current blocks and random new ones
                current_set = set(current_blocks)
                prioritized = [b for b in candidate_list if b in current_set]
                others = [b for b in candidate_list if b not in current_set]
                rng.shuffle(others)
                candidate_list = prioritized + others[:200 - len(prioritized)]
            
            solution = solve_milp(candidate_list, time_limit=8)
            
            if solution is not None and len(solution) < len(current_blocks):
                current_blocks = [tuple(b) for b in solution]
            else:
                # If no improvement, we might break or continue with different perturbations
                # To save time, break if we didn't improve in a few iterations?
                # Let's just continue but reduce time limit for subsequent MILPs if no improvement
                pass

        return [list(b) for b in current_blocks]

    # Main Logic
    best_blocks = None
    best_count = float('inf')
    
    # 1. Generate initial greedy solutions
    num_greedy = 5
    for i in range(num_greedy):
        if time.time() > deadline - 30.0:
            break
        blocks = greedy_cover(i)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    # 2. Apply iterative compression to the best greedy solution (and maybe a couple of others)
    # We have ~100s left. Each compression iter takes ~10-20s (MILP).
    # Let's try 3-5 iterations on the best solution.
    
    if best_blocks is not None:
        # Try compression on the best one
        compressed = iterative_compression(best_blocks, seed=42, max_iters=5)
        if compressed and len(compressed) < best_count:
            best_count = len(compressed)
            best_blocks = compressed
            
        # If time permits, try compression on the second best greedy solution
        if time.time() < deadline - 40.0:
            # Find second best
            # Re-run greedy to get distinct ones? Or just use seed 1 and 2
            for seed in [1, 2]:
                if time.time() > deadline - 30.0:
                    break
                blocks = greedy_cover(seed)
                if len(blocks) < best_count:
                    compressed = iterative_compression(blocks, seed=seed + 100, max_iters=3)
                    if compressed and len(compressed) < best_count:
                        best_count = len(compressed)
                        best_blocks = compressed

    # Fallback if MILP failed or didn't find anything better
    if best_blocks is None:
        # Simple greedy fallback
        blocks = greedy_cover(0)
        best_blocks = blocks

    # Final verification and cleanup
    # Ensure distinct blocks
    unique_blocks = []
    seen = set()
    for b in best_blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            unique_blocks.append(b)
            
    # Verify validity
    covered = set()
    for b in unique_blocks:
        for ts in itertools.combinations(b, t):
            covered.add(t_subset_to_idx[ts])
            
    if covered != set(range(n_t)):
        # Invalid, fallback to raw greedy
        blocks = greedy_cover(0)
        # Verify again
        covered = set()
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered.add(t_subset_to_idx[ts])
        if covered == set(range(n_t)):
            return blocks
        else:
            # Should not happen, but return something
            return blocks

    return unique_blocks
