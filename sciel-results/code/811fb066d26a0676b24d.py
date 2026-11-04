import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute t-subsets per block and reverse mapping
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)

    # block_t_subs[i] = list of t-subset indices covered by block i
    block_t_subs = []
    # reverse: t_sub_to_blocks[j] = list of block indices covering t-subset j
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]

    for i, block in enumerate(all_blocks):
        subs = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_subs.append(subs)
        for j in subs:
            t_sub_to_blocks[j].append(i)

    # Convert to numpy arrays for speed
    block_t_subs_arr = [np.array(s, dtype=np.int32) for s in block_t_subs]
    t_sub_to_blocks_arr = [np.array(s, dtype=np.int32) for s in t_sub_to_blocks]

    def greedy_cover(random_seed=None, deadline_greedy=None):
        """Randomized greedy set cover. Returns list of block indices."""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        uncovered = np.ones(num_t_subsets, dtype=np.int32)
        gains = np.zeros(num_blocks, dtype=np.int32)
        for i in range(num_blocks):
            gains[i] = len(block_t_subs_arr[i])  # all start uncovered
        
        selected = []
        while True:
            if deadline_greedy and time.time() > deadline_greedy:
                break
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            # Random tie-breaking: pick from top few
            top_threshold = max_gain
            candidates = np.where(gains == top_threshold)[0]
            if len(candidates) > 1:
                b_idx = int(np.random.choice(candidates))
            else:
                b_idx = int(candidates[0])
            
            selected.append(b_idx)
            covered = block_t_subs_arr[b_idx]
            for j in covered:
                if uncovered[j] == 1:
                    uncovered[j] = 0
                    for bi in t_sub_to_blocks_arr[j]:
                        gains[bi] -= 1
            
            if np.all(uncovered == 0):
                break
        
        return selected

    def is_valid_solution(selected_blocks):
        """Check if selected blocks cover all t-subsets."""
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            cover_counts[block_t_subs_arr[b_idx]] += 1
        return np.all(cover_counts >= 1)

    def deletion_pass(selected_blocks):
        """Remove redundant blocks. Returns new list."""
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            cover_counts[block_t_subs_arr[b_idx]] += 1
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            for i, b_idx in enumerate(selected_blocks):
                covered = block_t_subs_arr[b_idx]
                if np.all(cover_counts[covered] >= 2):
                    to_remove.append(i)
            if to_remove:
                for i in to_remove:
                    b_idx = selected_blocks[i]
                    cover_counts[block_t_subs_arr[b_idx]] -= 1
                remove_set = set(to_remove)
                selected_blocks = [b for i, b in enumerate(selected_blocks) if i not in remove_set]
                changed = True
        return selected_blocks

    def ils_step(selected_blocks, perturbation_strength, deadline_ils):
        """One ILS iteration: perturb, then local search (deletion + swap)."""
        # Perturbation: remove some random blocks and re-add via greedy
        n_remove = max(1, int(perturbation_strength * len(selected_blocks)))
        n_remove = min(n_remove, len(selected_blocks) - 1)
        
        removed = list(np.random.choice(len(selected_blocks), size=n_remove, replace=False))
        remaining = [b for i, b in enumerate(selected_blocks) if i not in set(removed)]
        
        # Greedily add back until valid
        if not is_valid_solution(remaining):
            # Find uncovered t-subsets
            cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
            for b_idx in remaining:
                cover_counts[block_t_subs_arr[b_idx]] += 1
            uncovered_mask = (cover_counts == 0)
            
            # Greedily add blocks that cover uncovered subsets
            while np.any(uncovered_mask):
                if time.time() > deadline_ils:
                    break
                best_block = -1
                best_gain = 0
                for b_idx in range(num_blocks):
                    if b_idx in set(remaining):
                        continue
                    covered = block_t_subs_arr[b_idx]
                    gain = np.sum(uncovered_mask[covered])
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                if best_block == -1 or best_gain == 0:
                    break
                remaining.append(best_block)
                cover_counts[block_t_subs_arr[best_block]] += 1
                uncovered_mask = (cover_counts == 0)
        
        # Local search: deletion pass
        remaining = deletion_pass(remaining)
        
        # Swap local search: try replacing blocks
        if time.time() < deadline_ils - 2.0:
            cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
            for b_idx in remaining:
                cover_counts[block_t_subs_arr[b_idx]] += 1
            
            improved = True
            while improved and time.time() < deadline_ils - 2.0:
                improved = False
                for i in range(len(remaining)):
                    if time.time() > deadline_ils - 2.0:
                        break
                    b_old = remaining[i]
                    # Try removing b_old and see if we can add a different block
                    cover_counts[block_t_subs_arr[b_old]] -= 1
                    if np.all(cover_counts >= 1):
                        # b_old is redundant, just remove it
                        remaining.pop(i)
                        improved = True
                        # Recompute cover_counts
                        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                        for b_idx in remaining:
                            cover_counts[block_t_subs_arr[b_idx]] += 1
                        break
                    else:
                        # Restore
                        cover_counts[block_t_subs_arr[b_old]] += 1
        
        return remaining

    # Phase 1: Multiple randomized greedy starts
    best_solution = None
    best_size = float('inf')
    
    num_greedy_starts = 5
    greedy_deadline = deadline - 60.0  # Reserve time for ILS
    
    for trial in range(num_greedy_starts):
        if time.time() > greedy_deadline:
            break
        seed = trial * 12345 + 67890
        sol = greedy_cover(random_seed=seed, deadline_greedy=greedy_deadline)
        sol = deletion_pass(sol)
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]
    
    if best_solution is None:
        # Fallback: deterministic greedy
        best_solution = greedy_cover(random_seed=0, deadline_greedy=deadline - 10.0)
        best_solution = deletion_pass(best_solution)
        best_size = len(best_solution)

    # Phase 2: ILS from the best solution
    ils_deadline = deadline - 5.0
    perturbation_strength = 0.1
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < ils_deadline and no_improve_count < max_no_improve:
        # Perturbation
        perturbed = ils_step(best_solution, perturbation_strength, ils_deadline)
        
        if len(perturbed) < best_size:
            best_solution = perturbed[:]
            best_size = len(perturbed)
            no_improve_count = 0
            # Reduce perturbation on improvement
            perturbation_strength = max(0.05, perturbation_strength * 0.9)
        else:
            no_improve_count += 1
            # Increase perturbation when stuck
            perturbation_strength = min(0.3, perturbation_strength * 1.1)
        
        # Time-adaptive: if we're running low on time, do smaller steps
        if time.time() > ils_deadline - 15.0:
            perturbation_strength = 0.05
            max_no_improve = 10

    # Final deletion pass
    best_solution = deletion_pass(best_solution)

    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in best_solution]
    return blocks
