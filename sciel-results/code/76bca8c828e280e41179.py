import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix: Cover[i, j] = 1 if t-subset j is in block i
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1

    # 2. Greedy Initialization
    uncovered = np.ones(n_tsubs, dtype=bool)
    selected_indices = []
    
    while np.sum(uncovered) > 0:
        if time.time() > deadline - 30.0:
            break
        gains = Cover @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
    
    current_blocks = list(selected_indices)
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in current_blocks:
        cover_counts += Cover[idx]
    
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # 3. Helper functions
    def reduce_redundant(blocks, counts):
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for idx in blocks:
                if np.any((Cover[idx] == 1) & (counts == 1)):
                    new_blocks.append(idx)
                else:
                    counts -= Cover[idx]
                    changed = True
            blocks = new_blocks
        return blocks

    def repair(blocks, counts):
        while np.any(counts == 0):
            if time.time() > deadline - 5.0:
                break
            uncovered_mask = (counts == 0)
            gains = Cover @ uncovered_mask
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            blocks.append(best_idx)
            counts += Cover[best_idx]
        return blocks

    def entropy_optimize(blocks, counts, budget=20.0):
        """Maximum entropy optimization: learn soft block weights via SGD."""
        if time.time() > deadline - 10.0:
            return blocks, counts
        
        block_set = set(blocks)
        # Only consider blocks in current solution and a few candidates
        # Use all blocks for simplicity but weight by relevance
        n = len(blocks)
        if n < 2:
            return blocks, counts
        
        # Initialize probabilities: high for blocks that cover unique subsets
        # Low for redundant blocks
        probs = np.ones(n, dtype=np.float64)
        for i, idx in enumerate(blocks):
            unique_covered = np.sum((Cover[idx] == 1) & (counts == 1))
            total_covered = np.sum(Cover[idx])
            if total_covered > 0:
                probs[i] = 0.5 + 0.5 * (unique_covered / total_covered)
        
        # Normalize to sum to 1
        probs = probs / probs.sum()
        
        lr = 0.01
        entropy_reg = 0.1
        coverage_weight = 1.0
        n_iter = min(500, int(budget * 20))  # ~20 steps per second
        
        for it in range(n_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Compute expected coverage: E[counts_j] = sum_i p_i * Cover[i,j]
            expected_counts = probs @ Cover  # shape (n_tsubs,)
            
            # Objective: maximize expected coverage of currently uncovered subsets
            # while maximizing entropy (spread out probability)
            # Penalty: penalize blocks that don't add unique coverage
            
            # Gradient of expected coverage w.r.t. log p_i
            # d/dlogp_i [sum_j max(0, counts_j - expected_j)] 
            # Simplified: focus on uncovered subsets
            uncovered_mask = (counts == 0)
            if np.any(uncovered_mask):
                # For each block i, gradient contribution = Cover[i, uncovered]
                grad_coverage = Cover[:, uncovered_mask].sum(axis=1) if np.any(uncovered_mask) else np.zeros(n)
            else:
                grad_coverage = np.zeros(n)
            
            # Entropy gradient: d/dlogp_i [H(p)] = 1 - log(p_i) - sum_j p_j log(p_j)
            log_probs = np.log(probs + 1e-10)
            H = -np.sum(probs * log_probs)
            grad_entropy = 1.0 - log_probs - H
            
            # Combined gradient
            grad = coverage_weight * grad_coverage + entropy_reg * grad_entropy
            
            # SGD update
            log_probs += lr * grad
            # Clip to prevent numerical issues
            log_probs = np.clip(log_probs, -20, 20)
            probs = np.exp(log_probs)
            probs = probs / probs.sum()
            
            # Periodically try to extract a discrete solution
            if it % 50 == 0:
                # Sort blocks by probability, take top ones
                sorted_idx = np.argsort(probs)[::-1]
                for trial_top in [n, max(1, n-1), max(1, n-2), max(1, n-3)]:
                    candidate = [blocks[i] for i in sorted_idx[:trial_top]]
                    cand_counts = np.zeros(n_tsubs, dtype=np.int32)
                    for idx in candidate:
                        cand_counts += Cover[idx]
                    
                    if np.all(cand_counts > 0):
                        # Valid! Try to reduce
                        cand_counts_copy = cand_counts.copy()
                        reduced = reduce_redundant(candidate, cand_counts_copy)
                        if len(reduced) < best_size:
                            return reduced, cand_counts_copy
                        elif len(reduced) < len(blocks):
                            return reduced, cand_counts_copy
        
        return blocks, counts

    # 4. Main Optimization Loop
    temperature = 10.0
    cooling = 0.95
    min_temp = 0.01
    
    # Initial reduction
    current_blocks = reduce_redundant(current_blocks, cover_counts)
    if len(current_blocks) < best_size:
        best_blocks = list(current_blocks)
        best_size = len(best_blocks)
        
    random.seed(42)
    
    use_entropy = True
    entropy_calls = 0
    
    while time.time() < deadline - 5.0:
        # 1. Reduce redundant blocks
        current_blocks = reduce_redundant(current_blocks, cover_counts)
        
        # Update best
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(best_blocks)
            
        # 2. Try entropy optimization periodically
        if use_entropy and entropy_calls < 5 and len(current_blocks) > 10:
            entropy_calls += 1
            new_blocks, new_counts = entropy_optimize(current_blocks, cover_counts, budget=15.0)
            if len(new_blocks) < len(current_blocks):
                current_blocks = new_blocks
                cover_counts = new_counts
                if len(current_blocks) < best_size:
                    best_blocks = list(current_blocks)
                    best_size = len(best_blocks)
                continue
        
        # 3. Perturb: Remove 1 to 3 random blocks
        num_removals = random.randint(1, 3)
        if len(current_blocks) <= num_removals:
            break
            
        removals = random.sample(current_blocks, num_removals)
        
        for idx in removals:
            cover_counts -= Cover[idx]
        current_blocks = [b for b in current_blocks if b not in removals]
        
        # 4. Repair
        current_blocks = repair(current_blocks, cover_counts)
        
        # 5. Acceptance
        new_size = len(current_blocks)
        delta = new_size - best_size
        
        if delta < 0:
            best_blocks = list(current_blocks)
            best_size = new_size
        elif temperature > min_temp:
            prob = np.exp(-delta / temperature)
            if random.random() < prob:
                pass
            else:
                current_blocks = list(best_blocks)
                cover_counts = np.zeros(n_tsubs, dtype=np.int32)
                for idx in current_blocks:
                    cover_counts += Cover[idx]
                
        temperature *= cooling

    # Final reduction
    final_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in best_blocks:
        final_counts += Cover[idx]
        
    final_blocks = reduce_redundant(best_blocks, final_counts)
    
    if np.any(final_counts == 0):
        final_blocks = best_blocks

    return [all_kblocks[i] for i in final_blocks]

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
