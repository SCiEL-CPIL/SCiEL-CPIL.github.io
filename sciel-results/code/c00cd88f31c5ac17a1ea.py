import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 143.0
    start_time = time.time()

    # Precompute all t-subsets and index
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all blocks and the t-subsets each covers
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # block_covered[b] = list of t-subset indices covered by block b
    block_covered = []
    for block in all_blocks:
        covered = [t_idx[ts] for ts in itertools.combinations(block, t)]
        covered.sort()
        block_covered.append(covered)

    # Reverse index: t-subset -> list of block indices that cover it
    t_to_blocks = [[] for _ in range(n_t)]
    for b_idx, cov in enumerate(block_covered):
        for j in cov:
            t_to_blocks[j].append(b_idx)

    # ---------- Coverage helpers ----------
    def coverage_counts(block_list):
        counts = np.zeros(n_t, dtype=np.int32)
        for b in block_list:
            for j in block_covered[b]:
                counts[j] += 1
        return counts

    def is_valid(counts):
        return np.all(counts >= 1)

    # ---------- Diverse Greedy Set Cover ----------
    def diverse_greedy(rng, time_budget, diversity_weight=1.0):
        """
        Select blocks one at a time. Score = new_cover / (1 + diversity_weight * overlap_with_existing).
        This penalizes blocks that overlap heavily with already-chosen blocks, encouraging diversity.
        """
        uncovered = set(range(n_t))
        chosen = []
        # For speed, we maintain a set of chosen blocks for overlap calculation
        chosen_set = set()
        
        while uncovered and time.time() < time_budget:
            # Pick a random uncovered t-subset to anchor the search for candidates
            # This avoids scanning all blocks every time
            anchor = rng.choice(tuple(uncovered))
            candidates = t_to_blocks[anchor]
            
            best_b = -1
            best_score = -1.0
            
            # Limit candidates to a random sample if too many, to save time
            # But usually anchor has < 1000 candidates for v=24,k=7
            for b in candidates:
                if b in chosen_set:
                    continue
                
                # Calculate new coverage
                new_cov = 0
                overlap = 0
                for j in block_covered[b]:
                    if j in uncovered:
                        new_cov += 1
                    else:
                        overlap += 1
                
                if new_cov == 0:
                    continue
                    
                score = new_cov / (1.0 + diversity_weight * overlap)
                
                if score > best_score:
                    best_score = score
                    best_b = b
                    
                    # Early exit if perfect score (covers all remaining, no overlap)
                    if new_cov == len(uncovered) and overlap == 0:
                        break
            
            if best_b < 0:
                break
                
            chosen.append(best_b)
            chosen_set.add(best_b)
            
            for j in block_covered[best_b]:
                uncovered.discard(j)
                
        return chosen

    # ---------- Reduction: remove redundant blocks ----------
    def reduce_blocks(block_list):
        """
        Greedily remove blocks that are fully redundant.
        We try to remove blocks with the least 'critical' coverage first.
        """
        current = list(block_list)
        if not current:
            return current
        
        # Sort by number of unique contributions? Or just iterate?
        # Let's iterate and check validity.
        # To be faster, we can compute counts once and update.
        counts = coverage_counts(current)
        
        # Indices to try removing
        # We prioritize blocks that cover fewer 'unique' (count==1) items
        unique_counts = []
        for b in current:
            u = 0
            for j in block_covered[b]:
                if counts[j] == 1:
                    u += 1
            unique_counts.append(u)
        
        # Sort indices by unique count (ascending)
        order = sorted(range(len(current)), key=lambda i: unique_counts[i])
        
        for i in order:
            if time.time() > deadline - 10.0:
                break
            if i >= len(current):
                continue
                
            b = current[i]
            
            # Check if removing b keeps validity
            removable = True
            for j in block_covered[b]:
                if counts[j] <= 1:
                    removable = False
                    break
            
            if removable:
                # Remove it
                for j in block_covered[b]:
                    counts[j] -= 1
                current.pop(i)
                # Adjust order? No, we just continue with next index in sorted list,
                # but indices shift. Since we are popping from a list, 
                # and 'order' contains original indices, this is tricky.
                # Simpler: just re-run the loop or use a set of indices to remove.
                # Given the small size (~76 blocks), re-computing counts is cheap.
                # But let's just break and restart the reduction loop for simplicity and correctness.
                return reduce_blocks(current)
        
        return current

    # ---------- Main Search Loop ----------
    best_sol = None
    best_len = float('inf')
    
    # We will run multiple iterations of Greedy + Reduce
    # Vary the diversity weight and seed to explore different landscapes
    
    max_iterations = 50
    iter_time_budget = 2.0  # seconds per greedy run
    
    for i in range(max_iterations):
        if time.time() > deadline - 15.0:
            break
            
        # Vary diversity weight: start low (standard greedy), increase to encourage diversity
        # Or randomize it.
        # Let's use a schedule: 0.5, 1.0, 1.5, 2.0...
        div_w = 0.5 + (i % 10) * 0.2
        seed = i
        
        rng = random.Random(seed)
        
        # Run greedy
        greedy_sol = diverse_greedy(rng, time.time() + iter_time_budget, diversity_weight=div_w)
        
        if not greedy_sol:
            continue
            
        # Check validity
        counts = coverage_counts(greedy_sol)
        if not is_valid(counts):
            # If greedy failed to cover (unlikely but possible with strict diversity),
            # we might need to add blocks. But standard greedy on anchor usually covers.
            # If not, skip or fix.
            # Let's fix by adding blocks for uncovered subsets.
            uncovered = np.where(counts == 0)[0]
            for u in uncovered:
                cands = t_to_blocks[u]
                if cands:
                    # Pick one not in solution
                    for c in cands:
                        if c not in greedy_sol:
                            greedy_sol.append(c)
                            break
            # Re-check
            counts = coverage_counts(greedy_sol)
            if not is_valid(counts):
                continue

        # Reduce
        reduced = reduce_blocks(greedy_sol)
        
        # Final validity check
        counts = coverage_counts(reduced)
        if is_valid(counts):
            if len(reduced) < best_len:
                best_len = len(reduced)
                best_sol = reduced
                
                # If we hit the theoretical lower bound or a known good bound, we can stop early?
                # Lower bound is 69. Best known 76.
                # If we get <= 76, that's good.
                if best_len <= 76:
                    # Continue a bit more to see if we can go lower
                    pass

    # Fallback if no solution found (should not happen)
    if best_sol is None:
        # Simple greedy without diversity
        rng = random.Random(0)
        best_sol = diverse_greedy(rng, time.time() + 10.0, diversity_weight=0.0)
        counts = coverage_counts(best_sol)
        if not is_valid(counts):
            # Emergency fix
            uncovered = np.where(counts == 0)[0]
            for u in uncovered:
                for c in t_to_blocks[u]:
                    if c not in best_sol:
                        best_sol.append(c)
                        break
        best_sol = reduce_blocks(best_sol)

    # Convert to result
    result = [all_blocks[i] for i in best_sol]
    result = list(set(result))

    # Final validation and fix (safety net)
    final_covered = np.zeros(n_t, dtype=bool)
    for b in result:
        for ts in itertools.combinations(b, t):
            final_covered[t_idx[ts]] = True

    if not final_covered.all():
        uncovered = np.where(~final_covered)[0]
        for ts_idx in uncovered:
            for b_idx in t_to_blocks[ts_idx]:
                b = all_blocks[b_idx]
                if b not in result:
                    result.append(b)
                    for ts in itertools.combinations(b, t):
                        final_covered[t_idx[ts]] = True
                    break

    return [list(b) for b in result]
