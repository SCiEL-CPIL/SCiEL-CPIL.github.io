import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    # Shape: (n_blocks, C(k, t))
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Compute how many blocks cover each triple (for weighted greedy)
    triple_counts = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_counts[block_triples[i]] += 1
    # Weight: rare triples get higher weight
    triple_weights = 1.0 / (triple_counts + 1).astype(np.float64)

    # Precompute triple_to_blocks for local search
    # triple_to_blocks[i] is a list of block indices covering triple i
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for tri_idx in block_triples[i]:
            triple_to_blocks[tri_idx].append(i)

    def greedy_cover(random_tiebreak=False):
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Weighted gain: sum of weights of uncovered triples in block
            gains = np.sum(uncovered[block_triples].astype(np.float64) * triple_weights[block_triples], axis=1)
            
            if random_tiebreak:
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
        
        return selected
    
    def remove_redundant(block_indices):
        """Remove blocks that are not needed to cover all triples."""
        if len(block_indices) == 0:
            return block_indices
        
        # For each block, check if removing it leaves all triples covered
        # Use a coverage count approach
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(result):
                if time.time() > deadline:
                    return result
                bi = result[i]
                # Check if any triple in this block would become uncovered
                triples_in_block = block_triples[bi]
                if np.any(coverage[triples_in_block] <= 1):
                    i += 1
                else:
                    # Safe to remove
                    coverage[triples_in_block] -= 1
                    result.pop(i)
                    changed = 1
                    # Do not increment i, check same position again
        
        return result

    def local_search(block_indices, max_iterations=100):
        """Try to replace blocks with alternatives that maintain coverage."""
        if len(block_indices) == 0:
            return block_indices
        
        # Build coverage count
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        current = list(block_indices)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                # Try to find a replacement block that covers all critical triples
                # Critical triples are those covered only by this block
                critical = block_triples[bi]
                critical_mask = coverage[critical] <= 1
                if not np.any(critical_mask):
                    # This block is redundant, remove it
                    coverage[critical] -= 1
                    current.pop(i)
                    improved = True
                    break # restart scan
                
                # Find blocks that cover all critical triples
                critical_triples = critical[critical_mask]
                
                # Intersect lists of blocks covering each critical triple
                # Start with the list for the first critical triple
                candidates = set(triple_to_blocks[critical_triples[0]])
                for tri in critical_triples[1:]:
                    candidates &= set(triple_to_blocks[tri])
                    if not candidates:
                        break
                
                # Remove all blocks currently in the set from candidates
                # We are replacing bi, so the new block must not be in current
                candidates -= set(current)
                
                if not candidates:
                    continue
                
                # Pick a candidate
                # Heuristic: Pick one randomly from the valid candidates
                candidate_list = list(candidates)
                bj = candidate_list[np.random.randint(len(candidate_list))]
                
                # Apply swap
                coverage[critical] -= 1
                coverage[block_triples[bj]] += 1
                current[i] = bj
                improved = True
                break # restart scan to re-evaluate criticals
        
            if not improved:
                break
                
        return current

    def simulated_annealing(block_indices, max_time=30.0):
        """Simulated annealing to minimize block count.
        
        Moves:
        1. Try to remove a random block and repair by adding one block that covers all newly-uncovered triples.
           If we can do this without increasing the total count, accept.
        2. Try to swap two blocks where the net effect reduces or maintains count.
        3. Random perturbation: remove a block, then greedily repair (may increase count temporarily).
        """
        if len(block_indices) == 0:
            return block_indices
        
        current = list(block_indices)
        best = list(current)
        best_size = len(current)
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in current:
            coverage[block_triples[bi]] += 1
        
        # Temperature schedule
        T = 1.0
        cooling = 0.995
        sa_start = time.time()
        
        while time.time() - sa_start < max_time:
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Move 1: Try to remove a block and repair with one block
                if len(current) <= 1:
                    T *= cooling
                    continue
                
                i = random.randint(0, len(current) - 1)
                bi = current[i]
                critical_triples = block_triples[bi][coverage[block_triples[bi]] <= 1]
                
                if len(critical_triples) == 0:
                    # Block is redundant, remove it
                    coverage[block_triples[bi]] -= 1
                    current.pop(i)
                    if len(current) < best_size:
                        best_size = len(current)
                        best = list(current)
                    T *= cooling
                    continue
                
                # Find a single block that covers all critical_triples and is not in current
                candidates = set(triple_to_blocks[critical_triples[0]])
                for tri in critical_triples[1:]:
                    candidates &= set(triple_to_blocks[tri])
                    if not candidates:
                        break
                
                candidates -= set(current)
                
                if candidates:
                    bj = random.choice(list(candidates))
                    # Check: after removing bi and adding bj, is everything covered?
                    # We need to verify no other triple becomes uncovered
                    # Temporarily remove bi
                    coverage[block_triples[bi]] -= 1
                    # Check if all triples still covered
                    if np.any(coverage < 1):
                        # Revert
                        coverage[block_triples[bi]] += 1
                        T *= cooling
                        continue
                    # Add bj
                    coverage[block_triples[bj]] += 1
                    current[i] = bj
                    # Size unchanged, but structure changed - keep as perturbation
                    T *= cooling
                else:
                    T *= cooling
                    
            elif move_type < 0.8:
                # Move 2: Swap two blocks (one in, one out) to try to reduce count
                # Actually, let's try: remove a block, then see if we can cover its unique triples
                # with existing blocks (shouldn't be possible by definition) or with one new block
                # This is similar to Move 1 but let's try a different angle:
                # Try removing two blocks and adding one block that covers all their unique triples
                if len(current) <= 2:
                    T *= cooling
                    continue
                
                i = random.randint(0, len(current) - 1)
                j = random.randint(0, len(current) - 1)
                if i == j:
                    T *= cooling
                    continue
                
                bi1 = current[i]
                bi2 = current[j]
                
                # Temporarily remove both
                coverage[block_triples[bi1]] -= 1
                coverage[block_triples[bi2]] -= 1
                
                # Find all uncovered triples
                uncovered_mask = coverage < 1
                if not np.any(uncovered_mask):
                    # Both were redundant! Remove both
                    current.pop(max(i, j))
                    current.pop(min(i, j))
                    if len(current) < best_size:
                        best_size = len(current)
                        best = list(current)
                    T *= cooling
                    continue
                
                uncovered_triples = np.where(uncovered_mask)[0]
                
                # Find a single block covering all uncovered triples
                if len(uncovered_triples) <= 5:  # Limit for efficiency
                    candidates = set(triple_to_blocks[uncovered_triples[0]])
                    for tri in uncovered_triples[1:]:
                        candidates &= set(triple_to_blocks[tri])
                        if not candidates:
                            break
                    candidates -= set(current)
                    
                    if candidates:
                        bj = random.choice(list(candidates))
                        coverage[block_triples[bj]] += 1
                        # Remove both old blocks, add one new
                        current.pop(max(i, j))
                        current.pop(min(i, j))
                        current.append(bj)
                        if len(current) < best_size:
                            best_size = len(current)
                            best = list(current)
                        T *= cooling
                        continue
                
                # Revert
                coverage[block_triples[bi1]] += 1
                coverage[block_triples[bi2]] += 1
                T *= cooling
                
            else:
                # Move 3: Perturbation - remove a random block, then greedily add back what's needed
                if len(current) <= 1:
                    T *= cooling
                    continue
                
                i = random.randint(0, len(current) - 1)
                bi = current[i]
                coverage[block_triples[bi]] -= 1
                current.pop(i)
                
                # Greedily repair
                repair_added = 0
                while np.any(coverage < 1):
                    if time.time() > deadline:
                        break
                    uncovered_mask = coverage < 1
                    # Find best block to add
                    gains = np.sum(uncovered_mask[block_triples].astype(np.float64) * triple_weights[block_triples], axis=1)
                    # Exclude blocks already in current
                    for ci in current:
                        gains[ci] = -1
                    bj = np.argmax(gains)
                    if gains[bj] <= 0:
                        break
                    coverage[block_triples[bj]] += 1
                    current.append(bj)
                    repair_added += 1
                
                # If we added fewer blocks than we removed (impossible since we removed 1)
                # or same number, accept with some probability
                if repair_added <= 1:
                    if len(current) < best_size:
                        best_size = len(current)
                        best = list(current)
                    T *= cooling
                else:
                    # Accept with probability exp(-delta/T) where delta = repair_added - 1
                    delta = repair_added - 1
                    if random.random() < np.exp(-delta / max(T, 0.01)):
                        if len(current) < best_size:
                            best_size = len(current)
                            best = list(current)
                    else:
                        # Revert: this is complex, so just rebuild from best
                        current = list(best)
                        coverage = np.zeros(n_triples, dtype=np.int32)
                        for bi in current:
                            coverage[block_triples[bi]] += 1
                    T *= cooling
            
            T *= cooling
        
        return best

    # First greedy pass
    selected = greedy_cover(random_tiebreak=False)
    selected = remove_redundant(selected)
    selected = local_search(selected, max_iterations=50)
    
    best_solution = [list(blocks[i]) for i in selected]
    best_size = len(best_solution)
    
    # Simulated annealing on the initial solution
    sa_time = min(40.0, max(10.0, deadline - time.time() - 15.0))
    sa_result = simulated_annealing(selected, max_time=sa_time)
    if len(sa_result) < best_size:
        best_size = len(sa_result)
        best_solution = [list(blocks[i]) for i in sa_result]
    
    # Multiple random restarts with SA
    remaining = deadline - time.time() - 5.0
    num_restarts = max(1, int(remaining / 8.0))
    for _ in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        selected = greedy_cover(random_tiebreak=True)
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iterations=20)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]
        
        # SA on this solution
        sa_time = min(5.0, max(2.0, deadline - time.time() - 5.0))
        if sa_time > 1.0:
            sa_result = simulated_annealing(selected, max_time=sa_time)
            if len(sa_result) < best_size:
                best_size = len(sa_result)
                best_solution = [list(blocks[i]) for i in sa_result]
    
    return best_solution
