import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Precompute triple frequencies for weighted greedy
    triple_freq = np.zeros(n_triples, dtype=np.float64)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
    triple_weight = 1.0 / np.maximum(triple_freq, 1.0)
    
    def greedy_cover(use_weighted=False, random_tiebreak=False, rng=None):
        """Run greedy set cover with optional weighting and random tie-breaking."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Calculate gains
            if use_weighted:
                gains = np.sum(uncovered[block_triples] * triple_weight[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            if random_tiebreak and rng is not None:
                candidates = np.where(gains == max_gain)[0]
                best_idx = rng.choice(candidates)
            else:
                best_idx = np.argmax(gains)
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected
    
    def peel_redundant(block_indices):
        """Remove redundant blocks from a solution."""
        if not block_indices:
            return list(block_indices)
        
        n_sel = len(block_indices)
        covered_by = np.zeros((n_triples, n_sel), dtype=bool)
        for i, bidx in enumerate(block_indices):
            covered_by[block_triples[bidx], i] = True
        
        keep = []
        for i in range(n_sel):
            my_triples = block_triples[block_indices[i]]
            still_covered = True
            for triple_idx in my_triples:
                if np.sum(covered_by[triple_idx]) <= 1:
                    still_covered = False
                    break
            if still_covered:
                continue
            else:
                keep.append(block_indices[i])
        
        return keep
    
    def simulated_annealing(block_indices, rng, max_iters=2000, T_start=1.0, T_end=0.01, cooling=0.999):
        """Simulated annealing to minimize block count while maintaining validity."""
        if not block_indices:
            return list(block_indices)
        
        # Build coverage count for each triple
        n_sel = len(block_indices)
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        current_blocks = list(block_indices)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        T = T_start
        iter_count = 0
        while time.time() < deadline - 5.0 and iter_count < max_iters and T > T_end:
            iter_count += 1
            T *= cooling
            
            if len(current_blocks) == 0:
                break
            
            # Choose a random move type
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try to remove a redundant block
                i = rng.randrange(len(current_blocks))
                bidx = current_blocks[i]
                my_triples = block_triples[bidx]
                redundant = True
                for triple_idx in my_triples:
                    if coverage_count[triple_idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove this block
                    coverage_count[my_triples] -= 1
                    current_blocks.pop(i)
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                    continue
                else:
                    # Not redundant, skip
                    continue
            
            elif move_type < 0.7:
                # Swap: replace one block with another
                i = rng.randrange(len(current_blocks))
                old_bidx = current_blocks[i]
                old_triples = block_triples[old_bidx]
                
                # Find critical triples (covered only by this block)
                critical_mask = (coverage_count == 1)
                critical_triples = old_triples[critical_mask[old_triples]]
                
                if len(critical_triples) == 0:
                    continue
                
                # Find candidate blocks that cover all critical triples
                best_candidate = -1
                best_gain = -1
                
                # Sample a subset of candidates for speed
                n_candidates = min(100, n_blocks)
                cand_indices = rng.sample(range(n_blocks), n_candidates)
                
                for cand_idx in cand_indices:
                    if cand_idx == old_bidx:
                        continue
                    cand_triples = block_triples[cand_idx]
                    
                    # Check if candidate covers all critical triples
                    covers_all = True
                    for ct in critical_triples:
                        if not np.isin(ct, cand_triples):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Calculate gain: how many new triples does this block cover?
                    new_coverage = 0
                    for ct in cand_triples:
                        if coverage_count[ct] == 0:
                            new_coverage += 1
                    
                    if new_coverage > best_gain:
                        best_gain = new_coverage
                        best_candidate = cand_idx
                
                if best_candidate >= 0:
                    # Perform the swap
                    old_triples_set = set(old_triples.tolist())
                    cand_triples_set = set(block_triples[best_candidate].tolist())
                    
                    # Update coverage counts
                    for tr in old_triples_set:
                        coverage_count[tr] -= 1
                    for tr in cand_triples_set:
                        coverage_count[tr] += 1
                    
                    current_blocks[i] = best_candidate
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
            
            else:
                # Perturbation: remove a random block and repair
                if len(current_blocks) > 1:
                    j = rng.randrange(len(current_blocks))
                    removed_bidx = current_blocks[j]
                    coverage_count[block_triples[removed_bidx]] -= 1
                    current_blocks.pop(j)
                    
                    # Rebuild to cover uncovered triples
                    uncovered_mask = (coverage_count == 0)
                    while np.any(uncovered_mask):
                        if time.time() > deadline - 5.0:
                            break
                        # Find best block to add
                        gains = np.sum(uncovered_mask[block_triples], axis=1)
                        best_add = np.argmax(gains)
                        if gains[best_add] == 0:
                            break
                        coverage_count[block_triples[best_add]] += 1
                        current_blocks.append(best_add)
                        uncovered_mask = (coverage_count == 0)
                    
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
        
        return best_blocks
    
    # Run multiple strategies
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Strategy 1: Deterministic greedy
    sel = greedy_cover(use_weighted=False, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Strategy 2: Weighted greedy
    sel = greedy_cover(use_weighted=True, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Strategy 3+: Random restarts with weighted greedy + simulated annealing
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        sel = greedy_cover(use_weighted=True, random_tiebreak=True, rng=rng)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            
            # Apply simulated annealing
            if time.time() < deadline - 10.0:
                sa_result = simulated_annealing(peeled, rng, max_iters=500)
                sa_size = len(sa_result)
                if sa_size < size:
                    size = sa_size
                    peeled = sa_result
            
            if size < best_size:
                best_size = size
                best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Final polish: try simulated annealing on the best solution found
    if best_solution is not None and time.time() < deadline - 5.0:
        # Convert back to indices
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        current_indices = []
        for b in best_solution:
            t = tuple(b)
            if t in block_to_idx:
                current_indices.append(block_to_idx[t])
        
        if current_indices:
            sa_result = simulated_annealing(current_indices, rng, max_iters=300)
            if len(sa_result) < len(best_solution):
                best_solution = [list(all_blocks[i]) for i in sa_result]
                best_size = len(sa_result)
    
    if best_solution is None:
        best_solution = []
    
    return best_solution
