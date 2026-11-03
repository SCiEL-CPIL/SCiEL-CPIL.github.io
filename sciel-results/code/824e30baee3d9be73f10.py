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
    
    # Precompute: for each triple, which blocks cover it
    # triple_blocks[j] is a list of block indices covering triple j
    triple_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for tr_idx in block_triples[i]:
            triple_blocks[tr_idx].append(i)
    
    def greedy_cover(use_weighted=False, random_tiebreak=False, rng=None, triple_based=False):
        """Run greedy set cover with optional weighting and random tie-breaking."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if triple_based:
                # Triple-based: pick an uncovered triple, then best block covering it
                uncov_indices = np.where(uncovered)[0]
                if len(uncov_indices) == 0:
                    break
                if random_tiebreak and rng is not None:
                    triple_idx = rng.choice(uncov_indices)
                else:
                    # Pick the rarest uncovered triple (most constrained)
                    freqs = triple_freq[uncov_indices]
                    min_freq = np.min(freqs)
                    candidates = uncov_indices[freqs == min_freq]
                    if random_tiebreak and rng is not None:
                        triple_idx = rng.choice(candidates)
                    else:
                        triple_idx = candidates[0]
                
                # Find best block covering this triple
                cand_blocks = triple_blocks[triple_idx]
                if not cand_blocks:
                    break
                    
                if use_weighted:
                    gains = np.array([np.sum(uncovered[block_triples[b]] * triple_weight[block_triples[b]]) for b in cand_blocks])
                else:
                    gains = np.array([np.sum(uncovered[block_triples[b]]) for b in cand_blocks])
                
                if random_tiebreak and rng is not None:
                    max_gain = np.max(gains)
                    best_cands = np.where(gains == max_gain)[0]
                    best_idx = cand_blocks[rng.choice(best_cands)]
                else:
                    best_idx = cand_blocks[np.argmax(gains)]
            else:
                # Block-based greedy
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
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        keep = []
        for i, bidx in enumerate(block_indices):
            my_triples = block_triples[bidx]
            redundant = True
            for triple_idx in my_triples:
                if coverage_count[triple_idx] <= 1:
                    redundant = False
                    break
            if not redundant:
                keep.append(bidx)
                # Don't update coverage_count here since we're just checking
        
        # Actually need to be more careful - do it properly
        keep = []
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        for i, bidx in enumerate(block_indices):
            my_triples = block_triples[bidx]
            redundant = True
            for triple_idx in my_triples:
                if coverage_count[triple_idx] <= 1:
                    redundant = False
                    break
            if not redundant:
                keep.append(bidx)
        
        return keep
    
    def local_search(block_indices, rng, max_iters=3000, time_budget=30.0):
        """Aggressive local search to minimize block count."""
        if not block_indices:
            return list(block_indices)
        
        ls_start = time.time()
        n_sel = len(block_indices)
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        current_blocks = list(block_indices)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        iter_count = 0
        no_improve_count = 0
        while time.time() - ls_start < time_budget and iter_count < max_iters:
            iter_count += 1
            
            if len(current_blocks) == 0:
                break
            
            improved = False
            move_type = rng.random()
            
            if move_type < 0.4:
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
                    coverage_count[my_triples] -= 1
                    current_blocks.pop(i)
                    improved = True
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
            
            elif move_type < 0.8:
                # Swap: replace one block with another that covers its critical triples
                i = rng.randrange(len(current_blocks))
                old_bidx = current_blocks[i]
                old_triples = block_triples[old_bidx]
                
                critical_mask = (coverage_count == 1)
                critical_triples = old_triples[critical_mask[old_triples]]
                
                if len(critical_triples) == 0:
                    # Can remove this block
                    coverage_count[old_triples] -= 1
                    current_blocks.pop(i)
                    improved = True
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                    continue
                
                # Find candidate blocks that cover all critical triples
                best_candidate = -1
                best_score = -1
                
                # Use the triple_blocks index to find candidates
                # All candidates must cover all critical triples
                if len(critical_triples) > 0:
                    # Start with blocks covering the first critical triple
                    cand_set = set(triple_blocks[critical_triples[0]])
                    for ct in critical_triples[1:]:
                        cand_set &= set(triple_blocks[ct])
                    
                    cand_list = list(cand_set)
                    if not cand_list:
                        continue
                    
                    # Sample if too many
                    if len(cand_list) > 200:
                        cand_list = rng.sample(cand_list, 200)
                    
                    for cand_idx in cand_list:
                        if cand_idx == old_bidx:
                            continue
                        cand_triples = block_triples[cand_idx]
                        
                        # Score: minimize overlap with existing blocks
                        overlap = 0
                        for ct in cand_triples:
                            if coverage_count[ct] > 0:
                                overlap += 1
                        
                        score = -overlap  # We want to minimize overlap
                        if score > best_score:
                            best_score = score
                            best_candidate = cand_idx
                
                if best_candidate >= 0:
                    # Perform the swap
                    old_triples_list = block_triples[old_bidx].tolist()
                    cand_triples_list = block_triples[best_candidate].tolist()
                    
                    for tr in old_triples_list:
                        coverage_count[tr] -= 1
                    for tr in cand_triples_list:
                        coverage_count[tr] += 1
                    
                    current_blocks[i] = best_candidate
                    improved = True
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
                        if time.time() - ls_start > time_budget:
                            break
                        # Find best block to add
                        gains = np.sum(uncovered_mask[block_triples], axis=1)
                        best_add = np.argmax(gains)
                        if gains[best_add] == 0:
                            break
                        coverage_count[block_triples[best_add]] += 1
                        current_blocks.append(best_add)
                        uncovered_mask = (coverage_count == 0)
                    
                    improved = True
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 100:
                    break
        
        return best_blocks

    def simulated_annealing(block_indices, rng, time_budget=20.0):
        """Simulated annealing to escape local optima."""
        if not block_indices:
            return list(block_indices)
        
        sa_start = time.time()
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        current_blocks = list(block_indices)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        T = 1.0
        cooling = 0.995
        T_min = 0.01
        
        while time.time() - sa_start < time_budget and T > T_min:
            T *= cooling
            
            if len(current_blocks) == 0:
                break
            
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
                    coverage_count[my_triples] -= 1
                    current_blocks.pop(i)
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
            
            elif move_type < 0.6:
                # Swap: replace one block with another that covers its critical triples
                i = rng.randrange(len(current_blocks))
                old_bidx = current_blocks[i]
                old_triples = block_triples[old_bidx]
                
                critical_mask = (coverage_count == 1)
                critical_triples = old_triples[critical_mask[old_triples]]
                
                if len(critical_triples) == 0:
                    # Can remove this block
                    coverage_count[old_triples] -= 1
                    current_blocks.pop(i)
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                    continue
                
                # Find candidate blocks that cover all critical triples
                best_candidate = -1
                best_score = -1
                
                if len(critical_triples) > 0:
                    cand_set = set(triple_blocks[critical_triples[0]])
                    for ct in critical_triples[1:]:
                        cand_set &= set(triple_blocks[ct])
                    
                    cand_list = list(cand_set)
                    if not cand_list:
                        continue
                    
                    if len(cand_list) > 200:
                        cand_list = rng.sample(cand_list, 200)
                    
                    for cand_idx in cand_list:
                        if cand_idx == old_bidx:
                            continue
                        cand_triples = block_triples[cand_idx]
                        
                        overlap = 0
                        for ct in cand_triples:
                            if coverage_count[ct] > 0:
                                overlap += 1
                        
                        score = -overlap
                        if score > best_score:
                            best_score = score
                            best_candidate = cand_idx
                
                if best_candidate >= 0:
                    old_triples_list = block_triples[old_bidx].tolist()
                    cand_triples_list = block_triples[best_candidate].tolist()
                    
                    for tr in old_triples_list:
                        coverage_count[tr] -= 1
                    for tr in cand_triples_list:
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
                    
                    uncovered_mask = (coverage_count == 0)
                    while np.any(uncovered_mask):
                        if time.time() - sa_start > time_budget:
                            break
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
                    else:
                        # Accept worse solution with probability exp(-delta/T)
                        delta = len(current_blocks) - best_size
                        if delta > 0:
                            accept_prob = np.exp(-delta / max(T, 1e-6))
                            if rng.random() > accept_prob:
                                # Revert: add back the removed block
                                coverage_count[block_triples[removed_bidx]] += 1
                                current_blocks.append(removed_bidx)
        
        return best_blocks
    
    # Run multiple strategies
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Strategy 1: Deterministic block-based greedy
    sel = greedy_cover(use_weighted=False, random_tiebreak=False, triple_based=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Strategy 2: Weighted block-based greedy
    sel = greedy_cover(use_weighted=True, random_tiebreak=False, triple_based=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Strategy 3: Triple-based greedy (deterministic)
    sel = greedy_cover(use_weighted=False, random_tiebreak=False, triple_based=True)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Strategy 4: Weighted triple-based greedy
    sel = greedy_cover(use_weighted=True, random_tiebreak=False, triple_based=True)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Strategy 5+: Random restarts with local search and simulated annealing
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Alternate between triple-based and block-based
        if r % 2 == 0:
            sel = greedy_cover(use_weighted=True, random_tiebreak=True, rng=rng, triple_based=True)
        else:
            sel = greedy_cover(use_weighted=True, random_tiebreak=True, rng=rng, triple_based=False)
        
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            
            # Apply local search
            if time.time() < deadline - 15.0:
                ls_result = local_search(peeled, rng, max_iters=2000, time_budget=8.0)
                ls_size = len(ls_result)
                if ls_size < size:
                    size = ls_size
                    peeled = ls_result
            
            # Apply simulated annealing for additional improvement
            if time.time() < deadline - 10.0:
                sa_result = simulated_annealing(peeled, rng, time_budget=5.0)
                sa_size = len(sa_result)
                if sa_size < size:
                    size = sa_size
                    peeled = sa_result
            
            if size < best_size:
                best_size = size
                best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Final polish: local search and simulated annealing on the best solution found
    if best_solution is not None and time.time() < deadline - 5.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        current_indices = []
        for b in best_solution:
            t = tuple(b)
            if t in block_to_idx:
                current_indices.append(block_to_idx[t])
        
        if current_indices:
            remaining_time = deadline - 5.0 - time.time()
            if remaining_time > 5.0:
                ls_result = local_search(current_indices, rng, max_iters=5000, time_budget=remaining_time/2)
                if len(ls_result) < len(best_solution):
                    best_solution = [list(all_blocks[i]) for i in ls_result]
                    best_size = len(ls_result)
                
                # Apply simulated annealing as final polish
                if time.time() < deadline - 3.0:
                    sa_result = simulated_annealing(current_indices, rng, time_budget=remaining_time/2)
                    if len(sa_result) < len(best_solution):
                        best_solution = [list(all_blocks[i]) for i in sa_result]
                        best_size = len(sa_result)
    
    if best_solution is None:
        best_solution = []
    
    return best_solution
