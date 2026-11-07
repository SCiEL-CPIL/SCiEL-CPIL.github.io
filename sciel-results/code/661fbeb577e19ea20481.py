import numpy as np
import time
import random

def construct():
    n = 12
    m = 40
    total_subsets = 1 << n  # 4096
    deadline = time.time() + 82.0  # Safety margin (90s limit)
    
    # Precompute subset sizes
    sizes = np.array([bin(i).count('1') for i in range(total_subsets)], dtype=np.int32)
    
    # Build the strict subset relation matrix M
    # M[i, j] = 1 if i is strict subset of j
    M = np.zeros((total_subsets, total_subsets), dtype=np.int8)
    
    chunk_size = 512
    for start in range(0, total_subsets, chunk_size):
        end = min(start + chunk_size, total_subsets)
        i_chunk = np.array(range(start, end), dtype=np.int32)
        j_all = np.array(range(total_subsets), dtype=np.int32)
        
        i_expanded = i_chunk[:, None]
        j_expanded = j_all[None, :]
        
        is_subset = (i_expanded & j_expanded) == i_expanded
        not_equal = i_expanded != j_expanded
        M[start:end, :] = (is_subset & not_equal).astype(np.int8)
    
    # Comparability matrix: C[i, j] = 1 if i and j are comparable
    C = np.zeros((total_subsets, total_subsets), dtype=np.int8)
    for start in range(0, total_subsets, chunk_size):
        end = min(start + chunk_size, total_subsets)
        i_chunk = np.array(range(start, end), dtype=np.int32)
        j_all = np.array(range(total_subsets), dtype=np.int32)
        
        i_expanded = i_chunk[:, None]
        j_expanded = j_all[None, :]
        
        j_i_subset = (j_expanded & i_expanded) == j_expanded
        not_equal2 = j_expanded != i_expanded
        M_ji = (j_i_subset & not_equal2).astype(np.int8)
        
        C[start:end, :] = ((M[start:end, :] | M_ji)).astype(np.int8)
    
    # Precompute degree of each node in the comparability graph
    degree = np.sum(C, axis=1).astype(np.int32)
    
    def compute_score_fast(family):
        """Faster score computation using vectorization."""
        if len(family) < 2:
            return 0
        fam_arr = np.array(family, dtype=np.int32)
        i_idx, j_idx = np.triu_indices(len(fam_arr), k=1)
        pairs = C[fam_arr[i_idx], fam_arr[j_idx]]
        return int(np.sum(pairs))
    
    def greedy_construct(init_family, deadline):
        """
        Greedy: repeatedly pick the candidate with highest comparability to current family.
        """
        family = list(init_family)
        used = set(family)
        current_score = compute_score_fast(family)
        
        if len(family) == 0:
            return family, 0
        
        fam_arr = np.array(family, dtype=np.int32)
        
        # comp_counts[c] = number of sets in family comparable with c
        comp_counts = np.zeros(total_subsets, dtype=np.int32)
        for c in range(total_subsets):
            if c not in used:
                comp_counts[c] = np.sum(C[fam_arr, c])
        
        available = np.array([x for x in range(total_subsets) if x not in used], dtype=np.int32)
        
        while len(family) < m:
            if time.time() > deadline - 3.0:
                break
            if len(available) == 0:
                break
            
            best_idx = np.argmax(comp_counts[available])
            best_mask = int(available[best_idx])
            
            family.append(best_mask)
            used.add(best_mask)
            current_score += int(comp_counts[best_mask])
            
            # Update: remove best_mask from available, update comp_counts for rest
            mask_pos = np.where(available == best_mask)[0]
            available = np.delete(available, mask_pos)
            if len(available) > 0:
                comp_counts[available] += C[available, best_mask]
        
        return family, current_score
    
    def local_search_polish(family, deadline, max_iters=200):
        """
        Local search: try swapping each element with a non-family element.
        """
        family = list(family)
        used = set(family)
        current_score = compute_score_fast(family)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            improved = False
            fam_arr = np.array(family, dtype=np.int32)
            
            for pos in range(len(family)):
                if time.time() > deadline - 3.0:
                    break
                
                removed = family[pos]
                contrib = int(np.sum(C[np.delete(fam_arr, pos), removed]))
                score_without = current_score - contrib
                
                non_family = np.array([x for x in range(total_subsets) if x not in used], dtype=np.int32)
                
                if len(non_family) == 0:
                    continue
                
                remaining_fam = np.delete(fam_arr, pos)
                
                # For each candidate, compute comparability with remaining family
                comp_with_remaining = np.zeros(len(non_family), dtype=np.int32)
                chunk = 256
                for start in range(0, len(non_family), chunk):
                    end = min(start + chunk, len(non_family))
                    chunk_arr = non_family[start:end]
                    comp_with_remaining[start:end] = C[chunk_arr[:, None], remaining_fam[None, :]].sum(axis=1)
                
                new_scores = score_without + comp_with_remaining
                best_new_idx = np.argmax(new_scores)
                best_new_mask = int(non_family[best_new_idx])
                best_new_score = int(new_scores[best_new_idx])
                
                if best_new_score > current_score:
                    family[pos] = best_new_mask
                    used.discard(removed)
                    used.add(best_new_mask)
                    current_score = best_new_score
                    improved = True
                    break
            
            if not improved:
                break
        
        return family, current_score
    
    def chain_based_init(seed):
        """
        Build an initial solution by constructing multiple chains and merging.
        A chain is a sequence where each set is a strict subset of the next.
        """
        rng = random.Random(seed)
        init = []
        used = set()
        
        # Build several chains
        num_chains = 5
        for chain_idx in range(num_chains):
            current = 0
            chain = []
            while current < (1 << n) - 1 and len(chain) < 10:
                chain.append(current)
                used.add(current)
                # Pick a random superset
                available_bits = [b for b in range(n) if not (current & (1 << b))]
                if not available_bits:
                    break
                num_add = rng.randint(1, min(3, len(available_bits)))
                bits_to_add = rng.sample(available_bits, num_add)
                for b in bits_to_add:
                    current |= (1 << b)
            init.extend(chain)
        
        # Fill remaining with high-degree nodes not yet used
        if len(init) < m:
            candidates = [x for x in range(total_subsets) if x not in used]
            candidates.sort(key=lambda x: -degree[x])
            for c in candidates:
                if len(init) >= m:
                    break
                init.append(c)
                used.add(c)
        
        return init[:m]
    
    # Run multiple restarts
    best_score = -1
    best_family = None
    
    num_restarts = 40
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        random.seed(restart * 13 + 7)
        
        # Different initialization strategies
        if restart % 4 == 0:
            # Chain-based initialization
            init = chain_based_init(restart)
        elif restart % 4 == 1:
            # Start with structured sets: empty, full, and some intermediate
            init = [0, (1 << n) - 1]
            # Add all singletons
            for b in range(n):
                init.append(1 << b)
            # Add all 2-element sets
            for b in range(n):
                for c in range(b+1, n):
                    if len(init) >= 15:
                        break
                    init.append((1 << b) | (1 << c))
                if len(init) >= 15:
                    break
        elif restart % 4 == 2:
            # Start with a long random chain
            init = []
            current = 0
            init.append(current)
            for _ in range(12):
                available_bits = [b for b in range(n) if not (current & (1 << b))]
                if not available_bits:
                    break
                current |= (1 << random.choice(available_bits))
                init.append(current)
            # Add some more random high-degree nodes
            remaining = [x for x in range(total_subsets) if x not in set(init)]
            remaining.sort(key=lambda x: -degree[x])
            for x in remaining[:20]:
                if len(init) >= m:
                    break
                init.append(x)
        else:
            # Random initialization biased toward high-degree nodes
            init = random.sample(range(total_subsets), 10)
        
        # Ensure distinct and valid
        init = list(dict.fromkeys(init))
        if len(init) > m:
            init = init[:m]
        
        # Greedy construction
        family, score = greedy_construct(init, deadline)
        
        if len(family) < m:
            used = set(family)
            remaining = [x for x in range(total_subsets) if x not in used]
            remaining.sort(key=lambda x: -degree[x])
            for x in remaining:
                if len(family) >= m:
                    break
                family.append(x)
        
        if len(family) < m:
            continue
        
        # Local search polish
        family, score = local_search_polish(family, deadline, max_iters=100)
        
        if score > best_score:
            best_score = score
            best_family = list(family)
    
    # Final polish on the best family if time allows
    if best_family is not None and time.time() < deadline - 3.0:
        for _ in range(3):
            if time.time() > deadline - 2.0:
                break
            polished, pol_score = local_search_polish(best_family, deadline, max_iters=80)
            if pol_score > best_score:
                best_score = pol_score
                best_family = polished
    
    # Convert to lists of integers
    if best_family is None:
        best_family = list(range(m))
    
    result = []
    for mask in best_family:
        mask = int(mask)
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result
