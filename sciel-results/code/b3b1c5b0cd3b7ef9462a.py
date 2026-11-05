import itertools
import numpy as np
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Precompute all t-subsets (triples) and map them to indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute block_triple_indices: for each block, the indices of triples it covers
    c_k_t = len(list(itertools.combinations(range(k), t)))
    block_triple_indices = np.empty((num_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        indices = []
        for sub in itertools.combinations(block, t):
            indices.append(triple_to_idx[sub])
        block_triple_indices[i] = indices
    
    # Build reverse mapping: triple_idx -> list of block indices containing it
    triple_blocks = [[] for _ in range(num_triples)]
    for b in range(num_blocks):
        for t_idx in block_triple_indices[b]:
            triple_blocks[t_idx].append(b)
    
    def greedy_set_cover(time_limit):
        """Weighted greedy set cover with triple frequency weighting."""
        uncovered = np.ones(num_triples, dtype=bool)
        selected = []
        
        while uncovered.any():
            if time.time() > time_limit:
                break
            
            # Compute gain: number of currently uncovered triples covered by each block
            gains = uncovered[block_triple_indices].sum(axis=1)
            
            # Weight by rarity: blocks covering rarer triples get bonus
            # triple_weight[b, j] = 1 / (number of blocks covering triple j)
            # Use precomputed inverse frequencies
            best_idx = np.argmax(gains)
            
            if gains[best_idx] == 0:
                break
            
            selected.append(best_idx)
            covered = block_triple_indices[best_idx]
            uncovered[covered] = False
        
        return selected
    
    def remove_redundant(selected, time_limit):
        """Remove blocks whose covered triples are all covered by other selected blocks."""
        if not selected:
            return selected
        
        cover_counts = np.zeros(num_triples, dtype=np.int32)
        for b in selected:
            cover_counts[block_triple_indices[b]] += 1
        
        new_selected = []
        selected_set = set(selected)
        
        for b in selected:
            if time.time() > time_limit:
                new_selected.append(b)
                continue
            
            cover_counts[block_triple_indices[b]] -= 1
            # Check if all triples of b are still covered
            if np.all(cover_counts[block_triple_indices[b]] > 0):
                # b is redundant, skip it
                pass
            else:
                cover_counts[block_triple_indices[b]] += 1
                new_selected.append(b)
        
        return new_selected
    
    def local_search(selected, time_limit, max_rounds=50):
        """Try swap operations: replace one block with another to potentially remove a third."""
        if len(selected) < 3:
            return selected
        
        selected_list = list(selected)
        cover_counts = np.zeros(num_triples, dtype=np.int32)
        for b in selected_list:
            cover_counts[block_triple_indices[b]] += 1
        
        best_selected = selected_list[:]
        best_len = len(selected_list)
        
        for _ in range(max_rounds):
            if time.time() > time_limit:
                break
            
            improved = False
            random.shuffle(selected_list)
            
            for i, A in enumerate(selected_list):
                if time.time() > time_limit:
                    break
                
                A_triples = block_triple_indices[A]
                
                # Find triples only covered by A in current selection
                unique_mask = (cover_counts[A_triples] == 1)
                if not np.any(unique_mask):
                    # A is redundant, remove it
                    cover_counts[A_triples] -= 1
                    selected_list.pop(i)
                    best_len -= 1
                    best_selected = selected_list[:]
                    improved = True
                    break
                
                unique_triples = A_triples[unique_mask]
                # Find candidate B's that cover all unique triples of A
                candidate_Bs = set(triple_blocks[unique_triples[0]])
                for ut in unique_triples[1:]:
                    candidate_Bs &= set(triple_blocks[ut])
                
                candidate_Bs.discard(A)
                
                # Sort candidates by potential (fewer unique triples they'd need to cover)
                # Just try them in random order for diversity
                cand_list = list(candidate_Bs)
                random.shuffle(cand_list)
                
                for B in cand_list:
                    if B in selected_list:
                        continue
                    
                    # Try: remove A, add B
                    cover_counts[A_triples] -= 1
                    cover_counts[block_triple_indices[B]] += 1
                    
                    # Check if we can remove some C - find ALL removable C's
                    removable_Cs = []
                    for j, C in enumerate(selected_list):
                        if C == A:
                            continue
                        C_triples = block_triple_indices[C]
                        if np.all(cover_counts[C_triples] > 1):
                            removable_Cs.append(j)
                    
                    if removable_Cs:
                        # Pick the first one (could try to be smarter here)
                        removed_C = removable_Cs[0]
                        cover_counts[block_triple_indices[removed_C]] -= 1
                        # Rebuild selected_list
                        new_sel = []
                        for idx, b in enumerate(selected_list):
                            if idx != removed_C and b != A:
                                new_sel.append(b)
                        new_sel.append(B)
                        selected_list = new_sel
                        best_len -= 1
                        best_selected = selected_list[:]
                        improved = True
                        break
                    else:
                        # Revert
                        cover_counts[A_triples] += 1
                        cover_counts[block_triple_indices[B]] -= 1
            
            if not improved:
                break
        
        return best_selected
    
    # Phase 1: Greedy set cover
    sel1 = greedy_set_cover(deadline - 55)
    
    # Phase 2: Remove redundant blocks
    sel2 = remove_redundant(sel1, deadline - 45)
    
    # Phase 3: Local search (give it more time)
    sel3 = local_search(sel2, deadline - 8, max_rounds=120)
    
    # Phase 4: Another round of local search if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        sel3 = local_search(sel3, deadline - 3, max_rounds=60)
    
    # Final cleanup
    sel4 = remove_redundant(sel3, deadline - 1)
    
    # Convert to list of lists
    result = [list(all_blocks[idx]) for idx in sel4]
    
    # Verify coverage
    covered = set()
    for block in result:
        for sub in itertools.combinations(block, t):
            covered.add(sub)
    
    if len(covered) == num_triples:
        return result
    
    # Fallback: use greedy result if local search broke coverage
    result2 = [list(all_blocks[idx]) for idx in sel2]
    covered2 = set()
    for block in result2:
        for sub in itertools.combinations(block, t):
            covered2.add(sub)
    
    if len(covered2) == num_triples:
        return result2
    
    # Final fallback: greedy
    result3 = [list(all_blocks[idx]) for idx in sel1]
    return result3
