import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)
        
    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)
            
    # Pre-compute bitsets for t-subsets to blocks
    # n_words = ceil(n_blocks / 64)
    n_words = (n_blocks + 63) // 64
    tsub_block_bitsets = np.zeros((n_t_subs, n_words), dtype=np.uint64)
    for s_idx, blocks in enumerate(tsub_to_blocks):
        for b_idx in blocks:
            word_idx = b_idx // 64
            bit_idx = b_idx % 64
            tsub_block_bitsets[s_idx, word_idx] |= (1 << bit_idx)
            
    def get_set_bits(bitset_array):
        """Yield indices of set bits in a numpy array of uint64."""
        for i, word in enumerate(bitset_array):
            while word:
                # Get index of least significant set bit
                lsb = word & -word
                bit_pos = (lsb.bit_length() - 1)
                yield i * 64 + bit_pos
                word ^= lsb

    # Helper: Greedy construction
    def greedy_init():
        uncovered = np.ones(n_t_subs, dtype=np.int8)
        total_uncovered = n_t_subs
        solution_set = set()
        solution_list = []
        
        uncovered_count_per_block = np.array([len(subs) for subs in block_to_tsubs], dtype=np.int32)
        
        while total_uncovered > 0:
            if time.time() > deadline:
                break
            best_b_idx = np.argmax(uncovered_count_per_block)
            if uncovered_count_per_block[best_b_idx] == 0:
                break
            
            solution_set.add(best_b_idx)
            solution_list.append(best_b_idx)
            
            for s in block_to_tsubs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
                    for b_idx in tsub_to_blocks[s]:
                        uncovered_count_per_block[b_idx] -= 1
                        
        return solution_list, solution_set

    # Helper: Local Search (Deletion and Swap) with Bitset Optimization
    def local_search(solution_list, solution_set):
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
                
        # Maintain solution bitset for fast "not in solution" check
        solution_bitset = np.zeros(n_words, dtype=np.uint64)
        for b_idx in solution_set:
            solution_bitset[b_idx // 64] |= (1 << (b_idx % 64))
            
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # Pass 1: Deletion
            to_remove = []
            for b_idx in solution_list:
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b_idx)
            
            if to_remove:
                for b_idx in to_remove:
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_set.remove(b_idx)
                    solution_bitset[b_idx // 64] &= ~(1 << (b_idx % 64))
                    solution_list.remove(b_idx)
                improved = True
                continue # Restart to re-evaluate swaps

            # Pass 2: Swap
            for i in range(len(solution_list)):
                b_idx = solution_list[i]
                critical_subsets = []
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        critical_subsets.append(s)
                            
                if not critical_subsets:
                    # Should have been deleted
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_set.remove(b_idx)
                    solution_bitset[b_idx // 64] &= ~(1 << (b_idx % 64))
                    solution_list.pop(i)
                    improved = True
                    break

                # Find candidates using bitsets
                # AND all bitsets for critical subsets
                candidate_mask = tsub_block_bitsets[critical_subsets[0]].copy()
                for s in critical_subsets[1:]:
                    candidate_mask = np.bitwise_and(candidate_mask, tsub_block_bitsets[s])
                
                # Exclude blocks already in solution
                available_mask = np.bitwise_and(candidate_mask, np.bitwise_not(solution_bitset))
                
                # Find any available candidate
                candidates = list(get_set_bits(available_mask))
                
                if candidates:
                    # Pick a random candidate to diversify
                    best_candidate = random.choice(candidates)
                    
                    # Perform Swap
                    # Decrement cover_count for b_idx
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    # Increment cover_count for best_candidate
                    for s in block_to_tsubs[best_candidate]:
                        cover_count[s] += 1
                            
                    # Update solution
                    solution_list[i] = best_candidate
                    solution_set.remove(b_idx)
                    solution_set.add(best_candidate)
                    
                    # Update bitset
                    solution_bitset[b_idx // 64] &= ~(1 << (b_idx % 64))
                    solution_bitset[best_candidate // 64] |= (1 << (best_candidate % 64))
                    
                    improved = True
                    break # Restart loop to re-evaluate deletions

        return solution_list, solution_set

    # Helper: Perturbation and Repair
    def perturb_and_repair(solution_list, solution_set):
        n_remove = random.randint(2, 5)
        if len(solution_list) <= n_remove:
            return greedy_init()
            
        indices_to_remove = random.sample(range(len(solution_list)), n_remove)
        indices_to_remove.sort(reverse=True)
        
        new_solution_set = set(solution_set)
        for idx in indices_to_remove:
            b_idx = solution_list[idx]
            new_solution_set.remove(b_idx)
            
        new_solution_list = [b for i, b in enumerate(solution_list) if i not in indices_to_remove]
        
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in new_solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
                
        uncovered_mask = (cover_count == 0)
        uncovered_indices = np.where(uncovered_mask)[0]
        
        if len(uncovered_indices) == 0:
            return new_solution_list, new_solution_set
            
        to_cover = set(uncovered_indices.tolist())
        
        while to_cover:
            if time.time() > deadline:
                break
            best_b_idx = -1
            max_cov = 0
            
            if to_cover:
                first_uncovered = next(iter(to_cover))
                candidates = tsub_to_blocks[first_uncovered]
                
                for b_idx in candidates:
                    if b_idx in new_solution_set:
                        continue
                    cnt = 0
                    for s in block_to_tsubs[b_idx]:
                        if s in to_cover:
                            cnt += 1
                            if cnt > max_cov:
                                max_cov = cnt
                                best_b_idx = b_idx
                                if cnt == len(to_cover):
                                    break
                    if max_cov == len(to_cover):
                        break
                if best_b_idx != -1:
                    new_solution_set.add(best_b_idx)
                    new_solution_list.append(best_b_idx)
                    for s in block_to_tsubs[best_b_idx]:
                        if s in to_cover:
                            to_cover.remove(s)
                else:
                    break
                    
        return new_solution_list, new_solution_set

    # Main ILS Loop
    best_solution_list, best_solution_set = greedy_init()
    best_solution_list, best_solution_set = local_search(best_solution_list, best_solution_set)
    best_len = len(best_solution_list)
    current_solution_list = best_solution_list
    current_solution_set = best_solution_set
    
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        pert_list, pert_set = perturb_and_repair(current_solution_list, current_solution_set)
        opt_list, opt_set = local_search(pert_list, pert_set)
        
        if len(opt_list) < best_len:
            best_len = len(opt_list)
            best_solution_list = opt_list
            best_solution_set = opt_set
            current_solution_list = opt_list
            current_solution_set = opt_set
        else:
            if len(opt_list) < len(current_solution_list):
                current_solution_list = opt_list
                current_solution_set = opt_set

    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution_list]
    return final_blocks
