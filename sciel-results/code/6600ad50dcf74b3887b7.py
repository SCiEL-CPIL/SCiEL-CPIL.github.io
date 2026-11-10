import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 125.0

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbors
    neighbor_indices = []
    neighbor_sets = []
    for w in all_words:
        nbs = {w}
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_word = list(w)
                    new_word[pos] = val
                    nbs.add(tuple(new_word))
        idxs = [word_to_idx[nei] for nei in nbs]
        neighbor_indices.append(idxs)
        neighbor_sets.append(set(idxs))

    def local_search(initial_codewords):
        codeword_indices = [word_to_idx[w] for w in initial_codewords]
        codeword_indices = list(set(codeword_indices))
        
        coverage = np.zeros(total_words, dtype=np.int32)
        for idx in codeword_indices:
            for ni in neighbor_indices[idx]:
                coverage[ni] += 1
                
        in_code = set(codeword_indices)
        
        # Phase 1: Greedy Removal (fast)
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(len(codeword_indices) - 1, -1, -1):
                if time.time() > deadline - 10.0:
                    break
                idx = codeword_indices[i]
                can_remove = True
                for ni in neighbor_indices[idx]:
                    if coverage[ni] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for ni in neighbor_indices[idx]:
                        coverage[ni] -= 1
                    codeword_indices.pop(i)
                    in_code.discard(idx)
                    improved = True
                    break 

        # Phase 2: 2-out, 1-in Move (slow but powerful)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            n_codewords = len(codeword_indices)
            found_move = False
            
            # Try random pairs if deterministic fails to escape local minima?
            # For now, deterministic scan.
            
            for i in range(n_codewords):
                if time.time() > deadline - 5.0:
                    break
                idx_i = codeword_indices[i]
                nbs_i = neighbor_indices[idx_i]
                set_i = neighbor_sets[idx_i]
                
                for j in range(i + 1, n_codewords):
                    idx_j = codeword_indices[j]
                    nbs_j = neighbor_indices[idx_j]
                    set_j = neighbor_sets[idx_j]
                    
                    uncov_set = []
                    for w in nbs_i:
                        if coverage[w] == 1:
                            uncov_set.append(w)
                    for w in nbs_j:
                        if coverage[w] == 1:
                            uncov_set.append(w)
                        
                    if not uncov_set:
                        for w in nbs_i:
                            coverage[w] -= 1
                        for w in nbs_j:
                            coverage[w] -= 1
                        codeword_indices.pop(j)
                        codeword_indices.pop(i)
                        in_code.discard(idx_i)
                        in_code.discard(idx_j)
                        improved = True
                        found_move = True
                        break
                    else:
                        candidate_set = set(neighbor_indices[uncov_set[0]])
                        for w in uncov_set[1:]:
                            candidate_set &= neighbor_sets[w]
                            if not candidate_set:
                                break
                        
                        if candidate_set:
                            candidate_set -= in_code
                            if candidate_set:
                                candidate_k = next(iter(candidate_set))
                                for w in nbs_i:
                                    coverage[w] -= 1
                                for w in nbs_j:
                                    coverage[w] -= 1
                                for w in neighbor_indices[candidate_k]:
                                    coverage[w] += 1
                                
                                if i < j:
                                    codeword_indices.pop(j)
                                    codeword_indices.pop(i)
                                else:
                                    codeword_indices.pop(i)
                                    codeword_indices.pop(j)
                                
                                in_code.discard(idx_i)
                                in_code.discard(idx_j)
                                in_code.add(candidate_k)
                                codeword_indices.append(candidate_k)
                                
                                improved = True
                                found_move = True
                                break
                                
                if found_move:
                    break
                    
        return [list(all_words[idx]) for idx in codeword_indices]

    # --- Strategy ---
    best_code = None
    best_size = float('inf')
    
    # 1. Tetracode Product
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                initial_codewords.append(tuple(h + list(tail)))
        
        code = local_search(initial_codewords)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
    
    # 2. Greedy Set Cover (Deterministic)
    if time.time() < deadline - 20.0:
        initial_codewords = []
        coverage = np.zeros(total_words, dtype=np.int32)
        uncovered = np.ones(total_words, dtype=bool)
        
        while np.any(uncovered) and time.time() < deadline - 15.0:
            best_idx = -1
            best_count = 0
            for idx, nbs in enumerate(neighbor_indices):
                count = 0
                for ni in nbs:
                    if uncovered[ni]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1:
                break
            initial_codewords.append(all_words[best_idx])
            for ni in neighbor_indices[best_idx]:
                coverage[ni] += 1
                uncovered[ni] = False
        
        if initial_codewords:
            code = local_search(initial_codewords)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code

    # 3. Randomized Greedy (if time permits)
    if time.time() < deadline - 10.0:
        random.seed(42)
        initial_codewords = []
        coverage = np.zeros(total_words, dtype=np.int32)
        uncovered = np.ones(total_words, dtype=bool)
        
        while np.any(uncovered) and time.time() < deadline - 5.0:
            best_idx = -1
            best_count = 0
            # Sample a few to speed up? No, just scan.
            for idx in range(total_words):
                nbs = neighbor_indices[idx]
                count = 0
                for ni in nbs:
                    if uncovered[ni]:
                        count += 1
                # Add noise
                noisy_count = count + random.random() * 0.1
                if noisy_count > best_count:
                    best_count = noisy_count
                    best_idx = idx
            if best_idx == -1:
                break
            initial_codewords.append(all_words[best_idx])
            for ni in neighbor_indices[best_idx]:
                coverage[ni] += 1
                uncovered[ni] = False
        
        if initial_codewords:
            code = local_search(initial_codewords)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code

    if best_code is None:
        best_code = [list(all_words[0])]

    return best_code
