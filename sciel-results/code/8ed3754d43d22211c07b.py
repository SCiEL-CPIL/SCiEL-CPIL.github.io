import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_space = q ** n

    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]

    # Precompute cover masks
    # mask[i] is a bitmask of indices of words within Hamming distance <= radius of word i
    cover_masks = [0] * total_space
    neighbors = [[] for _ in range(total_space)]
    reverse_cover = [[] for _ in range(total_space)]

    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        neigh = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
                    neigh.append(j)
        # For radius > 1, we would need more logic, but task specifies radius 1
        cover_masks[i] = mask
        neighbors[i] = neigh
        for j in neigh:
            reverse_cover[j].append(i)

    full_mask = (1 << total_space) - 1

    # Greedy Set Cover
    def greedy_cover(randomized=False):
        uncovered = full_mask
        code = []
        while uncovered:
            best_gain = 0
            candidates = []
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            if best_gain == 0:
                break
            if randomized:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
        return code

    def local_search_reduce(code, max_iter=50):
        if not code:
            return code

        current_code = code[:]
        
        # Build cover_count array: cover_count[w] = number of codewords covering word w
        cover_count = [0] * total_space
        for idx in current_code:
            for w in neighbors[idx]:
                cover_count[w] += 1
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                # mask = cover_masks[idx]
                
                # Check if all words covered by idx are covered by at least one other codeword
                removable = True
                for w_idx in neighbors[idx]:
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                
                if removable:
                    # Decrement cover_count and remove
                    for w_idx in neighbors[idx]:
                        cover_count[w_idx] -= 1
                    current_code.pop(i)
                    improved = True
                    break
                else:
                    # Try to find a swap
                    # Find words uniquely covered by idx
                    unique_words = []
                    for w_idx in neighbors[idx]:
                        if cover_count[w_idx] == 1:
                            unique_words.append(w_idx)
                    
                    if unique_words:
                        # Find a candidate j not in current_code that covers all words in unique_words
                        # Optimization: Start with the word with the smallest reverse_cover
                        unique_words.sort(key=lambda x: len(reverse_cover[x]))
                        
                        candidates = set(reverse_cover[unique_words[0]])
                        for w in unique_words[1:]:
                            candidates &= set(reverse_cover[w])
                            if not candidates:
                                break
                        
                        # Exclude current code
                        for c in current_code:
                            candidates.discard(c)
                        
                        if candidates:
                            # Choose a random candidate
                            new_idx = random.choice(list(candidates))
                            
                            # Update cover_count
                            # Remove old idx
                            for w_idx in neighbors[idx]:
                                cover_count[w_idx] -= 1
                            
                            # Add new idx
                            for w_idx in neighbors[new_idx]:
                                cover_count[w_idx] += 1
                            
                            current_code[i] = new_idx
                            improved = True
                            break
            
        return current_code

    def iterated_local_search(code, deadline):
        current_code = local_search_reduce(code)
        best_code = current_code[:]
        
        while time.time() < deadline - 2.0:
            # Perturb: remove a random few codewords
            k = random.randint(1, 3)
            if len(current_code) <= k:
                break
            to_remove = random.sample(current_code, k)
            perturbed = [c for c in current_code if c not in to_remove]
            
            # Repair: greedy cover the remaining holes
            # Compute uncovered
            covered_mask = 0
            for c in perturbed:
                covered_mask |= cover_masks[c]
            uncovered = full_mask & ~covered_mask
            
            if not uncovered:
                # No holes, just reduce again
                repaired = local_search_reduce(perturbed)
            else:
                # Greedy cover the holes
                code = perturbed[:]
                while uncovered:
                    best_gain = 0
                    candidates = []
                    for i in range(total_space):
                        if i in code:
                            continue
                        gain = (cover_masks[i] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            candidates = [i]
                        elif gain == best_gain and gain > 0:
                            candidates.append(i)
                    if best_gain == 0:
                        break
                    chosen = random.choice(candidates)
                    code.append(chosen)
                    uncovered &= ~cover_masks[chosen]
                repaired = local_search_reduce(code)
            
            if len(repaired) < len(best_code):
                best_code = repaired[:]
                current_code = best_code[:]
            else:
                # Simulated annealing style: accept worse with some probability?
                # For now, just keep current if not improved, or revert
                current_code = best_code[:]
                
            # Check time
            if time.time() > deadline - 5.0:
                break
                
        return best_code

    # Initial greedy construction
    best_code = greedy_cover()
    best_code = local_search_reduce(best_code)
    
    # Run Iterated Local Search
    if time.time() < deadline - 10.0:
        best_code = iterated_local_search(best_code, deadline)
    
    # Run a few restarts if time permits
    if time.time() < deadline - 20.0:
        for _ in range(5): # Try up to 5 restarts
            if time.time() > deadline - 5.0:
                break
            cand_code = greedy_cover(randomized=True)
            cand_code = local_search_reduce(cand_code)
            if len(cand_code) < len(best_code):
                best_code = cand_code
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
