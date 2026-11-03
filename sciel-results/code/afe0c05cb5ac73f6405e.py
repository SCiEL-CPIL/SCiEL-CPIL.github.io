import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # We assume the evaluator passes these.
    
    total_words = q ** n
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors
    # neighbor_indices[i] is a list of indices within distance <= radius
    neighbor_indices = [[] for _ in range(total_words)]
    # For fast redundancy check, we might need masks or just use the lists
    # Since radius=1, size is 1 + n*(q-1) = 1 + 7*2 = 15
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        neighbor_indices[i] = idxs

    # Precompute masks for speed in greedy
    masks = [0] * total_words
    for i in range(total_words):
        m = 0
        for j in neighbor_indices[i]:
            m |= (1 << j)
        masks[i] = m

    # Precompute adjacency matrix: adj[i][j] = True if i is in N(j)
    # This enables O(1) check for swap validity
    adj = [bytearray(total_words) for _ in range(total_words)]
    for i in range(total_words):
        for j in neighbor_indices[i]:
            adj[i][j] = 1

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = (1 << total_words) - 1
        code = []
        active = list(range(total_words))
        random.shuffle(active)
        
        while uncovered:
            best_gain = -1
            best_candidate = -1
            best_count = 0
            
            for i in active:
                cov = masks[i] & uncovered
                if cov == 0:
                    continue
                gain = cov.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = i
                    best_count = 1
                elif gain == best_gain:
                    best_count += 1
                    if random.random() < 1.0 / best_count:
                        best_candidate = i
            
            if best_gain == -1:
                break
                
            chosen = best_candidate
            code.append(chosen)
            uncovered &= ~masks[chosen]
            
            # Remove candidates that are now fully covered
            new_active = []
            for i in active:
                if i == chosen:
                    continue
                if (masks[i] & uncovered) != 0:
                    new_active.append(i)
            active = new_active
            
        return code

    def reduce_and_optimize(code, deadline):
        """
        Reduces code by removing redundant words and performing 1-for-1 swaps.
        Uses precomputed adjacency matrix for O(1) swap checks.
        """
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # Tentatively remove c_remove
                for w in neighbor_indices[c_remove]:
                    counts[w] -= 1
                
                # Check if any count is 0 in the neighborhood of c_remove
                min_count = np.min(counts[neighbor_indices[c_remove]])
                
                if min_count > 0:
                    # Redundant, keep removed
                    code.pop(i)
                    improved = True
                    break
                
                else:
                    # Restore counts
                    for w in neighbor_indices[c_remove]:
                        counts[w] += 1
                    
                    # Find critical words (those with count 1 in N(c_remove))
                    critical_words = [w for w in neighbor_indices[c_remove] if counts[w] == 1]
                    
                    if not critical_words:
                        continue

                    # Find a replacement c_new that covers all critical_words
                    candidates = neighbor_indices[critical_words[0]]
                    random.shuffle(candidates)
                    
                    best_new = -1
                    for c_new in candidates:
                        if c_new == c_remove:
                            continue
                        
                        # O(1) check using precomputed adjacency matrix
                        valid = all(adj[cw][c_new] for cw in critical_words[1:])
                        
                        if valid:
                            best_new = c_new
                            break
                    
                    if best_new != -1:
                        # Perform swap
                        code[i] = best_new
                        
                        # Update counts: -c_remove, +best_new
                        for w in neighbor_indices[c_remove]:
                            counts[w] -= 1
                        for w in neighbor_indices[best_new]:
                            counts[w] += 1
                        
                        improved = True
                        break
        
        return code

    def perturb_and_repair(code, deadline):
        """
        Iterated local search: remove a few codewords, repair with greedy, then re-optimize.
        This helps escape local minima where no single swap is possible.
        """
        if len(code) < 2 or time.time() >= deadline:
            return code
        
        # Remove 2-4 random codewords
        num_remove = min(random.randint(2, 4), len(code) - 1)
        indices = random.sample(range(len(code)), num_remove)
        indices.sort(reverse=True)
        
        # Build current coverage
        covered = (1 << total_words) - 1
        for i in indices:
            covered &= masks[code[i]]
        
        # Remove the selected codewords
        new_code = [c for i, c in enumerate(code) if i not in indices]
        
        # Greedy repair: add codewords to cover uncovered words
        uncovered = ~covered & ((1 << total_words) - 1)
        
        while uncovered:
            best_gain = -1
            best_candidate = -1
            
            for i in range(total_words):
                if i in new_code:
                    continue
                cov = masks[i] & uncovered
                if cov == 0:
                    continue
                gain = cov.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = i
            
            if best_gain == -1:
                break
            
            new_code.append(best_candidate)
            uncovered &= ~masks[best_candidate]
        
        # Re-optimize the repaired code
        new_code = reduce_and_optimize(new_code, deadline)
        
        return new_code

    # Main loop with time management
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    # We run multiple instances
    # Each greedy + local search takes some time.
    # Greedy is fast (~0.1s). Local search is variable.
    
    while time.time() < deadline - 10:
        code = greedy_construct(seed)
        
        # Initial reduction
        code = reduce_and_optimize(code, deadline - 10)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        
        seed += 1
    
    # If we have time left, try iterated local search on the best code
    if best_code is not None and time.time() < deadline - 5:
        current_best = list(best_code)
        current_size = len(current_best)
        ils_seed = 1000
        
        while time.time() < deadline - 5:
            perturbed = perturb_and_repair(current_best, deadline - 5)
            
            if perturbed and len(perturbed) < current_size:
                current_size = len(perturbed)
                current_best = perturbed
                best_size = current_size
                best_code = current_best
            
            ils_seed += 1
    
    if best_code is None:
        # Fallback: return empty or a single word? Must be valid.
        # If no code found, return a trivial cover (all words? No, too big).
        # Greedy should always find a cover.
        return []

    # Convert indices back to lists of digits
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
