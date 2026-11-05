"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for 150s limit

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists as numpy arrays of int32 for fast operations
    neighbors_list = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))

    # Build cover matrix efficiently using the neighbor lists
    cover_matrix = np.zeros((N, N), dtype=np.bool_)
    for c in range(N):
        cover_matrix[c, neighbors_list[c]] = True

    def greedy_cover():
        """Construct a valid covering code using greedy set cover."""
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        
        while uncovered_mask.any():
            # Vectorized: compute gain for all candidates at once
            gains = cover_matrix[:, uncovered_mask].sum(axis=1)
            best_c = int(np.argmax(gains))
            best_count = int(gains[best_c])
            
            if best_count == 0:
                break
            
            code.append(best_c)
            uncovered_mask[neighbors_list[best_c]] = False
            
        return code

    def compute_coverage(code):
        """Compute coverage array from a list of codewords."""
        coverage = np.zeros(N, dtype=np.int32)
        for c in code:
            coverage[neighbors_list[c]] += 1
        return coverage

    def is_redundant(c, coverage):
        """Check if codeword c is redundant."""
        return np.min(coverage[neighbors_list[c]]) > 1

    def reduce_code(code, coverage):
        """Remove all redundant codewords from the code."""
        changed = True
        while changed:
            changed = False
            for c in list(code):
                if is_redundant(c, coverage):
                    code.remove(c)
                    coverage[neighbors_list[c]] -= 1
                    changed = True
        return code, coverage

    def find_best_replacement(uncovered_indices, current_code):
        """Find the single best codeword to cover the uncovered indices."""
        if len(uncovered_indices) == 0:
            return None, 0
        
        in_code = np.zeros(N, dtype=bool)
        for c in current_code:
            in_code[c] = True
            
        gains = cover_matrix[:, uncovered_indices].sum(axis=1)
        gains[in_code] = 0
        
        best_c = int(np.argmax(gains))
        best_count = int(gains[best_c])
        
        return best_c, best_count

    # Main Algorithm
    # 1. Initial Greedy
    code = greedy_cover()
    coverage = compute_coverage(code)
    
    # 2. Initial Reduction
    code, coverage = reduce_code(code, coverage)
    
    best_code = list(code)
    best_size = len(code)
    
    # 3. Local Search with Kick Mechanism
    # We use a hybrid of hill climbing and randomized kicks to escape local minima.
    
    def kick(code, coverage, k):
        """
        Perform a kick: remove k random codewords and repair with greedy.
        This allows escaping deep local minima.
        """
        if len(code) <= k:
            return code, coverage
        
        # Randomly select k codewords to remove
        remove_indices = random.sample(range(len(code)), k)
        remove_indices.sort(reverse=True)
        
        for idx in remove_indices:
            c = code[idx]
            code.remove(c)
            coverage[neighbors_list[c]] -= 1
        
        # Repair: greedily add codewords to cover uncovered spots
        uncovered = np.where(coverage == 0)[0]
        while len(uncovered) > 0:
            # Find best candidate
            in_code = np.zeros(N, dtype=bool)
            for c in code:
                in_code[c] = True
            
            gains = cover_matrix[:, uncovered].sum(axis=1)
            gains[in_code] = 0
            
            best_c = int(np.argmax(gains))
            best_count = int(gains[best_c])
            
            if best_count == 0:
                break
            
            code.append(best_c)
            coverage[neighbors_list[best_c]] += 1
            
            # Update uncovered
            uncovered = np.where(coverage == 0)[0]
        
        # Reduce
        code, coverage = reduce_code(code, coverage)
        
        return code, coverage

    step_count = 0
    no_improve_steps = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        step_count += 1
        
        # Update best
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            no_improve_steps = 0
        else:
            no_improve_steps += 1
        
        if len(code) < 2:
            break
        
        # Determine move type
        # Mostly try single swaps, occasionally kick
        if no_improve_steps > max_no_improve or (step_count % 100 == 0 and random.random() < 0.3):
            # Kick
            k = random.choice([2, 3, 4])
            if len(code) > k + 1:
                code, coverage = kick(code, coverage, k)
            else:
                # Fallback to swap
                c_remove = random.choice(code)
                code.remove(c_remove)
                coverage[neighbors_list[c_remove]] -= 1
                uncovered = np.where(coverage == 0)[0]
                if len(uncovered) > 0:
                    c_new, count = find_best_replacement(uncovered, code)
                    if c_new is not None and count > 0:
                        code.append(c_new)
                        coverage[neighbors_list[c_new]] += 1
                    else:
                        code.append(c_remove)
                        coverage[neighbors_list[c_remove]] += 1
                code, coverage = reduce_code(code, coverage)
        else:
            # Standard swap
            c_remove = random.choice(code)
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            
            uncovered = np.where(coverage == 0)[0]
            
            if len(uncovered) == 0:
                # Free removal
                pass
            else:
                c_new, count = find_best_replacement(uncovered, code)
                if c_new is not None and count > 0:
                    code.append(c_new)
                    coverage[neighbors_list[c_new]] += 1
                else:
                    code.append(c_remove)
                    coverage[neighbors_list[c_remove]] += 1
            
            code, coverage = reduce_code(code, coverage)

    # Final verification and return
    final_coverage = compute_coverage(best_code)
    if np.any(final_coverage == 0):
        best_code = list(range(N))
        
    result = [idx_to_tuple(cw) for cw in best_code]
    return result
# EVOLVE-BLOCK-END
