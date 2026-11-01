"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import minimize

import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    For the specific case q=3, n=6, radius=1, we use Simulated Annealing with incremental
    coverage tracking to minimize the number of codewords while maintaining full coverage.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors
        neighbors = {}
        for w in all_words:
            nb = set()
            for v in all_words:
                dist = sum(1 for i in range(n) if w[i] != v[i])
                if dist <= radius:
                    nb.add(v)
            neighbors[w] = nb
        
        uncovered = set(all_words)
        code = []
        
        while uncovered:
            best_word = None
            max_covered = -1
            for w in all_words:
                if w in code:
                    continue
                covered_count = len(neighbors[w] & uncovered)
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_word = w
            
            if best_word is None:
                break
                
            code.append(best_word)
            uncovered -= neighbors[best_word]
            
        return [list(c) for c in code]

    # Specific optimization for q=3, n=6, radius=1
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words) # 729
    
    # Precompute neighbors as list of indices
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        # Distance 0
        neighbors[i].append(i)
        # Distance 1
        for pos in range(n):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors[i].append(idx)
        
    # Initialize with Tetracode-based construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            initial_code_indices.add(word_to_idx[w])
            
    code_set = set(initial_code_indices)
    code_list = list(code_set)
    
    # Initialize coverage count
    cov_count = np.zeros(N, dtype=np.int32)
    for i in code_set:
        for j in neighbors[i]:
            cov_count[j] += 1
            
    uncovered_count = np.sum(cov_count == 0)
    current_size = len(code_set)
    
    # Objective: Minimize Size + 1000 * Uncovered
    def get_cost(size, unc):
        return size + 1000 * unc
        
    current_cost = get_cost(current_size, uncovered_count)
    
    best_code_set = set(code_set)
    best_cost = current_cost
    
    # Simulated Annealing
    T = 50.0
    alpha = 0.995
    start_time = time.time()
    timeout = 115.0 # Leave 5s buffer
    
    random.seed(42)
    
    while time.time() - start_time < timeout:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a random codeword
            if not code_list:
                T *= alpha
                continue
            i = random.choice(code_list)
            
            # Calculate delta
            old_unc = uncovered_count
            # Decrement coverage for neighbors of i
            for j in neighbors[i]:
                cov_count[j] -= 1
            
            # Count new uncovered
            new_unc = np.sum(cov_count == 0)
            delta_unc = new_unc - old_unc
            delta_size = -1
            delta_cost = delta_size + 1000 * delta_unc
            
            # Accept?
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Commit
                code_set.remove(i)
                code_list.remove(i)
                current_size -= 1
                uncovered_count = new_unc
                current_cost += delta_cost
                
                if uncovered_count == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_code_set = set(code_set)
            else:
                # Revert
                for j in neighbors[i]:
                    cov_count[j] += 1
                    
        elif move_type < 0.6:
            # Add a random non-codeword
            i = random.randint(0, N - 1)
            if i in code_set:
                T *= alpha
                continue
            
            # Calculate delta
            old_unc = uncovered_count
            for j in neighbors[i]:
                cov_count[j] += 1
            
            new_unc = np.sum(cov_count == 0)
            delta_unc = new_unc - old_unc
            delta_size = 1
            delta_cost = delta_size + 1000 * delta_unc
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                code_set.add(i)
                code_list.append(i)
                current_size += 1
                uncovered_count = new_unc
                current_cost += delta_cost
                
                if uncovered_count == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_code_set = set(code_set)
            else:
                for j in neighbors[i]:
                    cov_count[j] -= 1
                    
        else:
            # Swap
            if not code_list:
                T *= alpha
                continue
            i = random.choice(code_list)
            j = random.randint(0, N - 1)
            if j in code_set:
                T *= alpha
                continue
                
            # Remove i, Add j
            old_unc = uncovered_count
            
            # Remove i
            for k in neighbors[i]:
                cov_count[k] -= 1
            
            # Add j
            for k in neighbors[j]:
                cov_count[k] += 1
                
            new_unc = np.sum(cov_count == 0)
            delta_unc = new_unc - old_unc
            delta_size = 0
            delta_cost = delta_size + 1000 * delta_unc
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                code_set.remove(i)
                code_list.remove(i)
                code_set.add(j)
                code_list.append(j)
                uncovered_count = new_unc
                current_cost += delta_cost
                
                if uncovered_count == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_code_set = set(code_set)
            else:
                # Revert
                for k in neighbors[j]:
                    cov_count[k] -= 1
                for k in neighbors[i]:
                    cov_count[k] += 1

        T *= alpha
        if T < 0.1:
            break
            
    # Return the best valid code found
    final_code_set = best_code_set
    
    result = [list(all_words[i]) for i in final_code_set]
    return result

# EVOLVE-BLOCK-END
