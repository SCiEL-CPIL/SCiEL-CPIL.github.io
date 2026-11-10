"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: use all words (worst case)
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # For q=3, n=7, radius=1
    total_words = q ** n  # 2187
    
    # Precompute neighbor lists for all words using vectorized approach
    # Each word has 1 + (q-1)*n = 1 + 2*7 = 15 neighbors
    all_neighbors = np.empty((total_words, 1 + (q-1)*n), dtype=np.int32)
    
    # Precompute digit positions
    positions = np.array([q**i for i in range(n)], dtype=np.int64)
    
    for w_int in range(total_words):
        neighbors = [w_int]
        for i in range(n):
            orig_digit = (w_int // positions[i]) % q
            for new_val in range(q):
                if new_val != orig_digit:
                    delta = (new_val - orig_digit) * positions[i]
                    neighbors.append(w_int + delta)
        all_neighbors[w_int] = neighbors
    
    # Initial code: direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    def word_to_int(word):
        result = 0
        for w in word:
            result = result * q + w
        return result
    
    code_ints = [word_to_int(c) for c in initial_code]
    code_set = set(code_ints)
    
    # Coverage count using numpy for speed
    covered_count = np.zeros(total_words, dtype=np.int32)
    
    # Vectorized coverage update
    for c_int in code_ints:
        covered_count[all_neighbors[c_int]] += 1
    
    # Simulated annealing with improved energy function
    # Use soft penalty for uncovered words to create smoother landscape
    
    def compute_energy(covered_count, num_codewords, soft_penalty=10.0):
        """Compute energy with soft penalty for uncovered words."""
        # Soft penalty: sum of exp(-coverage/soft_penalty) for uncovered
        # This creates a smooth surrogate for the hard constraint
        uncovered_penalty = np.sum(np.exp(-covered_count / soft_penalty))
        return uncovered_penalty + num_codewords
    
    # Start from current solution
    current_energy = compute_energy(covered_count, len(code_ints))
    best_code_ints = list(code_ints)
    best_energy = current_energy
    best_uncovered = int(np.sum(covered_count == 0))
    
    # Simulated annealing parameters - tuned for better exploration
    T_start = 2.0
    T_end = 0.05
    cooling_rate = 0.997
    
    T = T_start
    iteration = 0
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 15:
        iteration += 1
        
        # Try a move: remove a random codeword or add a random codeword
        move_type = random.random()
        
        if move_type < 0.55 and len(code_ints) > 0:
            # Try removing a codeword
            idx = random.randint(0, len(code_ints) - 1)
            c_int = code_ints[idx]
            
            # Temporarily remove
            covered_count[all_neighbors[c_int]] -= 1
            
            new_num = len(code_ints) - 1
            new_energy = compute_energy(covered_count, new_num)
            
            # Accept if better, or with probability exp(-(new_energy - current_energy)/T)
            delta_e = new_energy - current_energy
            if delta_e < 0 or (T > 0 and random.random() < np.exp(-delta_e / T)):
                code_ints.pop(idx)
                code_set.remove(c_int)
                current_energy = new_energy
                
                new_uncovered = int(np.sum(covered_count == 0))
                if current_energy < best_energy or (current_energy == best_energy and new_uncovered < best_uncovered):
                    best_energy = current_energy
                    best_uncovered = new_uncovered
                    best_code_ints = list(code_ints)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Roll back
                covered_count[all_neighbors[c_int]] += 1
        
        elif move_type < 0.75:
            # Try adding a codeword - prefer words that cover uncovered areas
            # Sample from uncovered words or their neighbors for better moves
            uncovered_mask = covered_count == 0
            if np.any(uncovered_mask):
                uncovered_indices = np.where(uncovered_mask)[0]
                if random.random() < 0.7:
                    # Pick a word near an uncovered word
                    target = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
                    new_int = all_neighbors[target][random.randint(0, len(all_neighbors[target]) - 1)]
                else:
                    new_int = random.randint(0, total_words - 1)
            else:
                new_int = random.randint(0, total_words - 1)
            
            if new_int in code_set:
                continue
            
            # Temporarily add
            covered_count[all_neighbors[new_int]] += 1
            
            new_num = len(code_ints) + 1
            new_energy = compute_energy(covered_count, new_num)
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or (T > 0 and random.random() < np.exp(-delta_e / T)):
                code_ints.append(new_int)
                code_set.add(new_int)
                current_energy = new_energy
                
                new_uncovered = int(np.sum(covered_count == 0))
                if current_energy < best_energy or (current_energy == best_energy and new_uncovered < best_uncovered):
                    best_energy = current_energy
                    best_uncovered = new_uncovered
                    best_code_ints = list(code_ints)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Roll back
                covered_count[all_neighbors[new_int]] -= 1
        
        else:
            # Try swap: remove one, add another
            if len(code_ints) == 0:
                continue
                
            remove_idx = random.randint(0, len(code_ints) - 1)
            remove_int = code_ints[remove_idx]
            
            # For swap, try to add a word that covers uncovered areas
            uncovered_mask = covered_count == 0
            if np.any(uncovered_mask):
                uncovered_indices = np.where(uncovered_mask)[0]
                if random.random() < 0.6:
                    target = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
                    add_int = all_neighbors[target][random.randint(0, len(all_neighbors[target]) - 1)]
                else:
                    add_int = random.randint(0, total_words - 1)
            else:
                add_int = random.randint(0, total_words - 1)
            
            if add_int in code_set:
                continue
            
            # Temporarily apply swap
            covered_count[all_neighbors[remove_int]] -= 1
            covered_count[all_neighbors[add_int]] += 1
            
            new_num = len(code_ints)
            new_energy = compute_energy(covered_count, new_num)
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or (T > 0 and random.random() < np.exp(-delta_e / T)):
                code_ints[remove_idx] = add_int
                code_set.remove(remove_int)
                code_set.add(add_int)
                current_energy = new_energy
                
                new_uncovered = int(np.sum(covered_count == 0))
                if current_energy < best_energy or (current_energy == best_energy and new_uncovered < best_uncovered):
                    best_energy = current_energy
                    best_uncovered = new_uncovered
                    best_code_ints = list(code_ints)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Roll back
                covered_count[all_neighbors[add_int]] -= 1
                covered_count[all_neighbors[remove_int]] += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        # Restart if stuck for too long
        if no_improve_count > max_no_improve:
            T = max(T_start * 0.5, T_end * 10)
            no_improve_count = 0
    
    # Use best solution found
    code_ints = best_code_ints
    
    # Final repair: ensure full coverage
    # Recompute coverage
    covered_count = np.zeros(total_words, dtype=np.int32)
    for c_int in code_ints:
        covered_count[all_neighbors[c_int]] += 1
    
    # Add codewords for uncovered words - prioritize words that cover most uncovered
    uncovered_words = [w for w in range(total_words) if covered_count[w] == 0]
    for w in uncovered_words:
        if time.time() < deadline - 8:
            code_ints.append(w)
            covered_count[all_neighbors[w]] += 1
    
    # Aggressive pruning pass - multiple rounds
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        removed_any = False
        for idx in range(len(code_ints)):
            c_int = code_ints[idx]
            # Check if removing this codeword would leave any word uncovered
            can_remove = True
            for nb in all_neighbors[c_int]:
                if covered_count[nb] <= 1:
                    can_remove = False
                    break
            if can_remove:
                covered_count[all_neighbors[c_int]] -= 1
                code_ints.pop(idx)
                removed_any = True
                break
        if not removed_any:
            break
    
    # Second pruning pass with different order
    for _ in range(100):
        if time.time() > deadline - 3:
            break
        removed_any = False
        for idx in range(len(code_ints) - 1, -1, -1):
            c_int = code_ints[idx]
            can_remove = True
            for nb in all_neighbors[c_int]:
                if covered_count[nb] <= 1:
                    can_remove = False
                    break
            if can_remove:
                covered_count[all_neighbors[c_int]] -= 1
                code_ints.pop(idx)
                removed_any = True
                break
        if not removed_any:
            break
    
    # Convert to list of words
    def int_to_word(val):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        word.reverse()
        return word
    
    return [int_to_word(c) for c in code_ints]
# EVOLVE-BLOCK-END
