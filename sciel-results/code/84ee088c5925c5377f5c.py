import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 105  # 15s margin before 120s limit
    
    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # For radius=1
    
    # Precompute word<->index conversions
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute balls as lists of indices
    balls = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                neighbors.append(word_to_idx(w))
            w[pos] = original
        balls.append(neighbors)
    
    # Precompute ball masks as Python integers (bitsets)
    ball_masks = []
    for c in range(total_words):
        mask = 0
        for nb in balls[c]:
            mask |= (1 << nb)
        ball_masks.append(mask)
    
    all_mask = (1 << total_words) - 1
    
    best_solution = None
    best_size = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        covered_mask = 0
        code = []
        
        while covered_mask != all_mask:
            best_gain = -1
            best_candidates = []
            
            for c in range(total_words):
                gain = (ball_masks[c] & ~covered_mask).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            covered_mask |= ball_masks[chosen]
            
            if time.time() > deadline:
                return None
        
        return code
    
    def reduce_cover(code):
        if not code:
            return code
            
        cover_count = [0] * total_words
        
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    changed = True
        
        return code
    
    def local_search(code):
        """
        Try to reduce size by removing redundant codewords and swapping.
        """
        if not code:
            return code
            
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
                
        in_code = [False] * total_words
        for c in code:
            in_code[c] = True
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # First pass: remove redundant codewords
            for i in range(len(code)):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code.pop(i)
                    improved = True
                    break  # Restart loop
            
            if improved:
                continue
            
            # Second pass: try swaps - for each codeword, find a replacement
            # that covers all its unique words, then check if it enables further reduction
            for i in range(len(code)):
                c = code[i]
                # Find words uniquely covered by c
                unique_words = []
                for w in balls[c]:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                
                if not unique_words:
                    continue
                
                # Build a bitmask of unique words for fast checking
                unique_mask = 0
                for w in unique_words:
                    unique_mask |= (1 << w)
                
                # Find a c' not in code that covers all unique_words
                best_c_prime = -1
                for c_prime in range(total_words):
                    if in_code[c_prime]:
                        continue
                    if (ball_masks[c_prime] & unique_mask) == unique_mask:
                        best_c_prime = c_prime
                        break
                
                if best_c_prime != -1:
                    # Perform swap
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code[i] = best_c_prime
                    
                    for w in balls[best_c_prime]:
                        cover_count[w] += 1
                    in_code[best_c_prime] = True
                    
                    # Check if this enabled a reduction
                    reduced = False
                    for j in range(len(code)):
                        cj = code[j]
                        redundant = True
                        for w in balls[cj]:
                            if cover_count[w] == 1:
                                redundant = False
                                break
                        if redundant:
                            for w in balls[cj]:
                                cover_count[w] -= 1
                            in_code[cj] = False
                            code.pop(j)
                            reduced = True
                            break
                    
                    if reduced:
                        improved = True
                        break  # Restart main while loop
                    else:
                        # Revert swap
                        for w in balls[best_c_prime]:
                            cover_count[w] -= 1
                        in_code[best_c_prime] = False
                        code[i] = c
                        for w in balls[c]:
                            cover_count[w] += 1
                        in_code[c] = True
                        
        return code
    
    def two_opt(code):
        """
        More aggressive local search: try replacing pairs of codewords with one.
        """
        if len(code) < 2:
            return code
            
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
                
        in_code = [False] * total_words
        for c in code:
            in_code[c] = True
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Remove redundant
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-opt: remove two, add one that covers their unique words
            for i in range(len(code)):
                for j in range(i + 1, len(code)):
                    c1 = code[i]
                    c2 = code[j]
                    
                    # Find words uniquely covered by c1 or c2
                    unique_mask = 0
                    for w in balls[c1]:
                        if cover_count[w] == 1:
                            unique_mask |= (1 << w)
                    for w in balls[c2]:
                        if cover_count[w] == 1:
                            unique_mask |= (1 << w)
                    
                    if unique_mask == 0:
                        continue
                    
                    # Find a c' not in code that covers all unique_words
                    best_c_prime = -1
                    for c_prime in range(total_words):
                        if in_code[c_prime]:
                            continue
                        if (ball_masks[c_prime] & unique_mask) == unique_mask:
                            best_c_prime = c_prime
                            break
                    
                    if best_c_prime != -1:
                        # Perform 2-opt: remove c1, c2, add c_prime
                        for w in balls[c1]:
                            cover_count[w] -= 1
                        in_code[c1] = False
                        for w in balls[c2]:
                            cover_count[w] -= 1
                        in_code[c2] = False
                        code.remove(c1)
                        code.remove(c2)
                        code.append(best_c_prime)
                        
                        for w in balls[best_c_prime]:
                            cover_count[w] += 1
                        in_code[best_c_prime] = True
                        
                        # Check for reductions
                        reduced = False
                        for k in range(len(code)):
                            ck = code[k]
                            redundant = True
                            for w in balls[ck]:
                                if cover_count[w] == 1:
                                    redundant = False
                                    break
                            if redundant:
                                for w in balls[ck]:
                                    cover_count[w] -= 1
                                in_code[ck] = False
                                code.pop(k)
                                reduced = True
                                break
                        
                        improved = True
                        break  # Restart main while loop
                
                if improved:
                    break
            
        return code
    
    def simulated_annealing(code, max_iter=5000):
        """
        Simulated annealing on the binary selection vector.
        Energy = big_penalty * uncovered + code_size
        Moves: add, remove, swap
        """
        if not code:
            return code
            
        # Initialize state
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        in_code = [False] * total_words
        for c in code:
            in_code[c] = True
        
        def get_uncovered_count():
            count = 0
            for w in range(total_words):
                if cover_count[w] == 0:
                    count += 1
            return count
        
        def get_energy():
            return 1000 * get_uncovered_count() + len(code)
        
        current_energy = get_energy()
        best_code = list(code)
        best_energy = current_energy
        
        T = 10.0  # Initial temperature
        cooling = 0.995
        
        rng = random.Random()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Choose a random move
            move_type = rng.random()
            if move_type < 0.4:
                # Remove a random codeword
                if len(code) == 0:
                    continue
                idx = rng.randrange(len(code))
                c = code[idx]
                
                # Try removal
                for w in balls[c]:
                    cover_count[w] -= 1
                in_code[c] = False
                new_energy = get_energy()
                
                # Check if valid (no uncovered words)
                if new_energy < current_energy:
                    # Accept
                    code.pop(idx)
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_code = list(code)
                else:
                    # Revert
                    for w in balls[c]:
                        cover_count[w] += 1
                    in_code[c] = True
                    code.insert(idx, c)
                    
                    # Metropolis acceptance
                    delta = new_energy - current_energy
                    if delta < 0:
                        pass  # Already accepted above
                    else:
                        if rng.random() < np.exp(-delta / max(T, 1e-10)):
                            # Accept worse solution
                            code.pop(idx)
                            current_energy = new_energy
                            if new_energy < best_energy:
                                best_energy = new_energy
                                best_code = list(code)
                        else:
                            # Revert (already done)
                            pass
            
            elif move_type < 0.7:
                # Add a random codeword
                c = rng.randrange(total_words)
                if in_code[c]:
                    continue
                
                for w in balls[c]:
                    cover_count[w] += 1
                in_code[c] = True
                new_energy = get_energy()
                
                if new_energy < current_energy:
                    # Accept
                    code.append(c)
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_code = list(code)
                else:
                    # Revert
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    
                    # Metropolis acceptance
                    delta = new_energy - current_energy
                    if delta < 0:
                        pass
                    else:
                        if rng.random() < np.exp(-delta / max(T, 1e-10)):
                            # Accept worse solution
                            code.append(c)
                            current_energy = new_energy
                            if new_energy < best_energy:
                                best_energy = new_energy
                                best_code = list(code)
                        else:
                            # Revert (already done)
                            pass
            
            else:
                # Swap: remove one, add another
                if len(code) == 0:
                    continue
                idx = rng.randrange(len(code))
                c_old = code[idx]
                c_new = rng.randrange(total_words)
                if in_code[c_new] and c_new != c_old:
                    continue
                
                # Remove c_old
                for w in balls[c_old]:
                    cover_count[w] -= 1
                in_code[c_old] = False
                
                # Add c_new
                for w in balls[c_new]:
                    cover_count[w] += 1
                in_code[c_new] = True
                
                code[idx] = c_new
                new_energy = get_energy()
                
                if new_energy < current_energy:
                    # Accept
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_code = list(code)
                else:
                    # Revert
                    for w in balls[c_new]:
                        cover_count[w] -= 1
                    in_code[c_new] = False
                    for w in balls[c_old]:
                        cover_count[w] += 1
                    in_code[c_old] = True
                    code[idx] = c_old
                    
                    # Metropolis acceptance
                    delta = new_energy - current_energy
                    if delta < 0:
                        pass
                    else:
                        if rng.random() < np.exp(-delta / max(T, 1e-10)):
                            # Accept worse solution
                            for w in balls[c_old]:
                                cover_count[w] -= 1
                            in_code[c_old] = False
                            for w in balls[c_new]:
                                cover_count[w] += 1
                            in_code[c_new] = True
                            code[idx] = c_new
                            current_energy = new_energy
                            if new_energy < best_energy:
                                best_energy = new_energy
                                best_code = list(code)
                        else:
                            # Revert (already done)
                            pass
            
            T *= cooling
        
        return best_code
    
    # Main loop: try multiple random seeds for greedy, then local search
    initial_code = greedy_cover(0)
    if initial_code:
        reduced_code = reduce_cover(initial_code)
        local_searched_code = local_search(reduced_code)
        best_solution = local_searched_code
        best_size = len(local_searched_code)
    
    seed = 1
    while time.time() < deadline:
        code = greedy_cover(seed)
        if code is None:
            break
        reduced_code = reduce_cover(code)
        local_searched_code = local_search(reduced_code)
        if len(local_searched_code) < best_size:
            best_solution = local_searched_code
            best_size = len(local_searched_code)
        
        # Try 2-opt on the best solution found so far
        if seed % 3 == 0 and best_solution:
            two_opt_code = two_opt(list(best_solution))
            if len(two_opt_code) < best_size:
                best_solution = two_opt_code
                best_size = len(two_opt_code)
        
        # Try simulated annealing on the best solution found so far
        if seed % 5 == 0 and best_solution:
            sa_code = simulated_annealing(list(best_solution), max_iter=3000)
            # Final reduction after SA
            sa_reduced = reduce_cover(sa_code)
            if len(sa_reduced) < best_size:
                best_solution = sa_reduced
                best_size = len(sa_reduced)
        
        seed += 1
    
    if best_solution is None:
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result
