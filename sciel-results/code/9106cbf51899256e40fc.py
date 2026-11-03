import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-(3,6,1) cases
    def construct_tetracode(q, n, radius):
        if q != 3:
            # Simple greedy for general case
            all_words = list(itertools.product(range(q), repeat=n))
            covered = [False] * (q ** n)
            word_to_idx = {w: i for i, w in enumerate(all_words)}
            code = []
            for w in all_words:
                if not covered[word_to_idx[w]]:
                    code.append(list(w))
                    # Mark all words within radius
                    for i in range(n):
                        for d in range(1, q):
                            nw = list(w)
                            nw[i] = (nw[i] + d) % q
                            # Only mark if within radius (simplified for radius=1)
                            if radius >= 1:
                                covered[word_to_idx[tuple(nw)]] = True
                    covered[word_to_idx[w]] = True
            return code
        
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n <= 4:
            return [h[:n] for h in tetracode]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if q != 3 or n != 6 or radius != 1:
        return construct_tetracode(q, n, radius)

    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precompute all words and their indices
    all_space = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_space)}
    N = 729

    # Precompute spheres for each word
    def get_sphere(w):
        w = tuple(w)
        sphere = [w]
        for i in range(6):
            for d in [-1, 1]:
                nc = list(w)
                nc[i] = (nc[i] + d) % 3
                sphere.append(tuple(nc))
        return sphere

    # Precompute all spheres
    spheres = [get_sphere(w) for w in all_space]

    # Initial code: direct-sum construction (81 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Verify initial coverage
    cover_count = np.zeros(N, dtype=np.int32)
    for c in code:
        for w in spheres[word_to_idx[tuple(c)]]:
            cover_count[word_to_idx[w]] += 1
    
    if not np.all(cover_count > 0):
        # Should not happen, but fallback
        return code

    # Phase 1: Greedy removal
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 10.0:
            break
        for i in range(len(code)):
            if time.time() > deadline - 5.0:
                break
            c = tuple(code[i])
            # Check if removing c leaves all words covered
            can_remove = True
            for w in spheres[word_to_idx[c]]:
                if cover_count[word_to_idx[w]] == 1:
                    can_remove = False
                    break
            if can_remove:
                for w in spheres[word_to_idx[c]]:
                    cover_count[word_to_idx[w]] -= 1
                code.pop(i)
                changed = True
                break

    # Phase 2: Simulated Annealing with swap and remove operations
    # State: set of codewords, cover_count array
    # Energy: number of codewords (to minimize)
    
    best_code = list(code)
    best_size = len(code)
    
    # Simulated Annealing parameters
    T_initial = 5.0
    T_final = 0.1
    cooling_rate = 0.995
    max_iterations = 50000
    
    current_code = list(code)
    current_cover = cover_count.copy()
    current_size = len(current_code)
    
    T = T_initial
    iteration = 0
    
    while T > T_final and iteration < max_iterations:
        if time.time() > deadline - 5.0:
            break
        
        # Choose a random operation: swap, remove, or add
        op = random.random()
        
        if op < 0.7 and len(current_code) > 0:
            # Swap: replace one codeword with another
            i = random.randint(0, len(current_code) - 1)
            old_c = tuple(current_code[i])
            
            # Try to find a new codeword that maintains coverage
            # Words only covered by old_c must be covered by new_c
            critical_words = []
            for w in spheres[word_to_idx[old_c]]:
                if current_cover[word_to_idx[w]] == 1:
                    critical_words.append(w)
            
            if not critical_words:
                # Can just remove
                for w in spheres[word_to_idx[old_c]]:
                    current_cover[word_to_idx[w]] -= 1
                current_code.pop(i)
                current_size -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best_code = list(current_code)
                
                T *= cooling_rate
                iteration += 1
                continue
            
            # Find a candidate new codeword that covers all critical words
            candidates = []
            for w in all_space:
                if w == old_c:
                    continue
                # Check if w covers all critical words
                sphere_w = set(spheres[word_to_idx[w]])
                if all(cw in sphere_w for cw in critical_words):
                    candidates.append(w)
            
            if not candidates:
                T *= cooling_rate
                iteration += 1
                continue
            
            new_c = tuple(random.choice(candidates))
            
            # Apply swap
            for w in spheres[word_to_idx[old_c]]:
                current_cover[word_to_idx[w]] -= 1
            for w in spheres[word_to_idx[new_c]]:
                current_cover[word_to_idx[w]] += 1
            
            # Verify no word becomes uncovered
            valid = True
            for w in all_space:
                if current_cover[word_to_idx[w]] < 0:
                    valid = False
                    break
            
            if valid:
                current_code[i] = list(new_c)
                # Try to remove more codewords after swap
                for j in range(len(current_code)):
                    if time.time() > deadline - 5.0:
                        break
                    cj = tuple(current_code[j])
                    can_remove = True
                    for w in spheres[word_to_idx[cj]]:
                        if current_cover[word_to_idx[w]] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for w in spheres[word_to_idx[cj]]:
                            current_cover[word_to_idx[w]] -= 1
                        current_code.pop(j)
                        current_size -= 1
                        if current_size < best_size:
                            best_size = current_size
                            best_code = list(current_code)
                        break
                
                T *= cooling_rate
                iteration += 1
            else:
                # Rollback
                for w in spheres[word_to_idx[new_c]]:
                    current_cover[word_to_idx[w]] -= 1
                for w in spheres[word_to_idx[old_c]]:
                    current_cover[word_to_idx[w]] += 1
                current_code[i] = list(old_c)
                T *= cooling_rate
                iteration += 1
        
        elif op < 0.85 and len(current_code) > 0:
            # Remove a codeword
            i = random.randint(0, len(current_code) - 1)
            c = tuple(current_code[i])
            
            can_remove = True
            for w in spheres[word_to_idx[c]]:
                if current_cover[word_to_idx[w]] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                for w in spheres[word_to_idx[c]]:
                    current_cover[word_to_idx[w]] -= 1
                current_code.pop(i)
                current_size -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best_code = list(current_code)
                
                T *= cooling_rate
                iteration += 1
            else:
                T *= cooling_rate
                iteration += 1
        
        else:
            # Add a codeword (to escape local minima)
            if len(current_code) < 100:
                # Find an uncovered word or a word with low coverage
                min_cover = min(current_cover)
                if min_cover > 0:
                    # All words covered, add a random word to explore
                    new_c = tuple(random.choice(all_space))
                    for w in spheres[word_to_idx[new_c]]:
                        current_cover[word_to_idx[w]] += 1
                    current_code.append(list(new_c))
                    current_size += 1
                    
                    T *= cooling_rate
                    iteration += 1
                else:
                    # Find an uncovered word
                    uncovered = [i for i in range(N) if current_cover[i] == 0]
                    if uncovered:
                        idx = random.choice(uncovered)
                        new_c = all_space[idx]
                        for w in spheres[word_to_idx[new_c]]:
                            current_cover[word_to_idx[w]] += 1
                        current_code.append(list(new_c))
                        current_size += 1
                        
                        T *= cooling_rate
                        iteration += 1
                    else:
                        T *= cooling_rate
                        iteration += 1
            else:
                T *= cooling_rate
                iteration += 1

    # Phase 3: Final greedy removal pass on best_code
    if time.time() < deadline - 2.0:
        cover_count = np.zeros(N, dtype=np.int32)
        for c in best_code:
            for w in spheres[word_to_idx[tuple(c)]]:
                cover_count[word_to_idx[w]] += 1
        
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 1.0:
                break
            for i in range(len(best_code)):
                if time.time() > deadline - 1.0:
                    break
                c = tuple(best_code[i])
                can_remove = True
                for w in spheres[word_to_idx[c]]:
                    if cover_count[word_to_idx[w]] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for w in spheres[word_to_idx[c]]:
                        cover_count[word_to_idx[w]] -= 1
                    best_code.pop(i)
                    changed = True
                    break
    
    return best_code
