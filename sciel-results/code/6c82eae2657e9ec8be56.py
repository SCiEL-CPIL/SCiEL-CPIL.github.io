import time
import random
import itertools
import sys

# Set recursion limit just in case, though we don't use deep recursion
sys.setrecursionlimit(10000)

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    # The evaluator passes these, but the logic is specific to the structure.
    # We will implement a general solver for small q, n, radius=1 using bitmasks/arrays.
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to the tetracode product for other parameters if needed
        # Though the task is specific to 7 matches ternary
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Generalize tetracode product for n > 4
        if n > 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # If n <= 4, just return all words or a perfect code if available
            # For n=4, tetracode is perfect (9 words)
            if n == 4:
                return tetracode
            # Fallback
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific logic for q=3, n=7, R=1
    total_words = 3**7
    # Precompute neighbors for each word index
    # Word index -> list of indices at distance <= 1
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        # Convert i to base-3 digits
        w = []
        tmp = i
        for _ in range(n):
            w.append(tmp % 3)
            tmp //= 3
        w.reverse()
        
        # Add self
        neighbors[i].append(i)
        
        # Add distance 1 neighbors
        for pos in range(n):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = w[:]
                new_w[pos] = val
                # Convert back to index
                idx = 0
                for d in new_w:
                    idx = idx * 3 + d
                neighbors[i].append(idx)
    
    # Initial solution: Tetracode product
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_code.append(h + list(tail))
    
    # Convert to indices
    def to_idx(w):
        idx = 0
        for d in w:
            idx = idx * 3 + d
        return idx

    code_indices = [to_idx(w) for w in init_code]
    best_code = code_indices[:]
    best_size = len(best_code)
    
    # cover_count[i] = number of codewords in C that cover word i
    cover_count = [0] * total_words
    
    def update_cover_counts(code, delta=1):
        for c in code:
            for nb in neighbors[c]:
                cover_count[nb] += delta

    # Initialize cover counts
    update_cover_counts(code_indices, 1)
    
    def is_redundant(c, code_set):
        # Check if c can be removed
        # c is redundant if for all nb in neighbors[c], cover_count[nb] > 1
        # Note: cover_count includes c itself. So we need cover_count[nb] >= 2
        for nb in neighbors[c]:
            if cover_count[nb] < 2:
                return False
        return True

    # Greedy removal
    def greedy_remove(code_list):
        # Shuffle to break symmetry
        random.shuffle(code_list)
        current_code = code_list[:]
        # We need to maintain cover_count correctly.
        # Reset cover_count
        for i in range(total_words):
            cover_count[i] = 0
        update_cover_counts(current_code, 1)
        
        removed = False
        i = 0
        while i < len(current_code):
            c = current_code[i]
            if is_redundant(c, set(current_code)):
                # Remove c
                update_cover_counts([c], -1)
                current_code.pop(i)
                removed = True
            else:
                i += 1
        
        if removed:
            return current_code
        return current_code

    # Apply initial greedy
    best_code = greedy_remove(best_code)
    best_size = len(best_code)
    
    # Local Search
    # Strategy: 
    # 1. Try to remove a codeword c.
    # 2. If not possible, identify the "blocking" words (those with cover_count == 1).
    # 3. Try to find a replacement c' that covers the blocking words.
    # 4. If c' covers all blocking words, swap c with c'.
    # 5. After swap, try to remove another codeword that became redundant.
    
    def try_improve(code_list):
        # Reset cover counts
        for i in range(total_words):
            cover_count[i] = 0
        update_cover_counts(code_list, 1)
        
        current_code = code_list[:]
        improved = False
        
        # Randomize order of attempts
        indices = list(range(len(current_code)))
        random.shuffle(indices)
        
        for idx in indices:
            c = current_code[idx]
            
            # Find words that would be uncovered if c is removed
            # These are nb in neighbors[c] where cover_count[nb] == 1
            critical_words = []
            for nb in neighbors[c]:
                if cover_count[nb] == 1:
                    critical_words.append(nb)
            
            if not critical_words:
                # c is redundant
                update_cover_counts([c], -1)
                current_code.pop(idx)
                improved = True
                # Continue trying to remove more from the new list? 
                # Let's just return the improved list and let the outer loop call greedy again
                # Or continue scanning? 
                # Scanning with modified list is tricky with indices. 
                # Let's restart the greedy phase on the new list.
                break
            
            # Try to find a replacement c' that covers all critical_words
            # c' must be in {0..2}^7
            # We can search candidates:
            # 1. Neighbors of critical words?
            # 2. Random words?
            
            # A good candidate set: 
            # The intersection of neighbors of all critical words?
            # If |critical_words| is small, this is fast.
            
            # Let's try a limited search for c'
            # Candidates: 
            # - The critical words themselves (if we place a codeword AT a critical word, it covers it)
            # - Neighbors of critical words
            
            candidate_set = set()
            for w in critical_words:
                candidate_set.add(w)
                for nb in neighbors[w]:
                    candidate_set.add(nb)
            
            # Also add some random candidates
            for _ in range(10):
                candidate_set.add(random.randint(0, total_words - 1))
            
            best_candidate = None
            best_gain = -1 # Number of critical words covered
            
            for c_prime in candidate_set:
                # Check if c_prime covers all critical_words
                # c_prime covers w if w in neighbors[c_prime]
                # Or distance(c_prime, w) <= 1
                
                # Optimization: Precomputed neighbors
                # We need to check if critical_words is a subset of neighbors[c_prime]
                covered_critical = 0
                for w in critical_words:
                    if w in neighbors[c_prime]: # This is O(1) if we use sets or bitsets, but list is O(15). 
                        # Actually, checking membership in a list of 15 is fast.
                        covered_critical += 1
                
                if covered_critical == len(critical_words):
                    # Check if c_prime is already in current_code
                    if c_prime not in current_code:
                        # Potential swap
                        # Gain: We cover the critical words.
                        # Cost: We add c_prime.
                        # Net size change: 0 (remove c, add c_prime)
                        # But maybe c_prime makes another codeword redundant?
                        
                        # Let's simulate the swap
                        # 1. Remove c
                        # 2. Add c_prime
                        # 3. Check if any other codeword is redundant
                        
                        # To do this efficiently, we need to update cover counts.
                        # This is expensive inside the loop.
                        
                        # Heuristic: Just accept the swap if it seems good?
                        # Or just try the swap and then run a quick greedy remove.
                        
                        best_candidate = c_prime
                        # We found a valid swap. Let's take it.
                        break
            
            if best_candidate is not None:
                # Perform swap
                c_prime = best_candidate
                # Remove c
                update_cover_counts([c], -1)
                current_code.pop(idx)
                # Add c_prime
                update_cover_counts([c_prime], 1)
                current_code.append(c_prime)
                improved = True
                # Break and restart greedy on new state
                break
        
        if improved:
            # Run greedy remove on the new code to clean up
            return greedy_remove(current_code)
        
        return current_code

    # Main Search Loop
    current_code = best_code[:]
    
    while time.time() - start_time < time_limit:
        # Try to improve
        new_code = try_improve(current_code)
        
        if len(new_code) < len(current_code):
            current_code = new_code
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code[:]
        
        # Random restart if stuck?
        # If no improvement in last N iterations, restart from best or random?
        # For now, just keep trying.
        
        # Occasionally, shuffle the current code to escape local minima in try_improve
        # try_improve already shuffles the order of checks.
        
        # If we get stuck, we can try a "kick":
        # Remove 2 random codewords and try to recover.
        # This is complex. Let's rely on the swap heuristic.
        
        # If time is running low, break
        if time.time() - start_time > time_limit - 5:
            break

    # Convert best_code indices back to list of lists
    result = []
    for idx in best_code:
        w = []
        tmp = idx
        for _ in range(n):
            w.append(tmp % 3)
            tmp //= 3
        w.reverse()
        result.append(w)
        
    return result
