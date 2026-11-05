import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # Problem parameters
    # q=3, n=8, radius=1
    # Total words = 3^8 = 6561
    # We represent words as integers 0..6560 in base 3
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters, though task specifies this one
        # Simple greedy for general case
        return _general_greedy(q, n, radius, start_time, timeout)

    # Specific fast path for q=3, n=8, r=1
    return _ternary_8_1_solver(start_time, timeout)

def _ternary_8_1_solver(start_time, timeout):
    q = 3
    n = 8
    radius = 1
    total_words = q ** n  # 6561
    
    # Precompute neighbors for each word in the space
    # neighbor[word] = list of words within Hamming distance 1 (including self)
    # There are 1 + n*(q-1) = 1 + 8*2 = 17 neighbors per word.
    
    # To speed up, we can generate neighbors on the fly or precompute.
    # Precomputation is cheap (6561 * 17 ops).
    neighbors = [[] for _ in range(total_words)]
    
    # Generate all words in base 3 representation
    # We need to map integer index to digits to find neighbors
    # Or better, iterate through all words and calculate neighbors by modifying digits
    
    # Let's create a helper to get digits
    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits
        
    # Precompute neighbors
    for i in range(total_words):
        digits = get_digits(i)
        # Self
        neighbors[i].append(i)
        # Mutations
        for pos in range(n):
            original = digits[pos]
            for val in range(q):
                if val == original: continue
                new_digits = digits[:]
                new_digits[pos] = val
                # Convert back to index
                idx = 0
                multiplier = 1
                for d in new_digits:
                    idx += d * multiplier
                    multiplier *= q
                neighbors[i].append(idx)
    # Note: The above conversion is slow. Let's optimize neighbor generation.
    
    # Optimized Neighbor Generation
    neighbors = [[] for _ in range(total_words)]
    # Powers of 3 for index calculation
    powers = [q ** i for i in range(n)]
    
    for i in range(total_words):
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % q
            temp //= q
            
        neighbors[i].append(i)
        
        for pos in range(n):
            orig = digits[pos]
            # For ternary, the other two values are (orig+1)%3 and (orig+2)%3
            # We can calculate index difference directly
            # delta = (new_val - orig) * powers[pos]
            for k in range(1, q):
                new_val = (orig + k) % q
                delta = (new_val - orig) * powers[pos]
                neighbors[i].append(i + delta)
                
    # Coverage tracking
    # covered_count[w] = number of codewords covering w
    covered_count = np.zeros(total_words, dtype=np.int32)
    covered_total = 0
    
    code = []
    code_set = set()
    
    # 1. Greedy Construction
    # Start with empty code
    # We want to add the word that covers the most uncovered words
    
    # To make greedy fast, we can use a heuristic:
    # Just pick random words? No, greedy is better.
    # But full greedy (finding max coverage each step) is O(M * N) where M is steps, N is total words.
    # M ~ 486, N ~ 6561. 486 * 6561 ~ 3M ops. Very fast.
    
    # However, we can speed up by maintaining a "potential" score for each candidate.
    # But simple greedy is likely fast enough.
    
    # Let's use a simpler initialization:
    # Start with a known good structure?
    # The seed used Tetracode product. Size 729.
    # Let's start with that, it's a valid code.
    
    # Generate Tetracode product code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in range(3 ** 4):
            # Convert tail to 4 digits
            d0 = tail % 3
            d1 = (tail // 3) % 3
            d2 = (tail // 9) % 3
            d3 = (tail // 27) % 3
            word_digits = h + [d0, d1, d2, d3]
            idx = 0
            for i in range(n):
                idx += word_digits[i] * powers[i]
            initial_code.append(idx)
            
    # Initialize coverage with this code
    for w in initial_code:
        for nb in neighbors[w]:
            covered_count[nb] += 1
            
    covered_total = np.sum(covered_count > 0)
    code = initial_code
    code_set = set(code)
    
    # 2. Local Search: Remove and Swap
    # Goal: Minimize len(code)
    
    # Helper: try to remove word w
    def try_remove(w):
        nonlocal covered_total
        # Check if any neighbor becomes uncovered
        # If min(covered_count[nb] for nb in neighbors[w]) > 1, safe to remove
        # If any becomes 0, unsafe.
        # We modify covered_count, check, and revert if needed.
        
        # Optimization: Precompute min count? No, dynamic.
        # Just check neighbors.
        for nb in neighbors[w]:
            if covered_count[nb] == 1:
                # This word would become uncovered
                return False
        
        # Safe to remove
        for nb in neighbors[w]:
            covered_count[nb] -= 1
        covered_total -= 0 # covered_total only changes if a word goes from covered to uncovered, which we checked doesn't happen
        # Wait, covered_total is count of words with count > 0.
        # Since no count went to 0, covered_total remains unchanged.
        return True

    # Helper: try to add word w (assuming not in code)
    def try_add(w):
        nonlocal covered_total
        newly_covered = 0
        for nb in neighbors[w]:
            if covered_count[nb] == 0:
                newly_covered += 1
            covered_count[nb] += 1
        covered_total += newly_covered
        return True

    # Helper: try to swap u (in code) with v (not in code)
    # Returns True if swap is valid (coverage maintained)
    def try_swap(u, v):
        nonlocal covered_total
        
        # 1. Remove u
        # Check if removing u breaks coverage
        # We need to know if any neighbor of u has count 1 AND is NOT covered by v.
        # Actually, simpler:
        # Temporarily remove u's coverage.
        # Then check if v covers all the holes created.
        
        # Let's do it step-by-step to be safe and simple, relying on the small number of neighbors (17).
        
        # Step 1: Remove u
        for nb in neighbors[u]:
            covered_count[nb] -= 1
            
        # Step 2: Check if any word is now uncovered (count == 0)
        # If yes, we must add v. But adding v might not cover all holes.
        # Let's collect holes.
        holes = []
        for nb in neighbors[u]:
            if covered_count[nb] == 0:
                holes.append(nb)
                
        if not holes:
            # No holes, u was redundant. We can just remove u.
            # But we want to swap to explore. 
            # If u is redundant, we should just remove it.
            # The swap logic here is for moving the code.
            # If no holes, removing u is strictly better (size -1).
            # We handle removal separately.
            # For swap, we assume u is NOT redundant, so there are holes.
            # If no holes, we revert and return False for swap (let remove handle it).
            for nb in neighbors[u]:
                covered_count[nb] += 1
            return False

        # Step 3: Add v
        # Check if v covers all holes.
        # Also, adding v increases coverage for its other neighbors (which is fine).
        
        # Check holes coverage
        for h in holes:
            if h not in neighbors[v]:
                # v doesn't cover this hole
                # Revert
                for nb in neighbors[u]:
                    covered_count[nb] += 1
                return False
        
        # If we reach here, v covers all holes created by removing u.
        # Execute the swap:
        # u is already removed from counts.
        # Add v to counts.
        for nb in neighbors[v]:
            covered_count[nb] += 1
            
        # Update code
        code.remove(u)
        code_set.remove(u)
        code.append(v)
        code_set.add(v)
        
        return True

    # Main Loop
    # Randomize order to escape symmetries
    random.seed(42)
    
    # First, do a pass of removals
    # Shuffle code to avoid bias
    while time.time() - start_time < timeout:
        # 1. Removal Phase
        removed_any = False
        # Iterate through a random permutation of the code
        current_code_copy = code[:]
        random.shuffle(current_code_copy)
        
        for w in current_code_copy:
            if time.time() - start_time > timeout:
                break
            if w not in code_set:
                continue
            if try_remove(w):
                code.remove(w)
                code_set.remove(w)
                removed_any = True
        
        if not removed_any:
            # 2. Swap Phase
            swapped_any = False
            # Try a few random swaps
            # We need u in code, v not in code.
            # To speed up, we can pick random u and random v from the space.
            # Probability v is in code is small (486/6561 ~ 7%).
            
            attempts = 0
            max_attempts = 1000 # Tune this
            
            while attempts < max_attempts:
                if time.time() - start_time > timeout:
                    break
                attempts += 1
                
                u = random.choice(code)
                v = random.randint(0, total_words - 1)
                
                if v in code_set:
                    continue
                
                if try_swap(u, v):
                    swapped_any = True
                    # After a swap, we might have created a redundant word.
                    # Go back to removal phase immediately.
                    break
            
            if not swapped_any:
                # Stuck? Try a more aggressive move?
                # For now, if no swap works in 1000 tries, we might be in a local minimum.
                # We can try to restart with a different random seed or perturb.
                # But given time constraints, we just keep trying.
                # Maybe increase attempts?
                # Or break if we've been stuck for a while?
                # Let's just keep looping, it's cheap.
                pass

    # Convert code back to lists of integers
    result = []
    for idx in code:
        digits = [0] * n
        temp = idx
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        result.append(digits)
        
    return result

def _general_greedy(q, n, radius, start_time, timeout):
    # Fallback for other parameters
    # Simple greedy: add word covering most uncovered
    total = q ** n
    powers = [q ** i for i in range(n)]
    
    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Precompute neighbors for general q, n, radius=1
    neighbors = [[] for _ in range(total)]
    for i in range(total):
        digits = get_digits(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig = digits[pos]
            for val in range(q):
                if val == orig: continue
                delta = (val - orig) * powers[pos]
                neighbors[i].append(i + delta)
                
    covered_count = np.zeros(total, dtype=np.int32)
    code = []
    code_set = set()
    
    # Greedy
    while time.time() - start_time < timeout:
        best_word = -1
        best_gain = 0
        # Sample a subset of candidates to speed up
        candidates = random.sample(range(total), min(100, total))
        for w in candidates:
            if w in code_set: continue
            gain = 0
            for nb in neighbors[w]:
                if covered_count[nb] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = w
        
        if best_word == -1 or best_gain == 0:
            break
            
        for nb in neighbors[best_word]:
            covered_count[nb] += 1
        code.append(best_word)
        code_set.add(best_word)
        
    # Reduction
    while time.time() - start_time < timeout:
        removed = False
        for w in code[:]:
            can_remove = True
            for nb in neighbors[w]:
                if covered_count[nb] == 1:
                    can_remove = False
                    break
            if can_remove:
                for nb in neighbors[w]:
                    covered_count[nb] -= 1
                code.remove(w)
                code_set.remove(w)
                removed = True
        if not removed:
            break
            
    result = []
    for idx in code:
        digits = get_digits(idx)
        result.append(digits)
    return result
