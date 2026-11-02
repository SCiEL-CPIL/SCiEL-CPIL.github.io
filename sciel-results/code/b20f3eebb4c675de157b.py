import random
import time
import itertools

def construct(q, n, radius):
    # We assume q=3, n=8, radius=1 based on the task description.
    # However, the function signature takes q, n, radius.
    # We should handle the general case or at least the specific case robustly.
    # Given the 120s limit, we should focus on the specific case q=3, n=8, R=1.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed-like behavior or simple greedy for other cases
        # But the task is specific to this instance.
        # For safety, if it's not the target case, return a valid code (maybe larger)
        # using a simple product construction if possible, or random.
        # But the prompt implies this function is called for this specific task.
        raise ValueError("Only q=3, n=8, radius=1 is supported")
    total_words = q ** n
    # Map word (tuple of ints) to integer index
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Pre-compute neighbors for each word in the space
    # neighbor[w] = list of words at distance <= 1 from w
    # Since q=3, n=8, total 6561 words.
    # Sphere size = 1 + n*(q-1) = 1 + 8*2 = 17.
    
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = to_word(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig = w[pos]
            for v in range(q):
                if v == orig:
                    continue
                w[pos] = v
                neighbors[i].append(to_int(w))
            w[pos] = orig

    # Greedy minimization function
    def minimize(code_set):
        # code_set is a set of integers
        # cov[w] = number of codewords in code_set that cover w
        cov = [0] * total_words
        
        # Initialize coverage
        for c in code_set:
            for w in neighbors[c]:
                cov[w] += 1
        
        # Check if all covered
        for w in range(total_words):
            if cov[w] == 0:
                # Not covered, need to add words. 
                # This function assumes input is a valid cover.
                # If not, we handle it outside.
                pass
        
        # Iteratively remove redundant codewords
        changed = True
        while changed:
            changed = False
            # Find a removable codeword
            # A codeword c is removable if for all w in neighbors[c], cov[w] > 1
            # We can pick one randomly from the removable ones to escape local minima
            removable = []
            for c in code_set:
                removable_flag = True
                for w in neighbors[c]:
                    if cov[w] <= 1:
                        removable_flag = False
                        break
                if removable_flag:
                    removable.append(c)
            
            if removable:
                # Pick one to remove
                c_to_remove = random.choice(removable)
                code_set.remove(c_to_remove)
                # Update coverage
                for w in neighbors[c_to_remove]:
                    cov[w] -= 1
                changed = True
            else:
                # No single removal possible. 
                # We could try swaps, but let's keep it simple first.
                # If we are stuck, we might be in a local minimum.
                break
                
        return code_set

    # Main Logic
    # 1. Try to find a valid cover of size 486
    # 2. If not, find a valid cover and minimize.
    
    # Strategy: 
    # Start with a random set of 486 words.
    # Check coverage.
    # If holes exist, add words to cover them.
    # Then minimize.
    # Repeat a few times with different random seeds.
    
    best_code = None
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Initial guess: The known optimal size is 486.
    # Let's try to construct a cover of size 486.
    
    # To ensure we don't waste time, we can use a heuristic initialization.
    # The seed code (729) is a valid cover.
    # We can start with the seed code and try to reduce it.
    # But reducing from 729 to 486 might be hard with simple removal.
    # Better: Start with random 486, fix, minimize.
    
    for attempt in range(20): # Try a few times
        if time.time() > deadline:
            break
            
        # Generate random set of 486 words
        # Using random.sample is fast
        target_size = 486
        random_set = set(random.sample(range(total_words), target_size))
        
        # Check coverage and add missing words
        cov = [0] * total_words
        for c in random_set:
            for w in neighbors[c]:
                cov[w] += 1
        
        # Find uncovered words
        uncovered = [w for w in range(total_words) if cov[w] == 0]
        
        # Add words to cover uncovered words
        # Greedy: pick a word that covers the most uncovered words
        while uncovered:
            if time.time() > deadline:
                break
            # Pick a random uncovered word
            w = random.choice(uncovered)
            # Find a codeword that covers w. 
            # Actually, we should pick a codeword that covers MANY uncovered words.
            # For each candidate c in neighbors[w], count how many uncovered words it covers.
            best_c = -1
            best_count = -1
            for c in neighbors[w]:
                count = 0
                for u in neighbors[c]:
                    if cov[u] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_c = c
            
            if best_c != -1:
                random_set.add(best_c)
                # Update coverage
                for u in neighbors[best_c]:
                    cov[u] += 1
                # Update uncovered list (slow if done naively, but n is small)
                # We can just re-calculate uncovered or update it
                # For efficiency, let's just re-calculate if the set is small
                # Or maintain a list
                # Since we are adding, we can just remove from uncovered if covered
                # But cov was updated, so:
                new_uncovered = []
                for u in uncovered:
                    if cov[u] == 0:
                        new_uncovered.append(u)
                uncovered = new_uncovered
            
        # Now random_set is a valid cover.
        # Minimize it
        min_code = minimize(random_set)
        
        if best_code is None or len(min_code) < len(best_code):
            best_code = min_code
            
        # If we reached 486, we can stop early? 
        # 486 is the known lower bound? No, 403 is lower bound. 486 is best known.
        # If we get 486, it's optimal (as far as we know).
        if len(best_code) <= 486:
            break

    # If we didn't find a good code, fall back to a known valid code (seed)
    if best_code is None:
        # Use seed construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Convert to ints
        tc_ints = set()
        for h in tetracode:
            val = 0
            for i in range(4):
                val = val * 3 + h[i]
            # Extend to 8: h + tail
            # We need to iterate over all tails
            # This is 9 * 81 = 729 words
            pass
        
        # Simpler fallback: just return the seed code directly
        seed_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=4):
                word = h + list(tail)
                val = 0
                for i in range(8):
                    val = val * 3 + word[i]
                seed_code.append(val)
        best_code = set(seed_code)
        
    # Convert best_code (set of ints) back to list of lists
    result = [to_word(c) for c in best_code]
    return result
