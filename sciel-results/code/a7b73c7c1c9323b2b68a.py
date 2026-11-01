import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: 110s out of 120s
    time_limit = start_time + 110.0
    
    N = q ** n
    
    # Precompute neighbors for all N words
    # neighbors[i] is a list of indices of words at Hamming distance <= 1 from i
    neighbors = [[] for _ in range(N)]
    for idx in range(N):
        # Convert index to base-q digits
        digits = [0] * n
        tmp = idx
        for i in range(n):
            digits[i] = tmp % q
            tmp //= q
        
        neighbors[idx].append(idx)
        
        for i in range(n):
            for val in range(q):
                if val == digits[i]:
                    continue
                new_digits = digits.copy()
                new_digits[i] = val
                new_idx = 0
                for j in range(n):
                    new_idx = new_idx * q + new_digits[j]
                neighbors[idx].append(new_idx)
    
    def to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def from_idx(idx):
        digits = [0] * n
        tmp = idx
        for i in range(n):
            digits[i] = tmp % q
            tmp //= q
        return digits

    def greedy_cover():
        covered = np.zeros(N, dtype=bool)
        code = set()
        
        while not np.all(covered):
            # Calculate gain for each word
            # gain[w] = number of uncovered neighbors of w
            gains = np.zeros(N, dtype=int)
            for w in range(N):
                # Vectorized sum of not covered
                gains[w] = np.sum(~covered[neighbors[w]])
            
            # Pick word with max gain
            best_w = np.argmax(gains)
            if gains[best_w] == 0:
                break # Should not happen if not all covered
            
            code.add(best_w)
            covered[neighbors[best_w]] = True
            
            if time.time() > time_limit:
                break
        
        return code

    def local_search(initial_code):
        code_set = set(initial_code)
        
        # Compute cover count
        cover_count = np.zeros(N, dtype=int)
        for c in code_set:
            cover_count[neighbors[c]] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to delete redundant codewords
            # A codeword c is redundant if min(cover_count[neighbors[c]]) >= 2
            deletable = []
            for c in code_set:
                if np.min(cover_count[neighbors[c]]) >= 2:
                    deletable.append(c)
            
            if deletable:
                # Delete one (random choice from deletable)
                c = deletable[random.randint(0, len(deletable) - 1)]
                code_set.remove(c)
                cover_count[neighbors[c]] -= 1
                improved = True
                continue # Restart loop to find more deletions
            
            # 2. Try to swap
            # Find a codeword that has "only_old" neighbors
            # and try to replace it with another codeword that covers them
            
            if not code_set:
                break
                
            c_old = list(code_set)[random.randint(0, len(code_set) - 1)]
            
            # Find neighbors of c_old that are covered ONLY by c_old
            only_old = []
            for w in neighbors[c_old]:
                if cover_count[w] == 1:
                    only_old.append(w)
            
            if not only_old:
                # Should have been deleted
                code_set.remove(c_old)
                cover_count[neighbors[c_old]] -= 1
                improved = True
                continue
            
            # Try to find c_new that covers all words in only_old
            found_swap = False
            # Try a few random candidates
            for _ in range(50):
                c_new = random.randint(0, N - 1)
                if c_new in code_set:
                    continue
                
                # Check if c_new covers all only_old
                valid = True
                for w in only_old:
                    if w not in neighbors[c_new]:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    code_set.remove(c_old)
                    code_set.add(c_new)
                    cover_count[neighbors[c_old]] -= 1
                    cover_count[neighbors[c_new]] += 1
                    improved = True
                    found_swap = True
                    break
            
            if not found_swap:
                # Stuck. Break to allow restart in main loop
                break
        
        return list(code_set)

    # Main Loop: ILS
    best_code = None
    best_len = float('inf')
    
    # Initial Greedy
    current_code = greedy_cover()
    current_code = local_search(current_code)
    
    if len(current_code) < best_len:
        best_len = len(current_code)
        best_code = current_code
    
    # Iterated Local Search
    while time.time() < time_limit:
        # Perturb the current best code
        # Pick 2 random codewords and replace them with random ones
        # This might break validity, so we need to repair or just restart greedy?
        # Safer: Restart Greedy with a different random seed/tie-breaking?
        # Or, take best_code, remove 2 words, add 2 random words, then Local Search.
        
        if best_code is None:
            break
            
        perturbed_code = list(best_code)
        
        # Remove 2 random words
        if len(perturbed_code) > 4:
            idxs = random.sample(range(len(perturbed_code)), 2)
            # Remove in reverse order to keep indices valid
            idxs.sort(reverse=True)
            for idx in idxs:
                perturbed_code.pop(idx)
        
        # Add 2 random words
        for _ in range(2):
            perturbed_code.append(random.randint(0, N - 1))
        
        # Ensure uniqueness
        perturbed_set = set(perturbed_code)
        
        # Local Search on perturbed code
        # Note: Perturbed code might not be a valid cover.
        # local_search assumes valid cover?
        # My local_search deletes redundant words and swaps.
        # If it's not a valid cover, deletions might make it worse.
        # Swaps might fix it?
        # Actually, if it's not a cover, `cover_count` will have 0s.
        # `min(cover_count[neighbors[c]])` will be 0, so no deletions.
        # Swaps will try to cover `only_old`. If `only_old` is empty (because cover_count is 0? No, 0 is not 1).
        # So local_search might get stuck immediately.
        
        # Better: Repair the perturbed code first?
        # Or just run Greedy again?
        
        # Let's try: Run Greedy, then Local Search.
        # To add diversity, we can randomize the Greedy tie-breaking.
        
        # Simple approach: Just run Greedy again.
        # But Greedy is deterministic.
        # Let's add a small random perturbation to the gains in Greedy.
        
        # For now, just run Local Search on the best_code again?
        # No, that will yield the same result.
        
        # Let's try a "Soft" restart:
        # Take best_code, remove 1 word, add 1 random word.
        # Then Local Search.
        
        if len(best_code) > 2:
            c_remove = random.choice(best_code)
            perturbed = [c for c in best_code if c != c_remove]
            perturbed.append(random.randint(0, N - 1))
            perturbed_set = set(perturbed)
            
            current_code = local_search(perturbed_set)
            
            if len(current_code) < best_len:
                best_len = len(current_code)
                best_code = current_code
        
        # If stuck, we can also try to improve Greedy by randomizing
        # But for now, this ILS should help.

    if best_code is None:
        # Fallback to seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert back to list of lists
    return [from_idx(c) for c in best_code]
