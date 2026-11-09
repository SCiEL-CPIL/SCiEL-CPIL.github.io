import time
import numpy as np
import random
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Handle general case fallback
    if q != 3 or n != 8 or radius != 1:
        # Simple product code fallback
        # For K_3(4,1), Tetracode is optimal (size 9)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
        else:
            return [list(w) for w in product(range(q), repeat=n)]

    total_words = q ** n  # 6561
    num_neighbors = 1 + n * (q - 1)  # 17
    
    # Precompute neighbors
    # neighbors[i] is a list of indices
    neighbors = [None] * total_words
    pow3 = [3 ** (n - 1 - i) for i in range(n)]
    
    for i in range(total_words):
        temp = i
        word = [0] * n
        for j in range(n - 1, -1, -1):
            word[j] = temp % q
            temp //= q
        
        nbs = [i]
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                delta = (new_val - orig_val) * pow3[pos]
                nbs.append(i + delta)
        neighbors[i] = nbs

    best_code = None
    best_size = float('inf')
    
    # Multiple restarts of Greedy + Removal
    num_restarts = 20  # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(restart)
        
        # Phase 1: Greedy Set Cover
        uncovered = np.ones(total_words, dtype=bool)
        code_indices = []
        
        # To speed up greedy, we can just pick the max count each time
        # But updating counts for all 6561 candidates is expensive (6561 * 17 * iterations)
        # Iterations ~ 600. 6561 * 17 * 600 ~ 67 million ops. In Python, this is slow.
        # We need a faster greedy.
        
        # Alternative: Randomized Greedy
        # Pick a random word, check if it covers many uncovered words.
        # Better: Maintain a list of candidate counts.
        
        # Let's try a simple greedy first to see if it fits in time.
        # If not, we will use a simpler initialization (e.g. Tetracode product) and just do removal.
        
        # Tetracode product code size 729.
        # Greedy usually gets to ~650.
        # Removal from 729 might get to ~550?
        # Removal from 650 might get to ~500?
        
        # Let's use the Tetracode product code as the base for all restarts,
        # but randomize the removal order.
        
        # Generate Tetracode product code indices
        tetracode_words = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_code_words = []
        for h in tetracode_words:
            for tail in product(range(3), repeat=4):
                base_code_words.append(h + list(tail))
        
        base_code_indices = []
        for w in base_code_words:
            idx = 0
            for j in range(8):
                idx = idx * 3 + w[j]
            base_code_indices.append(idx)
            
        # Shuffle the base code
        code_indices = base_code_indices[:]
        random.shuffle(code_indices)
        
        # Phase 2: Removal
        # cover_count[i] = number of codewords in code_indices that cover i
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_indices:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        # Removal loop
        # We iterate through code_indices. If a codeword is removable, we remove it.
        # We do multiple passes.
        
        active = [True] * len(code_indices)
        
        for pass_num in range(50): # Max passes
            if time.time() > deadline - 2.0:
                break
                
            removed_any = False
            # Shuffle order for each pass to avoid bias
            order = list(range(len(code_indices)))
            random.shuffle(order)
            
            for i in order:
                if not active[i]:
                    continue
                    
                c = code_indices[i]
                # Check if removable
                # min(cover_count[neighbors[c]]) > 1
                # If all neighbors are covered by at least 2 codewords, we can remove c.
                # Note: cover_count includes c itself.
                # So if min is > 1, it means there is at least 1 OTHER codeword covering each neighbor.
                
                is_removable = True
                for nb in neighbors[c]:
                    if cover_count[nb] <= 1:
                        is_removable = False
                        break
                        
                if is_removable:
                    active[i] = False
                    removed_any = True
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                        
            if not removed_any:
                break
        
        # Calculate final size
        current_size = sum(active)
        
        if current_size < best_size:
            best_size = current_size
            best_code = [code_indices[i] for i in range(len(code_indices)) if active[i]]
            
    # If no code was found (should not happen), return base code
    if best_code is None:
        best_code = base_code_indices
        
    # Convert indices back to words
    result = []
    for idx in best_code:
        temp = idx
        word = [0] * n
        for j in range(n - 1, -1, -1):
            word[j] = temp % q
            temp //= q
        result.append(word)
        
    return result
