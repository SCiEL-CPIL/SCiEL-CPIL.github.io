import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Parameters for this specific task instance
    # The evaluator passes q, n, radius. 
    # Task is K_3(8,1).
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other instances if any (though task is fixed)
        # Use direct sum if possible, else random
        pass

    # Precomputation
    total_words = q ** n
    # Map integer to tuple and vice versa
    # For speed, work with integers 0..total_words-1
    
    # Generate all words
    # neighbors[word] = list of words within Hamming distance <= radius
    # radius = 1
    # For ternary, neighbors are:
    # self
    # for each pos i, change val to val+1, val+2
    
    neighbors = []
    for w in range(total_words):
        # Convert to digits
        digits = [0] * n
        temp = w
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        
        neigh_set = {w}
        for i in range(n):
            orig = digits[i]
            # Change to other 2 values
            for new_val in range(q):
                if new_val != orig:
                    # Reconstruct number
                    new_w = w + (new_val - orig) * (q ** i)
                    neigh_set.add(new_w)
        neighbors.append(list(neigh_set))
    
    # Initial Code: Direct Sum
    # Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Convert tetracode to integers for the first 4 coords
    # And tails for the rest
    # This is the 729 word code.
    
    code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            # Combine
            word_int = 0
            # First 4 digits from h
            for i in range(4):
                word_int += h[i] * (3 ** i)
            # Next n-4 digits from tail
            for i in range(n - 4):
                word_int += tail[i] * (3 ** (4 + i))
            code.append(word_int)
    
    code_set = set(code)
    
    # Initialize cover_count
    cover_count = [0] * total_words
    for c in code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
    
    # Helper to check validity (should always be true if we maintain counts correctly)
    def is_valid():
        return all(c > 0 for c in cover_count)
    
    # Peeling
    def peel():
        nonlocal code, code_set, cover_count
        removed = True
        while removed:
            removed = False
            # Shuffle to break symmetry
            random.shuffle(code)
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if c is redundant
                redundant = True
                for nb in neighbors[c]:
                    if cover_count[nb] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    code.pop(i)
                    code_set.remove(c)
                    removed = True
                    if time.time() > deadline:
                        return
    
    # Initial Peeling
    peel()
    
    best_code = list(code)
    best_size = len(code)
    
    # 2-for-1 Exchange Search
    # Randomized
    while time.time() < deadline:
        # Pick random pair
        if len(code) < 2:
            break
        idx1 = random.randint(0, len(code) - 1)
        idx2 = random.randint(0, len(code) - 1)
        if idx1 == idx2:
            continue
        
        c1 = code[idx1]
        c2 = code[idx2]
        
        # Determine uncovered words if c1, c2 removed
        # uncovered = {w in N(c1) U N(c2) | cover_count[w] == 1 and w covered by c1 or c2}
        # Actually, if cover_count[w] == 1, w is covered by exactly one codeword.
        # If w in N(c1), and cover_count[w]==1, then c1 is the unique cover.
        # So w becomes uncovered if we remove c1.
        
        uncovered = set()
        for nb in neighbors[c1]:
            if cover_count[nb] == 1:
                uncovered.add(nb)
        for nb in neighbors[c2]:
            if cover_count[nb] == 1:
                uncovered.add(nb)
        
        if not uncovered:
            # We can just remove one of them?
            # If uncovered is empty, it means no word was uniquely covered by c1 or c2.
            # So we can remove c1 (or c2) safely.
            # Let's remove c1.
            for nb in neighbors[c1]:
                cover_count[nb] -= 1
            code.remove(c1)
            code_set.remove(c1)
            best_code = list(code)
            best_size = len(code)
            continue
        
        # Try to find c3 that covers all uncovered
        # c3 must be in code_set? No, can be any word.
        # But if c3 is not in code, we add it.
        # If c3 is in code, we just replace.
        
        # Optimization: 
        # The intersection of neighborhoods of all w in uncovered must contain c3.
        # Actually, c3 must be a neighbor of ALL w in uncovered.
        # So c3 must be in the intersection of {neighbors[w] for w in uncovered}.
        
        if len(uncovered) > 17:
            # Impossible for a single codeword to cover > 17 words?
            # No, a codeword covers 17 words.
            # So if len(uncovered) > 17, no single c3 can cover them.
            continue
            
        # Find intersection of neighbors of all w in uncovered
        # Start with neighbors of first w
        common_candidates = set(neighbors[list(uncovered)[0]])
        for w in list(uncovered)[1:]:
            common_candidates.intersection_update(neighbors[w])
            if not common_candidates:
                break
        
        if not common_candidates:
            continue
            
        # Pick a candidate c3
        # Prefer c3 that is already in code? Or random?
        # If c3 is in code, size decreases by 1 (remove 2, add 1).
        # If c3 is not in code, size decreases by 1 (remove 2, add 1).
        # So size always decreases by 1.
        
        # Randomly pick one
        c3 = random.choice(list(common_candidates))
        
        # Check if c3 is c1 or c2 (should not be, as they don't cover each other uniquely? 
        # Actually c1 covers c1. If c1 is in uncovered, c3 must cover c1.
        # If c3 == c1, we are replacing {c1, c2} with {c1}, which is just removing c2.
        # But we checked if c2 was redundant. If c2 was not redundant, removing it alone fails.
        # So c3 != c1 and c3 != c2 likely.
        
        # Perform exchange
        # Remove c1, c2
        for nb in neighbors[c1]:
            cover_count[nb] -= 1
        for nb in neighbors[c2]:
            cover_count[nb] -= 1
        code.remove(c1)
        code.remove(c2)
        code_set.discard(c1)
        code_set.discard(c2)
        
        # Add c3
        for nb in neighbors[c3]:
            cover_count[nb] += 1
        code.append(c3)
        code_set.add(c3)
        
        best_code = list(code)
        best_size = len(code)
        
        # Optional: Run a quick peel after exchange?
        # Might be too slow.
        
    return [[int(c // (3**i)) % 3 for i in range(n)] for c in best_code]
