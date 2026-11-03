import time
import random
import numpy as np

# Precomputation for q=3, n=6
Q = 3
N = 6
RADIUS = 1
SPACE_SIZE = Q ** N  # 729

# Convert word index to list of digits and vice versa
def idx_to_word(idx):
    word = []
    for _ in range(N):
        word.append(idx % Q)
        idx //= Q
    return word

def word_to_idx(word):
    idx = 0
    for i in range(N):
        idx += word[i] * (Q ** i)
    return idx

# Precompute neighbors (distance <= 1) for all words
# neighbors[idx] is a list of indices
neighbors = []
for i in range(SPACE_SIZE):
    w = idx_to_word(i)
    nbs = [i]
    for pos in range(N):
        orig_val = w[pos]
        for new_val in range(Q):
            if new_val == orig_val:
                continue
            w[pos] = new_val
            nbs.append(word_to_idx(w))
        w[pos] = orig_val
    neighbors.append(nbs)

# We can also precompute for each word, the set of codewords that cover it?
# No, that's dynamic.

def construct(q, n, radius):
    # The evaluator passes q, n, radius. 
    # We assume q=3, n=6, radius=1 based on the task description.
    # But we should handle general case or at least check.
    # The task says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # It implies it might vary? "football pool problem for 6 matches... ternary...".
    # "Interface: construct(q, n, radius)".
    # "Score: the number of codewords... upper bound on K_q(n, radius)".
    # The specific instance is K_3(6,1).
    # I will assume the evaluator calls construct(3, 6, 1).
    # If it calls something else, my precomputed neighbors are wrong.
    # I should check if q==3 and n==6. If not, fall back to a generic (slower) method or just the seed.
    # Given the strict time limit and specific known best, I will optimize for 3,6,1.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for other parameters, though unlikely
        import itertools
        if n < 4:
             # trivial
             return [list(t) for t in itertools.product(range(q), repeat=n)]
        # Simple greedy for general case
        # ... (implement a basic greedy)
        # For now, return seed-like result to be safe
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    best_code = None
    best_size = 100
    
    # Greedy Initialization
    def greedy_init(randomize=False):
        uncovered = set(range(SPACE_SIZE))
        code = []
        while uncovered:
            best_c = -1
            best_count = -1
            # To speed up, we don't need to check all 729 if we have a good candidate
            # But 729 is small.
            candidates = list(uncovered) if randomize else range(SPACE_SIZE)
            # Actually, candidates should be all possible codewords, not just uncovered ones.
            # A codeword can be anywhere.
            # But picking a codeword that is itself uncovered is a good heuristic.
            # Let's iterate through all 729.
            for c in range(SPACE_SIZE):
                # Count how many uncovered words are in neighbors[c]
                count = 0
                for nb in neighbors[c]:
                    if nb in uncovered:
                        count += 1
                if count > best_count or (randomize and count == best_count and random.random() < 0.5):
                    best_count = count
                    best_c = c
            code.append(best_c)
            # Remove covered
            for nb in neighbors[best_c]:
                if nb in uncovered:
                    uncovered.remove(nb)
            if time.time() > deadline:
                return code
        return code

    # Local Search
    def local_search(initial_code, max_iterations=10000):
        code = list(initial_code)
        counts = np.zeros(SPACE_SIZE, dtype=np.int32)
        
        # Initialize counts
        for c in code:
            for nb in neighbors[c]:
                counts[nb] += 1
        
        # Check if valid
        if np.min(counts) < 1:
             return None, len(code)
             
        size = len(code)
        
        # Helper to try removal
        def try_remove(idx_c):
            c = code[idx_c]
            # Decrement
            for nb in neighbors[c]:
                counts[nb] -= 1
            # Check min
            if np.min(counts) >= 1:
                # Accept
                code.pop(idx_c)
                return True
            else:
                # Restore
                for nb in neighbors[c]:
                    counts[nb] += 1
                return False

        # Helper to try swap
        def try_swap(idx_c, new_c):
            old_c = code[idx_c]
            if old_c == new_c:
                return False
            # Check if new_c is already in code (avoid duplicates)
            # For efficiency, we can assume we pick new_c randomly and check
            # But we need to ensure distinctness.
            # If new_c is in code, swap is a no-op or invalid (duplicate).
            # Let's just check if new_c is in code.
            if new_c in code:
                return False
            
            # Decrement old
            for nb in neighbors[old_c]:
                counts[nb] -= 1
            # Increment new
            for nb in neighbors[new_c]:
                counts[nb] += 1
            
            if np.min(counts) >= 1:
                # Accept
                code[idx_c] = new_c
                return True
            else:
                # Restore
                for nb in neighbors[new_c]:
                    counts[nb] -= 1
                for nb in neighbors[old_c]:
                    counts[nb] += 1
                return False

        # Main loop
        # We alternate between trying removals and swaps.
        # Removals reduce size, which is the primary goal.
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            # 1. Try to remove a codeword
            # Randomly pick an index to try removal? Or iterate?
            # Iterating through all is O(K * 13). K~73. 73*13=949 ops. Fast.
            removed = False
            for i in range(len(code)):
                if try_remove(i):
                    size -= 1
                    removed = True
                    break # Restart removal loop from 0? Or continue?
                    # Restarting is safer to find new removals enabled by this one.
                    # But for speed, maybe just break and let the outer loop handle it.
                    # Actually, if we remove one, the indices shift.
                    # So we must restart the removal scan.
                    break
            
            if removed:
                continue
            
            # 2. No removal possible. Try swaps to rearrange.
            # Pick a random codeword to replace.
            i = random.randint(0, len(code) - 1)
            # Pick a random new codeword.
            # To be smart, pick a new codeword that covers "tight" spots?
            # Or just random.
            new_c = random.randint(0, SPACE_SIZE - 1)
            
            if try_swap(i, new_c):
                # Swap accepted. Size unchanged.
                pass
            else:
                # Swap rejected.
                pass

        return code, size

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        # Initialize
        # Use a mix of greedy and random
        if num_restarts % 5 == 0:
            init_code = greedy_init(randomize=False)
        else:
            init_code = greedy_init(randomize=True)
        
        # Local search
        final_code, final_size = local_search(init_code)
        
        if final_code is not None and final_size < best_size:
            best_size = final_size
            best_code = final_code
            # If we hit the theoretical lower bound or known best, we might stop?
            # Known best is 73. Lower bound 57.
            # If we find 73, we are good. If we find < 73, we are new record.
            if best_size <= 73:
                # We can try to push for lower, but 73 is a strong target.
                # Let's keep searching a bit more to see if we can get 72 or 71.
                pass
        
        num_restarts += 1
        
        # If we have plenty of time, do more.
        # If we are close to deadline, ensure we return valid code.
        
    if best_code is None:
        # Fallback
        best_code = greedy_init(randomize=False)
        
    # Convert indices back to lists of digits
    return [idx_to_word(c) for c in best_code]
