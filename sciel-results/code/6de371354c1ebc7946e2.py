import itertools
import time
import random
import sys

# Constants for the specific problem instance if needed, 
# but the function signature is general construct(q, n, radius).
# However, the task is specific to q=3, n=6, radius=1.
# We will implement a general solver but optimized for this case.

def construct(q, n, radius):
    # The problem is small enough (3^6 = 729 words) 
    # to use bit-parallel set cover techniques.
    
    # If the problem size is too large for bit-parallel (e.g., q^n > 10000), 
    # we should fall back to a simpler greedy or return the seed.
    # But for q=3, n=6, 729 is fine.
    total_words = q ** n
    if total_words > 5000:
        # Fallback to seed strategy for larger instances to avoid timeout/memory
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Generic greedy fallback
            return _generic_greedy(q, n, radius)

    # Bit-parallel exact set cover solver for small spaces
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Map words to indices
    # We can iterate through all combinations
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute coverage masks
    # mask[i] is an integer where bit j is 1 if word j is within radius of word i
    masks = [0] * total_words
    for i, w in enumerate(words):
        mask = 0
        # Generate all words within Hamming distance 'radius'
        # For radius 1, this is w itself + neighbors changing 1 coord
        # For general radius, we can use BFS or combination generation
        if radius == 1:
            mask |= (1 << i)
            for k in range(n):
                for val in range(q):
                    if val == w[k]:
                        continue
                    new_w = list(w)
                    new_w[k] = val
                    idx = word_to_idx[tuple(new_w)]
                    mask |= (1 << idx)
        else:
            # Generic radius: iterate all words and check distance
            # This is O(N^2 * N) which is 729^2 * 6 ~ 3M ops, acceptable
            for j, w2 in enumerate(words):
                dist = sum(1 for a, b in zip(w, w2) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
        masks[i] = mask

    FULL_MASK = (1 << total_words) - 1

    # 3. Greedy Construction
    def greedy_construct():
        covered = 0
        codewords = []
        # To allow randomization, we can shuffle the initial order or tie-breaking
        # We'll use a priority list approach
        # For speed, we just scan all words each time. N=729, K~73. 729*73 ~ 50k ops. Fast.
        while covered != FULL_MASK:
            best_gain = -1
            best_idx = -1
            # Randomize tie-breaking
            # We can iterate in random order
            indices = list(range(total_words))
            random.shuffle(indices)
            
            for i in indices:
                # If this word is already covered, it might still be useful? 
                # No, in greedy set cover, we pick the word that covers the most *new* elements.
                # Even if the word itself is covered, its neighbors might not be.
                gain = (masks[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain:
                    # Random tie break is handled by shuffle, 
                    # but we can also just pick the first one found in shuffled order
                    pass
            
            if best_idx == -1:
                # Should not happen if FULL_MASK is reachable
                break
                
            codewords.append(words[best_idx])
            covered |= masks[best_idx]
            
        return codewords

    # 4. Reduction (Local Search)
    def is_valid_cover(codewords):
        cover = 0
        for w in codewords:
            idx = word_to_idx[w]
            cover |= masks[idx]
        return cover == FULL_MASK

    def reduce_cover(codewords):
        # Try to remove 1 or 2 codewords
        # We maintain the list of indices for faster mask lookup
        # But converting words to indices every time is overhead.
        # Let's store indices in the reduction loop.
        
        # Convert words to indices
        indices = [word_to_idx[w] for w in codewords]
        
        changed = True
        while changed:
            changed = False
            N = len(indices)
            
            # Try removing 1
            # Shuffle to break symmetry
            order = list(range(N))
            random.shuffle(order)
            
            for i in order:
                # Check if removing indices[i] leaves a valid cover
                cover = 0
                for j in range(N):
                    if j != i:
                        cover |= masks[indices[j]]
                if cover == FULL_MASK:
                    # Remove it
                    indices.pop(i)
                    changed = True
                    break # Restart loop with new N
            
            if changed:
                continue
                
            # Try removing 2
            # Only if N > 2
            if N > 2:
                # To avoid O(N^3) being too slow if N is large, but N~73, 73^3/2 ~ 200k checks.
                # Each check is O(N) mask ORs. 200k * 73 ~ 15M ops. Might be slow in Python.
                # Optimization: Precompute prefix/suffix covers?
                # Or just limit this phase.
                
                # Let's try a smarter 2-removal:
                # For each pair (i, j), check if cover without i and j is FULL.
                # We can precompute cover_without_i for all i.
                cover_without = [0] * N
                total_cover = 0
                for i in range(N):
                    total_cover |= masks[indices[i]]
                
                # Calculate cover_without[i]
                # cover_without[i] = total_cover XOR masks[indices[i]] is NOT correct for set union.
                # We need to recompute or use a trick.
                # Since N is small, we can just do the O(N^3) but with early termination or limited attempts.
                
                # Let's just do a few random pair removals to escape local minima
                # If we find one, we accept and restart.
                attempts = 0
                max_attempts = 100 # Limit to keep time down
                while attempts < max_attempts:
                    i = random.randint(0, N - 1)
                    j = random.randint(0, N - 1)
                    if i == j:
                        continue
                    attempts += 1
                    cover = 0
                    for k in range(N):
                        if k != i and k != j:
                            cover |= masks[indices[k]]
                    if cover == FULL_MASK:
                        # Remove both
                        # Remove larger index first to keep indices valid
                        if i > j:
                            i, j = j, i
                        indices.pop(i)
                        indices.pop(j)
                        changed = True
                        break
        
        return [words[i] for i in indices]

    best_code = None
    best_size = float('inf')
    
    # Run multiple iterations
    while time.time() < deadline:
        code = greedy_construct()
        code = reduce_cover(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
            # If we reached the theoretical lower bound or known optimum, we can stop
            # Known optimum for 3,6,1 is 73.
            if best_size <= 73:
                break
    
    if best_code is None:
        # Fallback
        return greedy_construct()
        
    return best_code

def _generic_greedy(q, n, radius):
    """Fallback for larger instances."""
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute distances or masks if small, else use direct distance check
    # For generic fallback, simple greedy
    covered = set()
    codewords = []
    
    while len(covered) < len(words):
        best_gain = -1
        best_w = None
        # Sample if too large? No, just iterate
        for w in words:
            # Calculate new coverage
            # For efficiency, we can't precompute all masks if N is huge.
            # But this is a fallback for cases where bit-parallel fails.
            # We assume N is manageable for O(N^2) or we just do a simple greedy.
            new_cov = 0
            for v in words:
                if v in covered: continue
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    new_cov += 1
            if new_cov > best_gain:
                best_gain = new_cov
                best_w = w
        
        if best_w is None:
            break
            
        codewords.append(best_w)
        # Update covered
        for v in words:
            if v in covered: continue
            dist = sum(1 for a, b in zip(best_w, v) if a != b)
            if dist <= radius:
                covered.add(v)
                
    return codewords
