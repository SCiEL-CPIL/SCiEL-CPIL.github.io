"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    # For the specific task: q=3, n=7, radius=1
    # We use a greedy construction with local improvement (add/remove/swap)
    # to find a small covering code.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: simple direct product for other cases
        # For general case, use a greedy approach
        return _generic_greedy(q, n, radius)
    
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 120s limit
    
    # Precompute all words
    all_words = np.array(np.arange(3**7)).reshape(-1, 7)
    # Convert to base-3 digits
    for i in range(7):
        all_words[:, i] = (all_words[:, i] // (3 ** (6 - i))) % 3
    
    # For each word, precompute its neighbors (distance <= 1)
    # A word is covered if it's in the ball of radius 1 around some codeword
    # Ball of radius 1: the word itself + words differing in exactly 1 position
    
    # Number of neighbors per word: 1 + 7*2 = 15
    # For each position i and each delta d in {1,2}, the neighbor is word with position i changed by d mod 3
    
    # Precompute neighbor indices for each word
    # word_idx: index in all_words array
    # For word w at index idx, neighbors are:
    # idx itself
    # For each position i, for each delta in {1,2}:
    #   new_word = w with w[i] changed to (w[i]+delta)%3
    #   Find its index: we can compute it directly
    
    # Actually, let's use a different approach:
    # Represent each word as an integer 0..2186
    # For word w (as integer), its neighbors are:
    # w itself
    # For each position i (0..6), for each delta in {1,2}:
    #   w + delta * 3^(6-i) if w[i] + delta < 3
    #   w + (delta - 3) * 3^(6-i) if w[i] + delta >= 3
    
    # Let's precompute a coverage array
    # covered[word_idx] = True if word is covered by current code
    
    # Start with a random initial code
    # Use a greedy seed: pick random words, add if they cover new words
    
    best_code = None
    best_size = 3**7  # worst case: all words
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Greedy construction
        covered = np.zeros(3**7, dtype=bool)
        code = []
        
        # Precompute: for each word index, which other word indices are in its ball
        # This is expensive to precompute fully, so let's compute on the fly
        # Actually, let's precompute the neighbor list for each word
        
        # For word with integer value v, positions are:
        # pos_i = (v // 3^(6-i)) % 3
        # Neighbor in position i with delta: 
        #   if pos_i + delta < 3: v + delta * 3^(6-i)
        #   else: v + (delta-3) * 3^(6-i)
        
        pow3 = [1, 3, 9, 27, 81, 243, 729]
        pow3_rev = [729, 243, 81, 27, 9, 3, 1]
        
        def get_neighbors(v):
            """Get all word indices in ball of radius 1 around word v."""
            neighbors = [v]
            for i in range(7):
                pos_val = (v // pow3_rev[i]) % 3
                for delta in [1, 2]:
                    new_pos = (pos_val + delta) % 3
                    diff = new_pos - pos_val
                    neighbors.append(v + diff * pow3_rev[i])
            return neighbors
        
        # Precompute neighbors for all words
        all_neighbors = [get_neighbors(v) for v in range(3**7)]
        
        # Greedy: repeatedly pick the word that covers the most uncovered words
        uncovered_count = np.ones(3**7, dtype=int)
        for v in range(3**7):
            for nb in all_neighbors[v]:
                uncovered_count[v] = uncovered_count[v]  # will update as we go
        
        # Actually, let's just do a simpler greedy:
        # Pick random words, add if they cover at least one new word
        # Then do local search
        
        # Initialize with a few random words
        code_set = set()
        covered = np.zeros(3**7, dtype=bool)
        
        # Add some initial words greedily
        for _ in range(200):
            if time.time() > deadline:
                break
            best_v = -1
            best_gain = 0
            # Sample some candidates to speed up
            candidates = random.sample(range(3**7), min(100, 3**7))
            for v in candidates:
                gain = sum(1 for nb in all_neighbors[v] if not covered[nb])
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_gain == 0:
                break
            code_set.add(best_v)
            for nb in all_neighbors[best_v]:
                covered[nb] = True
        
        # Check if complete
        if not covered.all():
            # Continue adding
            while not covered.all():
                if time.time() > deadline:
                    break
                best_v = -1
                best_gain = 0
                candidates = random.sample(range(3**7), min(200, 3**7))
                for v in candidates:
                    gain = sum(1 for nb in all_neighbors[v] if not covered[nb])
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_gain == 0:
                    # Try all words
                    for v in range(3**7):
                        gain = sum(1 for nb in all_neighbors[v] if not covered[nb])
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                if best_v == -1 or best_gain == 0:
                    break
                code_set.add(best_v)
                for nb in all_neighbors[best_v]:
                    covered[nb] = True
        
        if not covered.all():
            continue
        
        current_size = len(code_set)
        if current_size < best_size:
            best_size = current_size
            best_code = list(code_set)
        
        # Local search: try to remove words
        if time.time() > deadline:
            break
        
        # Try removing words
        code_list = list(code_set)
        improved = True
        while improved:
            improved = False
            for i in range(len(code_list)):
                if time.time() > deadline:
                    break
                # Try removing code_list[i]
                candidate = code_list[i]
                # Check if all its neighbors are still covered by other words
                all_covered = True
                for nb in all_neighbors[candidate]:
                    if not covered[nb]:
                        all_covered = False
                        break
                # We need to check if removing this word leaves all its neighbors covered
                # by other words in the code
                # This is expensive, so let's do a quick check
                # Count how many codewords cover each neighbor
                # For simplicity, just try removing and check
                temp_code = [c for j, c in enumerate(code_list) if j != i]
                # Recompute covered for temp_code is expensive
                # Instead, check: for each neighbor of candidate, is it covered by some other codeword?
                ok = True
                for nb in all_neighbors[candidate]:
                    # Check if nb is in ball of some other codeword
                    found = False
                    for c in temp_code:
                        if nb in all_neighbors[c]:
                            found = True
                            break
                    if not found:
                        ok = False
                        break
                if ok:
                    code_list = temp_code
                    code_set = set(code_list)
                    improved = True
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_set)
                    break  # restart loop after modification
        
        if best_code is not None and len(best_code) < best_size:
            best_size = len(best_code)
    
    if best_code is None:
        # Fallback: return all words (shouldn't happen)
        return [list(w) for w in all_words]
    
    # Convert integer indices back to word lists
    result = []
    for v in best_code:
        word = []
        for i in range(7):
            word.append((v // pow3_rev[i]) % 3)
        result.append(word)
    
    return result


def _generic_greedy(q, n, radius):
    """Generic greedy construction for arbitrary q, n, radius."""
    start_time = time.time()
    deadline = start_time + 110
    
    # For small spaces, brute force might work
    total = q ** n
    if total <= 100000:
        # Greedy set cover
        all_words = []
        for i in range(total):
            word = []
            v = i
            for j in range(n):
                word.append(v % q)
                v //= q
            all_words.append(tuple(word))
        
        # Precompute balls
        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        
        covered = set()
        code = []
        
        while len(covered) < total:
            if time.time() > deadline:
                break
            best_word = None
            best_gain = 0
            for w in all_words:
                gain = 0
                for u in all_words:
                    if u not in covered and hamming_dist(w, u) <= radius:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_gain == 0:
                break
            code.append(list(best_word))
            for u in all_words:
                if hamming_dist(best_word, u) <= radius:
                    covered.add(u)
        
        if len(covered) < total:
            # Add remaining uncovered words
            for u in all_words:
                if u not in covered:
                    code.append(list(u))
                    covered.add(u)
        
        return code
    
    # For large spaces, use a simple heuristic
    # Just return a random sample (likely invalid, but better than timeout)
    # Actually, let's try a basic greedy with sampling
    random.seed(42)
    code = []
    covered = set()
    
    def word_to_int(w):
        v = 0
        for i, x in enumerate(w):
            v += x * (q ** (n - 1 - i))
        return v
    
    def int_to_word(v):
        w = []
        for i in range(n):
            w.append(v % q)
            v //= q
        return w[::-1]
    
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
    
    # Greedy with random sampling
    for _ in range(5000):
        if time.time() > deadline:
            break
        best_word = None
        best_gain = 0
        for _ in range(100):
            w = tuple(random.randint(0, q-1) for _ in range(n))
            gain = 0
            for _ in range(50):
                u = tuple(random.randint(0, q-1) for _ in range(n))
                if u not in covered and hamming_dist(w, u) <= radius:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = w
        if best_word is None or best_gain == 0:
            break
        code.append(list(best_word))
        # Update covered (approximate)
    
    # Ensure validity by adding all words if needed (expensive but correct)
    if total <= 500000:
        all_words = []
        for i in range(total):
            all_words.append(int_to_word(i))
        
        covered_set = set()
        for cw in code:
            for w in all_words:
                if hamming_dist(cw, w) <= radius:
                    covered_set.add(tuple(w))
        
        for w in all_words:
            if tuple(w) not in covered_set:
                code.append(list(w))
    
    return code

# EVOLVE-BLOCK-END
