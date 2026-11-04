import time
import random
import itertools

def construct(q, n, radius):
    # Specific to q=3, n=6, radius=1
    # General fallback for other params if needed, but task is specific.
    if q != 3 or n != 6 or radius != 1:
        # Fallback to simple greedy or tetracode
        # Tetracode sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n
    # Map word tuple to index 0..728
    def to_idx(word):
        idx = 0
        for w in word:
            idx = idx * 3 + w
        return idx

    def to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute balls
    # ball[i] is a list of indices covered by word i
    balls = []
    for i in range(total_words):
        w = to_word(i)
        covered = {i}
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w[pos] = new_val
                covered.add(to_idx(w))
            w[pos] = orig
        balls.append(list(covered))

    # Initial Code: Tetracode Sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # counts[word_idx] = number of codewords in current set covering word_idx
    counts = [0] * total_words
    for c in initial_code:
        for w in balls[c]:
            counts[w] += 1
    
    best_code = list(initial_code)
    best_size = len(best_code)
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    current_code = list(initial_code)
    current_counts = list(counts)
    
    def try_remove(c, counts_arr):
        # Check if removing c is safe
        # We need to ensure all words in balls[c] have count > 1
        # If any word in balls[c] has count == 1, removing c makes it 0 -> invalid
        for w in balls[c]:
            if counts_arr[w] == 1:
                return False
        return True

    def do_remove(c, counts_arr):
        for w in balls[c]:
            counts_arr[w] -= 1

    def do_add(c, counts_arr):
        for w in balls[c]:
            counts_arr[w] += 1

    # Phase 1: Greedy Remove
    while True:
        removed_any = False
        # Randomize order for variety
        random.shuffle(current_code)
        for c in current_code[:]:
            if try_remove(c, current_counts):
                do_remove(c, current_counts)
                current_code.remove(c)
                removed_any = True
        if not removed_any:
            break
            
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)
        
    # Phase 2: Local Search with Add-Remove
    # We maintain current_code and current_counts.
    # If we add, size increases. Then we remove greedily.
    # If final size < best_size, update.
    # We also keep the best known state to restart from if we get stuck?
    # Actually, just keep improving the current state.
    
    # To escape local minima, we might need to restart from best_code occasionally.
    
    while time.time() < deadline:
        # Add a random word
        # Pick a word not in current_code
        # Efficient way: pick random index, if in set, pick another?
        # Set for membership check
        code_set = set(current_code)
        
        # Try a few random additions
        added = False
        for _ in range(10):
            rand_idx = random.randint(0, total_words - 1)
            if rand_idx not in code_set:
                current_code.append(rand_idx)
                do_add(rand_idx, current_counts)
                code_set.add(rand_idx)
                added = True
                break
        
        if not added:
            # Code is full? Unlikely.
            break
            
        # Greedy Remove
        random.shuffle(current_code)
        for c in current_code[:]:
            if try_remove(c, current_counts):
                do_remove(c, current_counts)
                current_code.remove(c)
                code_set.discard(c)
                
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
            
        # Optional: If we are stuck (no improvement in last N steps), reset to best?
        # For now, just keep going. The add step introduces randomness.
        
    # Convert best_code indices back to lists of ints
    return [to_word(idx) for idx in best_code]
