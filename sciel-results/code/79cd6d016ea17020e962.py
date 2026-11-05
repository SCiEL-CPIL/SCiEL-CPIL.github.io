import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n
    ball_size = 1 + n * (q - 1)  # 1 + 6*2 = 13
    
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

    # Precompute balls as numpy array
    balls_np = np.zeros((total_words, ball_size), dtype=np.int32)
    for i in range(total_words):
        w = to_word(i)
        balls_np[i, 0] = i
        col = 1
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w[pos] = new_val
                balls_np[i, col] = to_idx(w)
                col += 1
            w[pos] = orig

    # Initial Code: Tetracode Sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Use numpy for counts
    counts_np = np.zeros(total_words, dtype=np.int32)
    for c in initial_code:
        counts_np[balls_np[c]] += 1
    
    best_code = list(initial_code)
    best_size = len(best_code)
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    current_code = list(initial_code)
    current_counts = counts_np.copy()
    code_set = set(current_code)
    
    def try_remove(c):
        # Check if all words in ball have count > 1
        return np.all(current_counts[balls_np[c]] > 1)

    def do_remove(c):
        current_counts[balls_np[c]] -= 1
        code_set.discard(c)

    def do_add(c):
        current_counts[balls_np[c]] += 1
        code_set.add(c)

    def greedy_remove(code_list, counts_arr, set_local):
        """Remove as many codewords as possible in one pass."""
        code_list = list(code_list)
        random.shuffle(code_list)
        for c in code_list[:]:
            if c not in set_local:
                continue
            # Check if removable
            if np.all(counts_arr[balls_np[c]] > 1):
                counts_arr[balls_np[c]] -= 1
                set_local.discard(c)
        return code_list

    # Phase 1: Greedy Remove - multiple passes
    for _ in range(10):
        removed_any = False
        random.shuffle(current_code)
        for c in current_code[:]:
            if c in code_set and try_remove(c):
                do_remove(c)
                current_code.remove(c)
                removed_any = True
        if not removed_any:
            break
            
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)
    
    # Phase 2: Local search with swaps
    # We'll try: remove c1, add c2, then greedy remove
    # Use a more efficient approach: only try promising c2 candidates
    
    # Precompute for each codeword, which other codewords are "close" (distance 2)
    # This helps find good swap candidates
    
    while time.time() < deadline:
        if not current_code:
            break
            
        # Pick a random codeword to potentially remove
        c1 = random.choice(current_code)
        
        # Find candidate c2 to add
        # Strategy: pick a word that is well-covered (high count) to balance coverage
        # Or pick randomly from under-covered regions
        
        # Try multiple random c2 candidates
        found_improvement = False
        
        for attempt in range(3):
            c2 = random.randint(0, total_words - 1)
            if c2 in code_set:
                continue
            
            # Save state
            saved_code = list(current_code)
            saved_counts = current_counts.copy()
            saved_set = set(code_set)
            
            # Try removing c1
            can_remove_c1 = try_remove(c1)
            if can_remove_c1:
                do_remove(c1)
                current_code.remove(c1)
            
            # Add c2
            do_add(c2)
            current_code.append(c2)
            
            # Greedy remove
            random.shuffle(current_code)
            for c in current_code[:]:
                if c not in code_set:
                    continue
                if try_remove(c):
                    do_remove(c)
                    current_code.remove(c)
            
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                found_improvement = True
                break
            else:
                # Revert
                current_code = saved_code
                current_counts = saved_counts
                code_set = saved_set
        
        # If no improvement after several attempts, try a different strategy
        if not found_improvement and random.random() < 0.1:
            # Try adding a codeword and then removing two
            c_add = random.randint(0, total_words - 1)
            if c_add not in code_set:
                saved_code = list(current_code)
                saved_counts = current_counts.copy()
                saved_set = set(code_set)
                
                do_add(c_add)
                current_code.append(c_add)
                
                # Remove two codewords
                removed = 0
                random.shuffle(current_code)
                for c in current_code[:]:
                    if c not in code_set:
                        continue
                    if try_remove(c):
                        do_remove(c)
                        current_code.remove(c)
                        removed += 1
                        if removed >= 2:
                            break
                
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                else:
                    current_code = saved_code
                    current_counts = saved_counts
                    code_set = saved_set

    return [to_word(idx) for idx in best_code]
