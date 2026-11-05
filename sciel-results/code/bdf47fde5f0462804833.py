import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other params
        # Tetracode sum works for n=6, q=3
        # For general, return brute force or simple construction
        # Since task is specific, we focus on 3,6,1
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n
    # 729

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

    # Precompute balls (indices covered by each word)
    balls = []
    for i in range(total_words):
        w = to_word(i)
        covered = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w[pos] = new_val
                covered.append(to_idx(w))
            w[pos] = orig
        balls.append(covered)
        
    # Convert balls to numpy arrays for faster subtraction?
    # Or just lists. Lists are fine for size 13.
    
    # Initial Code: Tetracode Sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Sort for determinism? No, random is better.
    
    counts = [0] * total_words
    for c in initial_code_indices:
        for w in balls[c]:
            counts[w] += 1
            
    best_code = list(initial_code_indices)
    best_size = len(best_code)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    def greedy_reduce(code, counts_arr):
        """
        Attempts to remove words from code while maintaining coverage.
        Returns new code and counts.
        """
        # Work on copies? No, in-place is faster, but we need to rollback if we fail?
        # Greedy reduce is monotonic (size decreases). We only keep result if valid.
        # So we can modify in place and return.
        
        # To make it faster, we can pick removable words randomly.
        # A word is removable if min(counts[w] for w in balls[c]) > 1
        
        while True:
            removable = []
            for c in code:
                is_removable = True
                for w in balls[c]:
                    if counts_arr[w] == 1:
                        is_removable = False
                        break
                if is_removable:
                    removable.append(c)
            
            if not removable:
                break
            
            # Pick a random removable word
            c_to_remove = random.choice(removable)
            
            # Remove it
            code.remove(c_to_remove)
            for w in balls[c_to_remove]:
                counts_arr[w] -= 1
                
        return code, counts_arr

    def perturb(code, counts_arr, k=2):
        """
        Add k random words not in code.
        """
        code_set = set(code)
        added = 0
        attempts = 0
        while added < k and attempts < 100:
            rand_idx = random.randint(0, total_words - 1)
            if rand_idx not in code_set:
                code.append(rand_idx)
                code_set.add(rand_idx)
                for w in balls[rand_idx]:
                    counts_arr[w] += 1
                added += 1
            attempts += 1
        return code, counts_arr

    current_code = list(initial_code_indices)
    current_counts = list(counts)
    
    # Initial greedy reduce
    current_code, current_counts = greedy_reduce(current_code, current_counts)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)
        
    while time.time() < deadline:
        # Save state
        saved_code = list(current_code)
        saved_counts = list(current_counts)
        
        # Perturb
        current_code, current_counts = perturb(current_code, current_counts, k=2)
        
        # Reduce
        current_code, current_counts = greedy_reduce(current_code, current_counts)
        
        # Check improvement
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
            # If we hit 73, we can break? Or keep going to see if 72 is possible?
            # Best known is 73. Lower bound 71.
            # If we find 73, it's a huge jump.
            # We can try to find 72.
            
        # If we didn't improve, we might want to revert to the previous local minimum?
        # Or just keep the new state?
        # ILS usually reverts to the best known or the pre-perturbation state if no improvement.
        # Here, we just keep the current state (which is a local min) and perturb again.
        # This is a random walk on local minima.
        
    return [to_word(idx) for idx in best_code]
