import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # Precompute all words and their balls
    # Total words = q**n
    # For q=3, n=6, total = 729
    # We represent words as integers 0..q**n-1 for fast indexing, 
    # but we need to convert to/from lists of digits for the final output and distance calc.
    
    # It's easier to work with tuples of digits for ball generation, 
    # but map them to indices for the cover_count array.
    
    words = []
    word_to_idx = {}
    for i in range(q**n):
        # Convert i to base-q digits
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w = tuple(reversed(w))
        words.append(w)
        word_to_idx[w] = i
        
    # Precompute balls for each word index
    # ball[i] is a list of indices j such that d(words[i], words[j]) <= radius
    balls = [None] * (q**n)
    for i, w1 in enumerate(words):
        ball_i = []
        for j, w2 in enumerate(words):
            # Check Hamming distance
            dist = sum(1 for k in range(n) if w1[k] != w2[k])
            if dist <= radius:
                ball_i.append(j)
        balls[i] = ball_i

    # Initial solution: Use the seed's logic (Direct Sum)
    # The seed uses:
    # tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # This yields 81 words.
    
    import itertools
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_codewords.append(h + list(tail))
            
    # Convert init_codewords to indices
    current_set = set()
    for cw in init_codewords:
        idx = word_to_idx[tuple(cw)]
        current_set.add(idx)
        
    # Initialize cover counts
    cover_count = np.zeros(q**n, dtype=np.int32)
    for idx in current_set:
        for j in balls[idx]:
            cover_count[j] += 1
            
    uncovered_count = np.sum(cover_count == 0)
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Step 1: Greedy Redundancy Removal
    # Try to remove any codeword that is redundant
    # A codeword c is redundant if for all w in ball(c), cover_count[w] > 1 
    # (meaning another codeword also covers w)
    
    def is_redundant(idx):
        # Check if all words covered by idx are also covered by others
        # i.e., cover_count[w] >= 2 for all w in ball(idx)
        # Note: cover_count includes idx itself, so we need cover_count[w] > 1
        return np.all(cover_count[balls[idx]] > 1)

    changed = True
    while changed:
        changed = False
        # Randomize order to break symmetries
        items = list(current_set)
        random.shuffle(items)
        for idx in items:
            if idx in current_set and is_redundant(idx):
                # Remove it
                for j in balls[idx]:
                    cover_count[j] -= 1
                current_set.remove(idx)
                changed = True
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
        
        if time.time() > deadline:
            break

    # Step 2: Simulated Annealing
    # State: current_set, cover_count, uncovered_count
    # Moves:
    # 1. Remove a random codeword
    # 2. Add a random codeword
    # 3. Swap a random codeword with a random other codeword
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # We track energy = size * 1000 + uncovered_count * 10000
    # We prioritize covering, then minimize size.
    # Actually, we want valid solutions (uncovered=0).
    # Let's use a simpler metric:
    # If uncovered > 0, we are in a "bad" state. We want to minimize uncovered first.
    # If uncovered == 0, we want to minimize size.
    
    def get_energy(size, uncov):
        # Strong penalty for uncovered
        return uncov * 10000 + size
        
    current_energy = get_energy(len(current_set), uncovered_count)
    
    while time.time() < deadline and T > min_T:
        # Pick a move
        move_type = random.randint(0, 2)
        
        if move_type == 0: # Remove
            if len(current_set) == 0:
                continue
            idx_to_remove = random.choice(list(current_set))
            
            # Calculate delta
            # Removing idx affects cover_count for j in balls[idx]
            # new_cover_count[j] = cover_count[j] - 1
            # new_uncovered_count = uncovered_count + (number of j in balls[idx] where cover_count[j] == 1)
            # because those become 0
            
            # Efficient check:
            # We need to count how many j in balls[idx] have cover_count[j] == 1
            # and how many have cover_count[j] > 1
            # Actually, we just need the change in uncovered_count.
            # If cover_count[j] == 1, it becomes 0 -> uncovered increases by 1.
            # If cover_count[j] > 1, it becomes > 0 -> uncovered unchanged.
            
            delta_uncov = 0
            for j in balls[idx_to_remove]:
                if cover_count[j] == 1:
                    delta_uncov += 1
            
            delta_size = -1
            delta_energy = get_energy(len(current_set) + delta_size, uncovered_count + delta_uncov) - current_energy
            
            if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
                # Accept
                for j in balls[idx_to_remove]:
                    cover_count[j] -= 1
                current_set.remove(idx_to_remove)
                uncovered_count += delta_uncov
                current_energy += delta_energy
                
                if uncovered_count == 0 and len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    
        elif move_type == 1: # Add
            # Pick a random word not in current_set
            # To be fast, pick a random index from 0..q**n-1
            # If it's already in set, retry or pick another.
            # For small sets, rejection sampling is fine.
            
            while True:
                idx_to_add = random.randint(0, q**n - 1)
                if idx_to_add not in current_set:
                    break
            
            delta_uncov = 0
            for j in balls[idx_to_add]:
                if cover_count[j] == 0:
                    delta_uncov -= 1 # uncovered decreases
            
            delta_size = 1
            delta_energy = get_energy(len(current_set) + delta_size, uncovered_count + delta_uncov) - current_energy
            
            if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
                # Accept
                for j in balls[idx_to_add]:
                    cover_count[j] += 1
                current_set.add(idx_to_add)
                uncovered_count += delta_uncov
                current_energy += delta_energy
                
                if uncovered_count == 0 and len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    
        else: # Swap
            if len(current_set) == 0:
                continue
            idx_to_remove = random.choice(list(current_set))
            
            while True:
                idx_to_add = random.randint(0, q**n - 1)
                if idx_to_add not in current_set:
                    break
            
            # Calculate delta for removal
            delta_uncov_remove = 0
            for j in balls[idx_to_remove]:
                if cover_count[j] == 1:
                    delta_uncov_remove += 1
            
            # Calculate delta for addition
            # Note: We are adding idx_to_add to the set AFTER removing idx_to_remove?
            # Or simultaneously? 
            # If we do it simultaneously, the intermediate state doesn't matter for the final count,
            # but the delta calculation is easier if we think of it as:
            # 1. Remove idx_to_remove
            # 2. Add idx_to_add
            
            # Let's simulate the two steps.
            # Step 1: Remove idx_to_remove
            # cover_count_temp[j] = cover_count[j] - 1 for j in balls[idx_to_remove]
            # Step 2: Add idx_to_add
            # cover_count_final[j] = cover_count_temp[j] + 1 for j in balls[idx_to_add]
            
            # This is complex to do in one pass without temp array.
            # Given ball size is small (13), we can just update a temp array or do it carefully.
            # But for speed, let's just do the updates and rollback if rejected.
            # Since we are in a tight loop, let's stick to simple Add/Remove moves for now,
            # or implement swap by doing Remove then Add sequentially in the acceptance logic.
            
            # Actually, simple Add and Remove moves are sufficient for SA to explore the space.
            # Swaps are helpful but Add/Remove can simulate them (Remove then Add).
            # Given the time constraint, I'll stick to Add/Remove for robustness and simplicity.
            pass
        
        T *= cooling_rate
        
    # Final check: if best_set is empty, return initial? No, best_set is always valid if we started valid.
    # But what if SA broke validity and didn't recover? 
    # We track best_set only when uncovered_count == 0.
    # So best_set is always a valid covering code.
    
    # Convert best_set indices back to lists of digits
    result = []
    for idx in best_set:
        result.append(list(words[idx]))
        
    return result
