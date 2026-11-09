import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512

    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        mask: int where bit i is 1 if i is in the set.
        """
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        """Convert bitmask to list of integers."""
        res = []
        for i in range(1, 513):
            if (mask >> i) & 1:
                res.append(i)
        return res

    def list_to_mask(lst):
        """Convert list of integers to bitmask."""
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    candidates = []

    # 1. Generate Structured Algebraic Candidates
    
    # Family A: Union of two disjoint Arithmetic Progressions
    # A = {a + i * stepA}, B = {b + j * stepB}
    # We need 32 distinct elements. Let's try splitting 16+16 or other ratios.
    
    def gen_union_ap():
        sets = []
        # Try different splits
        for n1 in range(8, 25):
            n2 = N - n1
            if n2 < 4: continue
            
            # Step sizes must be large enough to avoid collisions and stay in range
            # Max range is 512. If we have n elements with step s, span is (n-1)*s.
            # We want to pack them densely enough to create APs but distinct enough.
            
            # Heuristic: Try steps that are divisors or near-divisors of 512, or random
            # Let's try specific structured steps
            possible_steps_a = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32]
            possible_steps_b = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32]
            
            # To keep search space small, we'll sample a few combinations
            for sa in possible_steps_a:
                if (n1 - 1) * sa > 500: continue
                for sb in possible_steps_b:
                    if (n2 - 1) * sb > 500: continue
                    
                    # Random offsets to shift the progressions
                    for _ in range(3):
                        offset_a = random.randint(1, 512 - (n1 - 1) * sa)
                        offset_b = random.randint(1, 512 - (n2 - 1) * sb)
                        
                        set_a = [offset_a + i * sa for i in range(n1)]
                        set_b = [offset_b + j * sb for j in range(n2)]
                        
                        combined = set_a + set_b
                        if len(combined) == N and all(1 <= x <= MAX_VAL for x in combined):
                            candidates.append(combined)
        
        # Limit candidates from this family
        return candidates[-100:] # Keep last 100 generated

    candidates.extend(gen_union_ap())

    # Family B: Cantor-like Base-4 Sets
    # Represent numbers in base 4. Select numbers with specific digit patterns.
    # For example, numbers where the 2nd digit is 0 or 1.
    
    def gen_cantor_like():
        sets = []
        # Base 4 representation of 512 is 31100 (approx). 512 = 4^4 * 3 + ...
        # Actually 4^4 = 256, 4^5 = 1024. So 512 fits in 5 digits (0-31100 in base 4? No, 512/256=2, rem 0. 20000 in base 4).
        
        # Let's generate sets by filtering numbers 1..512 based on their base-4 digits.
        # Pattern: Keep numbers where digit k is in {0, 1}.
        # This creates a "Cantor dust" which often has good additive properties.
        
        digits = 5 # Enough for 512
        
        for pos in range(digits):
            for allowed_digits in [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]]:
                # We need exactly 32 numbers.
                # Filtering might give too many or too few.
                # We can collect all matching numbers and then sample or take the first 32.
                
                matching = []
                for x in range(1, MAX_VAL + 1):
                    # Get digit at position pos (0 is least significant)
                    d = (x // (4 ** pos)) % 4
                    if d in allowed_digits:
                        matching.append(x)
                
                if len(matching) >= N:
                    # Take a random subset of size N to vary the structure
                    # Or take the first N? Random is better for optimization start.
                    # Let's take a few random samples
                    for _ in range(2):
                        sample = random.sample(matching, N)
                        candidates.append(sample)
        
        return candidates[-50:]

    candidates.extend(gen_cantor_like())

    # Family C: Mixed Modular Grids
    # Set = { x in 1..512 : x % m in S } for some modulus m and subset S
    # This creates a periodic structure.
    
    def gen_modular():
        sets = []
        for m in range(2, 20):
            # We need to pick a subset S of {0, ..., m-1} such that we get ~32 elements.
            # Density is |S|/m. Total elements ~ 512 * |S|/m.
            # We want 32, so |S|/m ~ 32/512 = 1/16.
            # So if m=16, |S|=1. If m=8, |S|=0.5 (impossible, so m=8, |S|=1 gives 64 elements).
            
            target_density = N / MAX_VAL
            for s_size in range(1, m):
                if abs(s_size / m - target_density) < 0.05:
                    # Generate random subsets S of size s_size
                    for _ in range(5):
                        S = set(random.sample(range(m), s_size))
                        matching = [x for x in range(1, MAX_VAL + 1) if x % m in S]
                        if len(matching) >= N:
                            sample = random.sample(matching, N)
                            candidates.append(sample)
                        elif len(matching) == N:
                            candidates.append(matching)
        return candidates[-50:]

    candidates.extend(gen_modular())

    # Family D: Pure Random (Baseline)
    for _ in range(20):
        candidates.append(random.sample(range(1, MAX_VAL + 1), N))

    # Family E: Dense Blocks
    # Two dense blocks of 16 numbers
    for _ in range(10):
        start1 = random.randint(1, 512 - 16)
        start2 = random.randint(1, 512 - 16)
        if abs(start1 - start2) > 20: # Ensure disjoint
            set1 = list(range(start1, start1 + 16))
            set2 = list(range(start2, start2 + 16))
            candidates.append(set1 + set2)

    # Evaluate initial candidates
    best_score = -1
    best_list = None
    best_mask = 0

    for lst in candidates:
        if lst is None or len(lst) != N:
            continue
        m = list_to_mask(lst)
        s = score_mask(m)
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m

    # 2. Fast Greedy Local Search Refinement
    # Unlike SA, this is a deterministic hill-climbing with random restarts if stuck.
    # It tries to swap elements to increase the score.
    
    current_list = best_list[:]
    current_score = best_score
    current_mask = best_mask
    current_set = set(current_list)
    
    # We will run a series of refinement passes.
    # Each pass tries to improve the current best.
    
    refinement_limit = 50 # Max refinements per restart
    
    # To avoid getting stuck in a local optimum, we will run this on the top few candidates
    top_candidates = []
    for lst in candidates:
        if lst is None or len(lst) != N: continue
        m = list_to_mask(lst)
        s = score_mask(m)
        top_candidates.append((s, lst))
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Process top 5 distinct candidates
    processed_masks = set()
    for init_score, init_list in top_candidates[:5]:
        if time.time() > deadline - 5:
            break
        
        m_init = list_to_mask(init_list)
        if m_init in processed_masks:
            continue
        processed_masks.add(m_init)
        
        cur_list = init_list[:]
        cur_set = set(cur_list)
        cur_mask = m_init
        cur_score = init_score
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try swapping each element with a random new element
            # We prioritize swaps that create new APs.
            
            # To speed up, we can sample swaps rather than trying all
            # But N=32 is small. Let's try a few random swaps per element.
            
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                    
                old_val = cur_list[i]
                
                # Sample 10 potential new values
                for _ in range(10):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val == old_val or new_val in cur_set:
                        continue
                    
                    new_mask = cur_mask ^ (1 << old_val) ^ (1 << new_val)
                    new_score = score_mask(new_mask)
                    
                    if new_score > cur_score:
                        # Accept
                        cur_list[i] = new_val
                        cur_set.remove(old_val)
                        cur_set.add(new_val)
                        cur_mask = new_mask
                        cur_score = new_score
                        improved = True
                        
                        if cur_score > best_score:
                            best_score = cur_score
                            best_list = cur_list[:]
                            best_mask = cur_mask
                            
                        break # Restart inner loop for this element? Or continue?
                        # Breaking restarts the sampling for this element, which is good.
                        # But we might want to try other elements too.
                        # Let's just break out of the swap attempts for this element.
                    
            # If no improvement in this full pass, we might be stuck.
            # To escape, we could do a random perturbation, but that risks dropping score.
            # Given the short time, simple hill climbing is safer.
            # If stuck, we exit the while loop for this candidate.
            
            # Actually, the `improved` flag only stays True if we found a swap in the current pass.
            # If we found a swap, we set improved=True and continue the outer while loop.
            # If we didn't, improved stays False (initially False, set to True only on swap).
            # Wait, `improved` is set to False at start of while.
            # If we find a swap, we set it to True.
            # So if we find NO swaps in the entire pass over all 32 elements, improved remains False.
            # Then the while loop terminates.
            # This is standard hill climbing.

    return best_list
# EVOLVE-BLOCK-END
