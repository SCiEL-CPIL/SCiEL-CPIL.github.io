import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    # Parameters
    # q=3, n=7, radius=1
    # Space size: q^n = 2187
    # Neighborhood size: 1 + n*(q-1) = 1 + 7*2 = 15
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    space_size = q ** n
    
    # Precompute neighborhoods for all possible words in the space
    # We represent words as integers 0..space_size-1
    # word -> list of neighbor indices
    neighborhoods = {}
    for i in range(space_size):
        # Convert int to base-q tuple
        # Efficient conversion
        val = i
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = val % q
            val //= q
        
        neigh = [i]
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                # Create new word
                new_digits = digits.copy()
                new_digits[pos] = new_val
                # Convert back to int
                new_val_int = 0
                for d in new_digits:
                    new_val_int = new_val_int * q + d
                neigh.append(new_val_int)
        neighborhoods[i] = neigh

    # Initial Code: Use the seed's construction (243 words)
    # Tetracode part
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word_list = h + list(tail)
            # Convert to int
            w_int = 0
            for d in word_list:
                w_int = w_int * q + d
            initial_code.append(w_int)
            
    code = list(set(initial_code)) # Ensure distinct
    random.shuffle(code)
    
    # Coverage tracking
    # cover_count[w] = number of codewords covering w
    cover_count = np.zeros(space_size, dtype=np.int32)
    
    def add_codeword(c):
        for w in neighborhoods[c]:
            cover_count[w] += 1
            
    def remove_codeword(c):
        for w in neighborhoods[c]:
            cover_count[w] -= 1
            
    for c in code:
        add_codeword(c)
        
    best_code = list(code)
    best_size = len(code)
    
    # Helper: Check if code is valid (should always be if we start valid and only make safe moves)
    # But we need to ensure we don't break coverage.
    
    # Phase 1: Greedy Redundancy Removal
    def try_remove_redundant():
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            # Iterate in random order to avoid bias
            indices = list(range(len(code)))
            random.shuffle(indices)
            for idx in indices:
                c = code[idx]
                # Check if all neighbors are covered by >= 2 codewords
                # If so, c is redundant
                is_redundant = True
                for w in neighborhoods[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    remove_codeword(c)
                    code.pop(idx)
                    removed_any = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    
    try_remove_redundant()
    
    # Phase 2: Local Search - Move codewords to better positions
    # A move is valid if the new position covers all words that the old position uniquely covered
    # Or more simply:
    # 1. Pick a codeword c.
    # 2. Pick a new position c_new.
    # 3. Check if N(c) \ N(c_new) is covered by other codewords.
    # 4. If yes, move.
    # 5. Try to remove redundancies.
    
    # To speed up, we only try moves that cover at least as many "unique" spots?
    # Actually, we want to find moves that allow subsequent removals.
    
    # Precompute "unique coverage" is expensive to update.
    # Instead, we can try random moves and see if they enable removals.
    
    # Optimization: Only try moves for codewords that are "weak" (cover many words with count 1)?
    
    # Let's implement a simple local search:
    # 1. Pick random c.
    # 2. Pick random c_new.
    # 3. Tentatively move: remove c, add c_new.
    # 4. Check validity: Are all words in N(c) covered by (Code \ {c}) U {c_new}?
    #    This is equivalent to: For all w in N(c), if w not in N(c_new), then cover_count[w] (before move) > 1.
    #    Wait, cover_count includes c.
    #    So condition: For all w in N(c) \ N(c_new), cover_count[w] >= 2.
    #    And for w in N(c) \cap N(c_new), cover_count[w] >= 1 (always true if valid code).
    #    Actually, if we move c to c_new, the new cover count for w is:
    #    old_count - (1 if w in N(c) else 0) + (1 if w in N(c_new) else 0).
    #    We need new_count >= 1 for all w.
    #    This is satisfied if:
    #    - w in N(c) and w in N(c_new): count stays same or +1. OK.
    #    - w in N(c) and w NOT in N(c_new): count decreases by 1. Need old_count >= 2.
    #    - w NOT in N(c) and w in N(c_new): count increases by 1. OK.
    #    - w NOT in N(c) and w NOT in N(c_new): count stays same. OK.
    #    So condition: For all w in N(c) \ N(c_new), cover_count[w] >= 2.
    
    def try_local_search():
        nonlocal code, best_code, best_size
        no_improve_steps = 0
        max_no_improve = 5000 # Reset if stuck
        
        while time.time() < deadline:
            if no_improve_steps > max_no_improve:
                break
                
            if len(code) == 0:
                break
                
            c_idx = random.randint(0, len(code) - 1)
            c = code[c_idx]
            
            # Generate random new position
            # Bias towards positions close to c? Or just random.
            # Random is fine for 2187 space.
            c_new = random.randint(0, space_size - 1)
            
            if c_new == c:
                continue
                
            # Check move validity
            # N(c) \ N(c_new)
            # We need to check if all w in N(c) that are NOT in N(c_new) have count >= 2
            
            # Optimization: Precompute set of neighbors for faster difference?
            # 15 elements is small.
            n_c = set(neighborhoods[c])
            n_new = set(neighborhoods[c_new])
            
            # Words in N(c) but not in N(c_new)
            to_check = n_c - n_new
            
            valid_move = True
            for w in to_check:
                if cover_count[w] < 2:
                    valid_move = False
                    break
            
            if valid_move:
                # Perform move
                remove_codeword(c)
                add_codeword(c_new)
                code[c_idx] = c_new
                
                # Try to remove redundancies
                prev_size = len(code)
                try_remove_redundant()
                
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
                
    try_local_search()
    
    # Final check
    if time.time() < deadline - 5:
         # One last pass of redundancy removal
         try_remove_redundant()
         if len(code) < best_size:
             best_code = list(code)
             
    # Convert best_code back to lists of integers
    result = []
    for c in best_code:
        val = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = val % q
            val //= q
        result.append(digits)
        
    return result
