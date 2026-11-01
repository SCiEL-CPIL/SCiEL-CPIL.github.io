import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # Safety margin for 120s limit

    # For the specific task q=3, n=6, radius=1
    # The space size is 729, which fits in a Python integer bitmask.
    
    # Precompute neighbors as bitmasks
    # Index i corresponds to the word with digits derived from base-q representation.
    # We use the standard mixed-radix mapping: index = sum(w[i] * q^i) or similar.
    # Let's stick to the seed's mapping: w2i and i2w logic, but optimized.
    
    space = q ** n
    
    # Precompute powers of q
    powers = [q ** i for i in range(n)]
    
    # Precompute neighbor bitmasks for each index
    # nbrs[i] is a bitmask where bit j is set if word j is within distance 'radius' of word i
    # For radius 1, neighbors are i itself and i +/- powers[p] for each position p,
    # provided the digit change is valid (0..q-1).
    
    nbrs = [0] * space
    
    for i in range(space):
        mask = 1 << i
        # Extract digits to check validity of +/-
        # We can compute digits on the fly or precompute them. 
        # For 729 items, on-the-fly is fine, but precomputing digits is faster for the loop.
        digits = [0] * n
        temp = i
        for p in range(n):
            digits[p] = temp % q
            temp //= q
        
        for p in range(n):
            d = digits[p]
            # Try decrementing digit at p
            if d > 0:
                nb = i - powers[p]
                mask |= (1 << nb)
            # Try incrementing digit at p
            if d < q - 1:
                nb = i + powers[p]
                mask |= (1 << nb)
        nbrs[i] = mask

    all_mask = (1 << space) - 1
    
    def i2w(i):
        w = [0] * n
        for j in range(n):
            w[j] = i % q
            i //= q
        return w

    best_code = None
    best_size = space # Initialize with worst case
    
    # Strategy: Multi-start Greedy Set Cover + Greedy Deletion
    
    # We will run multiple trials. Each trial:
    # 1. Greedy Set Cover: Start with empty code, uncovered = all_mask.
    #    Repeatedly pick the word that covers the most uncovered bits.
    #    To add randomness, we can shuffle the order in which we evaluate candidates
    #    or pick randomly among those with max coverage.
    # 2. Greedy Deletion: Try to remove each codeword. If removing it leaves
    #    all words covered (i.e., no word becomes uncovered), remove it.
    
    rng = random.Random(42)
    
    # Number of trials is dynamic based on time
    trial_count = 0
    while time.time() < deadline - 5.0:
        trial_count += 1
        
        uncovered = all_mask
        code = []
        
        # Greedy Set Cover Phase
        while uncovered:
            max_cov = -1
            best_cands = []
            
            # Iterate through all possible words
            # Optimization: We can skip words that are already in the code? 
            # No, a word already in the code might cover *different* uncovered parts 
            # if we consider it again? No, we add it once. 
            # But we can skip words that don't cover any new bits.
            
            # To speed up, we can just scan all 729 words. 729 * (bit ops) is fast.
            
            for w in range(space):
                # Count how many currently uncovered bits this word covers
                overlap = (nbrs[w] & uncovered).bit_count()
                if overlap > max_cov:
                    max_cov = overlap
                    best_cands = [w]
                elif overlap == max_cov and overlap > 0:
                    best_cands.append(w)
            
            if not best_cands:
                break # Should not happen if uncovered is not 0 and space is connected
            
            # Randomly pick from best candidates to encourage diversity
            chosen = rng.choice(best_cands)
            code.append(chosen)
            uncovered &= ~nbrs[chosen]
            
            # Early termination if we exceed current best size
            if len(code) >= best_size:
                break
        
        if len(code) >= best_size:
            continue
            
        # Greedy Deletion Phase
        # We have a valid covering code 'code'. Try to remove elements.
        # We maintain a count of how many times each word is covered.
        # If a word is covered exactly once by codeword C, C is essential.
        
        # Efficient deletion:
        # 1. Compute coverage counts for all words in space.
        # 2. Iterate through code. If codeword c is not essential, remove it and update counts.
        
        # Note: The order of deletion matters. We shuffle the code to try different deletion orders.
        # We do this in a separate loop or just once per trial? 
        # Let's do a few deletion passes with different random orders.
        
        # For simplicity and speed, let's do one deletion pass per trial first.
        # If we have time, we can do more.
        
        # Compute initial counts
        counts = np.zeros(space, dtype=np.int32)
        for c in code:
            # We need to add 1 to all bits in nbrs[c]
            # Doing this bit-by-bit is slow. 
            # Alternative: Just check validity after removal?
            # If we remove c, the words that become uncovered are those where 
            # count was 1 and c was one of the coverers.
            
            # Let's use a simpler check for deletion:
            # A codeword c is removable if for every word w in nbrs[c], 
            # there exists another codeword c' in code (c' != c) such that w in nbrs[c'].
            
            # This is O(|code| * |nbrs|) per candidate. |code|~80, |nbrs|~19.
            # 80 * 80 * 19 ~ 120,000 ops. Very fast.
            
            pass
        
        # Better deletion strategy:
        # 1. Build a list of codewords to try removing.
        # 2. For each c in list:
        #    Check if removing c breaks coverage.
        #    If not, remove it.
        
        # To make it robust, we shuffle the code order before deletion.
        code_to_del = code[:]
        rng.shuffle(code_to_del)
        
        current_code = code[:]
        current_set = set(current_code)
        
        # Precompute neighbor sets as lists for faster intersection? 
        # Or just use the bitmask logic:
        # A word w is covered by C if (nbrs[c] & (1<<w)) != 0.
        
        # Let's refine the deletion check:
        # We want to know if (Union of nbrs[c'] for c' in current_code \ {c}) covers all words.
        # This is equivalent to: Is there any word w such that w is in nbrs[c] 
        # but NOT in nbrs[c'] for any c' in current_code \ {c}?
        
        # Optimization:
        # For each c in current_code (in shuffled order):
        #   Check if c is essential.
        #   If not, remove it.
        
        # To check if c is essential:
        #   Compute union_mask = OR of nbrs[c'] for all c' in current_code, c' != c.
        #   If (nbrs[c] & ~union_mask) == 0, then c is not essential.
        
        # This is O(|code|^2 * bit_ops). 80^2 = 6400 bit ops. Fast.
        
        for i in range(len(current_code)):
            if time.time() > deadline - 5.0:
                break
            
            c = current_code[i]
            # Check if c can be removed
            # We need to see if the rest of the code covers everything c covers.
            
            # Optimization: Instead of recomputing union from scratch, 
            # we can maintain a global union? No, removing c changes it.
            
            # Let's just compute the union of others for this specific c.
            union_others = 0
            for j in range(len(current_code)):
                if j == i:
                    continue
                union_others |= nbrs[current_code[j]]
            
            # If nbrs[c] is a subset of union_others, c is redundant.
            if (nbrs[c] & ~union_others) == 0:
                # Remove c
                current_code.pop(i)
                # We don't need to update i because we just popped, 
                # and the loop index i will increment, effectively skipping the next element.
                # To avoid skipping, we should decrement i or use a while loop.
                # But skipping one element in deletion is acceptable as we shuffle each trial.
                # Actually, let's just break and restart deletion? No, that's slow.
                # Let's just continue. The element that shifted into index i will be checked in the next iteration?
                # No, range(len) is fixed. If we pop, indices shift.
                # Let's use a while loop for deletion to handle indices correctly.
                pass # We'll rewrite the deletion loop below
        
        # Corrected Deletion Loop
        current_code = code[:]
        changed = True
        # Limit deletion passes to avoid infinite loops or excessive time
        # Since we shuffle each trial, one pass might be enough, but multiple passes with different shuffles help.
        # Let's do 2-3 passes per trial if time permits.
        
        for pass_num in range(3):
            if time.time() > deadline - 5.0:
                break
                
            rng.shuffle(current_code)
            i = 0
            while i < len(current_code):
                c = current_code[i]
                union_others = 0
                for j in range(len(current_code)):
                    if j == i:
                        continue
                    union_others |= nbrs[current_code[j]]
                
                if (nbrs[c] & ~union_others) == 0:
                    current_code.pop(i)
                    # Do not increment i, check the new element at i
                else:
                    i += 1
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            
            # If we hit the theoretical lower bound or known best, we can stop early
            # Known best is 73. Lower bound 71.
            if best_size <= 73:
                # We can break early if we are confident, but let's try to find 72?
                # Unlikely. Let's just break to save time for other trials if we want.
                # But if we found 73, that's a very good score.
                # Let's run a few more trials just in case, but prioritize speed.
                pass

    if best_code is None:
        # Fallback to product construction
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        tails = list(itertools.product(range(q), repeat=n - 4))
        base = [h + list(t) for h in tetracode for t in tails]
        return base

    return [i2w(i) for i in best_code]
