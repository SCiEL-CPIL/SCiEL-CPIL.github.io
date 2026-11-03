import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s; leave safety margin for conversion and cleanup
    deadline = start_time + 142.0

    total_space = q ** n

    # 1. Precompute powers for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # 2. Precompute neighbor indices for each word (Hamming distance <= radius)
    # For radius=1, neighbors are the word itself + 1 word for each (pos, new_val)
    # We need to map (word, pos, new_val) to an index in 0..total_space-1
    # To speed up MILP matrix construction, we precompute the sparse structure
    
    # For each word w, list of indices it covers
    # Since total_space can be large (2187 for 3^7), we can afford a list of lists
    # Or better, build the constraint matrix A directly in sparse format.
    
    # A is (total_space, total_space) where A[i, j] = 1 if word i is covered by codeword j
    # We want sum_j x_j * A[i, j] >= 1 for all i
    
    # Let's build A as a scipy sparse matrix
    from scipy.sparse import lil_matrix, csr_matrix

    # Initialize sparse matrix
    # Rows: words to be covered (0..total_space-1)
    # Cols: candidate codewords (0..total_space-1)
    
    # We can exploit symmetry or just build it.
    # 2187x2187 is small enough for dense, but sparse is safer for generic n.
    # Let's use lil_matrix for construction then convert to csr.
    
    A = lil_matrix((total_space, total_space), dtype=np.int8)
    
    # Precompute neighbors for each word
    # For each word j (candidate), find all words i it covers
    # i is covered by j if dist(i, j) <= radius
    
    # Efficient way: iterate j, generate neighbors
    for j in range(total_space):
        # Get digits of j
        digits = [0] * n
        tmp = j
        for pos in range(n):
            digits[pos] = tmp % q
            tmp //= q
        
        # Generate all neighbors within radius
        # For radius=1: self + n*(q-1) neighbors
        # For general radius: combinatorial explosion, but n=7, q=3 is small.
        # Let's implement a general neighbor generator using recursion or itertools
        
        # Since n is small (7) and q is small (3), we can just iterate all i and check distance?
        # 2187 * 2187 = 4.7M ops. This is fast in Python if optimized, or slow if not.
        # Let's use the bitmask trick or direct generation.
        
        # Direct generation for radius=1 is very fast.
        # Let's assume radius is small (1 or 2).
        
        # Helper to check distance
        def dist(a, b):
            d = 0
            for k in range(n):
                if (a // powers[k]) % q != (b // powers[k]) % q:
                    d += 1
                    if d > radius:
                        return d
            return d

        # This O(N^2 * N) approach is 2187^2 * 7 ~ 33M ops. Might take a few seconds in Python.
        # Let's optimize: Precompute digits for all words.
        
    # Better: Precompute digits for all words into a numpy array
    digits_all = np.zeros((total_space, n), dtype=np.int8)
    for idx in range(total_space):
        val = idx
        for pos in range(n):
            digits_all[idx, pos] = val % q
            val //= q

    # Now compute A
    # A[i, j] = 1 if sum(|digits_all[i] - digits_all[j]|) <= radius
    
    # Vectorized distance calculation is hard for sparse A construction directly.
    # Let's stick to the loop but optimize it.
    
    # Actually, for q=3, n=7, radius=1:
    # Each codeword covers 1 + 7*(3-1) = 15 words.
    # Total non-zeros in A = 2187 * 15 = 32805.
    # This is very sparse.
    
    # Let's build A by iterating j and generating its specific neighbors.
    A = lil_matrix((total_space, total_space), dtype=np.int8)
    
    for j in range(total_space):
        # j covers itself
        A[j, j] = 1
        
        if radius >= 1:
            # Change one position
            for pos in range(n):
                orig_val = (j // powers[pos]) % q
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    # Calculate new index
                    # i = j + (new_val - orig_val) * powers[pos]
                    i = j + (new_val - orig_val) * powers[pos]
                    A[i, j] = 1
                    
        if radius >= 2:
            # Change two positions
            # This gets complex. Let's assume radius is small.
            # For n=7, q=3, radius=2, neighbors = 1 + 14 + 21*4 = 115.
            # We can generate these by iterating pairs of positions.
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    v1 = (j // powers[pos1]) % q
                    v2 = (j // powers[pos2]) % q
                    for nv1 in range(q):
                        if nv1 == v1: continue
                        for nv2 in range(q):
                            if nv2 == v2: continue
                            i = j + (nv1 - v1) * powers[pos1] + (nv2 - v2) * powers[pos2]
                            A[i, j] = 1

    A = A.tocsr()

    # 3. Formulate MILP
    # Minimize sum(x_j)
    # Subject to: A @ x >= 1
    # x_j in {0, 1}
    
    c = np.ones(total_space)
    
    # Constraints: A @ x >= 1
    # scipy.milp uses LinearConstraint(A, lb, ub)
    # A @ x >= 1  =>  A @ x in [1, inf)
    constraints = LinearConstraint(A, lb=1, ub=np.inf)
    
    integrality = np.ones(total_space) # All binary
    bounds = Bounds(lb=0, ub=1)
    
    # 4. Solve
    # time_limit in seconds
    time_left = deadline - start_time - 2.0 # Leave 2s for post-processing
    if time_left < 5:
        # Not enough time, fallback to greedy
        pass
    else:
        try:
            res = milp(c, integrality=integrality, constraints=constraints, bounds=bounds, 
                       options={"time_limit": time_left})
            
            if res.status == 0 or res.status == 1: # Optimal or Feasible
                x = res.x
                # x might be float, round it
                x = np.round(x).astype(int)
                
                # Extract selected codewords
                selected_indices = np.where(x == 1)[0]
                
                # Convert to tuples
                best_words = []
                for idx in selected_indices:
                    word = [0] * n
                    val = idx
                    for pos in range(n):
                        word[pos] = val % q
                        val //= q
                    best_words.append(word)
                
                # Sanity check: ensure distinct and valid
                # The MILP ensures distinctness via binary variables (index uniqueness)
                # and validity via constraints.
                
                # If MILP fails to find a feasible solution within time, or returns empty,
                # we need a fallback.
                if len(best_words) > 0:
                    return best_words
        except Exception as e:
            pass # Fall through to fallback

    # 5. Fallback: Greedy Set Cover
    # If MILP fails or times out without a solution, use the greedy approach from the seed.
    # We need a fast greedy implementation.
    
    # Precompute masks for greedy (using Python ints as bitsets)
    # 2187 bits fits in ~35 64-bit words, or just Python big ints.
    
    masks = [0] * total_space
    for j in range(total_space):
        m = 0
        # j covers itself
        m |= (1 << j)
        
        if radius >= 1:
            for pos in range(n):
                orig_val = (j // powers[pos]) % q
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    i = j + (new_val - orig_val) * powers[pos]
                    m |= (1 << i)
                    
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    v1 = (j // powers[pos1]) % q
                    v2 = (j // powers[pos2]) % q
                    for nv1 in range(q):
                        if nv1 == v1: continue
                        for nv2 in range(q):
                            if nv2 == v2: continue
                            i = j + (nv1 - v1) * powers[pos1] + (nv2 - v2) * powers[pos2]
                            m |= (1 << i)
                            
        masks[j] = m

    full_mask = (1 << total_space) - 1
    remaining = full_mask
    selected = []
    
    # Greedy
    while remaining and time.time() < deadline - 1:
        best_gain = -1
        best_word = -1
        
        # Optimization: only check words that cover at least 1 remaining element
        # Simple loop is O(N^2) which is ~4.7M iterations. 
        # In Python, this might take 1-2 seconds per pass.
        # We have ~140s left. Greedy might need ~100-200 words.
        # 100 * 4.7M = 470M ops. Too slow in pure Python loop.
        
        # Vectorized greedy?
        # gains = np.array([(masks[w] & remaining).bit_count() for w in range(total_space)])
        # This list comprehension is still Python-level loop.
        
        # Let's try to speed up the greedy step.
        # We can use numpy to count bits? No, bit_count is fast in C, but the loop is Python.
        
        # Alternative: Use the MILP result even if not optimal?
        # If res.status == 2 (Infeasible) or 3 (Unbounded) or 4 (Iteration limit) 
        # but res.x is not None, we might have a partial solution? 
        # scipy.milp usually returns res.x=None if no feasible solution found.
        
        # Let's implement a faster greedy using precomputed neighbor lists and a "coverage count" array.
        # This is the "Polish" approach but used for construction.
        
        # Actually, the seed program's greedy was the bottleneck.
        # Let's just use a randomized greedy with a time limit.
        
        # To make it faster, we can sample candidates? No, greedy needs max gain.
        
        # Let's rely on the fact that MILP is very fast for this size.
        # 2187 vars, 2187 constraints. HiGHS should solve this in milliseconds to seconds.
        # If it fails, we return an empty list? No, must be valid.
        
        # Fallback: Return all words? Score is N, so 2187. Bad score but valid.
        # Better Fallback: A simple greedy that runs fast enough.
        
        # Let's optimize the greedy loop by breaking early if gain is 0.
        # And we can precompute the list of neighbors for each word to speed up "gain" calculation?
        # gain = number of neighbors of w that are in remaining.
        
        # Let's just run the greedy loop, but it might be slow.
        # If time is tight, we return a trivial solution (all words) to ensure validity, 
        # though the score will be poor.
        
        # However, the prompt says "Keep the function names...". 
        # If MILP succeeds, we are done. 
        # If MILP fails, we need a valid solution.
        
        # Let's try to run a limited greedy.
        
        # To speed up greedy, we can use a "lazy" evaluation.
        # But for now, if MILP fails, we fall back to a simple heuristic.
        
        # Since MILP is highly likely to succeed for this size, the fallback is rarely used.
        # We will implement a very simple fallback: return a random subset if greedy is too slow?
        # No, must cover all.
        
        # Let's just run the greedy loop but with a strict time check inside.
        
        while remaining and time.time() < deadline - 0.5:
            best_gain = 0
            best_word = -1
            
            # Sample a subset of words to speed up search? 
            # No, greedy needs the best.
            # Let's just iterate.
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    
            if best_word == -1:
                break # Should not happen if remaining is not 0
                
            selected.append(best_word)
            remaining &= ~masks[best_word]
            
            # If we are running out of time, break and hope we covered everything?
            # No, we must return a valid cover.
            # If we break early, the solution is invalid.
            # So we must ensure we finish.
            # 2187 words * ~150 avg size? No, ~180 words.
            # 180 * 2187 = ~400k iterations.
            # Each iteration does a bit_count on a 2187-bit int.
            # This should be fast enough in C/Python hybrid.
            
        # Final check: if remaining != 0, we have a problem.
        # Add all remaining words to selected to ensure validity?
        # That would blow up the size.
        # But validity is more important than score (score 0 for invalid).
        
        if remaining:
            # Add words that cover the most remaining elements until done
            while remaining and time.time() < deadline - 0.1:
                best_gain = 0
                best_word = -1
                for w in range(total_space):
                    if w in selected: continue
                    gain = (masks[w] & remaining).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w
                if best_word == -1: break
                selected.append(best_word)
                remaining &= ~masks[best_word]
                
            # If still remaining, add ALL uncovered words directly?
            # No, we need codewords.
            # If we can't cover them, we are stuck.
            # But greedy always terminates with a valid cover if we don't break early.
            # The break condition `time.time() < deadline` is the risk.
            # To be safe, if we hit the deadline, we return the current selected 
            # only if it covers everything. Otherwise, we return a trivial valid set (all words).
            
            if remaining:
                # Fallback to all words
                selected = list(range(total_space))
                remaining = 0

        # Convert to tuples
        best_words = []
        for idx in selected:
            word = [0] * n
            val = idx
            for pos in range(n):
                word[pos] = val % q
                val //= q
            best_words.append(word)
            
        return best_words
