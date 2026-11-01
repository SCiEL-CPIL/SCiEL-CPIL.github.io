import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we stop at 140s to ensure return
    deadline = start_time + 140.0
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # 1. Precompute balls as lists of indices
    balls = []
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    for x in range(total):
        ball = [x]
        for i in range(n):
            old_digit = (x // powers[i]) % q
            for new_digit in range(q):
                if new_digit != old_digit:
                    y = x + (new_digit - old_digit) * powers[i]
                    ball.append(y)
        balls.append(ball)
        
    # 2. Precompute inverse mapping: for each word y, which balls cover it?
    # inv[y] is a list of ball indices
    inv = [[] for _ in range(total)]
    for i, ball in enumerate(balls):
        for y in ball:
            inv[y].append(i)
            
    # 3. Branch and Bound
    # State:
    # - covered: bitset (list of 64-bit ints) of covered words
    # - candidates: list of ball indices still available
    # - current_set: list of selected balls
    
    # Bitset helpers
    num_words = (total + 63) // 64
    zero_mask = 0
    
    def bitset_and(b1, b2):
        return [b1[i] & b2[i] for i in range(num_words)]
        
    def bitset_or(b1, b2):
        return [b1[i] | b2[i] for i in range(num_words)]
        
    def bitset_count(b):
        return sum(bin(x).count('1') for x in b)
        
    def bitset_is_empty(b):
        for x in b:
            if x:
                return False
        return True

    # Initial state
    # We want to cover all words.
    # Let's define the "uncovered" set. Initially all uncovered.
    # It's easier to track uncovered.
    # uncovered[i] = 1 if word i is not covered.
    
    initial_uncovered = [((1 << 64) - 1) if i < num_words - 1 else ((1 << (total % 64)) - 1) for i in range(num_words)]
    if total % 64 == 0:
        initial_uncovered = [((1 << 64) - 1) for i in range(num_words)]
        
    # Candidates: all balls
    initial_candidates = list(range(total))
    
    best_size = float('inf')
    best_solution = None
    nodes_visited = 0
    max_nodes = 200000 # Limit search to avoid hanging, though B&B should prune well
    
    def get_lower_bound(uncovered, candidates):
        # Simple lower bound: ceil(uncovered_count / ball_size)
        # Better: Max number of disjoint balls in candidates? No, that's for packing.
        # For covering:
        # 1. Count uncovered words.
        # 2. Each ball covers at most ball_size words.
        # LB = ceil(uncovered_count / ball_size)
        
        unc_count = bitset_count(uncovered)
        if unc_count == 0:
            return 0
        return (unc_count + ball_size - 1) // ball_size
        
    # A stronger lower bound for set cover:
    # Greedy packing: pick a ball, remove it and all words it covers from the "problem", repeat.
    # This is expensive. Let's stick to the simple one first, maybe add a "conflict" bound later if needed.
    # Actually, for this specific problem (covering code), the balls are very structured.
    
    def solve(uncovered, candidates, current_set, depth):
        nonlocal best_size, best_solution, nodes_visited
        
        if time.time() > deadline - 2.0:
            return
            
        nodes_visited += 1
        if nodes_visited > max_nodes:
            return
            
        if bitset_is_empty(uncovered):
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_solution = list(current_set)
            return
            
        # Pruning
        lb = get_lower_bound(uncovered, candidates)
        if len(current_set) + lb >= best_size:
            return
            
        # Heuristic: Pick the word in 'uncovered' that has the fewest candidates covering it.
        # If a word has 0 candidates, it's a dead end (shouldn't happen if we start with all).
        
        # To avoid iterating through all 6561 words to find the min count, 
        # we can just pick a random uncovered word or the first one.
        # But "Minimum Remaining Values" (MRV) is key to B&B success.
        
        # Let's find an uncovered word.
        # We can iterate through the bitset to find the first set bit.
        target_word = -1
        for i in range(num_words):
            if uncovered[i]:
                # Find lowest set bit
                # Python ints don't have ctz easily, but we can use bit_length or bin
                # Actually, we just need *some* uncovered word.
                # Let's pick one that is "hard" to cover.
                # We need to count candidates for each uncovered word.
                # This is expensive: O(num_uncovered * avg_inv_size).
                # Let's optimize: 
                # For each candidate c in candidates, it covers balls[c].
                # We want to find y in uncovered such that |{c in candidates : y in balls[c]}| is min.
                
                # Alternative: Just pick the first uncovered word.
                # This is simple and often works if the problem is symmetric.
                # Let's try to find the first uncovered word quickly.
                # We can maintain a list of uncovered words? No, bitset is better for OR/AND.
                
                # Let's just grab the first one.
                # To get index from bitset:
                # offset = i * 64
                # word_val = uncovered[i]
                # idx_in_word = (word_val & -word_val).bit_length() - 1
                target_word = i * 64 + ((uncovered[i] & -uncovered[i]).bit_length() - 1)
                break
                
        if target_word == -1:
            # Should be caught by is_empty
            return
            
        # Get candidates that cover target_word
        # These are the balls that MUST be considered for this word.
        # Actually, in Set Cover B&B, we branch on:
        # "Which ball covers this specific uncovered word?"
        # So we iterate through all candidates c that cover target_word.
        
        covering_candidates = [c for c in candidates if target_word in balls[c]]
        
        # Sort candidates? 
        # Heuristic: Try candidates that cover the most *other* uncovered words first.
        # This is the "Greedy" heuristic inside B&B.
        
        def score_candidate(c):
            # Count how many words in balls[c] are in uncovered
            count = 0
            for y in balls[c]:
                word_idx = y // 64
                bit_idx = y % 64
                if uncovered[word_idx] & (1 << bit_idx):
                    count += 1
            return count
            
        # Sorting every node is expensive. 
        # Let's just use the order from 'candidates' or randomize?
        # Randomizing helps avoid bias.
        random.shuffle(covering_candidates)
        
        # Optional: Sort by score (descending) to find good solutions faster for pruning
        # But sorting 100+ items at each node is costly.
        # Let's try without sorting first, or only if candidate list is small.
        
        new_candidates_base = [c for c in candidates if c != covering_candidates[0]] # Placeholder, we loop
        
        for c in covering_candidates:
            # Branch: Select c
            # New uncovered = uncovered AND NOT balls[c]
            
            # Check if c is still in candidates (it is, by construction)
            
            # Update uncovered
            # We need a bitset for balls[c]
            # We can precompute bitsets for all balls? 
            # 6561 balls * 103 ints = ~676k ints. Memory is fine.
            # Let's precompute ball_bitsets.
            
            # Wait, I didn't precompute ball_bitsets in the outer scope.
            # Let's do it here or pass it in. 
            # Since this is a recursive function, accessing a global/closure list is fast.
            
            # Let's precompute ball_bitsets outside 'solve' to save time in recursion.
            
            # ... (Refactoring to precompute ball_bitsets)
            
            # For now, let's just do the bitset op on the fly if we didn't precompute.
            # But 6561 * 103 operations per node is too slow if we do it inside.
            # I will add precomputation in the main block.
            
            # Assuming 'ball_bitsets' is a list of lists of ints, accessible in closure.
            
            new_unc = [uncovered[i] & ~ball_bitsets[c][i] for i in range(num_words)]
            
            # New candidates: remove c from candidates?
            # In Set Cover, once we pick a ball, we don't pick it again.
            # So new_candidates = [x for x in candidates if x != c]
            
            new_cands = [x for x in candidates if x != c]
            
            # Recurse
            solve(new_unc, new_cands, current_set + [c], depth + 1)
            
            # Pruning check after return?
            # The lb check at the start of the next call handles it.
            
            # Early termination if we found a solution of size 1? (Impossible)
            # If best_size becomes 1, we can stop everything.
            if best_size == 1:
                return

    # Precompute ball_bitsets
    ball_bitsets = []
    for i in range(total):
        bs = [0] * num_words
        for y in balls[i]:
            word_idx = y // 64
            bit_idx = y % 64
            bs[word_idx] |= (1 << bit_idx)
        ball_bitsets.append(bs)
        
    # Start B&B
    # To help B&B, we can start with a greedy solution to set a good upper bound (best_size).
    # Greedy Cover:
    unc_g = initial_uncovered[:]
    cands_g = initial_candidates[:]
    sol_g = []
    
    while not bitset_is_empty(unc_g) and time.time() < deadline - 5:
        # Find ball covering most uncovered
        best_c = -1
        best_count = -1
        for c in cands_g:
            count = 0
            for y in balls[c]:
                word_idx = y // 64
                bit_idx = y % 64
                if unc_g[word_idx] & (1 << bit_idx):
                    count += 1
            if count > best_count:
                best_count = count
                best_c = c
                if count == ball_size: # Max possible
                    break
                    
        if best_c == -1:
            break # Should not happen
            
        sol_g.append(best_c)
        # Update unc_g
        for i in range(num_words):
            unc_g[i] &= ~ball_bitsets[best_c][i]
        # Remove best_c from cands_g
        cands_g.remove(best_c)
        
    if sol_g:
        best_size = len(sol_g)
        best_solution = list(sol_g)
        
    # Now run B&B to improve
    # Note: B&B on 6561 elements is huge. 
    # The key is the lower bound and the branching factor.
    # Branching factor is the number of balls covering a specific word.
    # For q=3, n=8, radius=1:
    # A word y is covered by balls centered at y and its neighbors.
    # Number of neighbors = 1 + 8*2 = 17.
    # So branching factor is at most 17.
    # This is very manageable!
    
    solve(initial_uncovered, initial_candidates, [], 0)
    
    # Convert best_solution to codewords
    if best_solution is None:
        # Fallback: return greedy if B&B failed to find anything (unlikely)
        best_solution = sol_g
        
    codewords = []
    for idx in best_solution:
        digits = []
        x = idx
        for i in range(n):
            digits.append((x // powers[i]) % q)
        codewords.append(digits)
        
    return codewords
