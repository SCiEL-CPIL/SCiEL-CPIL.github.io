"""
Local Search Strategy for Ternary Covering Codes.

Strategy:
1. Generate an initial valid covering using a fast greedy heuristic (seed-like but optimized).
2. Define a cost function: Cost = |C| + P * |Uncovered|.
   We want to minimize |C| while keeping |Uncovered| = 0.
   By setting P high, we prioritize validity, but allow temporary invalid states if it leads to
   a better |C| later.
3. Perform Local Search (Hill Climbing / Simulated Annealing):
   - Neighborhood operations:
     a. Remove a codeword: If it doesn't uncover any new words (redundant), remove it.
     b. Swap a codeword: Replace one codeword with another. If the new set is smaller or same size
        and has <= uncovered words, accept.
     c. Add a codeword: Only if it reduces uncovered count significantly (less useful if already valid).
   
   Since the problem is small (N=2187), we can afford to check coverage efficiently.
   
   Key Insight: The "best known" 186 is a global optimum or near-optimum. Greedy often gets stuck
   at ~190-200. Local search can "shake out" redundant codewords and find better replacements.
"""

import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # 1. Precomputation
    total = q ** n
    # Represent words as integers 0..total-1
    # Precompute digits for all words to speed up ball generation?
    # Or just compute balls on the fly? 
    # N=2187, so precomputing balls as bitsets or lists is fast.
    
    # Ball precomputation: ball[c] = list of words covered by codeword c
    # Radius 1: c + (q-1)*n words.
    steps = [1] * n
    for i in range(1, n):
        steps[i] = steps[i - 1] * q
        
    def int_to_digits(x):
        return [x // steps[i] % q for i in range(n)]
        
    def digits_to_int(d):
        return sum(d[i] * steps[i] for i in range(n))

    # Precompute all balls
    # ball[c] is a set of integers
    balls = []
    for c in range(total):
        digits = int_to_digits(c)
        covered = {c}
        for i in range(n):
            orig = digits[i]
            for d in range(q):
                if d == orig:
                    continue
                digits[i] = d
                w = digits_to_int(digits)
                covered.add(w)
            digits[i] = orig
        balls.append(covered)
        
    # Convert balls to sorted lists for faster intersection? 
    # Or use bitsets? 
    # With N=2187, a bitset of 2187 bits fits in ~272 8-byte integers or similar.
    # Let's use Python sets first, it's simple. If slow, switch to bitsets.
    # Actually, for local search, we need to check "is word w covered by C?".
    # We can maintain a coverage count array: cover_count[w] = number of codewords in C that cover w.
    # If we add c, we increment cover_count for all w in balls[c].
    # If we remove c, we decrement.
    # Uncovered words are those with cover_count[w] == 0.
    
    # 2. Initial Solution: Greedy Set Cover
    # This is the "seed" part, but we'll run it fast to get a valid starting point.
    
    uncovered_counts = np.ones(total, dtype=np.int8) # 1 if uncovered, 0 if covered? No, let's use 1 for uncovered.
    # Actually, let's just use a set for the greedy phase, it's only ~190 iterations.
    
    uncovered_set = set(range(total))
    code = []
    
    # Greedy Selection
    # To speed up, we can precompute the "gain" for each c.
    # gain[c] = len(balls[c] & uncovered_set)
    
    # Since we do this ~200 times, and each check is O(N), total is 200 * 2187 * 2187? No.
    # balls[c] has size 1 + 2*n = 15 for q=3, n=7.
    # So gain calculation is O(15).
    # Total greedy time: 200 * 2187 * 15 = ~6.5M ops. Very fast.
    
    while uncovered_set:
        if time.time() > deadline - 20: # Reserve 20s for local search
            break
        best_c = -1
        best_gain = 0
        for c in range(total):
            # Calculate gain
            gain = 0
            for w in balls[c]:
                if w in uncovered_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
            if gain == 0:
                continue
        if best_c == -1:
            break
        code.append(best_c)
        # Remove covered
        for w in balls[best_c]:
            if w in uncovered_set:
                uncovered_set.remove(w)
                
    # If we didn't finish greedy, we have an invalid code. 
    # For local search to work well, we want a valid start.
    # If invalid, we can just add random codewords until valid (slow) or use a fallback.
    # Let's assume greedy finishes. If not, we'll try to repair.
    
    if uncovered_set:
        # Emergency repair: add random codewords covering uncovered
        while uncovered_set and time.time() < deadline - 5:
            # Pick a random uncovered word
            w = random.choice(list(uncovered_set))
            # Find a codeword that covers w and covers max uncovered
            best_c = -1
            best_gain = 0
            for c in range(total):
                if w in balls[c]:
                    gain = 0
                    for ww in balls[c]:
                        if ww in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
            if best_c != -1:
                code.append(best_c)
                for ww in balls[best_c]:
                    if ww in uncovered_set:
                        uncovered_set.remove(ww)
            else:
                break
                
    # If still invalid, return a known valid structure (tetracode direct sum) as fallback
    if uncovered_set:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # 3. Local Search Phase
    # Current code is a list of integers.
    # We maintain cover_count array to quickly determine uncovered.
    
    # Initialize cover counts
    cover_count = np.zeros(total, dtype=np.int8)
    for c in code:
        for w in balls[c]:
            cover_count[w] += 1
            
    def get_uncovered():
        return np.where(cover_count == 0)[0]
        
    # Objective: Minimize |code| + P * |uncovered|
    # We want to remove redundant codewords.
    # A codeword c is redundant if removing it doesn't increase the number of uncovered words.
    # i.e., if for all w in balls[c], cover_count[w] > 1.
    
    # We will perform iterations of:
    # 1. Try to remove a redundant codeword.
    # 2. Try to swap a codeword with another that allows further removals or is just better?
    #    Swapping is complex. Let's stick to "Remove Redundant" first.
    #    If no redundant codewords exist, we are at a local minimum for removal.
    #    To escape, we can "kick" by swapping one codeword with a random one,
    #    accepting if it doesn't increase the cost (|C| + P*|U|).
    
    P = 1000  # Penalty for uncovered words
    
    def calculate_cost(code_list):
        # Rebuild cover counts? No, we maintain them.
        # But for evaluation of a *candidate* move, we need to check the change.
        pass
        
    # To make local search efficient, we'll implement moves that update cover_count incrementally.
    
    # State:
    # code: list of ints
    # cover_count: np.array
    # uncovered_count: int (sum of cover_count == 0)
    
    uncovered_count = len(get_uncovered())
    current_cost = len(code) + P * uncovered_count
    
    # Precompute for each c, the list of words it covers (balls[c])
    
    # Helper: Check if removing c is safe (doesn't uncover new words)
    def is_redundant(c):
        for w in balls[c]:
            if cover_count[w] == 1:
                return False
        return True
        
    # Helper: Remove c
    def remove_codeword(c):
        nonlocal code, cover_count, uncovered_count, current_cost
        code.remove(c)
        for w in balls[c]:
            cover_count[w] -= 1
            if cover_count[w] == 0:
                uncovered_count += 1
        current_cost = len(code) + P * uncovered_count
        
    # Helper: Add c
    def add_codeword(c):
        nonlocal code, cover_count, uncovered_count, current_cost
        code.append(c)
        for w in balls[c]:
            if cover_count[w] == 0:
                uncovered_count -= 1
            cover_count[w] += 1
        current_cost = len(code) + P * uncovered_count
        
    # Local Search Loop
    # We will try to find redundant codewords and remove them.
    # If we reach a state with no redundant codewords, we try "kicks".
    
    # Optimization: Instead of scanning all codewords for redundancy every time,
    # we can just scan. |Code| is ~190, balls size is 15. 190*15 = 2850 checks. Very fast.
    
    max_iter = 10000
    iter_count = 0
    no_improve_count = 0
    
    while time.time() < deadline - 5 and iter_count < max_iter:
        iter_count += 1
        improved = False
        
        # Phase 1: Remove redundant codewords
        # We iterate through the code. If we find a redundant one, remove it and restart scan.
        # This is a "greedy reduction" step.
        i = 0
        while i < len(code):
            c = code[i]
            if is_redundant(c):
                remove_codeword(c)
                improved = True
                no_improve_count = 0
                # Restart scan from beginning as indices shifted
                i = 0
            else:
                i += 1
                
        if improved:
            continue
            
        # Phase 2: If no redundant codewords, we are at a local minimum.
        # Try to kick: Swap a random codeword c_old with a random c_new.
        # Accept if new_cost <= old_cost.
        
        if len(code) == 0:
            break
            
        c_old = random.choice(code)
        c_new = random.randint(0, total - 1)
        
        if c_old == c_new:
            continue
            
        # Simulate swap
        # 1. Remove c_old
        # 2. Add c_new
        
        # To avoid modifying state if we reject, we can calculate the delta.
        # But modification is cheap (15 ops). Let's just do it and revert if worse.
        
        old_cost = current_cost
        
        remove_codeword(c_old)
        add_codeword(c_new)
        
        new_cost = current_cost
        
        if new_cost <= old_cost:
            # Accept
            no_improve_count = 0
        else:
            # Revert
            remove_codeword(c_new)
            add_codeword(c_old)
            no_improve_count += 1
            
            # If stuck for a while, try a "double kick" or just break?
            # For now, just continue.
            
        # Occasionally, if we are stuck (no_improve > 100), try a more aggressive move?
        # Maybe remove a non-redundant codeword (creating uncovered) and then add a better one?
        # This is effectively what the swap does, but if we restrict to "swap", we might get stuck.
        # Let's allow "Remove + Add" as separate steps if we are stuck?
        # Actually, the swap is the most direct way to change the set size or composition.
        
    # Final cleanup: Remove any remaining redundant codewords
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        i = 0
        while i < len(code):
            c = code[i]
            if is_redundant(c):
                remove_codeword(c)
                improved = True
                i = 0
            else:
                i += 1
                
    # Check validity
    if uncovered_count > 0:
        # This should not happen if we started valid and only accepted cost-non-increasing moves
        # where P is large. But if P is too small, we might end up invalid.
        # Let's check.
        # If invalid, we must repair.
        # Repair: Add codewords to cover uncovered words.
        # This increases |C|, but ensures validity.
        
        while uncovered_count > 0 and time.time() < deadline - 1:
            # Find an uncovered word
            unc_words = get_uncovered()
            if len(unc_words) == 0:
                break
            w = unc_words[0]
            # Find best c to add
            best_c = -1
            best_gain = 0
            for c in range(total):
                if w in balls[c]:
                    gain = 0
                    for ww in balls[c]:
                        if cover_count[ww] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
            if best_c == -1:
                break
            add_codeword(best_c)
            
    # Convert back to list of lists of ints
    final_code = [int_to_digits(c) for c in code]
    
    # Ensure distinctness (should be, but just in case)
    # The code list maintains distinctness via add/remove logic?
    # remove_codeword uses list.remove, which removes the first occurrence.
    # add_codeword appends.
    # We never add a duplicate in the swap logic?
    # Wait, if c_new is already in code, we add a duplicate.
    # We should check for duplicates.
    
    # Let's verify distinctness and remove duplicates if any.
    seen = set()
    unique_code = []
    for c in code:
        if c not in seen:
            seen.add(c)
            unique_code.append(c)
            
    # If we removed duplicates, we need to re-verify coverage?
    # Duplicates don't affect coverage (cover_count would be >1, but removing one still leaves >0 if it was 2).
    # However, our cover_count array reflects the *duplicated* state.
    # If we have duplicates, cover_count is higher than it should be for the unique set.
    # So we must rebuild cover_count if we found duplicates.
    
    if len(unique_code) != len(code):
        code = unique_code
        cover_count = np.zeros(total, dtype=np.int8)
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        # Re-check redundancy and remove
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            i = 0
            while i < len(code):
                c = code[i]
                # Check redundancy with current cover_count
                redundant = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove
                    code.pop(i)
                    for w in balls[c]:
                        cover_count[w] -= 1
                    improved = True
                    i = 0
                else:
                    i += 1
                    
        final_code = [int_to_digits(c) for c in code]
        
    return final_code

# EVOLVE-BLOCK-END
