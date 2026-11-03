"""Strategy: Simulated Annealing for Covering Code K_3(6,1).
Uses a direct sum construction as a valid starting point (size 81),
then attempts to reduce the size to the known optimum of 73 via
swap and remove moves, accepting worsening moves probabilistically
to escape local minima.
"""
import itertools
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-target parameters
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = 110.0 # Safety margin for 120s limit

    # 1. Precomputation
    # Map between integer indices (0..728) and base-3 vectors
    # We use integer indices for the code set to speed up operations.
    
    # Precompute neighbors for each index (words at distance <= 1)
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    neighbors = [[] for _ in range(729)]
    for i in range(729):
        vec = [0] * 6
        num = i
        for k in range(6):
            vec[k] = num % 3
            num //= 3
        
        # The word itself
        neighbors[i].append(i)
        
        # Words at distance 1
        for pos in range(6):
            orig_val = vec[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    # Calculate new index efficiently
                    # index = sum(v[k] * 3^k)
                    # delta = (new_val - orig_val) * 3^pos
                    diff = new_val - orig_val
                    new_idx = i + diff * (3 ** pos)
                    neighbors[i].append(new_idx)
                    
    # 2. Initial Construction: Direct Sum of Tetracode
    # Tetracode: 9 codewords of length 4.
    # Direct sum with length 2 tail: 9 * 3^2 = 81 codewords.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            vec = h + list(tail)
            idx = 0
            for k in range(6):
                idx += vec[k] * (3 ** k)
            initial_code.add(idx)
            
    current_code = list(initial_code)
    code_set = set(current_code)
    
    # 3. State Tracking
    # coverage_count[j] = number of codewords in current_code that cover j
    coverage_count = np.zeros(729, dtype=np.int32)
    for c in current_code:
        for j in neighbors[c]:
            coverage_count[j] += 1
            
    # Uncovered words (should be none initially)
    uncovered_mask = coverage_count == 0
    uncovered_count = np.sum(uncovered_mask)
    
    best_size = len(current_code)
    best_code = list(current_code)
    
    # 4. Simulated Annealing
    # Temperature schedule: start high, decay exponentially
    T = 1.0
    T_min = 1e-5
    alpha = 0.9995
    
    # Move types:
    # 1. Remove: Pick c from code, remove it. Valid if no uncovered words created.
    #    Cost: -1 (if valid). If invalid, reject (or accept with penalty? No, must be valid).
    #    Actually, SA usually requires a defined cost. Here, validity is a hard constraint.
    #    We can define "energy" as the number of uncovered words.
    #    Target: Energy = 0, Minimize Size.
    #    
    #    Better approach: 
    #    Objective: Minimize Size.
    #    Constraint: Coverage is complete.
    #    
    #    Let's use a penalty method or just strict validity with "repair".
    #    Given the small size, strict validity with random restarts/SA on valid states is safer.
    #    
    #    However, to escape local minima, we might need to temporarily allow uncovered words?
    #    If we allow uncovered words, we must have a repair mechanism.
    #    Let's stick to valid moves (Remove/Swap) but use SA to accept "worse" valid states?
    #    What is a "worse" valid state? 
    #    If we only remove, size decreases. 
    #    If we swap, size stays same. 
    #    How does SA help if all moves must be valid and reduce/maintain size?
    #    It helps if we define a secondary cost, or if we allow temporary increases in size?
    #    
    #    Actually, the standard approach for set cover is:
    #    Start with valid solution.
    #    Try to remove elements. If stuck, we are in a local minimum.
    #    To escape, we might need to ADD an element (increasing size) to enable REMOVING two others?
    #    This is a "2-exchange" or similar.
    #    
    #    Let's implement a "Swapping" move that maintains size but changes the set, 
    #    potentially leading to a state where a removal is possible.
    #    And we use SA to accept swaps that don't immediately lead to removal, 
    #    guided by a "potential" for future removals?
    #    
    #    Simpler: 
    #    1. Try Remove. If valid, accept (always improves size).
    #    2. Try Swap. If valid, accept with probability exp(-delta/T). 
    #       What is delta? 
    #       If we just swap, size is same. We need a heuristic to prefer swaps that 
    #       "loosen" the constraints. 
    #       Let's define delta = 0 for all valid swaps? Then we do random walks.
    #       Random walks on valid states of fixed size might find a "hole" for removal.
    #       
    #    Let's refine:
    #    State: Set of codewords S.
    #    Move 1: Remove c. If valid, new_size = size - 1. Always accept.
    #    Move 2: Swap c in S with c' not in S. If valid, new_size = size. 
    #            Accept with probability 1 (random walk) or based on some potential.
    #            Let's just accept all valid swaps to explore the space of size-N codes.
    #            This allows us to find a different size-N code that might allow a removal.
    #            
    #    This is effectively a Random Walk with "Remove" as the goal.
    #    SA isn't strictly needed if we accept all valid moves. 
    #    But maybe we should bias swaps?
    #    
    #    Let's try:
    #    While time < deadline:
    #       1. Try to remove a random codeword.
    #          - Calculate change in coverage.
    #          - If no uncovered words created, accept removal. (Size decreases)
    #       2. If removal failed (or randomly, to explore), try a swap.
    #          - Pick c in S, c' not in S.
    #          - Check if (S - {c} + {c'}) is valid.
    #          - If valid, accept swap.
    #          
    #    This is a valid stochastic local search.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # 1. Attempt Removal
        if len(current_code) > 73:
            c_idx = random.randint(0, len(current_code) - 1)
            c = current_code[c_idx]
            
            # Check validity of removing c
            # A word j becomes uncovered if coverage_count[j] == 1 and c covers j.
            # So we check if any j in neighbors[c] has coverage_count[j] == 1.
            
            removable = True
            for j in neighbors[c]:
                if coverage_count[j] == 1:
                    removable = False
                    break
            
            if removable:
                # Accept removal
                for j in neighbors[c]:
                    coverage_count[j] -= 1
                
                current_code.pop(c_idx)
                code_set.remove(c)
                
                best_size = len(current_code)
                best_code = list(current_code)
                
                # If we reached 73, we can stop or try to go lower (unlikely)
                if best_size <= 73:
                    break
                    
        # 2. Attempt Swap (to escape local minima)
        # Only if we are stuck or to explore
        # To avoid infinite loops on the same state, we do swaps frequently.
        if len(current_code) > 73:
            # Pick a random codeword to remove
            c_idx = random.randint(0, len(current_code) - 1)
            c = current_code[c_idx]
            
            # Pick a random candidate to add (not in code)
            # We can pick from all 729, or just those that cover some "tight" words?
            # Random is fine for now.
            c_new = random.randint(0, 728)
            
            if c_new in code_set:
                continue # Skip if same
            
            # Check validity of swap: (S - {c} + {c_new})
            # This is valid if:
            # 1. All words covered only by c are also covered by c_new? No.
            # 2. Coverage counts after swap must be >= 1 for all words.
            
            # Let's simulate the change in coverage.
            # remove c: coverage_count[j] -= 1 for j in neighbors[c]
            # add c_new: coverage_count[j] += 1 for j in neighbors[c_new]
            
            # We need to check if any j ends up with 0.
            # This happens if:
            # - j is in neighbors[c] but NOT in neighbors[c_new], and coverage_count[j] was 1.
            # - j is in neighbors[c] AND neighbors[c_new], and coverage_count[j] was 1? No, if it's 1 and we remove then add, it stays 1.
            #   Wait. If j is in both, count goes 1 -> 0 -> 1. Valid.
            #   If j is in c but not c_new, count goes 1 -> 0. Invalid.
            #   If j is in c_new but not c, count goes k -> k+1. Valid (k>=0).
            
            # So the only danger is words in neighbors[c] \ neighbors[c_new] with coverage_count[j] == 1.
            
            valid_swap = True
            # Optimization: Iterate over neighbors[c]
            for j in neighbors[c]:
                if coverage_count[j] == 1:
                    # If this word is ONLY covered by c, we must ensure c_new covers it.
                    if j not in neighbors[c_new]: # This check is O(1) if neighbors are sets? No, lists.
                         # We need fast lookup. 
                         # Precompute neighbor sets? Or bitsets?
                         # 729 words. Bitset of 729 bits = 92 bytes. 729*92 = 67KB. Very small.
                         # Let's use Python integers as bitsets for neighbor checks?
                         # Or just a set for neighbors?
                         pass
            
            # The 'in' check on a list is O(N). N=19. 19*19 is small.
            # But doing this in a tight loop is slow in Python.
            
            # Let's optimize the swap check.
            # Precompute neighbor sets as Python sets for O(1) lookup?
            # Or better, precompute a boolean matrix? 729x729 is 500k bools. Numpy bool is 1 byte. 500KB.
            
            # Let's create a numpy boolean matrix for "covers" relation.
            # covers[i, j] = True if i covers j (dist <= 1)
            
            # We didn't do this in precomputation. Let's do it now if needed?
            # No, let's stick to the list but optimize the check.
            
            # Actually, let's just use a set for neighbors[c] for the check?
            # We can convert neighbors to sets once.
            
            # Let's restart the precomputation part to use sets for neighbors for faster 'in' checks during swaps.
            # Or, since we are in the loop, let's just assume the list check is fast enough for 120s?
            # 120s / (time per swap). 
            # If swap check is ~100us, we can do 1.2M swaps.
            # List check: 19 iterations * 'in' check (avg 10 comparisons) = 190 ops.
            # Python op is ~50ns? 10us. So 100k swaps/sec. 12M swaps in 120s. Plenty.
            
            # Let's implement the check properly.
            
            # We need to know if j is in neighbors[c_new].
            # To make this fast, let's convert neighbors to sets.
            
            # Wait, I am inside the function. I can't easily change the precomputation without restructuring.
            # I will create a set version of neighbors on the fly? No, that's slow.
            
            # Let's just use the list and 'in'.
            
            # Check:
            # For each j in neighbors[c]:
            #   if coverage_count[j] == 1:
            #       if j not in neighbors[c_new]:
            #           valid_swap = False
            #           break
            
            # If valid, perform swap.
            
            # To avoid O(N) 'in' check, we can precompute a set for neighbors[c_new] once per swap attempt?
            # No, c_new changes.
            
            # Let's just do it.
            
            valid_swap = True
            # We only care about j where coverage_count[j] == 1.
            # Let's find those first?
            
            # Actually, if we maintain a list of "critical" words (count==1), it might be faster?
            # But counts change.
            
            # Let's just iterate.
            for j in neighbors[c]:
                if coverage_count[j] == 1:
                    # Check if c_new covers j
                    # j is in neighbors[c_new]?
                    if j not in neighbors[c_new]:
                        valid_swap = False
                        break
            
            if valid_swap:
                # Perform swap
                # Update coverage counts
                for j in neighbors[c]:
                    coverage_count[j] -= 1
                for j in neighbors[c_new]:
                    coverage_count[j] += 1
                
                # Update code set
                current_code[c_idx] = c_new
                code_set.remove(c)
                code_set.add(c_new)
                
                # Note: We don't update best_code here because size didn't change.
                # We only update best_code when we remove.
                
        # Decay temperature? 
        # We aren't really using T for acceptance probability of swaps since we accept all valid swaps.
        # If we wanted to bias swaps, we could use T.
        # But random valid swaps are a good exploration strategy.
        
        # If we get stuck (no removals for a long time), we are exploring via swaps.
        
    # Final check
    if len(best_code) < 81:
        # Verify
        final_cover = np.zeros(729, dtype=np.int32)
        for c in best_code:
            for j in neighbors[c]:
                final_cover[j] += 1
        if np.all(final_cover > 0):
             return [ [ (c // (3**k)) % 3 for k in range(6) ] for c in best_code ]
    
    # Fallback
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

# EVOLVE-BLOCK-END
