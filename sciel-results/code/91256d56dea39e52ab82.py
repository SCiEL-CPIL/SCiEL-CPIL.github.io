import time
import itertools
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety

    # Specific case optimization: q=3, n=6, radius=1
    # General fallback if parameters differ, though the task implies this specific instance.
    
    total_words = q ** n
    
    # If the problem is too large for bitmask approach, we'd need a different strategy.
    # For q=3, n=6, total_words = 729. Bitmask is 729 bits, fits in a Python int easily.
    if total_words > 10000:
        # Fallback: simple greedy, no local search (too slow for large N with bitmasks if N is huge, 
        # but here N is small. If N were large, we'd use a different data structure).
        # For this specific task, we assume small N.
        pass

    def encode(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def decode(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Generate all words
    all_words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Precompute covers for all possible codewords
    # covers[c] is a bitmask of words covered by codeword c
    covers = [0] * total_words
    
    # Efficiently compute covers
    # For each codeword c, find all words v such that dist(c, v) <= radius
    for c_idx, c_word in enumerate(all_words):
        mask = 0
        for v_idx, v_word in enumerate(all_words):
            # Calculate Hamming distance
            dist = 0
            for a, b in zip(c_word, v_word):
                if a != b:
                    dist += 1
                    if dist > radius:
                        break
            if dist <= radius:
                mask |= (1 << v_idx)
        covers[c_idx] = mask

    # Greedy Set Cover
    uncovered = (1 << total_words) - 1
    solution = []
    candidates = list(range(total_words))
    
    # Optimization: Sort candidates by coverage size initially? 
    # Or just pick best each time.
    # 729 candidates, 729 bits.
    
    while uncovered and time.time() < deadline - 5.0: # Reserve 5s for local search
        best_c = -1
        best_gain = 0
        # Heuristic: pick c that maximizes (covers[c] & uncovered)
        for c in candidates:
            gain = bin(covers[c] & uncovered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_c = c
                if gain == bin(uncovered).count('1'):
                    break
        if best_c == -1:
            break
        solution.append(best_c)
        candidates.remove(best_c)
        uncovered &= ~covers[best_c]
        
    # Local Search: Simulated Annealing style with specific moves
    # We want to minimize the number of codewords.
    # Validity constraint: Union of covers == (1 << total_words) - 1
    
    sol_set = set(solution)
    current_size = len(sol_set)
    best_size = current_size
    best_sol = sol_set.copy()
    
    # Precompute union of all covers? No, it changes.
    # We need a fast way to check if a set is valid.
    # Valid if (OR of covers[c] for c in sol_set) == all_ones
    
    all_ones = (1 << total_words) - 1
    
    def is_valid(s_set):
        union_mask = 0
        for c in s_set:
            union_mask |= covers[c]
        return union_mask == all_ones

    # Optimization for local search:
    # Instead of checking full validity every time (O(N)), we can track uncovered words.
    # But for small N (729), O(N) bit operations are very fast.
    
    # Moves:
    # 1. Remove: If removing c keeps validity, do it.
    # 2. Swap: Replace c_old with c_new. If valid, do it.
    # 3. Add-Remove 2: Add c_new. Then try to remove two codewords c_a, c_b such that the set is still valid.
    #    This effectively swaps 2 codewords for 1, reducing size by 1.
    
    # We will run a loop of these moves until time runs out.
    # To escape local minima, we can use random restarts or randomization in the search order.
    # However, "Add-Remove 2" is a deterministic local improvement if we find the right pair.
    # To make it stochastic, we can randomize the order of c_new, c_a, c_b.
    
    # Let's implement a randomized local search.
    # Temperature is not strictly needed if we only accept improvements (size reduction).
    # But if we get stuck, we might need to accept a temporary size increase?
    # No, the problem is a minimization with a hard constraint (validity).
    # We can only accept moves that keep the solution valid.
    # If we are stuck at size K, we need a move that reduces size to K-1.
    # "Add 1, Remove 2" is the standard way to find such moves.
    
    # Algorithm:
    # While time remains:
    #   1. Try to remove any redundant codeword.
    #   2. Try to find a 1-for-1 swap that allows a removal? No, 1-for-1 swap keeps size same.
    #      Actually, a 1-for-1 swap might enable a subsequent removal.
    #      So: Try Swap (c_old -> c_new) if valid. Then try Remove.
    #   3. Try Add-Remove 2: Add c_new. If valid, try to remove 2. If successful, net -1.
    
    # To be efficient, we precompute "uniquely covered" words for each codeword?
    # That's expensive to update.
    
    # Let's stick to a robust randomized search.
    
    rng = random.Random(42)
    
    # Helper to check redundancy
    def can_remove(c, s_set):
        # Check if s_set - {c} is valid
        union_mask = 0
        for other in s_set:
            if other != c:
                union_mask |= covers[other]
        return union_mask == all_ones

    # Helper to check if swap is valid
    def can_swap(c_old, c_new, s_set):
        # Check if (s_set - {c_old} + {c_new}) is valid
        union_mask = 0
        for c in s_set:
            if c != c_old:
                union_mask |= covers[c]
        union_mask |= covers[c_new]
        return union_mask == all_ones

    # Helper to check if add-then-remove-2 is possible
    # This is O(N^2) per candidate c_new if we check all pairs.
    # N is ~70-80. N^2 is ~6000. 729 candidates * 6000 = 4M ops. Might be slow in Python.
    # Optimization: Only try if adding c_new covers some "hard" words.
    
    # Let's use a simpler heuristic for the search loop:
    # 1. Randomly pick a codeword to remove (if possible, do it).
    # 2. Randomly pick a swap. If valid, do it.
    # 3. Randomly pick an add. Then try to remove 2.
    
    # To speed up "Try to remove 2", we can iterate through the set and check redundancy.
    # If we add c_new, some codewords might become redundant.
    # We can find all redundant codewords in the new set, and if there are at least 2, remove them.
    
    def find_redundant(s_set):
        redundant = []
        # To speed up, we can check in random order
        items = list(s_set)
        rng.shuffle(items)
        for c in items:
            if can_remove(c, s_set):
                redundant.append(c)
                # We can remove it immediately for the purpose of finding the next one?
                # No, redundancy is relative to the current set.
                # If c is redundant in S, it might not be in S-{c'} for c' != c.
                # But we just need to find 2 that can be removed.
                # Actually, if we remove one, the set changes.
                # We want to remove 2 such that the final set is valid.
                # This is equivalent to: Does there exist a pair {a, b} such that S - {a, b} is valid?
                # We can just iterate pairs? O(N^2).
                pass
        return redundant

    # Better approach for Add-Remove 2:
    # Add c_new.
    # Iterate through all pairs (a, b) in s_set.
    # If S - {a, b} + {c_new} is valid, accept.
    # To make this faster, we can precompute "essential" words.
    
    # Given the time limit (140s), and N=729, we can afford some O(N^2) checks if done sparingly.
    # However, 729 candidates for c_new * (70^2 / 2) pairs is too much.
    
    # Alternative:
    # 1. Add c_new.
    # 2. Find all redundant codewords in S_new = S + {c_new}.
    # 3. If count >= 2, remove 2 of them.
    # This is O(N^2) to find redundant (N checks, each O(N)).
    # If we do this for every c_new, it's O(N^3) per step.
    
    # Let's optimize the search:
    # Instead of trying all c_new, try random c_new.
    # And limit the number of attempts per iteration.
    
    # Also, we can use the "1-swap" move more aggressively.
    # A 1-swap might not reduce size, but it changes the structure.
    # Then a subsequent "Remove" might work.
    
    # Strategy:
    # Loop:
    #   1. Try to remove any redundant. (O(N^2))
    #   2. If no removal, try random swaps. (O(K) swaps, each O(N))
    #   3. If no improvement, try random Add-Remove 2. (O(K) adds, each O(N^2) check for redundancy)
    
    # Let's implement this.
    
    current_sol = sol_set.copy()
    
    # Precompute list of all codewords not in current_sol?
    # It changes, so we just iterate range(total_words) and check membership.
    
    # To speed up redundancy check, we can use the fact that if a word is only covered by one codeword,
    # that codeword is essential.
    # Let's maintain a count of how many times each word is covered.
    # cover_counts[w] = number of codewords in sol_set that cover w.
    # A codeword c is redundant if for all w in covers[c], cover_counts[w] > 1.
    
    def build_counts(s_set):
        counts = [0] * total_words
        for c in s_set:
            # Iterate bits in covers[c]
            mask = covers[c]
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                mask ^= lsb
        return counts

    def is_redundant_fast(c, counts):
        mask = covers[c]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            if counts[idx] <= 1:
                return False
            mask ^= lsb
        return True

    # Update counts when adding/removing
    def update_counts_add(counts, c):
        mask = covers[c]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            mask ^= lsb

    def update_counts_remove(counts, c):
        mask = covers[c]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            counts[idx] -= 1
            mask ^= lsb

    # Initialize counts
    counts = build_counts(current_sol)
    
    # Main search loop
    # We will alternate between different move types to avoid getting stuck.
    
    no_improvement_count = 0
    max_no_improvement = 100 # Reset randomization if stuck
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove redundant
        # Shuffle to randomize order
        items = list(current_sol)
        rng.shuffle(items)
        for c in items:
            if is_redundant_fast(c, counts):
                # Remove c
                current_sol.remove(c)
                update_counts_remove(counts, c)
                improved = True
                if len(current_sol) < best_size:
                    best_size = len(current_sol)
                    best_sol = current_sol.copy()
                break # Restart loop to rebuild items list? Or continue?
                # Breaking is safer to ensure counts are consistent for next step if we do multiple removals.
                # But we can just continue checking others if we update counts.
                # Let's just break to be safe and fast.
        
        if improved:
            no_improvement_count = 0
            continue
            
        # 2. Try Swaps
        # Pick a random c_old from current_sol
        # Pick a random c_new from all_words not in current_sol
        # If valid, swap.
        
        # To make this effective, we should prioritize c_new that covers "weak" words (words with low count).
        # Words with count == 1 are "critical".
        # If we swap out a codeword that covers a critical word, we must ensure the new codeword also covers it.
        
        # Simple random swap:
        c_old = rng.choice(list(current_sol))
        # Find candidates for c_new
        # We can filter c_new to those that cover the words uniquely covered by c_old?
        # Words uniquely covered by c_old: w in covers[c_old] where counts[w] == 1.
        
        unique_to_old = 0
        mask = covers[c_old]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            if counts[idx] == 1:
                unique_to_old |= lsb
            mask ^= lsb
        
        # If unique_to_old is 0, c_old is redundant (should have been caught in step 1).
        # So unique_to_old is not 0.
        
        # We need c_new such that (unique_to_old & ~covers[c_new]) == 0.
        # i.e., c_new covers all words that c_old uniquely covers.
        
        # Iterate through potential c_new
        # To speed up, we can precompute for each word w, the list of codewords covering it?
        # That's 729 lists of ~19 items (1 + 2*6). Total ~13k entries.
        # Let's do it.
        
        # Precompute coverers: coverers[w] = list of c such that w in covers[c]
        # We can do this once.
        
        # Actually, we can just iterate c_new and check.
        # But 729 checks per swap attempt is fine.
        
        # Let's try a few random swaps
        for _ in range(10):
            c_new = rng.randint(0, total_words - 1)
            if c_new in current_sol:
                continue
            # Check if c_new covers unique_to_old
            if (unique_to_old & ~covers[c_new]) != 0:
                continue
            
            # Check if swap is valid
            # Temporarily remove c_old, add c_new
            # Check if any word becomes uncovered (count 0)
            
            # Words affected: covers[c_old] and covers[c_new]
            # We can just check the union?
            # Faster: 
            # 1. Remove c_old from counts.
            # 2. Check if any count in covers[c_old] became 0? (If so, invalid)
            # 3. Add c_new to counts.
            # 4. If valid, keep it.
            
            # To avoid modifying global counts if invalid, we can simulate.
            # Or modify and revert.
            
            # Simulate:
            # The only words that can become uncovered are those in covers[c_old] \ covers[c_new].
            # For each such word w, if counts[w] == 1, then removing c_old makes it 0.
            # So we need (covers[c_old] & ~covers[c_new]) to have no words with count 1.
            
            lost_mask = covers[c_old] & ~covers[c_new]
            valid = True
            mask = lost_mask
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    valid = False
                    break
                mask ^= lsb
            
            if valid:
                # Perform swap
                current_sol.remove(c_old)
                update_counts_remove(counts, c_old)
                current_sol.add(c_new)
                update_counts_add(counts, c_new)
                improved = True
                # Check for new redundancies
                # We don't update best_sol here because size didn't change.
                break
        
        if improved:
            no_improvement_count = 0
            continue
            
        # 3. Try Add-Remove 2
        # Add c_new, then try to remove 2.
        # This is the most powerful move.
        
        # Pick random c_new
        c_new = rng.randint(0, total_words - 1)
        if c_new in current_sol:
            continue
            
        # Add c_new
        current_sol.add(c_new)
        update_counts_add(counts, c_new)
        
        # Now try to remove 2 codewords.
        # We want to remove a and b such that the set remains valid.
        # This means no word has count 0 after removal.
        
        # Strategy: Find all redundant codewords in the new set.
        # If there are >= 2, remove them.
        
        redundant_candidates = []
        items = list(current_sol)
        rng.shuffle(items)
        for c in items:
            if is_redundant_fast(c, counts):
                redundant_candidates.append(c)
                if len(redundant_candidates) >= 2:
                    break
        
        if len(redundant_candidates) >= 2:
            # Remove 2
            c_rem1 = redundant_candidates[0]
            c_rem2 = redundant_candidates[1]
            
            # Check if removing both is valid?
            # is_redundant_fast(c1) means S - {c1} is valid.
            # It does NOT guarantee S - {c1, c2} is valid.
            # We must check S - {c1, c2}.
            
            # Let's verify validity of S - {c1, c2} + {c_new} (which is current_sol - {c1, c2})
            # We can simulate removal of c1 and c2.
            
            # Words that might become uncovered are in (covers[c1] | covers[c2]) \ covers[c_new]?
            # Actually, just check if any word in (covers[c1] | covers[c2]) has count <= 2?
            # No, count is in the current set (which includes c1, c2, c_new).
            # If we remove c1 and c2, the count for w becomes counts[w] - (1 if w in covers[c1] else 0) - (1 if w in covers[c2] else 0).
            # We need this to be >= 1 for all w.
            
            valid_removal = True
            mask = covers[c_rem1] | covers[c_rem2]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                # Current count
                cnt = counts[idx]
                # Reduction
                red = 0
                if covers[c_rem1] & lsb:
                    red += 1
                if covers[c_rem2] & lsb:
                    red += 1
                if cnt - red < 1:
                    valid_removal = False
                    break
                mask ^= lsb
            
            if valid_removal:
                # Perform removals
                current_sol.remove(c_rem1)
                update_counts_remove(counts, c_rem1)
                current_sol.remove(c_rem2)
                update_counts_remove(counts, c_rem2)
                
                improved = True
                if len(current_sol) < best_size:
                    best_size = len(current_sol)
                    best_sol = current_sol.copy()
            else:
                # Revert add of c_new
                current_sol.remove(c_new)
                update_counts_remove(counts, c_new)
        else:
            # Revert add of c_new
            current_sol.remove(c_new)
            update_counts_remove(counts, c_new)
            
        if improved:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # If stuck for a long time, we might restart or change strategy?
        # For now, just keep trying random moves.
        
    # Return the best solution found
    return [decode(c) for c in sorted(best_sol)]

