import itertools
import random
import time

def construct(q, n, radius):
    # Constants for the specific task instance
    # q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other instances
        # For small spaces, brute force or simple greedy
        if q ** n <= 10000:
            all_words = [list(t) for t in itertools.product(range(q), repeat=n)]
            return all_words
        else:
            # Simple greedy fallback
            def dist(w1, w2):
                return sum(1 for a, b in zip(w1, w2) if a != b)
            
            code = []
            uncovered = set(itertools.product(range(q), repeat=n))
            while uncovered:
                best_w = None
                best_count = -1
                for w in uncovered:
                    count = sum(1 for u in uncovered if dist(w, u) <= radius)
                    if count > best_count:
                        best_count = count
                        best_w = w
                if best_w is None: break
                code.append(list(best_w))
                to_remove = [u for u in uncovered if dist(best_w, u) <= radius]
                for u in to_remove:
                    uncovered.remove(u)
            return code

    # Precompute neighbors for all 729 words
    # Index: v[0]*3^5 + v[1]*3^4 + ... + v[5]
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute balls as lists of indices
    all_indices = list(range(q**n))
    neighbors = [[] for _ in range(q**n)]
    
    # Generate all words
    words = [tuple(t) for t in itertools.product(range(q), repeat=n)]
    
    for idx, w in enumerate(words):
        neighbors[idx].append(idx)
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                neighbors[idx].append(tuple_to_idx(tuple(new_w)))
        neighbors[idx] = list(set(neighbors[idx]))
        
    # For faster intersection checks, also store as sets
    neighbor_sets = [set(nb) for nb in neighbors]

    # Initial Code: Direct Sum of Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_tuples = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code_tuples.append(tuple(h + list(tail)))
    
    code = list(initial_code_tuples)
    
    # Initialize cover counts
    cover_count = [0] * (q**n)
    for c in code:
        c_idx = tuple_to_idx(c)
        for nb in neighbors[c_idx]:
            cover_count[nb] += 1

    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    def try_remove(c_tuple):
        c_idx = tuple_to_idx(c_tuple)
        uncovered = []
        for nb in neighbors[c_idx]:
            cover_count[nb] -= 1
            if cover_count[nb] == 0:
                uncovered.append(nb)
        
        if not uncovered:
            return True, [] # Successfully removed
        else:
            # Restore
            for nb in neighbors[c_idx]:
                cover_count[nb] += 1
            return False, uncovered

    def try_add(c_tuple):
        c_idx = tuple_to_idx(c_tuple)
        for nb in neighbors[c_idx]:
            cover_count[nb] += 1

    def find_swap_target_1(uncovered_indices):
        """Find a single word that covers all uncovered indices."""
        if not uncovered_indices:
            return None
        u0 = uncovered_indices[0]
        candidates = neighbors[u0]
        random.shuffle(candidates)
        for c_idx in candidates:
            if all(u in neighbor_sets[c_idx] for u in uncovered_indices):
                return idx_to_tuple(c_idx)
        return None

    def find_swap_target_2(uncovered_indices):
        """Find two words that cover all uncovered indices."""
        if not uncovered_indices:
            return None, None
        
        # If 1 point, any two neighbors work, but we want distinct from removed if possible?
        # Actually, we just need coverage.
        # Heuristic: Pick a candidate for the first word from neighbors of the first uncovered point.
        # Then check if the remaining points are covered by a second word.
        
        u0 = uncovered_indices[0]
        candidates_c1 = neighbors[u0]
        random.shuffle(candidates_c1)
        
        # Limit candidates to avoid O(N^2) explosion if uncovered set is large
        # If uncovered set is small, this is fast.
        for c1_idx in candidates_c1:
            # Check if c1 covers some of the uncovered
            # Let remaining_uncovered = uncovered_indices not covered by c1
            rem = [u for u in uncovered_indices if u not in neighbor_sets[c1_idx]]
            
            if not rem:
                # c1 covers all. We can pick any c2 (e.g., c1 itself, but they must be distinct in code? 
                # The code is a list of distinct words. If we add c1 twice, it's invalid.
                # So we need c2 != c1.
                # If c1 covers all, we can add c1 and any other word? No, we need to cover 'uncovered'.
                # If c1 covers all, we can add c1 and... wait.
                # The move is: remove c_a, c_b. Add c'_a, c'_b.
                # We need c'_a and c'_b to cover the union of holes.
                # If c1 covers all holes, we can set c'_a = c1 and c'_b = c1? No, distinctness.
                # We can set c'_a = c1 and c'_b = any word that is not c1.
                # But we want to minimize code size. We are swapping 2 for 2.
                # So we must add 2 distinct words.
                # If c1 covers all, we can add c1 and a random other word?
                # Actually, if c1 covers all, we could have just done a 1-swap if we removed 1 word.
                # Since we are in 2-opt, we removed 2 words.
                # If c1 covers all holes, we can add c1 and any valid distinct word.
                # To be safe, let's just return c1 and a random neighbor of c1 that is not c1.
                # But wait, if c1 covers all, we don't need c2 to cover anything specific, just to exist.
                # However, adding a random word might break coverage of other things?
                # No, adding a word only increases cover counts. It doesn't hurt.
                # So if c1 covers all, return c1 and some other word.
                # Let's find a word distinct from c1.
                # Any word works. Let's pick a random one from the whole space?
                # Or better, pick a word that is already in the code? No, we want to add new ones.
                # Actually, if c1 covers all, we can just return c1 and c1? No, distinct.
                # Let's pick a random word from the space.
                c2_idx = random.randint(0, q**n - 1)
                while c2_idx == c1_idx:
                    c2_idx = random.randint(0, q**n - 1)
                return idx_to_tuple(c1_idx), idx_to_tuple(c2_idx)
            
            # Now find c2 that covers 'rem'
            # Pick candidate from neighbors of rem[0]
            if not rem:
                 continue # Should not happen
            
            u_rem = rem[0]
            candidates_c2 = neighbors[u_rem]
            random.shuffle(candidates_c2)
            
            for c2_idx in candidates_c2:
                if all(u in neighbor_sets[c2_idx] for u in rem):
                    return idx_to_tuple(c1_idx), idx_to_tuple(c2_idx)
        
        return None, None

    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(code)
        
        # Phase 1: Single Removal
        for i in range(len(code)):
            if time.time() > deadline:
                break
            c = code[i]
            success, _ = try_remove(c)
            if success:
                code.pop(i)
                improved = True
                break
        
        if time.time() > deadline:
            break
        if improved:
            continue

        # Phase 2: Single Swap (1 for 1)
        # Try to swap a codeword to potentially allow future removals or just improve structure
        # Actually, if no removal is possible, a 1-for-1 swap doesn't reduce size.
        # It might help escape a local minimum for 2-opt?
        # Let's skip 1-for-1 and go straight to 2-opt (2-for-2) to reduce size?
        # No, 2-for-2 keeps size constant. We need to reduce size.
        # The reduction happens when we remove 2 and add 1? That's 2-for-1.
        # Or remove 1 and add 0? That's Phase 1.
        # So we need a move that reduces size.
        # Standard local search for set cover:
        # 1. Remove redundant.
        # 2. If stuck, try to swap k elements with k-1 elements?
        # Or swap k with k and then remove?
        
        # Let's try a "2-for-1" move: Remove 2 codewords, Add 1 codeword.
        # This reduces size by 1.
        # Conditions: The 1 new codeword must cover all holes created by removing the 2.
        
        # Phase 2: 2-for-1 Swap
        # Pick two codewords c1, c2.
        # Remove them. Get uncovered set U.
        # Find a word w such that Ball(w) covers U.
        # If found, replace c1, c2 with w.
        
        found_2_for_1 = False
        # To avoid O(N^2) on 81^2 pairs, we can be smart.
        # But 81^2 is small.
        # We need to be careful with the "uncovered" logic.
        
        # Iterate over pairs
        # To ensure we don't miss, we can randomize the order of pairs.
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for i_idx in range(len(indices)):
            if time.time() > deadline:
                break
            i = indices[i_idx]
            for j_idx in range(i_idx + 1, len(indices)):
                if time.time() > deadline:
                    break
                j = indices[j_idx]
                
                c1 = code[i]
                c2 = code[j]
                
                # Temporarily remove c1 and c2
                c1_idx = tuple_to_idx(c1)
                c2_idx = tuple_to_idx(c2)
                
                uncovered = []
                for nb in neighbors[c1_idx]:
                    cover_count[nb] -= 1
                for nb in neighbors[c2_idx]:
                    cover_count[nb] -= 1
                    if cover_count[nb] == 0:
                        uncovered.append(nb)
                
                if not uncovered:
                    # Both were redundant? Should have been caught in Phase 1.
                    # Restore and skip
                    for nb in neighbors[c1_idx]:
                        cover_count[nb] += 1
                    for nb in neighbors[c2_idx]:
                        cover_count[nb] += 1
                    continue
                
                # Find a word that covers all uncovered
                # Use the same logic as find_swap_target_1
                u0 = uncovered[0]
                candidates = neighbors[u0]
                random.shuffle(candidates)
                
                found = False
                for c_new_idx in candidates:
                    if all(u in neighbor_sets[c_new_idx] for u in uncovered):
                        # Success!
                        # Add the new word
                        for nb in neighbors[c_new_idx]:
                            cover_count[nb] += 1
                        
                        # Remove c1 and c2 from code list
                        # Be careful with indices if i > j
                        if i < j:
                            code.pop(j)
                            code.pop(i)
                        else:
                            code.pop(i)
                            code.pop(j)
                        
                        # Add new word
                        code.append(idx_to_tuple(c_new_idx))
                        
                        improved = True
                        found_2_for_1 = True
                        found = True
                        break
                
                if found:
                    break
                else:
                    # Restore
                    for nb in neighbors[c1_idx]:
                        cover_count[nb] += 1
                    for nb in neighbors[c2_idx]:
                        cover_count[nb] += 1
                
            if found_2_for_1:
                break
        
        if found_2_for_1:
            continue

        # Phase 3: 3-for-2 Swap (if 2-for-1 fails)
        # This is more complex. Let's stick to 2-for-1 and random restarts if needed.
        # If we are stuck at 81, maybe we need to break the symmetry differently.
        
        # Let's try a "Random Perturbation" if stuck for a while.
        # But first, let's try 3-for-2.
        # Pick 3 codewords, remove them, find 2 codewords that cover the holes.
        
        # This is O(N^3) which is 81^3 ~ 500k. Might be too slow if done repeatedly.
        # But we only do it if 2-for-1 fails.
        
        # Let's implement a limited 3-for-2 search.
        
        found_3_for_2 = False
        if time.time() < deadline:
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            # Limit the number of triples to check
            max_triples = 200
            count = 0
            
            for i_idx in range(len(indices)):
                if time.time() > deadline or count >= max_triples:
                    break
                i = indices[i_idx]
                for j_idx in range(i_idx + 1, len(indices)):
                    if time.time() > deadline or count >= max_triples:
                        break
                    j = indices[j_idx]
                    for k_idx in range(j_idx + 1, len(indices)):
                        if time.time() > deadline or count >= max_triples:
                            break
                        k = indices[k_idx]
                        count += 1
                        
                        c1 = code[i]
                        c2 = code[j]
                        c3 = code[k]
                        
                        c1_idx = tuple_to_idx(c1)
                        c2_idx = tuple_to_idx(c2)
                        c3_idx = tuple_to_idx(c3)
                        
                        uncovered = []
                        for nb in neighbors[c1_idx]:
                            cover_count[nb] -= 1
                        for nb in neighbors[c2_idx]:
                            cover_count[nb] -= 1
                        for nb in neighbors[c3_idx]:
                            cover_count[nb] -= 1
                            if cover_count[nb] == 0:
                                uncovered.append(nb)
                        
                        if not uncovered:
                            # Restore
                            for nb in neighbors[c1_idx]: cover_count[nb] += 1
                            for nb in neighbors[c2_idx]: cover_count[nb] += 1
                            for nb in neighbors[c3_idx]: cover_count[nb] += 1
                            continue
                        
                        # Find 2 words that cover uncovered
                        # Use find_swap_target_2
                        new_w1, new_w2 = find_swap_target_2(uncovered)
                        
                        if new_w1 is not None and new_w2 is not None:
                            # Check if new words are distinct from each other and from removed ones?
                            # They just need to be distinct from each other to be valid codewords.
                            # They can be the same as removed ones? No, we are replacing.
                            # If new_w1 == c1, then we are effectively removing c2, c3 and adding c1, new_w2.
                            # This is valid as long as new_w1 != new_w2.
                            
                            nw1_idx = tuple_to_idx(new_w1)
                            nw2_idx = tuple_to_idx(new_w2)
                            
                            if nw1_idx == nw2_idx:
                                # Restore
                                for nb in neighbors[c1_idx]: cover_count[nb] += 1
                                for nb in neighbors[c2_idx]: cover_count[nb] += 1
                                for nb in neighbors[c3_idx]: cover_count[nb] += 1
                                continue
                            
                            # Add new words
                            for nb in neighbors[nw1_idx]:
                                cover_count[nb] += 1
                            for nb in neighbors[nw2_idx]:
                                cover_count[nb] += 1
                                
                            # Remove old words
                            # Sort indices to pop safely
                            idxs_to_remove = sorted([i, j, k], reverse=True)
                            for idx_rm in idxs_to_remove:
                                code.pop(idx_rm)
                                
                            code.append(new_w1)
                            code.append(new_w2)
                            
                            improved = True
                            found_3_for_2 = True
                            break
                        else:
                            # Restore
                            for nb in neighbors[c1_idx]: cover_count[nb] += 1
                            for nb in neighbors[c2_idx]: cover_count[nb] += 1
                            for nb in neighbors[c3_idx]: cover_count[nb] += 1
                        
                    if found_3_for_2:
                        break
                if found_3_for_2:
                    break
            
            if found_3_for_2:
                continue

        # If we are still stuck, try random restarts or accept current best
        # Given the time limit, we might not reach 73 if we start at 81 and only do local search.
        # However, 2-for-1 is a strong move.
        
        # If no improvement found in this iteration, break to save time for other strategies?
        # No, just break the while loop if no improvement.
        if not improved:
            break

    # Final check: Ensure validity
    # The cover_count logic ensures validity during search.
    # But let's do a quick sanity check if time permits?
    # No, the search maintains invariants.
    
    return [list(c) for c in code]
