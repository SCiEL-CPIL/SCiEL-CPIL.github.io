import random
import numpy as np
import time
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Optimized for q=3, n=8, radius=1.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word

    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors
    neighbors = [None] * total_words
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.append(word_to_int(new_word, q))
        neighbors[w] = neigh
    
    neighbors_np = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    # Precompute a bitmask representation for fast subset checks
    # For each word, create a frozenset of its neighbors for O(1) subset checks
    neighbors_set = [frozenset(neigh) for neigh in neighbors]
    
    start_time = time.time()
    time_limit = 115  # Safety margin

    def fast_greedy():
        """Greedy set cover with criticality-based tie-breaking."""
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        uncovered_count = total_words
        used_words = set()
        
        # Precompute for each word, how many other words can cover it (degree in hypergraph)
        # This helps with criticality-based ordering
        cover_degree = np.zeros(total_words, dtype=np.int32)
        for w in range(total_words):
            cover_degree[w] = len(neighbors[w])  # Always 17 for q=3,n=8,r=1
        
        while uncovered_count > 0:
            best_word = -1
            best_score = -1
            
            start_idx = random.randint(0, total_words - 1)
            
            for i in range(total_words):
                w = (start_idx + i) % total_words
                if w in used_words:
                    continue
                
                neigh_arr = neighbors_np[w]
                new_cov = np.sum(~covered[neigh_arr])
                
                if new_cov == 0:
                    continue
                
                # Criticality-based scoring: prefer words that cover rare (hard-to-cover) words
                # Score = new_cov + small bonus for covering words with low cover_degree
                # But for simplicity and speed, just use new_cov with random tie-breaking
                score = new_cov
                # Add tiny random perturbation to break ties
                score += random.random() * 0.001
                
                if score > best_score:
                    best_score = score
                    best_word = w
                    if new_cov == uncovered_count:
                        break
            
            if best_word == -1:
                break
            
            codewords.append(best_word)
            used_words.add(best_word)
            covered[neighbors_np[best_word]] = True
            uncovered_count -= np.sum(~covered[neighbors_np[best_word]])
        
        return codewords

    def optimize(codewords, time_limit):
        if not codewords:
            return codewords
            
        coverage_count = np.zeros(total_words, dtype=np.int32)
        cw_set = set(codewords)
        
        for cw in codewords:
            coverage_count[neighbors_np[cw]] += 1
            
        def try_remove_redundant():
            """Vectorized redundancy removal: find all redundant codewords at once."""
            removed_any = False
            changed = True
            while changed:
                changed = False
                n_cw = len(codewords)
                if n_cw == 0:
                    break
                
                # For each codeword, check if all its neighbors have coverage >= 2
                # We can vectorize this: for each cw, check if min(coverage_count[neigh]) >= 2
                to_remove = []
                for i in range(n_cw):
                    cw = codewords[i]
                    neigh_arr = neighbors_np[cw]
                    if np.all(coverage_count[neigh_arr] >= 2):
                        to_remove.append(i)
                
                if not to_remove:
                    break
                
                # Remove all redundant codewords at once
                for i in reversed(to_remove):
                    cw = codewords[i]
                    coverage_count[neighbors_np[cw]] -= 1
                    codewords.pop(i)
                    cw_set.discard(cw)
                    changed = True
                    removed_any = True
            
            return removed_any

        def try_2opt_move():
            """Try to remove two codewords and add one that covers all their unique words."""
            n_cw = len(codewords)
            if n_cw < 2:
                return False
            
            # Pick two random codewords
            i, j = random.sample(range(n_cw), 2)
            if i > j:
                i, j = j, i
            
            cw1 = codewords[i]
            cw2 = codewords[j]
            neigh1 = neighbors_np[cw1]
            neigh2 = neighbors_np[cw2]
            
            # Words uniquely covered by cw1 or cw2 (after removing both)
            # A word is uniquely covered if its coverage count equals the number of these two cws that cover it
            # After removing both, a word is uncovered if it was covered only by these two or one of them
            
            # Unique to cw1: covered by cw1 but not cw2, and coverage_count == 1
            # Unique to cw2: covered by cw2 but not cw1, and coverage_count == 1
            # Covered by both: coverage_count == 2, will become uncovered
            
            # Find all words that will become uncovered
            # These are words where coverage_count equals the number of {cw1, cw2} that cover it
            # and that number > 0
            
            in_both = np.isin(neigh1, neigh2)
            only_cw1 = neigh1[~in_both]
            only_cw2 = neigh2[~np.isin(neigh2, neigh1)]
            both_words = neigh1[in_both]
            
            # Words that will be uncovered:
            # - only_cw1 where coverage_count == 1
            # - only_cw2 where coverage_count == 1  
            # - both_words where coverage_count == 2
            
            uncov_mask = np.zeros(total_words, dtype=bool)
            uncov_mask[only_cw1] = coverage_count[only_cw1] == 1
            uncov_mask[only_cw2] = coverage_count[only_cw2] == 1
            uncov_mask[both_words] = coverage_count[both_words] == 2
            
            uncov_words = np.where(uncov_mask)[0]
            if len(uncov_words) == 0:
                # Both are redundant, should have been caught by try_remove_redundant
                return False
            
            # Find a word that covers all uncov_words
            # Use the first uncov_word to filter candidates
            u0 = uncov_words[0]
            candidates = neighbors[u0]
            
            best_new_cw = -1
            best_cost = float('inf')
            
            for new_cw in candidates:
                if new_cw in cw_set:
                    continue
                
                new_neigh = neighbors_np[new_cw]
                
                # Check if new_cw covers all uncov_words
                if not np.all(np.isin(uncov_words, new_neigh)):
                    continue
                
                # Cost: number of words in new_neigh that would become uniquely covered
                # After the move: remove cw1, cw2; add new_cw
                # For words in new_neigh:
                #   If in neigh1 or neigh2: net change is -1 (if in one) or -2 (if in both) + 1
                #   If not in neigh1 or neigh2: net change is +1
                
                # Simplified: count words in new_neigh that have low coverage after the move
                # This is complex, so use a simpler heuristic:
                # Prefer new_cw that overlaps more with neigh1 and neigh2
                
                overlap1 = np.sum(np.isin(new_neigh, neigh1))
                overlap2 = np.sum(np.isin(new_neigh, neigh2))
                cost = -(overlap1 + overlap2)  # Prefer more overlap
                
                if cost < best_cost:
                    best_cost = cost
                    best_new_cw = new_cw
                    if best_cost <= -30:  # Very good overlap
                        break
            
            if best_new_cw != -1:
                # Perform the move
                coverage_count[neigh1] -= 1
                coverage_count[neigh2] -= 1
                new_neigh = neighbors_np[best_new_cw]
                coverage_count[new_neigh] += 1
                
                # Remove cw1 and cw2 (remove higher index first)
                codewords.pop(j)
                codewords.pop(i)
                cw_set.discard(cw1)
                cw_set.discard(cw2)
                
                # Add new codeword
                codewords.append(best_new_cw)
                cw_set.add(best_new_cw)
                
                try_remove_redundant()
                return True
            
            return False

        # Initial redundancy removal
        try_remove_redundant()
        
        # Local Search
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
                
            if len(codewords) == 0:
                break
            
            # Try 2-opt moves first (higher impact)
            if random.random() < 0.3:
                if try_2opt_move():
                    continue
            
            # Try swaps
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            old_neigh = neighbors_np[old_cw]
            unique_mask = coverage_count[old_neigh] == 1
            unique_words = old_neigh[unique_mask]
            
            if len(unique_words) == 0:
                coverage_count[old_neigh] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            best_new_cw = -1
            best_cost = float('inf')
            
            # Use the first unique word to filter candidates
            u0 = unique_words[0]
            candidates = neighbors[u0]
            
            for new_cw in candidates:
                if new_cw in cw_set:
                    continue
                if new_cw == old_cw:
                    continue
                
                new_neigh = neighbors_np[new_cw]
                
                if not np.all(np.isin(unique_words, new_neigh)):
                    continue
                
                in_both = np.isin(new_neigh, old_neigh)
                mask_only_new = ~in_both
                cost = np.sum((coverage_count[new_neigh] == 0) & mask_only_new)
                
                if cost < best_cost:
                    best_cost = cost
                    best_new_cw = new_cw
                    if cost == 0:
                        break
            
            if best_new_cw != -1:
                coverage_count[old_neigh] -= 1
                new_neigh = neighbors_np[best_new_cw]
                coverage_count[new_neigh] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
            else:
                # Perturb: remove a random codeword and fill the gap
                if random.random() < 0.15:
                    rem_idx = random.randint(0, len(codewords) - 1)
                    rem_cw = codewords[rem_idx]
                    coverage_count[neighbors_np[rem_cw]] -= 1
                    codewords.pop(rem_idx)
                    cw_set.discard(rem_cw)
                    
                    uncovered_mask = coverage_count == 0
                    uncovered_words = np.where(uncovered_mask)[0]
                    
                    if len(uncovered_words) > 0:
                        best_fill = -1
                        best_cov = -1
                        for w in range(total_words):
                            if w in cw_set:
                                continue
                            neigh_arr = neighbors_np[w]
                            cov = np.sum(uncovered_mask[neigh_arr])
                            if cov > best_cov:
                                best_cov = cov
                                best_fill = w
                                if best_cov == len(uncovered_words):
                                    break
                        
                        if best_fill != -1:
                            codewords.append(best_fill)
                            cw_set.add(best_fill)
                            coverage_count[neighbors_np[best_fill]] += 1
                            try_remove_redundant()
        
        try_remove_redundant()
        return codewords

    best_codewords = []
    
    while time.time() - start_time < time_limit - 5:
        cw = fast_greedy()
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            cw = optimize(cw, remaining_time - 5)
        
        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw
            if len(cw) <= 403:
                break

    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
