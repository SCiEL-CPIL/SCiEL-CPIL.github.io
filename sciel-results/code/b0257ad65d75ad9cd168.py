import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    # The evaluator passes the task's alphabet size q, length n and covering radius.
    # We only need to handle q=3, n=8, radius=1 per the task contract.
    # Return a list of codewords, each a list of n ints.
    # Score is the number of codewords (minimized).
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: just return all words (correct but large)
        return [list(word) for word in itertools.product(range(q), repeat=n)]
    
    # q=3, n=8, radius=1
    # Space: 6561 words
    # Ball: each codeword covers 17 words (itself + 8*2 neighbors)
    # Best known: 486, lower bound: 403
    # Seed score 1.1990 means value ≈ 6561/1.1990 ≈ 5474? No wait.
    # Score = seed/value where seed is the seed program's value.
    # If seed score is 1.1990, then value_seed/value_current = 1.1990
    # So current value = value_seed / 1.1990
    # The seed program returns some number of codewords.
    
    # Let me think about what the seed program produces.
    # The current program (score 1.1990) uses greedy + local search.
    # The "seed program" is the baseline. Score = seed/value.
    # So if seed produces 6561 codewords (all of them), and current produces ~5474, then score = 6561/5474 ≈ 1.199.
    # But that's a lot. Let me reconsider.
    
    # Actually, re-reading: "Score: the number of codewords, minimized; it is an upper bound on K_q(n, radius)."
    # And "The evaluator measures the raw value (lower is better). The search score is seed/value"
    # So value = number of codewords returned.
    # Score = seed_value / current_value
    # If score is 1.1990, then seed_value / current_value = 1.1990
    # The seed program likely returns 6561 (all words) or some fixed large number.
    # If seed returns 6561, then current returns 6561/1.1990 ≈ 5472 codewords.
    # That seems high. The best known is 486.
    
    # Hmm, maybe the seed is something else. Let me just focus on making a good algorithm.
    # The goal is to minimize the number of codewords.
    
    N = 3 ** n  # 6561
    # Precompute neighbors using index arithmetic (no need to store all words as tuples)
    # word i in base-3 with n digits
    # neighbors: for each position, change to the other 2 values
    
    # Convert index to word and back
    def idx_to_word(idx):
        w = []
        x = idx
        for _ in range(n):
            w.append(x % 3)
            x //= 3
        return tuple(w)
    
    def word_to_idx(w):
        idx = 0
        for i, v in enumerate(w):
            idx += v * (3 ** i)
        return idx
    
    # Precompute neighbors as array of shape (N, 17)
    # 17 = 1 + n*(q-1) = 1 + 8*2
    neighbors = np.zeros((N, 17), dtype=np.int32)
    for i in range(N):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        col = 1
        for pos in range(n):
            for new_val in range(3):
                if new_val == w[pos]:
                    continue
                w2 = list(w)
                w2[pos] = new_val
                neighbors[i, col] = word_to_idx(tuple(w2))
                col += 1
    
    # Also precompute for each point, the list of codewords that cover it (for reverse lookup)
    # point_to_codewords[j] = list of i such that j is in neighbors[i]
    # This is useful for finding which codewords become redundant when we remove one.
    # But 6561 * 17 = 111537 entries, manageable.
    # Actually, we can compute this on the fly or use the neighbors array.
    
    start_time = time.time()
    TIME_LIMIT = 145.0  # Safety margin from 150s
    
    def time_left():
        return TIME_LIMIT - (time.time() - start_time)
    
    # Phase 1: Greedy set cover
    uncovered = np.ones(N, dtype=bool)
    C = []  # list of codeword indices
    C_set = set()  # for O(1) membership
    
    while np.any(uncovered) and time_left() > 5.0:
        # For each candidate, count how many uncovered points it covers
        counts = uncovered[neighbors].sum(axis=1)  # shape (N,)
        best_idx = int(np.argmax(counts))
        best_count = counts[best_idx]
        
        if best_count == 0:
            break
        
        C.append(best_idx)
        C_set.add(best_idx)
        uncovered[neighbors[best_idx]] = False
    
    # Phase 2: Reduction - remove redundant codewords
    def reduce_code(C, C_set, counts):
        """Remove codewords that are fully covered by others."""
        if not C:
            return C, C_set, counts
        current_C = list(C)
        current_C_set = set(C_set)
        current_counts = counts.copy()
        
        changed = True
        while changed:
            changed = False
            next_C = []
            for c in current_C:
                # Check if all neighbors of c are covered by at least 2 codewords
                # (meaning c is redundant)
                if np.all(current_counts[neighbors[c]] >= 2):
                    current_counts[neighbors[c]] -= 1
                    current_C_set.discard(c)
                    changed = True
                else:
                    next_C.append(c)
            current_C = next_C
        return current_C, current_C_set, current_counts
    
    # Compute counts
    counts = np.zeros(N, dtype=int)
    for c in C:
        counts[neighbors[c]] += 1
    
    C, C_set, counts = reduce_code(C, C_set, counts)
    
    best_C = list(C)
    best_C_set = set(C_set)
    best_counts = counts.copy()
    best_size = len(C)
    
    # Phase 3: Local search with swaps
    def try_swap(C, C_set, counts, c_remove, c_add):
        """Try removing c_remove and adding c_add. Return (new_C, new_C_set, new_counts) or None."""
        if c_add in C_set:
            return None
        # Check coverage after removal
        # Remove c_remove
        counts2 = counts.copy()
        counts2[neighbors[c_remove]] -= 1
        # Add c_add
        counts2[neighbors[c_add]] += 1
        if np.any(counts2 < 1):
            return None
        new_C = list(C)
        new_C.remove(c_remove)
        new_C.append(c_add)
        new_C_set = set(C_set)
        new_C_set.discard(c_remove)
        new_C_set.add(c_add)
        return new_C, new_C_set, counts2
    
    def find_candidates_for_point(point, C_set):
        """Find codewords that cover `point`."""
        candidates = []
        for c in C_set:
            if np.any(neighbors[c] == point):
                candidates.append(c)
        return candidates
    
    # Faster: precompute for each point, the list of codewords covering it
    # But this is expensive to maintain. Instead, use the neighbors array.
    
    # Actually, for finding candidates that cover a set of points, we can:
    # For each point p in the set, find all codewords c such that p in neighbors[c]
    # This is: for each c, check if p is in neighbors[c]
    # We can precompute a reverse mapping: for each point p, list of c where p in neighbors[c]
    
    # Let's precompute reverse_neighbors: for each point, the set of codewords that cover it
    # This is a list of sets, or we can use a 2D boolean array
    # N=6561, 17 entries per codeword, total 111537 edges
    # reverse_adj: list of lists, reverse_adj[p] = list of c such that p in neighbors[c]
    
    reverse_adj = [[] for _ in range(N)]
    for c in range(N):
        for p in neighbors[c]:
            reverse_adj[p].append(c)
    
    # Now finding candidates that cover a set of points is:
    # Intersection of reverse_adj[p] for p in points
    # Then filter out those in C_set
    
    def find_covering_candidates(points, C_set):
        """Find all codewords that cover all points in `points`."""
        if not points:
            return []
        if len(points) == 1:
            candidates = set(reverse_adj[points[0]])
        else:
            candidates = set(reverse_adj[points[0]])
            for p in points[1:]:
                candidates &= set(reverse_adj[p])
                if not candidates:
                    break
        candidates -= C_set
        return candidates
    
    # Main local search loop
    stall_count = 0
    max_stall = 30
    
    while time_left() > 1.0:
        # Reduction
        old_size = len(C)
        C, C_set, counts = reduce_code(C, C_set, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = list(C)
            best_C_set = set(C_set)
            best_counts = counts.copy()
        
        if len(C) == old_size:
            # No reduction, try swaps
            improved = False
            
            # Try random swaps
            C_list = list(C)
            random.shuffle(C_list)
            
            for c_remove in C_list:
                if time_left() < 0.5:
                    break
                if improved:
                    break
                
                # Find points uniquely covered by c_remove
                u_mask = counts[neighbors[c_remove]] == 1
                unique_points = neighbors[c_remove][u_mask].tolist()
                
                if len(unique_points) == 0:
                    # c_remove is redundant (should have been caught by reduce)
                    C.remove(c_remove)
                    C_set.discard(c_remove)
                    counts[neighbors[c_remove]] -= 1
                    improved = True
                    continue
                
                if len(unique_points) > 3:
                    continue  # Too many unique points, unlikely to find a good replacement
                
                # Find candidates that cover all unique_points and are not in C
                candidates = find_covering_candidates(unique_points, C_set)
                
                if not candidates:
                    continue
                
                # Try each candidate (or a random subset if too many)
                cand_list = list(candidates)
                if len(cand_list) > 50:
                    random.shuffle(cand_list)
                    cand_list = cand_list[:50]
                
                random.shuffle(cand_list)
                
                for c_add in cand_list:
                    result = try_swap(C, C_set, counts, c_remove, c_add)
                    if result is None:
                        continue
                    new_C, new_C_set, new_counts = result
                    
                    # Reduce after swap
                    new_C, new_C_set, new_counts = reduce_code(new_C, new_C_set, new_counts)
                    
                    if len(new_C) < len(C):
                        C = new_C
                        C_set = new_C_set
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = list(C)
                            best_C_set = set(C_set)
                            best_counts = counts.copy()
                        improved = True
                        break
                    elif len(new_C) < old_size:
                        # Same size but we made progress (reduced from old_size)
                        C = new_C
                        C_set = new_C_set
                        counts = new_counts
                        improved = True
                        break
                    else:
                        # Accept if same size (for diversity), but don't break
                        # Actually, let's only accept if strictly better or equal
                        # To avoid cycling, let's be conservative
                        pass
            
            if not improved:
                stall_count += 1
                # Random kick
                if C:
                    c_idx = random.randint(0, len(C) - 1)
                    c_remove = C[c_idx]
                    # Pick a random codeword near c_remove
                    cand = neighbors[c_remove][random.randint(0, 16)]
                    if cand not in C_set:
                        result = try_swap(C, C_set, counts, c_remove, cand)
                        if result is not None:
                            new_C, new_C_set, new_counts = result
                            new_C, new_C_set, new_counts = reduce_code(new_C, new_C_set, new_counts)
                            C = new_C
                            C_set = new_C_set
                            counts = new_counts
                    else:
                        # Try a completely random codeword
                        cand = random.randint(0, N - 1)
                        if cand not in C_set:
                            result = try_swap(C, C_set, counts, c_remove, cand)
                            if result is not None:
                                new_C, new_C_set, new_counts = result
                                new_C, new_C_set, new_counts = reduce_code(new_C, new_C_set, new_counts)
                                C = new_C
                                C_set = new_C_set
                                counts = new_counts
            else:
                stall_count = 0
        else:
            stall_count = 0
        
        # If we've been stuck for a long time, restart with a new random greedy
        if stall_count > max_stall and time_left() > 20.0:
            # Try a different initialization
            uncovered = np.ones(N, dtype=bool)
            C = []
            C_set = set()
            
            # Randomized greedy: add some randomness
            while np.any(uncovered) and time_left() > 10.0:
                counts = uncovered[neighbors].sum(axis=1)
                # Add noise
                counts_noisy = counts.astype(np.float64)
                noise = random.random() * 3
                counts_noisy += noise
                best_idx = int(np.argmax(counts_noisy))
                best_count = counts[best_idx]
                
                if best_count == 0:
                    break
                
                C.append(best_idx)
                C_set.add(best_idx)
                uncovered[neighbors[best_idx]] = False
            
            counts = np.zeros(N, dtype=int)
            for c in C:
                counts[neighbors[c]] += 1
            C, C_set, counts = reduce_code(C, C_set, counts)
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = list(C)
                best_C_set = set(C_set)
                best_counts = counts.copy()
            
            stall_count = 0
    
    # Final reduction
    best_C, best_C_set, best_counts = reduce_code(best_C, best_C_set, best_counts)
    
    # Convert indices to words
    result = []
    for idx in best_C:
        result.append(list(idx_to_word(idx)))
    
    return result

