import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Parameters for the specific task instance
    # q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1()
    
    # Fallback for other parameters (simple greedy)
    total_words = q ** n
    if total_words > 100000:
        # Too large for simple fallback, return empty or minimal
        return []
        
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    ball_size = 1 + n * (q - 1)
    
    # Precompute balls
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
        
    for i in range(total_words):
        w = all_words[i]
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors.append(idx)
        balls[i] = neighbors
        
    is_covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    while not np.all(is_covered):
        if time.time() > deadline - 5:
            break
        newly_covered = (~is_covered)[balls]
        scores = newly_covered.sum(axis=1)
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
        codewords.append(best_idx)
        is_covered[balls[best_idx]] = True
        
    return [list(all_words[c]) for c in codewords]

def _solve_ternary_8_1():
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17
    
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for return
    
    # Precompute all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    # Precompute balls using base-3 arithmetic
    balls = np.zeros((total, ball_size), dtype=np.int32)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # Precompute overlap graph as bit-parallel integers
    # overlap[w] is a big int where bit i is set iff words w and i are within distance 2
    # (i.e., their radius-1 balls overlap, meaning they can't both be in a valid code)
    # Two words have overlapping balls iff their distance <= 2
    # We compute this using the balls: overlap[w] = union of balls of all words in balls[w]
    # Actually simpler: w and v have overlapping balls iff dist(w,v) <= 2
    # Let's compute dist(w,v) <= 2 for all pairs efficiently.
    
    # For each word w, the set of words within distance 2 is:
    # w itself, all words at distance 1 (16 words), and all words at distance 2
    # Distance 2: choose 2 positions to change (C(8,2)=28), each with (q-1)^2=4 options = 112 words
    # Total: 1 + 16 + 112 = 129 words within distance 2
    
    # Compute overlap masks as Python big integers (6561 bits each)
    # overlap_mask[w] has bit i set if dist(w, i) <= 2
    # We can compute this from balls: words within distance 2 of w are 
    # the union of balls[x] for x in balls[w], minus... no that's not right.
    # Actually, dist(w,v) <= 2 means v is in the "radius-2 ball" of w.
    # Let's just compute it directly.
    
    # Precompute: for each word, the list of words within distance 2
    # dist2_neighbors[w] = list of v such that dist(w,v) <= 2
    # This includes w, dist-1 neighbors (16), and dist-2 neighbors (112)
    
    # Build dist-2 neighbor lists
    # For distance 2: pick 2 positions, change both
    dist2_lists = []
    for w in range(total):
        # Get digits
        digits = []
        tmp = w
        for pos in range(n):
            digits.append((tmp // weights[pos]) % 3)
        
        neighbors = [w]  # distance 0
        # Distance 1
        for pos in range(n):
            for new_digit in range(q):
                if new_digit == digits[pos]:
                    continue
                new_w = w + (new_digit - digits[pos]) * weights[pos]
                neighbors.append(new_w)
        # Distance 2
        for p1 in range(n):
            for p2 in range(p1 + 1, n):
                for d1 in range(q):
                    if d1 == digits[p1]:
                        continue
                    for d2 in range(q):
                        if d2 == digits[p2]:
                            continue
                        new_w = w + (d1 - digits[p1]) * weights[p1] + (d2 - digits[p2]) * weights[p2]
                        neighbors.append(new_w)
        
        # Deduplicate (shouldn't be needed but safe)
        dist2_lists.append(set(neighbors))
    
    # Build overlap masks as big integers
    # overlap_mask[w] = big int with bit v set if v in dist2_lists[w]
    # Using Python integers for bit operations
    overlap_masks = [0] * total
    for w in range(total):
        mask = 0
        for v in dist2_lists[w]:
            mask |= (1 << v)
        overlap_masks[w] = mask
    
    # Also store as numpy array of Python objects for indexing
    # Actually, let's keep them as a list of Python ints for speed
    
    # Bit-parallel greedy Maximum Independent Set
    # A valid covering code is an independent set in the overlap graph
    # (no two codewords are within distance 2 of each other)
    # AND it must cover all words (each word is within distance 1 of some codeword)
    
    # Greedy MIS: repeatedly pick the vertex with the fewest remaining neighbors
    # (or most isolated), add to independent set, remove it and all its neighbors
    
    def greedy_mis(seed_val):
        np.random.seed(seed_val)
        random.seed(seed_val)
        
        # active[v] = True if v is still in the graph
        active = [True] * total
        
        # remaining_neighbors[v] = overlap_masks[v] & active_mask
        # We'll track this incrementally
        
        # For efficiency, maintain a count of how many active neighbors each vertex has
        # degree[v] = number of active vertices in overlap_masks[v]
        
        # Initial degrees
        degree = [0] * total
        for w in range(total):
            # Count active neighbors of w
            mask = overlap_masks[w]
            # Count bits set in mask
            # Python int bit_count() is available in 3.8+
            degree[w] = mask.bit_count()
        
        independent_set = []
        remaining = total
        
        while remaining > 0:
            if time.time() > deadline - 15:
                break
            
            # Find vertex with minimum degree among active vertices
            # For speed, we can sample or use a priority structure
            # Simple: find min degree among active
            min_deg = total + 1
            best_v = -1
            for v in range(total):
                if active[v] and degree[v] < min_deg:
                    min_deg = degree[v]
                    best_v = v
            
            if best_v == -1:
                break
            
            # Add best_v to independent set
            independent_set.append(best_v)
            active[best_v] = False
            remaining -= 1
            
            # Remove best_v and all its active neighbors from the graph
            # For each active neighbor u of best_v:
            #   active[u] = False
            #   For each active neighbor v of u: degree[v] -= 1
            #   remaining -= 1
            
            # Get active neighbors of best_v
            mask = overlap_masks[best_v]
            # Iterate through bits set in mask that are still active
            m = mask
            while m:
                # Get lowest set bit
                low_bit = m & (-m)
                u = low_bit.bit_length() - 1
                m ^= low_bit
                
                if not active[u]:
                    continue
                
                active[u] = False
                remaining -= 1
                
                # Decrease degree of all active neighbors of u
                um = overlap_masks[u]
                um2 = um
                while um2:
                    lb = um2 & (-um2)
                    v = lb.bit_length() - 1
                    um2 ^= lb
                    if active[v]:
                        degree[v] -= 1
        
        return independent_set
    
    # Verify coverage and clean up
    def verify_and_cleanup(codewords):
        """Check if codewords form a valid cover, remove redundant ones."""
        if not codewords:
            return codewords
        
        cover_counts = np.zeros(total, dtype=np.int32)
        for c in codewords:
            cover_counts[balls[c]] += 1
        
        # Remove redundant codewords
        i = 0
        while i < len(codewords):
            c = codewords[i]
            cover_counts[balls[c]] -= 1
            if not np.any(cover_counts == 0):
                codewords.pop(i)
            else:
                cover_counts[balls[c]] += 1
                i += 1
        
        return codewords
    
    # Local search polish: try swaps to reduce size
    def local_search_polish(codewords, time_limit):
        """Try to reduce codeword count via swaps that enable removals."""
        if not codewords:
            return codewords
        
        cover_counts = np.zeros(total, dtype=np.int32)
        for c in codewords:
            cover_counts[balls[c]] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Phase 1: Remove redundant
            i = 0
            while i < len(codewords):
                c = codewords[i]
                cover_counts[balls[c]] -= 1
                if not np.any(cover_counts == 0):
                    codewords.pop(i)
                    improved = True
                else:
                    cover_counts[balls[c]] += 1
                    i += 1
            
            if not improved:
                # Phase 2: Try swaps
                swap_found = False
                for i in range(len(codewords)):
                    if swap_found:
                        break
                    c = codewords[i]
                    
                    # Find words uniquely covered by c
                    unique_mask = (cover_counts == 1)
                    unique_in_ball = unique_mask[balls[c]]
                    unique_words = balls[c][unique_in_ball]
                    
                    if len(unique_words) == 0:
                        continue
                    
                    # Find candidates b that cover all unique_words
                    candidates = set(balls[unique_words[0]].tolist())
                    for w in unique_words[1:]:
                        if not candidates:
                            break
                        candidates &= set(balls[w].tolist())
                    
                    if not candidates:
                        continue
                    
                    current_set = set(codewords)
                    candidates -= current_set
                    if not candidates:
                        continue
                    
                    candidate_list = list(candidates)
                    random.shuffle(candidate_list)
                    
                    for b in candidate_list:
                        cover_counts[balls[c]] -= 1
                        cover_counts[balls[b]] += 1
                        
                        if not np.any(cover_counts == 0):
                            # Valid swap. Check if it enables a removal.
                            removal_found = False
                            for j in range(len(codewords)):
                                if j == i:
                                    continue
                                c_j = codewords[j]
                                cover_counts[balls[c_j]] -= 1
                                if not np.any(cover_counts == 0):
                                    removal_found = True
                                    cover_counts[balls[c_j]] += 1
                                    break
                                else:
                                    cover_counts[balls[c_j]] += 1
                            
                            if removal_found:
                                codewords[i] = b
                                improved = True
                                swap_found = True
                                break
                            else:
                                cover_counts[balls[b]] -= 1
                                cover_counts[balls[c]] += 1
                        else:
                            cover_counts[balls[b]] -= 1
                            cover_counts[balls[c]] += 1
        
        return codewords
    
    # SA polish with vectorized cleanup
    def sa_polish(codewords, time_limit):
        """Simulated annealing polish."""
        if not codewords:
            return codewords
        
        cover_counts = np.zeros(total, dtype=np.int32)
        for c in codewords:
            cover_counts[balls[c]] += 1
        
        n_cw = len(codewords)
        best_size = n_cw
        best_codewords = codewords[:]
        best_cover = cover_counts.copy()
        
        T0 = 2.0
        T_min = 0.01
        step = 0
        max_steps = 300000
        
        while step < max_steps and time.time() < time_limit:
            T = T0 * (T_min / T0) ** (step / max_steps)
            n_cw = len(codewords)
            
            if n_cw == 0:
                break
                
            i = np.random.randint(n_cw)
            c = codewords[i]
            
            move_type = np.random.choice(3, p=[0.5, 0.35, 0.15])
            
            if move_type == 0:
                cover_counts[balls[c]] -= 1
                uncovered = (cover_counts == 0).sum()
                
                if uncovered == 0:
                    codewords.pop(i)
                    n_cw -= 1
                    if n_cw < best_size:
                        best_size = n_cw
                        best_codewords = codewords[:]
                        best_cover = cover_counts.copy()
                else:
                    accept_prob = np.exp(-uncovered / T)
                    if np.random.random() < accept_prob:
                        codewords.pop(i)
                    else:
                        cover_counts[balls[c]] += 1
            
            elif move_type == 1:
                b = np.random.randint(total)
                if b in codewords:
                    step += 1
                    continue
                
                cover_counts[balls[c]] -= 1
                cover_counts[balls[b]] += 1
                uncovered = (cover_counts == 0).sum()
                
                if uncovered == 0:
                    codewords[i] = b
                else:
                    accept_prob = np.exp(-uncovered / T)
                    if np.random.random() < accept_prob:
                        codewords[i] = b
                    else:
                        cover_counts[balls[b]] -= 1
                        cover_counts[balls[c]] += 1
            
            else:
                b = np.random.randint(total)
                if b in codewords:
                    step += 1
                    continue
                
                codewords.append(b)
                n_cw += 1
                cover_counts[balls[b]] += 1
            
            step += 1
            
            # Periodic cleanup
            if step % 500 == 0:
                n_cw = len(codewords)
                j = 0
                while j < n_cw:
                    c_j = codewords[j]
                    cover_counts[balls[c_j]] -= 1
                    if (cover_counts == 0).sum() == 0:
                        codewords.pop(j)
                        n_cw -= 1
                        if n_cw < best_size:
                            best_size = n_cw
                            best_codewords = codewords[:]
                            best_cover = cover_counts.copy()
                    else:
                        cover_counts[balls[c_j]] += 1
                        j += 1
        
        return best_codewords, best_size, best_cover
    
    best_codewords = None
    best_size = total
    
    # Run multiple MIS restarts
    seed = 0
    while time.time() < deadline - 20:
        mis_set = greedy_mis(seed)
        
        if not mis_set:
            seed += 1
            continue
        
        # Verify coverage
        cover_counts = np.zeros(total, dtype=np.int32)
        for c in mis_set:
            cover_counts[balls[c]] += 1
        
        if np.any(cover_counts == 0):
            # Not a valid cover, try to fix by adding missing coverage
            # Find uncovered words and add codewords to cover them
            uncovered_idx = np.where(cover_counts == 0)[0]
            for w_idx in uncovered_idx:
                # Find a codeword that covers this word
                # Any word in balls[w_idx] works
                # Pick one not already in the set
                candidates = balls[w_idx]
                added = False
                for cand in candidates:
                    if cand not in set(mis_set):
                        mis_set.append(cand)
                        cover_counts[balls[cand]] += 1
                        added = True
                        break
                if not added:
                    # All candidates are already in the set, which shouldn't happen
                    pass
        
        # Clean up redundant codewords
        mis_set = verify_and_cleanup(mis_set)
        
        if len(mis_set) < best_size:
            best_size = len(mis_set)
            best_codewords = mis_set[:]
        
        # Polish with local search
        if time.time() < deadline - 10:
            polished = local_search_polish(best_codewords[:], deadline - 5)
            if len(polished) < best_size:
                best_size = len(polished)
                best_codewords = polished[:]
        
        # SA polish if time allows
        if time.time() < deadline - 8 and best_codewords:
            sa_result, sa_size, _ = sa_polish(best_codewords[:], deadline - 3)
            if sa_size < best_size:
                best_size = sa_size
                best_codewords = sa_result[:]
        
        seed += 1
        
        if best_size <= 486:
            break
    
    if best_codewords is None:
        return []
        
    return [list(all_words[c]) for c in best_codewords]
