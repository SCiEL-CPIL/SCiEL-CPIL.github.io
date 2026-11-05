import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 118.0

    total_words = q ** n
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbors as lists of ints
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = neigh

    # Precompute neighbor sets for fast intersection
    neighbor_sets = [set(neigh) for neigh in neighbors]

    def run_greedy():
        covered = np.zeros(total_words, dtype=bool)
        code_list = []
        count = np.full(total_words, 17, dtype=np.int32)
        
        while not covered.all() and time.time() < deadline - 60.0:
            c = int(np.argmax(count))
            if count[c] == 0:
                break
            code_list.append(c)
            for x in neighbors[c]:
                if not covered[x]:
                    covered[x] = True
                    for y in neighbors[x]:
                        count[y] -= 1
        return code_list

    def local_search(code, max_time):
        if not code:
            return code[:]
            
        # Build coverage counts
        cov_count = np.zeros(total_words, dtype=np.int32)
        code_set = set(code)
        for c in code:
            for x in neighbors[c]:
                cov_count[x] += 1
                
        current_code = code[:]
        current_size = len(current_code)
        best_code = current_code[:]
        best_size = current_size
        best_cov_count = cov_count.copy()
        
        random.seed(42)
        stuck_counter = 0
        end_time = time.time() + max_time
        
        # Precompute neighbor list indices for faster access
        # We'll use the existing 'neighbors' list
        
        while time.time() < end_time:
            if current_size == 0:
                break
                
            # Strategy: Prioritize removal of codewords that cover the most "unique" points (cov_count == 1)
            # This is the "dual packing" idea: remove codewords that are most critical to coverage
            # but also try to find swaps that allow removal.
            
            # First, try to find a removable codeword (all neighbors have cov_count > 1)
            # We'll scan for the one with the most "redundancy" (lowest max cov_count among its neighbors)
            # to maximize future removal opportunities.
            
            best_removal_idx = -1
            best_removal_score = -1
            
            # Sample a subset of codewords to avoid O(n^2) scanning every iteration
            # Use random sampling for speed, but also check a few fixed positions
            sample_size = min(current_size, 50)
            sampled_indices = set()
            for _ in range(sample_size):
                idx = random.randint(0, current_size - 1)
                sampled_indices.add(idx)
            
            for idx in sampled_indices:
                c = current_code[idx]
                removable = True
                min_cov = 999
                for x in neighbors[c]:
                    if cov_count[x] <= 1:
                        removable = False
                        break
                    if cov_count[x] < min_cov:
                        min_cov = cov_count[x]
                
                if removable:
                    # Score: prefer removing codewords that free up the most "tight" coverage
                    # Lower min_cov means the neighbors are more tightly covered, so removing
                    # this codeword doesn't leave them too exposed.
                    score = -min_cov  # Lower min_cov is better
                    if score > best_removal_score:
                        best_removal_score = score
                        best_removal_idx = idx
            
            if best_removal_idx >= 0:
                c = current_code[best_removal_idx]
                for x in neighbors[c]:
                    cov_count[x] -= 1
                current_code.pop(best_removal_idx)
                current_size -= 1
                stuck_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
                    best_cov_count = cov_count.copy()
                continue
            
            # If no removable codeword found in sample, try swaps
            # Pick a random codeword and try to swap it with another that covers its unique points
            idx = random.randint(0, current_size - 1)
            c = current_code[idx]
            
            # Find risky points: neighbors of c with cov_count == 1
            risky_points = []
            for x in neighbors[c]:
                if cov_count[x] == 1:
                    risky_points.append(x)
            
            if not risky_points:
                # Should be removable, but maybe logic error or race condition? 
                # Actually if no risky points, it means all neighbors have count > 1, 
                # which is the removable condition. So this shouldn't happen if removable was False.
                # Unless removable check failed due to some other reason? No, it's the same.
                # So if removable is False, risky_points is non-empty.
                pass
            else:
                # Find candidates that cover all risky points
                # Candidate must be in intersection of neighbors[r] for all r in risky_points
                candidates = neighbor_sets[risky_points[0]].copy()
                for r in risky_points[1:]:
                    candidates &= neighbor_sets[r]
                    if not candidates:
                        break
                
                if candidates:
                    # Exclude current code words
                    candidates.difference_update(current_code)
                    if candidates:
                        # Choose candidate that maximizes "packing" benefit
                        # i.e., covers the most points with cov_count == 1 (other than c's risky points)
                        best_c2 = None
                        best_benefit = -1
                        
                        # Sample candidates if too many
                        cand_list = list(candidates)
                        if len(cand_list) > 20:
                            random.shuffle(cand_list)
                            cand_list = cand_list[:20]
                        
                        for c2 in cand_list:
                            # Calculate benefit: number of points with cov_count == 1 that c2 covers
                            # (excluding those already covered by c's risky points)
                            benefit = 0
                            for x in neighbors[c2]:
                                if cov_count[x] == 1 and x not in risky_points:
                                    benefit += 1
                            
                            if benefit > best_benefit:
                                best_benefit = benefit
                                best_c2 = c2
                        
                        if best_c2 is not None:
                            c2 = best_c2
                            
                            # Perform swap
                            # Remove c
                            for x in neighbors[c]:
                                cov_count[x] -= 1
                            # Add c2
                            for x in neighbors[c2]:
                                cov_count[x] += 1
                            
                            current_code[idx] = c2
                            stuck_counter = 0
                            # Size doesn't change, so no best update here unless we allow size-increasing swaps?
                            # No, we only care about minimizing size. Swaps are to enable future removals.
                            continue
            
            # 3. Two-swap: remove 2, add 1
            if stuck_counter > 10 and current_size >= 2:
                i1 = random.randint(0, current_size - 1)
                i2 = random.randint(0, current_size - 1)
                if i2 == i1:
                    i2 = (i2 + 1) % current_size
                
                c1 = current_code[i1]
                c2 = current_code[i2]
                
                # Find points that become uncovered if we remove c1 and c2
                # These are points with cov_count == 1 that are neighbors of c1 or c2
                risky = set()
                for x in neighbors[c1]:
                    if cov_count[x] == 1:
                        risky.add(x)
                for x in neighbors[c2]:
                    if cov_count[x] == 1:
                        risky.add(x)
                
                if not risky:
                    # Just remove one of them?
                    # If no risky points, we could have removed c1 in step 1.
                    # Maybe the random pick in step 1 failed?
                    # Let's just remove c1.
                    for x in neighbors[c1]:
                        cov_count[x] -= 1
                    if i1 < i2:
                        current_code.pop(i1)
                    else:
                        current_code.pop(i2)
                    current_size -= 1
                    stuck_counter = 0
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                        best_cov_count = cov_count.copy()
                    continue
                else:
                    # Find candidate c3 that covers all risky points
                    risky_list = list(risky)
                    candidates = neighbor_sets[risky_list[0]].copy()
                    for r in risky_list[1:]:
                        candidates &= neighbor_sets[r]
                        if not candidates:
                            break
                    
                    if candidates:
                        candidates.difference_update(current_code)
                        if candidates:
                            # Choose candidate that maximizes packing benefit
                            cand_list = list(candidates)
                            if len(cand_list) > 20:
                                random.shuffle(cand_list)
                                cand_list = cand_list[:20]
                            
                            best_c3 = None
                            best_benefit = -1
                            
                            for c3 in cand_list:
                                benefit = 0
                                for x in neighbors[c3]:
                                    if cov_count[x] == 1 and x not in risky:
                                        benefit += 1
                                
                                if benefit > best_benefit:
                                    best_benefit = benefit
                                    best_c3 = c3
                            
                            if best_c3 is not None:
                                c3 = best_c3
                                
                                # Remove c1 and c2, add c3
                                for x in neighbors[c1]:
                                    cov_count[x] -= 1
                                for x in neighbors[c2]:
                                    cov_count[x] -= 1
                                for x in neighbors[c3]:
                                    cov_count[x] += 1
                                
                                # Update list
                                if i1 < i2:
                                    current_code.pop(i2)
                                    current_code.pop(i1)
                                else:
                                    current_code.pop(i1)
                                    current_code.pop(i2)
                                current_code.append(c3)
                                current_size -= 1
                                stuck_counter = 0
                                if current_size < best_size:
                                    best_size = current_size
                                    best_code = current_code[:]
                                    best_cov_count = cov_count.copy()
                                continue
            
            stuck_counter += 1
        
        return best_code

    # Phase 1: Greedy
    initial_code = run_greedy()
    
    # Phase 2: Local Search
    time_left = deadline - time.time() - 5.0
    if time_left > 1.0:
        final_code = local_search(initial_code, time_left)
    else:
        final_code = initial_code
    
    # Final verification and cleanup
    # Rebuild cov_count for final_code to ensure validity
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for c in final_code:
        for x in neighbors[c]:
            final_cov_count[x] += 1
            
    # If invalid (shouldn't happen with our logic, but just in case), try to fix or return best
    if not np.all(final_cov_count > 0):
        # Fallback to initial greedy if it was valid
        init_cov = np.zeros(total_words, dtype=np.int32)
        for c in initial_code:
            for x in neighbors[c]:
                init_cov[x] += 1
        if np.all(init_cov > 0):
            final_code = initial_code
        else:
            # Last resort: full space (expensive but valid)
            final_code = list(range(total_words))

    return [list(digits_of[i]) for i in final_code]
