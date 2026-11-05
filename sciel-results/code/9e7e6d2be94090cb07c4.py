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

    # Phase 1: Greedy
    covered = np.zeros(total_words, dtype=bool)
    code_list = []
    
    # Precompute uncovered counts for faster greedy?
    # Standard greedy: pick word covering most uncovered points.
    # We can maintain an array 'uncovered_count' for each word.
    # Initially all 1. When we pick a word, we decrement counts for all its neighbors.
    # This is O(N * |N|) per step, which is fine for N=6561.
    
    # Actually, the matrix-vector product approach in the original was O(N^2) per step if done naively, 
    # but with numpy it's fast. However, maintaining a count array is more efficient for sequential updates.
    
    # Let's stick to a simpler greedy that is fast enough:
    # 1. Start with all points uncovered.
    # 2. Pick word with max new coverage.
    # 3. Mark covered.
    # 4. Repeat.
    
    # Optimization: Use a heap or just argmax on a count array.
    # Count of uncovered points covered by word i.
    # Initially, count[i] = len(neighbors[i]) = 17.
    # When we pick word c, for every neighbor x of c, if x was uncovered, 
    # we need to decrement the count of all words that cover x.
    # This is complex. 
    # Alternative: Just use the numpy matrix product for the first few steps or just simple greedy.
    # Given N=6561, simple greedy with numpy argmax on a temporary vector is fine.
    
    # Let's use the count array approach for O(1) update per neighbor if we precompute reverse neighbors.
    # reverse_neighbors[x] = list of words that cover x.
    # This is the same as neighbors[x] in a symmetric graph? 
    # Yes, Hamming distance is symmetric. If y is in neighbors[x], then x is in neighbors[y].
    # So neighbors is symmetric.
    
    # Let's use the count array method.
    # count[i] = number of currently uncovered points in neighbors[i].
    # Initially count[i] = 17 for all i.
    # When we select c:
    #   For each x in neighbors[c]:
    #     If not covered[x]:
    #       covered[x] = True
    #       For each y in neighbors[x]:
    #         count[y] -= 1
    #   This is O(17 * 17) = 289 operations per step.
    #   Total steps ~ 400. Total ops ~ 100,000. Very fast.
    
    count = np.full(total_words, 17, dtype=np.int32)
    while not covered.all() and time.time() < deadline - 40.0:
        c = int(np.argmax(count))
        if count[c] == 0:
            break
        code_list.append(c)
        for x in neighbors[c]:
            if not covered[x]:
                covered[x] = True
                for y in neighbors[x]:
                    count[y] -= 1

    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size

    # Precompute neighbor sets as Python sets for fast intersection in local search
    neighbor_sets = [set(neigh) for neigh in neighbors]
    
    # Initialize coverage counts
    current_cov_count = np.zeros(total_words, dtype=np.int32)
    for c in current_code:
        for x in neighbors[c]:
            current_cov_count[x] += 1

    random.seed(42)

    # Phase 2: Local Search with removal, swap, and two-swap moves
    stuck_counter = 0
    two_swap_counter = 0
    while time.time() < deadline - 5.0:
        if current_size == 0:
            break
        
        # Try removal
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]
        
        removable = True
        for x in neighbors[c]:
            if current_cov_count[x] <= 1:
                removable = False
                break
        
        if removable:
            for x in neighbors[c]:
                current_cov_count[x] -= 1
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            two_swap_counter = 0
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
        
        # Try swap if stuck
        if stuck_counter > 3 and time.time() < deadline - 10.0 and current_size > 0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            
            risky_points = []
            for x in neighbors[c1]:
                if current_cov_count[x] == 1:
                    risky_points.append(x)
            
            if not risky_points:
                for x in neighbors[c1]:
                    current_cov_count[x] -= 1
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                two_swap_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                candidate_set = neighbor_sets[risky_points[0]].copy()
                for x in risky_points[1:]:
                    candidate_set &= neighbor_sets[x]
                    if not candidate_set:
                        break
                
                if candidate_set:
                    candidate_set.discard(c1)
                    for c_code in current_code:
                        if c_code != c1:
                            candidate_set.discard(c_code)
                    
                    if candidate_set:
                        c2 = random.choice(list(candidate_set))
                        for x in neighbors[c1]:
                            current_cov_count[x] -= 1
                        for x in neighbors[c2]:
                            current_cov_count[x] += 1
                        current_code[c1_idx] = c2
                        stuck_counter = 0
                        two_swap_counter = 0
        
        # Try two-swap: remove two codewords, add one
        if stuck_counter > 15 and current_size >= 2 and time.time() < deadline - 15.0:
            i1 = random.randint(0, current_size - 1)
            i2 = random.randint(0, current_size - 1)
            if i2 == i1:
                i2 = (i2 + 1) % current_size
            
            c1 = current_code[i1]
            c2 = current_code[i2]
            
            # Find points that would become uncovered if we remove both c1 and c2
            # These are points with count == 1 that are neighbors of either c1 or c2
            risky = set()
            for x in neighbors[c1]:
                if current_cov_count[x] == 1:
                    risky.add(x)
            for x in neighbors[c2]:
                if current_cov_count[x] == 1:
                    risky.add(x)
            
            if risky:
                # Need a replacement that covers all risky points
                candidate_set = neighbor_sets[list(risky)[0]].copy()
                for x in list(risky)[1:]:
                    candidate_set &= neighbor_sets[x]
                    if not candidate_set:
                        break
                
                if candidate_set:
                    candidate_set.discard(c1)
                    candidate_set.discard(c2)
                    for c_code in current_code:
                        if c_code != c1 and c_code != c2:
                            candidate_set.discard(c_code)
                    
                    if candidate_set:
                        c3 = random.choice(list(candidate_set))
                        
                        # Remove c1 and c2
                        for x in neighbors[c1]:
                            current_cov_count[x] -= 1
                        for x in neighbors[c2]:
                            current_cov_count[x] -= 1
                        # Add c3
                        for x in neighbors[c3]:
                            current_cov_count[x] += 1
                        
                        # Remove c2 first (higher index), then c1
                        if i1 < i2:
                            current_code.pop(i2)
                            current_code.pop(i1)
                        else:
                            current_code.pop(i1)
                            current_code.pop(i2)
                        current_code.append(c3)
                        current_size -= 1
                        stuck_counter = 0
                        two_swap_counter = 0
                        if current_size < best_size:
                            best_size = current_size
                            best_code = current_code[:]
            else:
                # No risky points, just remove one
                for x in neighbors[c1]:
                    current_cov_count[x] -= 1
                current_code.pop(i1)
                current_size -= 1
                stuck_counter = 0
                two_swap_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
        
        two_swap_counter += 1

    # Final polish: try to remove more, with multiple passes
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        for x in neighbors[idx]:
            final_cov_count[x] += 1
            
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            removable = True
            for x in neighbors[c]:
                if final_cov_count[x] <= 1:
                    removable = False
                    break
            if removable:
                for x in neighbors[c]:
                    final_cov_count[x] -= 1
                final_code.pop(i)
                improved = True
        
        # Also try two-swap in final polish
        if time.time() < deadline - 3.0 and len(final_code) >= 2:
            for i1 in range(len(final_code) - 1):
                if time.time() >= deadline - 3.0:
                    break
                for i2 in range(i1 + 1, len(final_code)):
                    c1 = final_code[i1]
                    c2 = final_code[i2]
                    
                    risky = set()
                    for x in neighbors[c1]:
                        if final_cov_count[x] == 1:
                            risky.add(x)
                    for x in neighbors[c2]:
                        if final_cov_count[x] == 1:
                            risky.add(x)
                    
                    if not risky:
                        # Just remove one
                        for x in neighbors[c1]:
                            final_cov_count[x] -= 1
                        final_code.pop(i1)
                        improved = True
                        break
                    elif len(risky) <= 3:
                        candidate_set = neighbor_sets[list(risky)[0]].copy()
                        for x in list(risky)[1:]:
                            candidate_set &= neighbor_sets[x]
                            if not candidate_set:
                                break
                        
                        if candidate_set:
                            candidate_set.discard(c1)
                            candidate_set.discard(c2)
                            for c_code in final_code:
                                if c_code != c1 and c_code != c2:
                                    candidate_set.discard(c_code)
                            
                            if candidate_set:
                                c3 = random.choice(list(candidate_set))
                                for x in neighbors[c1]:
                                    final_cov_count[x] -= 1
                                for x in neighbors[c2]:
                                    final_cov_count[x] -= 1
                                for x in neighbors[c3]:
                                    final_cov_count[x] += 1
                                final_code.pop(i2)
                                final_code.pop(i1)
                                final_code.append(c3)
                                improved = True
                                break
                if improved:
                    break
                
    # Verify validity
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
