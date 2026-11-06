import numpy as np
import itertools
import time
import random
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Generate all words and their indices
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors for radius 1
    # Each word has 1 (itself) + n*(q-1) neighbors
    num_neighbors = 1 + n * (q - 1)
    
    neighbors = []
    neighbor_array = np.zeros((N, num_neighbors), dtype=np.int32)
    
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbors.append(neigh)
        neighbor_array[i, :] = neigh
    
    # 3. Precompute reverse neighbors: for each point, which words cover it
    # covered_by[i] is the list of codewords that cover word i
    covered_by = [[] for _ in range(N)]
    for i in range(N):
        for nb in neighbors[i]:
            covered_by[nb].append(i)
            
    # Convert to numpy arrays for vectorized operations where possible
    # But lists are often faster for small n in Python loops
    # Let's keep lists for flexibility in local search, but maybe arrays for bulk checks
    
    # 4. Precompute neighbor sets for fast membership testing
    # This is memory heavy (N * num_neighbors), but N=2187, num_neighbors=15, so ~32k entries. Fine.
    neighbor_sets = [set(neigh) for neigh in neighbors]

    # 5. Helper functions
    
    def get_counts(code_list):
        """Calculate coverage counts for all points given a code."""
        if not code_list:
            return np.zeros(N, dtype=np.int32)
        counts = np.zeros(N, dtype=np.int32)
        for c in code_list:
            for nb in neighbors[c]:
                counts[nb] += 1
        return counts
    
    def is_valid(code_list):
        """Check if code covers all points."""
        if not code_list:
            return N == 0
        counts = get_counts(code_list)
        return np.min(counts) >= 1

    def prune_code(code_list):
        """Remove redundant codewords (those whose removal doesn't uncover any point)."""
        if not code_list:
            return []
        counts = get_counts(code_list)
        code_list = list(code_list)
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow popping
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                # Check if all points covered by c are covered by at least one other codeword
                if all(counts[nb] > 1 for nb in neighbors[c]):
                    code_list.pop(i)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    changed = True
        return code_list
    
    def greedy_set_cover(time_limit, use_packing_penalty=True):
        """
        Greedy set cover.
        If use_packing_penalty is True, we modify the gain calculation to prefer
        candidates that are "far" from existing codewords, or rather, that cover
        unique areas not well-covered by others. 
        Actually, the prompt suggests: "penalizing candidates that have high Hamming distance overlap with already selected codewords".
        Wait, "high Hamming distance overlap" is a bit contradictory. 
        Usually, packing means maximizing distance. 
        If we want to maximize average pairwise distance, we should prefer candidates that are far from existing ones.
        However, in set cover, we pick based on marginal gain.
        A common heuristic for packing is to subtract a penalty proportional to the number of neighbors shared with existing codewords?
        Or simply: gain = marginal_coverage - lambda * (overlap_with_existing_code).
        
        Let's implement: 
        raw_gain = number of uncovered points covered by candidate.
        penalty = sum over existing codewords c' of |ball(c) intersect ball(c')| / ball_size?
        Or simpler: count how many of the candidate's neighbors are already covered by >= 1 codeword?
        No, that's just redundancy.
        
        Let's try the specific instruction: "penalizing candidates that have high Hamming distance overlap with already selected codewords".
        This phrasing is tricky. "High Hamming distance" usually means "far". "Overlap" usually means "close".
        If it means "penalize candidates that are close to selected ones", that promotes packing.
        Let's assume it means: Penalize candidates that are close to selected codewords (i.e., have many common neighbors).
        
        Implementation:
        For each candidate c:
        gain = count of uncovered points in ball(c)
        penalty = 0
        for c' in code:
            # Count intersection of ball(c) and ball(c')
            # Since balls are small (15), this is fast.
            # We can precompute intersection sizes? N^2 * 15*15 is too big.
            # But we only do this for top candidates in each step?
            # Greedy step: we compute gains for all N candidates.
            # If we add penalty, we need to compute penalty for all N candidates against all |code| selected.
            # |code| grows. N=2187.
            # Step 1: |code|=0, penalty=0.
            # Step 10: |code|=10. 2187 * 10 * (15*15/4?) ~ 100k ops. Fast.
            # Step 100: 2187 * 100 * ... ~ 1M ops. Still fast in Python? Maybe borderline.
            # Let's optimize: 
            # Instead of full intersection, note that if c and c' are close, they share neighbors.
            # We can approximate penalty by the number of neighbors of c that are already covered by code.
            # This is basically "redundancy".
            # Let's stick to the vectorized gain first, and only apply penalty to a small set of top candidates to save time.
        """
        code = []
        covered = np.zeros(N, dtype=bool)
        remaining = N
        
        # Precompute neighbor arrays as lists of lists for fast iteration in penalty calc
        # neighbors is already list of lists.
        
        while remaining > 0 and time.time() < deadline - time_limit:
            # Vectorized gain computation
            gains = (~covered)[neighbor_array].sum(axis=1)
            
            if not use_packing_penalty:
                max_gain = int(np.max(gains))
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                # Find top K candidates by raw gain
                # If max gain is 0, break
                max_gain = int(np.max(gains))
                if max_gain == 0:
                    break
                
                # Get candidates with max gain
                top_candidates = np.where(gains == max_gain)[0]
                
                # If there are too many, or if we have a small code, just pick randomly
                # If code is small, penalty is small.
                # Let's pick top 10 candidates by gain (ties broken randomly)
                # Actually, gains are integers.
                # Let's just take all with max_gain.
                # If len(top_candidates) > 20, sample 20.
                if len(top_candidates) > 20:
                    top_candidates = top_candidates[random.sample(range(len(top_candidates)), 20)]
                
                best_idx = -1
                best_score = -1000000
                
                for c in top_candidates:
                    c = int(c)
                    score = max_gain
                    if code:
                        # Calculate penalty
                        # Penalty = number of neighbors of c that are already in the union of balls of code
                        # This is equivalent to: for each nb in ball(c), if covered[nb] is True, add 1?
                        # No, covered[nb] is True if ANY codeword covers it.
                        # This penalizes covering already covered points.
                        # This is a standard "redundancy" penalty.
                        # Is it "packing"? Not exactly, but it prevents clustering.
                        penalty = 0
                        for nb in neighbors[c]:
                            if covered[nb]:
                                penalty += 1
                        # Normalize penalty?
                        # Score = gain - 0.1 * penalty
                        score = max_gain - 0.1 * penalty
                    
                    if score > best_score:
                        best_score = score
                        best_idx = c
                
                if best_idx == -1:
                    best_idx = int(top_candidates[0])

            code.append(int(best_idx))
            for nb in neighbors[best_idx]:
                if not covered[nb]:
                    covered[nb] = True
                    remaining -= 1
        
        return code
    
    def local_search(code_list, max_time=5.0):
        """Local search with 1-exchange and 2-for-1 moves."""
        if not code_list:
            return []
        
        code_set = set(code_list)
        counts = get_counts(code_list)
        improved = True
        ls_start = time.time()
        max_time = min(max_time, deadline - 5 - time.time())
        
        if max_time <= 0:
            return prune_code(code_list)
        
        while improved and time.time() - ls_start < max_time and time.time() < deadline - 5:
            improved = False
            code_list = prune_code(code_list)
            code_set = set(code_list)
            counts = get_counts(code_list)
            
            # Try 1-for-1 swaps
            code_order = list(code_list)
            random.shuffle(code_order)
            
            for c in code_order:
                if time.time() >= deadline - 5:
                    break
                if c not in code_set:
                    continue
                
                # Find points only covered by c
                crit = [nb for nb in neighbors[c] if counts[nb] == 1]
                
                if not crit:
                    # c is redundant
                    code_list.remove(c)
                    code_set.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    improved = True
                    break
                
                # Try to find a replacement that covers all critical points
                p = crit[0]
                candidates = [x for x in neighbors[p] if x not in code_set]
                
                best_w = -1
                best_score = -1
                
                for w in candidates:
                    if not all(k in neighbor_sets[w] for k in crit):
                        continue
                    # Score: how many points with count==2 does w cover (creating redundancy)
                    # Higher redundancy is better as it allows future removals
                    score = sum(1 for nb in neighbors[w] if counts[nb] == 2)
                    if score > best_score:
                        best_score = score
                        best_w = w
                
                if best_w != -1:
                    code_list.remove(c)
                    code_set.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    code_list.append(best_w)
                    code_set.add(best_w)
                    for nb in neighbors[best_w]:
                        counts[nb] += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-for-1 exchanges
            if len(code_list) >= 2 and time.time() < deadline - 5:
                code_copy = list(code_list)
                random.shuffle(code_copy)
                
                found = False
                for i in range(len(code_copy)):
                    if found:
                        break
                    c1 = code_copy[i]
                    if c1 not in code_set:
                        continue
                    crit1 = [nb for nb in neighbors[c1] if counts[nb] == 1]
                    
                    for j in range(i + 1, len(code_copy)):
                        c2 = code_copy[j]
                        if c2 not in code_set:
                            continue
                        crit2 = [nb for nb in neighbors[c2] if counts[nb] == 1]
                        
                        all_crit = list(set(crit1 + crit2))
                        
                        if not all_crit:
                            # Both redundant
                            code_list.remove(c1)
                            code_set.remove(c1)
                            for nb in neighbors[c1]:
                                counts[nb] -= 1
                            code_list.remove(c2)
                            code_set.remove(c2)
                            for nb in neighbors[c2]:
                                counts[nb] -= 1
                            improved = True
                            found = True
                            break
                        
                        p = all_crit[0]
                        candidates = [x for x in neighbors[p] if x not in code_set and x != c1 and x != c2]
                        
                        best_w = -1
                        best_score = -1
                        
                        for w in candidates:
                            if not all(k in neighbor_sets[w] for k in all_crit):
                                continue
                            score = sum(1 for nb in neighbors[w] if counts[nb] == 2)
                            if score > best_score:
                                best_score = score
                                best_w = w
                        
                        if best_w != -1:
                            code_list.remove(c1)
                            code_set.remove(c1)
                            for nb in neighbors[c1]:
                                counts[nb] -= 1
                            code_list.remove(c2)
                            code_set.remove(c2)
                            for nb in neighbors[c2]:
                                counts[nb] -= 1
                            code_list.append(best_w)
                            code_set.add(best_w)
                            for nb in neighbors[best_w]:
                                counts[nb] += 1
                            improved = True
                            found = True
                            break
                        
                        if time.time() >= deadline - 5:
                            break
        
        return code_list
    
    def perturb(code_list, strength=0.1):
        """Perturb by removing random codewords and repairing."""
        if not code_list:
            return []
        
        code_set = set(code_list)
        code_list = list(code_list)
        num_remove = max(1, int(len(code_list) * strength))
        
        to_remove = random.sample(code_list, min(num_remove, len(code_list)))
        for c in to_remove:
            code_list.remove(c)
            code_set.discard(c)
        
        # Repair
        counts = get_counts(code_list)
        uncovered = counts < 1
        uncovered_count = int(np.sum(uncovered))
        
        while uncovered_count > 0 and time.time() < deadline - 5:
            gains = uncovered[neighbor_array].sum(axis=1)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            code_list.append(best_idx)
            code_set.add(best_idx)
            for nb in neighbors[best_idx]:
                if counts[nb] < 1:
                    uncovered_count -= 1
                counts[nb] += 1
            uncovered = counts < 1
        
        return code_list
    
    def run_ils(seed, use_packing_penalty=True):
        """Iterated Local Search."""
        random.seed(seed)
        
        # Initial greedy solution
        code = greedy_set_cover(time_limit=20.0, use_packing_penalty=use_packing_penalty)
        code = prune_code(code)
        
        best_local = list(code)
        best_size = len(best_local)
        
        # ILS loop
        ils_start = time.time()
        max_ils_time = min(8.0, deadline - 15 - time.time())
        
        while time.time() - ils_start < max_ils_time and time.time() < deadline - 10:
            # Perturb
            strength = random.uniform(0.05, 0.15)
            perturbed = perturb(best_local, strength)
            
            # Local search
            ls_time = min(3.0, max(0.5, (deadline - 10 - time.time()) * 0.5))
            optimized = local_search(perturbed, max_time=ls_time)
            optimized = prune_code(optimized)
            
            if len(optimized) < best_size:
                best_local = optimized
                best_size = len(optimized)
            
            # If no improvement for a while, try stronger perturbation
            if random.random() < 0.3:
                strength = random.uniform(0.1, 0.25)
                perturbed = perturb(best_local, strength)
                ls_time = min(2.0, max(0.5, (deadline - 10 - time.time()) * 0.4))
                optimized = local_search(perturbed, max_time=ls_time)
                optimized = prune_code(optimized)
                if len(optimized) < best_size:
                    best_local = optimized
                    best_size = len(optimized)
        
        # Final validity check
        counts = get_counts(best_local)
        if np.min(counts) < 1:
            return None
        
        return best_local
    
    # Main loop: multiple restarts
    best_code = None
    best_size = float('inf')
    
    num_restarts = 0
    while time.time() < deadline - 15:
        seed = num_restarts * 7919 + 13
        # Alternate between with and without packing penalty to explore different landscapes
        use_penalty = (num_restarts % 2 == 0)
        current_code = run_ils(seed, use_packing_penalty=use_penalty)
        num_restarts += 1
        
        if current_code is not None:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
        
        # Stop condition: if we've done enough restarts and time is running out
        if num_restarts >= 6 and time.time() > deadline - 25:
            break
        if num_restarts >= 10:
            break
    
    if best_code is None:
        # Fallback: greedy only
        code = greedy_set_cover(time_limit=10.0, use_packing_penalty=False)
        best_code = prune_code(code)
        if not best_code:
            return []
    
    best_code = prune_code(best_code)
    
    # Final validity check
    if best_code:
        counts = get_counts(best_code)
        if np.min(counts) < 1:
            # Emergency repair
            uncovered = counts < 1
            while np.any(uncovered) and time.time() < deadline - 1:
                gains = uncovered[neighbor_array].sum(axis=1)
                max_gain = int(np.max(gains))
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = int(candidates[0])
                best_code.append(best_idx)
                for nb in neighbors[best_idx]:
                    counts[nb] += 1
                uncovered = counts < 1
            best_code = prune_code(best_code)
    
    return [list(words[i]) for i in best_code]
