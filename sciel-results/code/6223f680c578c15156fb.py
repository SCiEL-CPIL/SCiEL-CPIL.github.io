import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n

    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists as numpy arrays
    # For q=3, n=8, radius=1: 1 + 8*(3-1) = 17 neighbors
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
        neighbors[i] = np.array(neigh, dtype=np.int32)
        
    # Also precompute a set of neighbors for faster membership checks? 
    # Or just use numpy boolean masks.
    
    # --- Phase 1: Fast Greedy Initialization ---
    # We use a greedy set cover approach with potential tracking.
    
    def run_greedy_init():
        cov_count = np.zeros(total_words, dtype=np.int32)
        is_code = np.zeros(total_words, dtype=bool)
        code_list = []
        
        # Potential: number of uncovered neighbors
        potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
        uncovered_remaining = total_words
        
        while uncovered_remaining > 0 and time.time() < deadline - 60.0:
            c = np.argmax(potential)
            if potential[c] == 0:
                break
                
            code_list.append(c)
            is_code[c] = True
            potential[c] = 0 # Prevent re-picking
            
            neigh_c = neighbors[c]
            for x in neigh_c:
                if cov_count[x] == 0:
                    cov_count[x] = 1
                    uncovered_remaining -= 1
                    # Update potentials of neighbors of x
                    # Only if x was previously uncovered, we reduce potential of all who cover x
                    potential[neighbors[x]] -= 1
                else:
                    cov_count[x] += 1
                    
        return code_list, cov_count, is_code

    code, cov_count, is_code = run_greedy_init()
    
    best_code = code[:]
    best_size = len(code)
    
    # --- Phase 2: Local Search ---
    
    def try_reduce(code, cov_count, is_code):
        """
        Attempts to remove any redundant codeword.
        Returns True if a removal occurred, False otherwise.
        """
        if not code:
            return False
            
        # Shuffle indices to avoid bias towards early indices
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for idx in indices:
            c = code[idx]
            # Check if all neighbors are covered by more than 1 codeword
            if np.all(cov_count[neighbors[c]] > 1):
                # Remove c
                cov_count[neighbors[c]] -= 1
                is_code[c] = False
                code.pop(idx)
                return True
        return False

    def find_swap_candidate(c, risky_points, code_set, cov_count):
        """
        Finds a candidate c' not in code that covers all risky_points.
        Heuristic: Prefer candidates that cover many other points (high degree of overlap with existing coverage)
        to facilitate future deletions.
        """
        if not risky_points:
            return None
            
        # Intersection of neighbors of all risky points
        # Start with neighbors of first risky point
        candidates_set = set(neighbors[risky_points[0]].tolist())
        
        for r in risky_points[1:]:
            candidates_set &= set(neighbors[r].tolist())
            if not candidates_set:
                return None
        
        # Filter out current code words
        valid_candidates = [c2 for c2 in candidates_set if c2 not in code_set]
        
        if not valid_candidates:
            return None
            
        # Heuristic: Pick the candidate that maximizes the sum of cov_count of its neighbors
        # This suggests it overlaps well with the existing code, making it easier to remove later.
        # However, calculating this for all candidates is expensive.
        # Alternative: Pick random.
        # Better: Pick one that is "central" or has high potential.
        
        # Let's just pick random for now, as the reduction step will filter out bad moves eventually.
        # But we can add a small bias: prefer candidates that are not too "isolated".
        
        return random.choice(valid_candidates)

    random.seed(42)
    stuck_counter = 0
    kick_counter = 0
    
    while time.time() < deadline - 10.0:
        current_size = len(code)
        
        # 1. Try to reduce
        if try_reduce(code, cov_count, is_code):
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
            stuck_counter = 0
            kick_counter = 0
            continue
            
        # 2. If no reduction, try a swap
        if current_size == 0:
            break
            
        # Find a codeword to swap
        # Prioritize codewords with few unique points (risky points)
        # Calculate risky points for all? Too expensive.
        # Pick a random codeword, check its risky points.
        
        idx = random.randint(0, current_size - 1)
        c = code[idx]
        
        # Find risky points: neighbors of c where count is 1
        neigh_c = neighbors[c]
        risky_mask = cov_count[neigh_c] == 1
        risky_points = neigh_c[risky_mask]
        
        if len(risky_points) == 0:
            # This should have been caught by try_reduce
            # Force remove if logic error
            cov_count[neighbors[c]] -= 1
            is_code[c] = False
            code.pop(idx)
            continue
            
        # If too many risky points, swap is unlikely to be helpful or valid
        if len(risky_points) > 4:
            stuck_counter += 1
            continue
            
        code_set = set(code)
        c_prime = find_swap_candidate(c, risky_points, code_set, cov_count)
        
        if c_prime is not None:
            # Perform swap
            # 1. Remove c
            cov_count[neighbors[c]] -= 1
            is_code[c] = False
            code[idx] = c_prime
            
            # 2. Add c_prime
            cov_count[neighbors[c_prime]] += 1
            is_code[c_prime] = True
            
            stuck_counter = 0
        else:
            stuck_counter += 1
            
        # 3. If stuck for a while, perform a "kick"
        # Remove a few random codewords and re-greedy the gaps
        if stuck_counter > 50:
            # Kick: remove 5% of code or 5 words, whichever is smaller
            kick_size = max(5, current_size // 20)
            kick_indices = random.sample(range(len(code)), kick_size)
            for k_idx in sorted(kick_indices, reverse=True):
                k_c = code[k_idx]
                cov_count[neighbors[k_c]] -= 1
                is_code[k_c] = False
                code.pop(k_idx)
                
            # Re-greedy fill the gaps
            # We need to restart the greedy logic on the remaining uncovered points
            # But our greedy function resets everything. 
            # Instead, we can just run a local greedy step:
            # While there are uncovered points, pick best candidate.
            
            # Check for uncovered points
            # We need to identify uncovered points.
            # cov_count == 0 means uncovered.
            
            # Potential array needs to be updated. 
            # Let's just re-run a simplified greedy loop that only adds words.
            
            # Update potential for remaining code?
            # Actually, it's safer to just re-run the full greedy init if we kick hard,
            # but that loses the local search progress.
            # Let's do a "soft" re-greedy:
            
            # Calculate potential for all non-code words based on current coverage
            # potential[x] = number of neighbors of x that are uncovered (cov_count == 0)
            
            # This is O(N * 17). Acceptable.
            potential = np.zeros(total_words, dtype=np.int32)
            for x in range(total_words):
                if not is_code[x]:
                    # Count uncovered neighbors
                    potential[x] = np.sum(cov_count[neighbors[x]] == 0)
            
            # Greedy fill
            while time.time() < deadline - 10.0:
                # Find uncovered points
                if not np.any(cov_count == 0):
                    break
                    
                c_new = np.argmax(potential)
                if potential[c_new] == 0:
                    break
                    
                code.append(c_new)
                is_code[c_new] = True
                potential[c_new] = 0
                
                neigh_new = neighbors[c_new]
                for x in neigh_new:
                    if cov_count[x] == 0:
                        cov_count[x] = 1
                        potential[neighbors[x]] -= 1
                    else:
                        cov_count[x] += 1
                        
            stuck_counter = 0
            kick_counter += 1
            # After kick, try to reduce again immediately
            if try_reduce(code, cov_count, is_code):
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]

    # --- Phase 3: Final Polish ---
    # Run reduction on the best code found
    final_code = best_code[:]
    final_counts = np.zeros(total_words, dtype=np.int32)
    for c in final_code:
        final_counts[neighbors[c]] += 1
        
    # Simple reduction pass
    changed = True
    while changed:
        changed = False
        next_code = []
        for c in final_code:
            if np.all(final_counts[neighbors[c]] > 1):
                final_counts[neighbors[c]] -= 1
                changed = True
            else:
                next_code.append(c)
        final_code = next_code
        
    # Verify validity
    if not np.all(final_counts > 0):
        # Fallback to best_code if polish broke it
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
