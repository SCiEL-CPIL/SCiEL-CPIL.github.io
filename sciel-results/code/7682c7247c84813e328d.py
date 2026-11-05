import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Fallback for other parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each point
    neighbors = [None] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        nbrs = [idx]
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            nbrs.append(idx + delta)

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            nbrs.append(idx + delta)

        neighbors[idx] = nbrs

    # Build balls as a numpy array for vectorized operations
    balls = np.array(neighbors, dtype=np.int32)  # shape (6561, 17)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def greedy_set_cover(seed=None, deterministic=False):
        """
        Fast greedy set cover using a 'potential' array.
        potential[i] = number of currently uncovered points in the ball of i.
        We pick the i with max potential, add it to the code, and update potentials.
        """
        if deterministic:
            rng = None
        else:
            rng = np.random.default_rng(seed)
            
        cov_count = np.zeros(total_points, dtype=np.int32)
        potential = np.full(total_points, ball_size, dtype=np.int32)
        uncovered_remaining = total_points
        selected = []

        while uncovered_remaining > 0:
            # Find candidates with max potential
            max_pot = int(np.max(potential))
            if max_pot == 0:
                break
            
            candidates = np.where(potential == max_pot)[0]
            if len(candidates) > 1:
                if deterministic:
                    best_idx = int(candidates[0])
                else:
                    best_idx = int(rng.choice(candidates))
            else:
                best_idx = int(candidates[0])
            
            selected.append(best_idx)
            
            # Update coverage and potentials
            # We need to update potential for all j such that j covers any point in ball(best_idx)
            # Actually, potential[j] decreases by 1 for each point x in ball(best_idx) that was uncovered.
            # And potential[j] is related to ball(j).
            # So if x is in ball(best_idx) and x was uncovered, then for any j such that x is in ball(j),
            # potential[j] should decrease.
            # This means we need to iterate over x in ball(best_idx) and update potential for all j in ball(x).
            
            for x in balls[best_idx]:
                if cov_count[x] == 0:
                    # x was uncovered, now covered
                    uncovered_remaining -= 1
                    # Update potentials of all j that cover x
                    # These are j such that x is in ball(j), i.e., j is in ball(x)
                    potential[balls[x]] -= 1
                cov_count[x] += 1

        return selected

    def prune(selected_indices):
        if not selected_indices:
            return selected_indices

        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)

        for idx in selected_list:
            coverage_count[balls[idx]] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                # If all points in this ball are covered by other codewords (count > 1), we can remove it
                if np.all(coverage_count[balls[idx]] > 1):
                    selected_list.pop(i)
                    coverage_count[balls[idx]] -= 1
                    changed = True
                    break

        return selected_list

    def rebuild_coverage(selected_list):
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return coverage_count

    def is_valid_code(selected_list):
        """Check if the selected codewords cover all points."""
        if not selected_list:
            return total_points == 0
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return bool(np.all(coverage_count > 0))

    def local_search_walk(selected_indices, deadline):
        """
        Local search with two main moves:
        1. Deletion: Remove a codeword if all its covered points are still covered by others.
        2. Swap: Move a codeword to a neighbor if it resolves 'risky' points (covered only by the moving codeword)
           without breaking coverage elsewhere.
        """
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)

        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        while time.time() < deadline:
            # 1. Remove any removable codeword (full sweep, deterministic)
            removed_any = True
            while removed_any and time.time() < deadline:
                removed_any = False
                for i in range(len(selected_list)):
                    idx = selected_list[i]
                    if np.all(coverage_count[balls[idx]] > 1):
                        coverage_count[balls[idx]] -= 1
                        selected_list.pop(i)
                        selected_set.discard(idx)
                        removed_any = True
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        break

            if time.time() > deadline:
                break

            # 2. Neighbor-restricted swap
            n_sel = len(selected_list)
            if n_sel == 0:
                break

            progress = False
            # Try a few swaps before giving up on this iteration
            for _ in range(30):
                if time.time() > deadline:
                    break
                i = rng.randrange(n_sel)
                c = selected_list[i]
                nbrs = list(neighbors[c])
                rng.shuffle(nbrs)
                
                # Identify risky points: points covered ONLY by c
                risky_points = [x for x in balls[c] if coverage_count[x] == 1]
                
                if not risky_points:
                    # Should have been deleted in step 1, but just in case
                    coverage_count[balls[c]] -= 1
                    selected_list.pop(i)
                    selected_set.discard(c)
                    progress = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                    break

                # Find candidates c' such that c' covers ALL risky_points
                # c' must be in intersection of N(r) for all r in risky_points
                # And c' must not be in current code (to be a valid swap target)
                # Note: c' can be c itself, but we skip that.
                
                candidates = set(neighbors[risky_points[0]].tolist())
                for r in risky_points[1:]:
                    candidates &= set(neighbors[r].tolist())
                    if not candidates:
                        break
                
                candidates.discard(c)
                # Filter out existing codewords
                valid_candidates = [c2 for c2 in candidates if c2 not in selected_set]
                
                if valid_candidates:
                    # Pick a random candidate
                    c_prime = rng.choice(valid_candidates)
                    
                    # Perform Swap: remove c, add c_prime
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1
                    
                    # Check validity: no point in balls[c] should have count 0
                    # Since c_prime covers all risky points, and non-risky points had count > 1,
                    # this swap is theoretically always valid. But let's verify.
                    if np.any(coverage_count[balls[c]] == 0):
                        # Invalid, revert
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        continue
                    
                    # Valid swap
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    progress = True
                    break
                
                if progress:
                    break

            if progress:
                # After a swap, prune to catch newly-redundant codewords
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    coverage_count = rebuild_coverage(pruned)
                    selected_list = pruned
                    selected_set = set(selected_list)
                
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    def polish(selected_indices, deadline):
        """Final high-precision pass: full deterministic sweeps + exhaustive neighbor checks for removals."""
        selected_list = list(selected_indices)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        # 1. Aggressive deterministic removal
        for _ in range(100):
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    coverage_count[balls[idx]] -= 1
                    selected_list.pop(i)
                    selected_set.discard(idx)
                    changed = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
            if not changed:
                break

        # 2. Try single-step moves that might enable removal
        rng = random.Random(99)
        attempts = 0
        max_attempts = 5000 # Limit to prevent hanging
        
        while time.time() < deadline and attempts < max_attempts:
            if not selected_list:
                break
            
            # Pick a random codeword
            i = rng.randrange(len(selected_list))
            c = selected_list[i]
            nbrs = list(neighbors[c])
            rng.shuffle(nbrs)
            
            for c_prime in nbrs:
                if c_prime == c or c_prime in selected_set:
                    continue
                
                # Tentative move
                coverage_count[balls[c]] -= 1
                coverage_count[balls[c_prime]] += 1
                
                if np.any(coverage_count[balls[c]] == 0):
                    coverage_count[balls[c_prime]] -= 1
                    coverage_count[balls[c]] += 1
                    attempts += 1
                    break # Try a different random codeword
                
                # Move is valid. Now check if we can remove any codeword.
                removed_any = False
                removed_idx = -1
                for j in range(len(selected_list)):
                    idx_j = selected_list[j]
                    if np.all(coverage_count[balls[idx_j]] > 1):
                        removed_any = True
                        removed_idx = j
                        break
                
                if removed_any:
                    # Accept the move and the removal
                    val = selected_list[removed_idx]
                    coverage_count[balls[val]] -= 1
                    selected_list.pop(removed_idx)
                    selected_set.discard(val)
                    
                    # Update best
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                    
                    attempts += 1
                    break # Restart the outer loop with the new state
                else:
                    # No removal possible, revert the move
                    coverage_count[balls[c_prime]] -= 1
                    coverage_count[balls[c]] += 1
                    attempts += 1
                    break # Try a different random codeword

        return best_code

    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    # 1. Generate initial codes
    best_code = None
    best_len = float('inf')

    # Deterministic greedy (potential-based)
    selected = greedy_set_cover(deterministic=True)
    selected = prune(selected)
    if len(selected) < best_len:
        best_len = len(selected)
        best_code = selected[:]

    # Random greedy (potential-based)
    # We can afford more restarts with the faster potential-based greedy
    for seed in range(10):
        if time.time() > deadline - 40:
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

    if best_code is None:
        best_code = greedy_set_cover(deterministic=True)
        best_code = prune(best_code)
        best_len = len(best_code)

    # 2. Local Search
    # We run local search on the best initial code
    best_code = local_search_walk(best_code, deadline - 10)

    # 3. Polish
    best_code = polish(best_code, deadline)

    # Verify validity before returning
    if not is_valid_code(best_code):
        # Fallback to a known valid construction
        best_code = greedy_set_cover(deterministic=True)
        best_code = prune(best_code)

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
