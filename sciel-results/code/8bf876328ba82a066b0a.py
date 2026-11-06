"""Combined program: vectorized greedy + enhanced local search for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists as numpy arrays of int32 for fast operations
    neighbors_list = []
    neighbor_sets = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbor_sets.append(set(nb))

    # ---------- Phase 1: Greedy Set-Cover with vectorized gain computation ----------
    # Instead of iterating over all candidates and computing np.sum(uncovered_mask[neighbors[c]]),
    # we accumulate gains from uncovered points: for each uncovered point, add 1 to all its neighbors.
    # This is much faster because it only touches uncovered points, not all N candidates.

    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()

    while np.any(uncovered):
        if time.time() > deadline - 30:
            break

        # Compute gains: for each candidate codeword, how many uncovered points does it cover?
        # Accumulate from uncovered points
        gains = np.zeros(N, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            gains[neighbors_list[idx]] += 1

        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]

        if max_gain == 0:
            break

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors_list[best_idx]] = False

    # ---------- Phase 2: Local Search ----------
    # Initialize coverage counts
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1

    best_size = len(code)
    best_code = list(code)

    # Precompute a sorted list of codewords for faster redundancy checks
    # and a helper to quickly find which codewords cover a given point.
    # For K_3(7,1), each point has 15 neighbors. We can precompute for each point
    # which of the current codewords are its neighbors, but that changes as code changes.
    # Instead, we'll use the counts array which is already maintained.

    def greedy_repair(uncovered_set):
        """Given a set of uncovered points, greedily add codewords to cover them."""
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)

        while temp_uncovered:
            best_c = -1
            best_count = 0

            # Build candidate pool from neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbor_sets[w])

            # Find best candidate
            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = 0
                for x in neighbors_list[c]:
                    if x in temp_uncovered:
                        cnt += 1
                        if cnt > best_count:
                            best_count = cnt
                            best_c = c
                        if best_count == len(temp_uncovered):
                            break
                if best_count == len(temp_uncovered):
                    break

            if best_c == -1:
                break

            added.append(best_c)
            for w in neighbors_list[best_c]:
                temp_uncovered.discard(w)

        return added

    def try_remove():
        """Remove all redundant codewords (those whose neighbors are all covered by >1 codeword)."""
        removed = False
        # Create a snapshot of the current code to iterate over
        code_snapshot = list(code)
        for c in code_snapshot:
            if c not in code_set:
                continue
            # Check if all neighbors of c have count > 1
            # This means c is redundant
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                removed = True
        return removed

    def try_swap(c):
        """Try to swap codeword c with a better-positioned codeword.
        Strategy 1: Find a replacement covering all critical points of c.
        Strategy 2: Relocation - move c to cover would-be-uncovered points.
        Returns True if a swap was made."""
        # Check if c is currently removable
        if np.min(counts[neighbors_list[c]]) > 1:
            return False

        # Get critical points for c: points in neighbors[c] with counts == 1
        crit_points = [i for i in neighbors_list[c] if counts[i] == 1]

        if not crit_points:
            return False

        # Strategy 1: Find replacement covering all critical points
        # We need a new codeword w (not in code_set) such that w covers all crit_points.
        # This means crit_points ⊆ neighbors[w].
        # Since |crit_points| can be large, we use intersection of neighbor sets.
        
        # Start with the neighbor set of the first critical point
        if len(crit_points) == 0:
            return False
            
        # Use the intersection of neighbor sets of all critical points
        # but restrict to candidates not in code_set
        # We'll iterate over candidates from the first critical point's neighbors
        first_crit = crit_points[0]
        candidates = [x for x in neighbor_sets[first_crit] if x not in code_set]
        
        best_w = -1
        best_score = -1
        
        for w in candidates:
            # Check if w covers all critical points
            valid = True
            for k in crit_points[1:]:  # Skip first_crit since w is already in its neighbors
                if k not in neighbor_sets[w]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Score: maximize redundancy creation (points that would go from count 1 to 2)
            # when we add w and remove c
            # New count for point p = old_count[p] - 1 (if p in neighbors[c]) + 1 (if p in neighbors[w])
            # We want to maximize the number of points that were count 1 and become count 2
            # This happens if p in neighbors[w] and p not in neighbors[c] and old_count[p] == 1
            # OR p in both neighbors[w] and neighbors[c] and old_count[p] == 1 (count stays 1, not helpful)
            # Actually, simpler: just count how many points in neighbors[w] have old_count == 1
            # and are NOT in neighbors[c] (so they gain coverage)
            
            # A simpler and effective score: how many points does w cover that have count == 1?
            score = int(np.sum(counts[neighbors_list[w]] == 1))
            
            if score > best_score:
                best_score = score
                best_w = w
        
        if best_w != -1:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1

            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors_list[best_w]] += 1
            return True

        # Strategy 2: Relocation - try to move c to cover would-be-uncovered points
        temp_counts = counts.copy()
        temp_counts[neighbors_list[c]] -= 1

        would_be_uncovered = [i for i in neighbors_list[c] if temp_counts[i] == 0]

        if not would_be_uncovered:
            return False

        all_needed = set(crit_points) | set(would_be_uncovered)

        best_w = -1
        best_score = -1

        # Use the first needed point to generate candidates
        first_point = list(all_needed)[0]
        cands = [x for x in neighbor_sets[first_point] if x not in code_set]

        for w in cands:
            valid = True
            for k in all_needed:
                if k not in neighbor_sets[w]:
                    valid = False
                    break
            if not valid:
                continue

            score = int(np.sum(counts[neighbors_list[w]] == 1))
            if score > best_score:
                best_score = score
                best_w = w

        if best_w != -1:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1

            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors_list[best_w]] += 1
            return True

        return False

    def kick():
        """Remove a random codeword and try to repair. Accept if size decreases."""
        if not code:
            return False

        old_size = len(code)

        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]

        temp_coverage = counts.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            code.remove(c_remove)
            code_set.remove(c_remove)
            counts[neighbors_list[c_remove]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - 1 + len(added)

        if new_size < old_size:
            code.remove(c_remove)
            code_set.remove(c_remove)
            counts[neighbors_list[c_remove]] -= 1
            for c in added:
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1
            return True
        else:
            return False

    def kick2():
        """Remove two random codewords and try to repair. Accept if size decreases."""
        if len(code) < 2:
            return False

        old_size = len(code)

        i1 = random.randint(0, len(code) - 1)
        i2 = random.randint(0, len(code) - 1)
        if i1 == i2:
            i2 = (i2 + 1) % len(code)

        c1 = code[i1]
        c2 = code[i2]

        temp_coverage = counts.copy()
        temp_coverage[neighbors_list[c1]] -= 1
        temp_coverage[neighbors_list[c2]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]
        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - 2 + len(added)

        if new_size < old_size:
            code.remove(c1)
            code_set.remove(c1)
            counts[neighbors_list[c1]] -= 1
            code.remove(c2)
            code_set.remove(c2)
            counts[neighbors_list[c2]] -= 1
            for c in added:
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1
            return True
        else:
            return False

    # ---------- Main Local Search Loop ----------
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False

        # 1. Exhaustive removal pass
        while try_remove():
            improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 10:
            break

        # 2. Swap pass - try to improve each codeword's position
        shuffled_code = list(code)
        random.shuffle(shuffled_code)

        for c in shuffled_code:
            if time.time() > deadline - 10:
                break
            if c not in code_set:
                continue
            if try_swap(c):
                improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 10:
            break

        # 3. Kick pass
        kicks_tried = 0
        max_kicks = 30
        while kicks_tried < max_kicks and time.time() < deadline - 10:
            if kick():
                improved = True
                kicks_tried = 0
            else:
                kicks_tried += 1

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 10:
            break

        # 4. 2-removal kick pass
        if len(code) > 2:
            kicks2_tried = 0
            max_kicks2 = 15
            while kicks2_tried < max_kicks2 and time.time() < deadline - 10:
                if kick2():
                    improved = True
                    kicks2_tried = 0
                else:
                    kicks2_tried += 1

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
        
        # 5. Additional polish: run a few more removal passes after kicks
        if time.time() < deadline - 5:
            while try_remove():
                pass
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

    # If we found a better code, restore it
    if len(best_code) < len(code):
        code = best_code
        code_set = set(best_code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

    # Convert back to list of lists
    result = [idx_to_tuple(cw) for cw in code]

    return result
# EVOLVE-BLOCK-END
