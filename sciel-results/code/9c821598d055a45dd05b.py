import time
import random
import heapq
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for 120s limit

    # Fallback for other parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187
    K = 1 + 2 * n  # 15 neighbors per codeword

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

    # Precompute neighbors as a flat array for fast vectorized access
    neighbors_list = np.zeros((N, K), dtype=np.int32)
    for i in range(N):
        w = idx_to_tuple(i)
        neighbors_list[i, 0] = i
        count = 1
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    neighbors_list[i, count] = tuple_to_idx(new_w)
                    count += 1

    best_code_size = float('inf')
    best_code = None

    # Multi-start strategy
    num_starts = 3

    for run_id in range(num_starts):
        if time.time() > deadline:
            break
        
        # 1. Greedy Set Cover with heap-based priority queue
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        in_code = np.zeros(N, dtype=bool)
        
        # Initialize heap with all candidates
        # We want max-heap, so we negate the gain
        # gain[c] = number of currently uncovered points that c covers
        gains = np.zeros(N, dtype=np.int32)
        for c in range(N):
            gains[c] = uncovered_mask[neighbors_list[c]].sum()
        
        # Max-heap: store (-gain, c)
        heap = [(-int(gains[c]), c) for c in range(N)]
        heapq.heapify(heap)
        
        # Lazy deletion: we track current gain and only process if it matches
        current_gain = gains.copy()
        
        while uncovered_mask.any() and time.time() < deadline - 5:
            # Pop until we find a valid candidate
            best_c = -1
            best_gain = -1
            while heap:
                neg_g, c = heapq.heappop(heap)
                if in_code[c]:
                    continue
                if -neg_g != current_gain[c]:
                    continue
                if current_gain[c] <= 0:
                    break
                best_c = c
                best_gain = current_gain[c]
                break
            
            if best_c == -1:
                break
            
            # Randomize among top candidates for diversity
            # Find all candidates with the same gain (within a small tolerance)
            # But for efficiency, just pick randomly from top few
            top_candidates = [best_c]
            # Check a few more from the heap for ties
            extra_heap = []
            for _ in range(10):
                if not heap:
                    break
                neg_g, c = heapq.heappop(heap)
                extra_heap.append((neg_g, c))
                if -neg_g < best_gain:
                    break
                if -neg_g == best_gain and not in_code[c] and current_gain[c] == best_gain:
                    top_candidates.append(c)
            
            # Push back what we didn't use
            for item in extra_heap:
                heapq.heappush(heap, item)
            
            if len(top_candidates) > 1:
                best_c = random.choice(top_candidates)
            
            # Select best_c
            code.append(best_c)
            code_set.add(best_c)
            in_code[best_c] = True
            
            # Update coverage and uncovered_mask
            covered = neighbors_list[best_c]
            coverage[covered] += 1
            newly_covered = covered[uncovered_mask[covered]]
            uncovered_mask[newly_covered] = False
            
            # Update gains for all candidates
            # For each candidate c, gain changes by -1 for each neighbor of best_c that was uncovered
            # i.e., for each point p in newly_covered, for each candidate c that covers p, gain[c] -= 1
            # This is expensive to do naively. Let's use a different approach.
            
            # Actually, let's just recompute gains periodically or use a smarter update.
            # For now, let's just push updated gains for candidates that might be affected.
            # A candidate c's gain changes if it shares a neighbor with best_c.
            # The neighbors of best_c are covered. So any candidate that covers any of these
            # points had its gain reduced by 1 (for each newly covered point it also covers).
            
            # This is still expensive. Let's just recompute gains from scratch every 100 iterations
            # or when the heap becomes stale.
            
            # For simplicity and correctness, let's just recompute gains for all candidates
            # when the heap is about to be exhausted or periodically.
            # Actually, the lazy deletion approach requires us to push new entries when gains change.
            # Let's take a different approach: just recompute the entire gain array and rebuild the heap
            # every 50 selections.
            
            if len(code) % 50 == 0 and len(code) > 0:
                # Rebuild heap
                current_gain = uncovered_mask[neighbors_list].sum(axis=1).astype(np.int32)
                current_gain[in_code] = 0
                heap = []
                for c in range(N):
                    if not in_code[c] and current_gain[c] > 0:
                        heapq.heappush(heap, (-int(current_gain[c]), c))
                # We don't need to heapify if we push one by one, but it's fine
        
        # 2. Prune redundant (vectorized)
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            mins = coverage[neighbors_list].min(axis=1)
            redundant = mins > 1
            for c in code[:]:
                if redundant[c]:
                    code.remove(c)
                    code_set.discard(c)
                    in_code[c] = False
                    coverage[neighbors_list[c]] -= 1
                    changed = True

        # 3. Local Search with 1-exchange moves
        def try_exchange():
            """Try to swap one codeword for another to reduce count or improve coverage."""
            if len(code) < 2:
                return False
            # Pick a random codeword to remove
            idx = random.randint(0, len(code) - 1)
            removed = code[idx]
            
            # Compute coverage if we remove this codeword
            temp_coverage = coverage.copy()
            temp_coverage[neighbors_list[removed]] -= 1
            
            uncovered = np.where(temp_coverage == 0)[0]
            if len(uncovered) == 0:
                # Can just remove
                code.pop(idx)
                code_set.discard(removed)
                in_code[removed] = False
                coverage[neighbors_list[removed]] -= 1
                return True
            
            # Try to find a replacement that covers all uncovered
            uncovered_set = set(uncovered.tolist())
            # Candidate replacements: neighbors of uncovered words that are not in code
            candidate_pool = set()
            for w in uncovered:
                candidate_pool.update(neighbors_list[w].tolist())
            candidate_pool -= code_set
            
            best_c = -1
            best_gain = 0
            for c in candidate_pool:
                covered = len(set(neighbors_list[c].tolist()) & uncovered_set)
                if covered > best_gain:
                    best_gain = covered
                    best_c = c
                    if best_gain == len(uncovered_set):
                        break
            
            if best_c == -1 or best_gain < len(uncovered_set):
                return False
            
            # Apply exchange
            code.pop(idx)
            code_set.discard(removed)
            in_code[removed] = False
            coverage[neighbors_list[removed]] -= 1
            code.append(best_c)
            code_set.add(best_c)
            in_code[best_c] = True
            coverage[neighbors_list[best_c]] += 1
            return True

        def try_kick(k):
            """Try to remove k codewords and add fewer than k to repair."""
            if len(code) < k:
                return False
            indices = random.sample(range(len(code)), k)
            removed = [code[i] for i in sorted(indices, reverse=True)]
            
            temp_coverage = coverage.copy()
            for c in removed:
                temp_coverage[neighbors_list[c]] -= 1
                
            uncovered = np.where(temp_coverage == 0)[0]
            
            if len(uncovered) == 0:
                for c in removed:
                    code.remove(c)
                    code_set.discard(c)
                    in_code[c] = False
                    coverage[neighbors_list[c]] -= 1
                return True
            
            uncovered_set = set(uncovered.tolist())
            added = []
            temp_uncovered = set(uncovered_set)
            
            while temp_uncovered and len(added) < k:
                candidate_pool = set()
                for w in temp_uncovered:
                    candidate_pool.update(neighbors_list[w].tolist())
                candidate_pool -= code_set
                
                best_c = -1
                best_count = 0
                for c in candidate_pool:
                    cnt = len(set(neighbors_list[c].tolist()) & temp_uncovered)
                    if cnt > best_count:
                        best_count = cnt
                        best_c = c
                        if best_count == len(temp_uncovered):
                            break
                if best_c == -1:
                    break
                added.append(best_c)
                temp_uncovered.difference_update(set(neighbors_list[best_c].tolist()))
            
            if len(added) < len(removed):
                for c in removed:
                    code.remove(c)
                    code_set.discard(c)
                    in_code[c] = False
                    coverage[neighbors_list[c]] -= 1
                for c in added:
                    if c not in code_set:
                        code.append(c)
                        code_set.add(c)
                        in_code[c] = True
                        coverage[neighbors_list[c]] += 1
                return True
            return False

        no_improvement = 0
        while time.time() < deadline - 5:
            # Prune
            changed = True
            while changed:
                changed = False
                mins = coverage[neighbors_list].min(axis=1)
                redundant = mins > 1
                for c in code[:]:
                    if redundant[c]:
                        code.remove(c)
                        code_set.discard(c)
                        in_code[c] = False
                        coverage[neighbors_list[c]] -= 1
                        changed = True
            
            # Try moves: exchange first (cheaper, often effective), then kicks
            improved = False
            if try_exchange():
                improved = True
            elif try_kick(1):
                improved = True
            elif try_kick(2):
                improved = True
            elif try_kick(3):
                improved = True
            
            if improved:
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement > 50:
                    break

        if len(code) < best_code_size:
            best_code_size = len(code)
            best_code = code[:]

    if best_code is None:
        best_code = [0]
    
    return [idx_to_tuple(c) for c in best_code]
