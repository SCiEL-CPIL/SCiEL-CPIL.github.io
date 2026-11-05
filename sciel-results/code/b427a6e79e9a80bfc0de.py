import time
import random
import heapq
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    d = 7
    n = 2 ** d

    # Precompute all edges as (u, v, bit) where u < v
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))

    num_edges = len(all_edges)
    edge_index = {}
    for idx, (x, y, bit) in enumerate(all_edges):
        edge_index[(x, y)] = idx

    # Precompute all C4s as tuples of 4 edge indices
    all_c4s = []
    seen_c4s = set()
    edge_to_c4s = defaultdict(list)

    for idx in range(num_edges):
        x, y, bit_i = all_edges[idx]
        for j in range(d):
            if j == bit_i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            e2_key = (min(c, e), max(c, e))
            e3_key = (min(x, c), max(x, c))
            e4_key = (min(y, e), max(y, e))

            e2_idx = edge_index.get(e2_key)
            e3_idx = edge_index.get(e3_key)
            e4_idx = edge_index.get(e4_key)

            if e2_idx is not None and e3_idx is not None and e4_idx is not None:
                c4 = tuple(sorted([idx, e2_idx, e3_idx, e4_idx]))
                if c4 not in seen_c4s:
                    seen_c4s.add(c4)
                    c4_idx = len(all_c4s)
                    all_c4s.append(c4)
                    for e in c4:
                        edge_to_c4s[e].append(c4_idx)

    num_c4s = len(all_c4s)

    def is_valid_add_fast(chosen_set, edge_idx, c4_count):
        """Check if adding edge_idx creates a C4 using precomputed C4 counts."""
        for c4_idx in edge_to_c4s[edge_idx]:
            c4 = all_c4s[c4_idx]
            # Count how many of the other 3 edges are in chosen
            count = 0
            for e in c4:
                if e != edge_idx and e in chosen_set:
                    count += 1
            if count == 3:
                return False
        return True

    def greedy_mcs(seed):
        """
        Maximum Cardinality Search: pick edge with minimum blocked C4s.
        Uses C4-count tracking for efficiency.
        """
        rng = random.Random(seed)

        c4_selected_count = [0] * num_c4s
        blocked_count = [0] * num_edges
        chosen = set()
        out = []
        unselected = set(range(num_edges))

        # Build initial heap
        heap = []
        for i in range(num_edges):
            heap.append((0, rng.random(), i))
        heapq.heapify(heap)

        in_heap = [True] * num_edges

        while unselected:
            if time.time() - start_time > time_limit - 5.0:
                break

            # Pop the edge with minimum blocked_count
            found = False
            while heap:
                bc, _, idx = heapq.heappop(heap)
                if not in_heap[idx] or idx not in unselected:
                    continue
                if bc != blocked_count[idx]:
                    continue
                found = True
                break

            if not found:
                # Rebuild heap
                heap = []
                for i in unselected:
                    heap.append((blocked_count[i], rng.random(), i))
                heapq.heapify(heap)
                if not heap:
                    break
                bc, _, idx = heapq.heappop(heap)

            in_heap[idx] = False
            unselected.discard(idx)

            if blocked_count[idx] > 0:
                continue

            chosen.add(idx)
            out.append([all_edges[idx][0], all_edges[idx][1]])

            # Update C4 counts and blocked counts
            for c4_idx in edge_to_c4s[idx]:
                c4 = all_c4s[c4_idx]
                c4_selected_count[c4_idx] += 1
                new_count = c4_selected_count[c4_idx]

                if new_count == 3:
                    for e in c4:
                        if e not in chosen and e in unselected:
                            blocked_count[e] += 1
                            heapq.heappush(heap, (blocked_count[e], rng.random(), e))

        return out, chosen, c4_selected_count

    def local_search_sa(out, chosen, rng, time_limit):
        """
        Simulated Annealing local search with C4-count tracking.
        Moves: add random valid edge, or remove+add to escape local optima.
        """
        current_score = len(chosen)
        best_score = current_score
        best_out = [list(e) for e in out]

        # Track C4 selected counts
        c4_selected_count = [0] * num_c4s
        for idx in chosen:
            for c4_idx in edge_to_c4s[idx]:
                c4_selected_count[c4_idx] += 1

        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9995

        while time.time() - start_time < time_limit - 3.0:
            # Try to add a random edge
            added = False
            for _ in range(15):
                candidate = rng.randint(0, num_edges - 1)
                if candidate not in chosen:
                    if is_valid_add_fast(chosen, candidate, c4_selected_count):
                        chosen.add(candidate)
                        out.append([all_edges[candidate][0], all_edges[candidate][1]])
                        current_score += 1
                        for c4_idx in edge_to_c4s[candidate]:
                            c4_selected_count[c4_idx] += 1
                        added = True
                        if current_score > best_score:
                            best_score = current_score
                            best_out = [list(e) for e in out]
                        break

            if not added:
                # Stuck: remove a random edge, then try to add
                if out:
                    idx = rng.randint(0, len(out) - 1)
                    ex, ey = out[idx]
                    edge_key = (ex, ey)
                    edge_idx = edge_index[edge_key]

                    # Remove
                    chosen.discard(edge_idx)
                    out.pop(idx)
                    current_score -= 1
                    for c4_idx in edge_to_c4s[edge_idx]:
                        c4_selected_count[c4_idx] -= 1

                    # Try to add edges back
                    added_count = 0
                    for _ in range(25):
                        candidate = rng.randint(0, num_edges - 1)
                        if candidate not in chosen:
                            if is_valid_add_fast(chosen, candidate, c4_selected_count):
                                chosen.add(candidate)
                                out.append([all_edges[candidate][0], all_edges[candidate][1]])
                                current_score += 1
                                for c4_idx in edge_to_c4s[candidate]:
                                    c4_selected_count[c4_idx] += 1
                                added_count += 1
                                if current_score > best_score:
                                    best_score = current_score
                                    best_out = [list(e) for e in out]
                                if added_count >= 2:
                                    break

            T *= cooling_rate
            if T < T_min:
                T = T_min

        return best_out, best_score

    # Run multiple restarts: MCS init + SA local search
    best_result = None
    best_count = 0

    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break

        rng = random.Random(restart * 7919 + 12345)
        out, chosen, _ = greedy_mcs(restart)

        if len(out) > best_count:
            best_count = len(out)
            best_result = [list(e) for e in out]

        # Local search improvement
        if time.time() - start_time < time_limit - 5.0:
            out, score = local_search_sa(out, chosen, rng, time_limit)
            if score > best_count:
                best_count = score
                best_result = out

    if best_result is None:
        # Fallback
        chosen = set()
        best_result = []
        for idx in range(num_edges):
            x, y, bit_i = all_edges[idx]
            valid = True
            for c4_idx in edge_to_c4s[idx]:
                c4 = all_c4s[c4_idx]
                count = 0
                for e in c4:
                    if e != idx and e in chosen:
                        count += 1
                if count == 3:
                    valid = False
                    break
            if valid:
                chosen.add(idx)
                best_result.append([x, y])

    return best_result
