"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    def is_valid_extension(v, path, used):
        if v in used:
            return False
        for i in range(len(path) - 1):
            if (v ^ path[i]).bit_count() == 1:
                return False
        return True

    def count_valid_extensions(v, path, used):
        """Count how many valid extensions v has (for lookahead heuristic)."""
        count = 0
        for b in range(N):
            w = v ^ (1 << b)
            if is_valid_extension(w, path + [v], used | {v}):
                count += 1
        return count

    def greedy_extend(path, used, bit_order, use_lookahead=False):
        while True:
            if time.time() > deadline:
                return path
            cur = path[-1]
            candidates = []
            for b in bit_order:
                v = cur ^ (1 << b)
                if is_valid_extension(v, path, used):
                    candidates.append(v)
            if not candidates:
                return path
            if use_lookahead and len(candidates) > 1:
                # Choose the candidate with the most future extensions
                best_v = max(candidates, key=lambda v: count_valid_extensions(v, path, used))
            else:
                best_v = candidates[0]
            path.append(best_v)
            used.add(best_v)

    def backtrack_extend(path, used, bit_order, max_backtrack_depth=3):
        """Greedy with limited backtracking to escape local optima."""
        while True:
            if time.time() > deadline:
                return path
            cur = path[-1]
            candidates = []
            for b in bit_order:
                v = cur ^ (1 << b)
                if is_valid_extension(v, path, used):
                    candidates.append(v)
            if not candidates:
                # Try backtracking
                if len(path) <= max_backtrack_depth:
                    return path
                # Remove last few vertices and try different choices
                backtrack_len = min(max_backtrack_depth, len(path) - 1)
                for i in range(backtrack_len):
                    removed = path.pop()
                    used.discard(removed)
                    cur = path[-1]
                    candidates = []
                    for b in bit_order:
                        v = cur ^ (1 << b)
                        if is_valid_extension(v, path, used):
                            candidates.append(v)
                    if candidates:
                        # Pick a different candidate than before
                        # Try to pick one that wasn't the first choice
                        if len(candidates) > 1:
                            best_v = candidates[-1]  # pick last to differ
                        else:
                            best_v = candidates[0]
                        path.append(best_v)
                        used.add(best_v)
                        break
                else:
                    return path
                continue
            path.append(candidates[0])
            used.add(candidates[0])

    best_path = []

    # Strategy 1: Multiple random bit orderings from vertex 0
    for _ in range(50):
        if time.time() > deadline:
            break
        random_bit_order = list(range(N))
        random.shuffle(random_bit_order)
        path = [0]
        used = {0}
        path = greedy_extend(path, used, random_bit_order)
        if len(path) > len(best_path):
            best_path = path

    # Strategy 2: Try different starting vertices
    if time.time() < deadline:
        start_vertices = [0, 1, 2, 4, 8, 16, 32, 64, 128]
        random.shuffle(start_vertices)
        for start_v in start_vertices:
            if time.time() > deadline:
                break
            random_bit_order = list(range(N))
            random.shuffle(random_bit_order)
            path = [start_v]
            used = {start_v}
            path = greedy_extend(path, used, random_bit_order)
            if len(path) > len(best_path):
                best_path = path

    # Strategy 3: Lookahead heuristic from vertex 0
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            random_bit_order = list(range(N))
            random.shuffle(random_bit_order)
            path = [0]
            used = {0}
            path = greedy_extend(path, used, random_bit_order, use_lookahead=True)
            if len(path) > len(best_path):
                best_path = path

    # Strategy 4: Backtracking from vertex 0
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            random_bit_order = list(range(N))
            random.shuffle(random_bit_order)
            path = [0]
            used = {0}
            path = backtrack_extend(path, used, random_bit_order, max_backtrack_depth=5)
            if len(path) > len(best_path):
                best_path = path

    return best_path
# EVOLVE-BLOCK-END
