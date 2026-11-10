"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: all words (valid but large)
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific case: q=3, n=7, radius=1
    N = 3 ** 7  # 2187

    # Precompute integer <-> tuple conversions
    def int_to_tuple(x):
        t = []
        for _ in range(n):
            t.append(x % 3)
            x //= 3
        return tuple(t)

    def tuple_to_int(t):
        x = 0
        for i in range(n):
            x += t[i] * (3 ** i)
        return x

    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        w = int_to_tuple(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig = w[pos]
            for d in range(3):
                if d == orig:
                    continue
                new_w = list(w)
                new_w[pos] = d
                j = tuple_to_int(tuple(new_w))
                neighbors[i].append(j)

    def run_greedy(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = [False] * N
        coverage_count = [15] * N  # 1 self + 14 neighbors
        uncovered_set = set(range(N))
        code = []

        while uncovered_set:
            if time.time() > deadline - 30:
                break

            # Find the word in uncovered_set with max coverage_count
            best_word = None
            max_cov = -1
            for w in uncovered_set:
                c = coverage_count[w]
                if c > max_cov:
                    max_cov = c
                    best_word = w
                elif c == max_cov and seed is not None:
                    # Random tie-breaking
                    if random.random() < 0.3:
                        best_word = w

            if best_word is None:
                break

            code.append(best_word)

            # Mark best_word and its neighbors as covered
            for v in neighbors[best_word]:
                if not covered[v]:
                    covered[v] = True
                    uncovered_set.discard(v)
                    # Update coverage_count for neighbors of v
                    for u in neighbors[v]:
                        if not covered[u]:
                            coverage_count[u] -= 1

        return [int_to_tuple(c) for c in code]

    def verify_coverage(code_tuples):
        """Check if code covers all words."""
        code_indices = [tuple_to_int(t) for t in code_tuples]
        covered = [False] * N
        for c in code_indices:
            for v in neighbors[c]:
                covered[v] = True
        return all(covered)

    def local_optimize(code_tuples, max_time=30):
        """Apply local optimization: redundancy removal and swaps."""
        if not verify_coverage(code_tuples):
            return code_tuples

        code_indices = [tuple_to_int(t) for t in code_tuples]
        cover_count = [0] * N  # How many codewords cover each word

        # Build coverage counts
        for c in code_indices:
            for v in neighbors[c]:
                cover_count[v] += 1

        improved = True
        while improved and time.time() < max_time:
            improved = False

            # Try removing redundant codewords
            for i in range(len(code_indices)):
                if time.time() > max_time:
                    break
                c = code_indices[i]
                redundant = True
                for v in neighbors[c]:
                    if cover_count[v] <= 1:
                        redundant = False
                        break
                if redundant:
                    for v in neighbors[c]:
                        cover_count[v] -= 1
                    code_indices.pop(i)
                    improved = True
                    break

            # Try swaps: replace one codeword with another
            if not improved:
                for i in range(len(code_indices)):
                    if time.time() > max_time:
                        break
                    c = code_indices[i]
                    # Find words only covered by this codeword
                    lost = []
                    for v in neighbors[c]:
                        if cover_count[v] == 1:
                            lost.append(v)

                    if not lost:
                        # Redundant, remove
                        for v in neighbors[c]:
                            cover_count[v] -= 1
                        code_indices.pop(i)
                        improved = True
                        break

                    # Try to find a replacement that covers all lost words
                    best_replace = None
                    best_gain = -1
                    for w in range(N):
                        if w in code_indices:
                            continue
                        ball_w = neighbors[w]
                        # Check if ball_w covers all lost words
                        covers_all = all(v in ball_w for v in lost)
                        if not covers_all:
                            continue
                        # Count new coverage gain
                        new_gain = sum(1 for v in ball_w if cover_count[v] == 0)
                        if new_gain > best_gain:
                            best_gain = new_gain
                            best_replace = w

                    if best_replace is not None:
                        # Perform swap
                        for v in neighbors[c]:
                            cover_count[v] -= 1
                        code_indices[i] = best_replace
                        for v in neighbors[best_replace]:
                            cover_count[v] += 1
                        improved = True
                        break

        return [int_to_tuple(c) for c in code_indices]

    # Run multiple greedy restarts with different seeds
    best_code = None
    best_len = float('inf')

    # First run: deterministic (seed=0)
    code = run_greedy(0)
    code = local_optimize(code, max_time=20)
    if verify_coverage(code):
        best_len = len(code)
        best_code = code

    # Additional random restarts
    num_restarts = 30
    for i in range(1, num_restarts + 1):
        if time.time() > deadline - 40:
            break
        code = run_greedy(i)
        code = local_optimize(code, max_time=15)
        if verify_coverage(code) and len(code) < best_len:
            best_len = len(code)
            best_code = code

    # Final polish: try more aggressive local search on best code
    if best_code is not None and time.time() < deadline - 10:
        best_code = local_optimize(best_code, max_time=deadline - time.time() - 5)
        if not verify_coverage(best_code):
            # Revert if optimization broke coverage
            best_code = run_greedy(0)
            best_code = local_optimize(best_code, max_time=10)

    if best_code is None:
        # Fallback: tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(w) for w in best_code]
# EVOLVE-BLOCK-END
