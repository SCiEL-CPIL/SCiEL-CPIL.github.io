import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill

    N = q ** n

    # Helper functions
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        base = 1
        for i in range(n):
            idx += w[i] * base
            base *= q
        return idx

    # Precompute neighbors for each word index (radius=1 case is most common)
    neighbors = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
    if radius == 1:
        for idx in range(N):
            w = idx_to_word(idx)
            neighbors[idx, 0] = idx
            pos = 1
            for p in range(n):
                orig = w[p]
                for val in range(q):
                    if val == orig:
                        continue
                    w[p] = val
                    neighbors[idx, pos] = word_to_idx(w)
                    pos += 1
                w[p] = orig
    elif radius == 0:
        for idx in range(N):
            neighbors[idx, 0] = idx
    else:
        # Fallback: brute force
        for idx in range(N):
            w1 = idx_to_word(idx)
            pos = 0
            for jdx in range(N):
                w2 = idx_to_word(jdx)
                dist = sum(1 for a, b in zip(w1, w2) if a != b)
                if dist <= radius:
                    if pos < neighbors.shape[1]:
                        neighbors[idx, pos] = jdx
                        pos += 1

    ball_size = 1 + n * (q - 1) if radius == 1 else N

    def is_covered(code_indices):
        """Check if all words are covered by the given code."""
        covered = np.zeros(N, dtype=np.uint8)
        for c in code_indices:
            covered[neighbors[c, :ball_size]] = 1
        return covered.all()

    def remove_redundant(code_list, max_iters=5000):
        """Iteratively remove codewords that don't reduce coverage."""
        code_set = set(code_list)
        code_arr = list(code_list)
        changed = True
        iters = 0
        while changed and iters < max_iters:
            if time.time() > deadline:
                break
            changed = False
            iters += 1
            # Try removing each codeword
            for i in range(len(code_arr)):
                if time.time() > deadline:
                    break
                candidate = code_arr[i]
                # Check if removing candidate still covers all
                remaining = [c for j, c in enumerate(code_arr) if j != i]
                if not remaining:
                    continue
                # Vectorized coverage check
                covered = np.zeros(N, dtype=np.uint8)
                for c in remaining:
                    covered[neighbors[c, :ball_size]] = 1
                if covered.all():
                    code_arr.pop(i)
                    changed = True
                    break
        return code_arr

    def local_search(code_list):
        """Try to improve code by swapping or removing codewords."""
        code_arr = list(code_list)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try removing redundant
            code_arr = remove_redundant(code_arr, max_iters=200)
            # Try swapping: replace one codeword with another
            for i in range(len(code_arr)):
                if time.time() > deadline:
                    break
                for c in range(N):
                    if c in code_arr:
                        continue
                    # Try replacing code_arr[i] with c
                    new_code = code_arr[:i] + [c] + code_arr[i+1:]
                    covered = np.zeros(N, dtype=np.uint8)
                    for cc in new_code:
                        covered[neighbors[cc, :ball_size]] = 1
                    if covered.all():
                        # Check if this allows further reduction
                        test_arr = remove_redundant(new_code, max_iters=100)
                        if len(test_arr) < len(code_arr):
                            code_arr = test_arr
                            improved = True
                            break
                if improved:
                    break
        return code_arr

    # Structured initialization using linear code structure
    # For q=3, n=8: use a [8,k] linear code over F_3
    def structured_init():
        """Initialize with a structured linear code."""
        # Use a systematic generator matrix approach
        # For n=8, q=3: try k=4 systematic code
        if n == 8 and q == 3:
            # Generator matrix for [8,4] code over F_3
            # G = [I_4 | P] where P is 4x4
            # Use a good parity check structure
            G = np.array([
                [1, 0, 0, 0, 1, 1, 2, 1],
                [0, 1, 0, 0, 1, 2, 1, 2],
                [0, 0, 1, 0, 2, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 2]
            ], dtype=np.int32) % 3

            code_indices = []
            for msg in itertools.product(range(3), repeat=4):
                msg_arr = np.array(msg, dtype=np.int32)
                codeword = (msg_arr @ G) % 3
                idx = word_to_idx(codeword.tolist())
                code_indices.append(idx)

            # Convert to list of words
            return [idx_to_word(idx) for idx in code_indices]
        else:
            # Fallback: direct sum construction
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
            return code

    best_code = None
    best_size = N + 1

    # Try multiple structured initializations
    for seed in range(5):
        if time.time() > deadline:
            break

        random.seed(seed)

        # Get initial code
        code = structured_init()

        # Convert to indices for efficiency
        code_indices = [word_to_idx(w) for w in code]

        # Local search to improve
        code_indices = local_search(code_indices)

        if len(code_indices) < best_size:
            best_size = len(code_indices)
            best_code = [idx_to_word(idx) for idx in code_indices]

    # If local search didn't help, try greedy + local search hybrid
    if time.time() < deadline - 10:
        def greedy_init():
            """Greedy set cover initialization."""
            coverage = np.zeros(N, dtype=np.int32)
            uncovered = N
            code_indices = []

            while uncovered > 0:
                if time.time() > deadline:
                    break

                best_c = -1
                max_new = -1

                for c in range(N):
                    new_covers = np.sum(coverage[neighbors[c, :ball_size]] == 0)
                    if new_covers > max_new:
                        max_new = new_covers
                        best_c = c
                        if max_new == 0:
                            break

                if best_c == -1 or max_new == 0:
                    break

                code_indices.append(best_c)
                prev_uncovered = coverage[neighbors[best_c, :ball_size]] == 0
                uncovered -= np.sum(prev_uncovered)
                coverage[neighbors[best_c, :ball_size]] += 1

            return code_indices

        greedy_code = greedy_init()
        if greedy_code:
            greedy_code = local_search(greedy_code)
            if len(greedy_code) < best_size:
                best_size = len(greedy_code)
                best_code = [idx_to_word(idx) for idx in greedy_code]

    if best_code is None:
        # Final fallback
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    return best_code
