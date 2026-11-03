import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total_words = q ** n
    powers = [q ** i for i in range(n)]
    ball_size = 1 + n * (q - 1)

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute balls as lists and as numpy arrays for vectorized ops
    balls_list = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d, q))
            d[pos] = orig
        balls_list[i] = neighbors

    balls = np.array(balls_list, dtype=np.int32)

    # Precompute inverse index: for each word w, list of centers whose ball contains w
    centers_covering = [None] * total_words
    for i in range(total_words):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total_words):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    def structured_init():
        """
        Deterministic structured initialization using tetracode product.
        For q=3, n=8, radius=1: use product of two length-4 ternary codes.
        """
        if q != 3 or n != 8 or radius != 1:
            return None

        # Tetracode: 6 codewords of length 4 over {0,1,2}, covering radius 1
        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
        ]

        # Product: all pairs (a, b) from tetracode x tetracode
        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)

        code_ints = [digits_to_int(c, q) for c in product_code]
        code_ints = list(set(code_ints))

        # Compute cover_count
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1

        # Greedy extension: add centers to cover remaining uncovered words
        uncovered = (cover_count == 0)
        while np.any(uncovered):
            if time.time() > deadline:
                break
            gains = uncovered[balls].sum(axis=1)
            gains[code_ints] = -1
            max_gain = gains.max()
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)

        return code_ints, cover_count

    def greedy_cover(seed_offset=0):
        """Greedy set cover with randomization for diversity."""
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)

        while np.any(uncovered):
            if time.time() > deadline:
                break

            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break

            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                best_center = int(best_centers[random.randint(0, len(best_centers) - 1)])

            code.append(best_center)
            uncovered[balls[best_center]] = False
            cover_count[balls[best_center]] += 1

        return code, cover_count

    def trim_code(code, cover_count):
        """Remove redundant codewords (those that don't uniquely cover any word)."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.any(cover_count[balls[c]] == 1):
                    continue
                to_remove.append(i)

            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True
        return code, cover_count

    def targeted_local_search(code, cover_count, max_iter=500000):
        """
        Aggressive targeted local search using inverse index for candidate enumeration.
        For each codeword, find candidates that cover its uniquely-covered words.
        """
        code_set = set(code)

        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iter:
            improved = False
            iter_count += 1

            # Shuffle code order for diversity
            code_indices = list(range(len(code)))
            random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    break
                i = code[ci]
                i_ball = balls[i]
                only_by_i = i_ball[cover_count[i_ball] == 1]

                if len(only_by_i) == 0:
                    # Can remove i
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                # Find candidates that cover ALL of only_by_i
                candidate_set = None
                for w in only_by_i:
                    if candidate_set is None:
                        candidate_set = set(centers_covering[w])
                    else:
                        candidate_set &= centers_covering[w]
                    if not candidate_set:
                        break

                if not candidate_set:
                    continue

                # Remove current codeword from candidates
                candidate_set.discard(i)
                if not candidate_set:
                    continue

                # Try each candidate
                for c_add in list(candidate_set):
                    if c_add in code_set:
                        continue

                    # Check if adding c_add and removing i keeps coverage
                    # Words that would lose coverage: only_by_i
                    # Words that would gain coverage: balls[c_add]
                    # Need: for all w in only_by_i, w is in balls[c_add] (already ensured)
                    # Also need: no word loses coverage that wasn't in only_by_i
                    # Since only_by_i are the ONLY words with cover_count==1 for i,
                    # removing i only affects those words. Adding c_add covers them.
                    # So the swap is valid!

                    code[ci] = c_add
                    cover_count[i_ball] -= 1
                    cover_count[balls[c_add]] += 1
                    code_set.discard(i)
                    code_set.add(c_add)
                    improved = True
                    break

                if improved:
                    break

        return code, cover_count

    def multi_start_search(code, cover_count, num_starts=3):
        """Try multiple restarts with perturbation to escape local optima."""
        best_code = list(code)
        best_count = cover_count.copy()
        best_len = len(code)

        for start in range(num_starts):
            if time.time() > deadline:
                break

            # Perturb: randomly swap a few codewords
            perturbed = list(code)
            perturbed_count = cover_count.copy()
            perturbed_set = set(perturbed)

            num_swaps = min(len(perturbed) // 3, 5)
            for _ in range(num_swaps):
                if time.time() > deadline:
                    break
                idx = random.randint(0, len(perturbed) - 1)
                old = perturbed[idx]
                new = random.randint(0, total_words - 1)
                if new in perturbed_set:
                    continue
                # Check validity
                perturbed_count[balls[old]] -= 1
                if np.any(perturbed_count < 0):
                    perturbed_count[balls[old]] += 1
                    continue
                perturbed[idx] = new
                perturbed_count[balls[new]] += 1
                perturbed_set.discard(old)
                perturbed_set.add(new)

            # Trim and local search
            perturbed, perturbed_count = trim_code(perturbed, perturbed_count)
            perturbed, perturbed_count = targeted_local_search(perturbed, perturbed_count, max_iter=100000)

            if len(perturbed) < best_len:
                best_code = perturbed
                best_count = perturbed_count
                best_len = len(perturbed)

        return best_code, best_count

    # Phase 1: Structured initialization
    result = structured_init()
    if result is not None:
        code, cover_count = result
    else:
        code, cover_count = greedy_cover()

    # Phase 2: Trim redundant codewords
    code, cover_count = trim_code(code, cover_count)

    # Phase 3: Targeted local search
    code, cover_count = targeted_local_search(code, cover_count, max_iter=200000)

    # Phase 4: Multi-start with perturbation
    code, cover_count = multi_start_search(code, cover_count, num_starts=3)

    # Final validation
    if time.time() < deadline:
        final_cover = np.zeros(total_words, dtype=np.int32)
        for c in code:
            final_cover[balls[c]] += 1
        if np.any(final_cover == 0):
            # Fallback: complete with greedy
            uncovered = (final_cover == 0)
            while np.any(uncovered):
                if time.time() > deadline:
                    break
                gains = uncovered[balls].sum(axis=1)
                gains[code] = -1
                max_gain = gains.max()
                if max_gain <= 0:
                    break
                best_center = int(np.argmax(gains))
                code.append(best_center)
                final_cover[balls[best_center]] += 1
                uncovered = (final_cover == 0)

    return [int_to_digits(idx, n, q) for idx in code]
