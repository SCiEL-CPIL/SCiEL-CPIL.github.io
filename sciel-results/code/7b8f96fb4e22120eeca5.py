import numpy as np
import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 145.0

    base = 51975
    divs = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_divs = len(divs)

    def time_left():
        return DEADLINE - time.time()

    def greedy_build(order):
        """Greedy: for each modulus in order, pick residue covering most uncovered."""
        cover_count = np.zeros(base, dtype=np.int32)
        classes = []
        for m in order:
            if time_left() < 2:
                break
            zeros = np.where(cover_count == 0)[0]
            if len(zeros) == 0:
                classes.append([m, 0])
                continue
            A = np.bincount(zeros % m, minlength=m)
            best_r = int(np.argmax(A))
            classes.append([m, best_r])
            cover_count[best_r::m] += 1
        score = base - int(np.sum(cover_count > 0))
        return classes, score, cover_count

    def local_search(classes, cover_count, cur_score):
        """Local search: for each modulus, try reassigning residue to maximize gain."""
        moduli = [c[0] for c in classes]
        residues = [c[1] for c in classes]
        n = len(moduli)

        # Precompute position arrays
        pos_arrays = {}
        for i in range(n):
            m = moduli[i]
            pos_arrays[m] = [np.arange(r, base, m) for r in range(m)]

        improved = True
        while improved and time_left() > 4:
            improved = False
            indices = list(range(n))
            random.shuffle(indices)

            # Precompute zeros once per sweep
            zeros = np.where(cover_count == 0)[0]
            if len(zeros) == 0:
                break

            for i in indices:
                if time_left() < 2:
                    break
                m = moduli[i]
                r = residues[i]

                A = np.bincount(zeros % m, minlength=m)
                best_r_in_A = int(np.argmax(A))

                # If current residue is already the best, skip
                if best_r_in_A == r:
                    continue

                old_pos = pos_arrays[m][r]
                L_fixed = int(np.sum(cover_count[old_pos] == 1))

                # Try best new residue
                best_gain = int(A[best_r_in_A]) - L_fixed

                if best_gain > 0:
                    new_pos = pos_arrays[m][best_r_in_A]
                    cover_count[old_pos] -= 1
                    cover_count[new_pos] += 1
                    residues[i] = best_r_in_A
                    cur_score -= best_gain
                    improved = True
                    # Update zeros since coverage changed
                    zeros = np.where(cover_count == 0)[0]
                    if len(zeros) == 0:
                        break

        return [[moduli[i], residues[i]] for i in range(n)], cur_score

    def polish_pass(classes, cover_count, cur_score):
        """Final high-precision polish: try all single changes with exact gain."""
        moduli = [c[0] for c in classes]
        residues = [c[1] for c in classes]
        n = len(moduli)

        pos_arrays = {}
        for i in range(n):
            m = moduli[i]
            pos_arrays[m] = [np.arange(r, base, m) for r in range(m)]

        for _ in range(5):
            if time_left() < 3:
                break
            improved = False
            zeros = np.where(cover_count == 0)[0]
            if len(zeros) == 0:
                break

            for i in range(n):
                if time_left() < 1.5:
                    break
                m = moduli[i]
                r = residues[i]
                old_pos = pos_arrays[m][r]
                L_fixed = int(np.sum(cover_count[old_pos] == 1))

                A = np.bincount(zeros % m, minlength=m)

                best_gain = 0
                best_r = r
                for r2 in range(m):
                    if r2 == r:
                        continue
                    gain = int(A[r2]) - L_fixed
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r2

                if best_gain > 0:
                    new_pos = pos_arrays[m][best_r]
                    cover_count[old_pos] -= 1
                    cover_count[new_pos] += 1
                    residues[i] = best_r
                    cur_score -= best_gain
                    improved = True
                    zeros = np.where(cover_count == 0)[0]
                    if len(zeros) == 0:
                        break

            if not improved:
                break

        return [[moduli[i], residues[i]] for i in range(n)], cur_score

    best_score = base
    best_classes = None
    best_cover_count = None

    # Generate diverse orderings
    orderings = []

    # 1. Ascending (small moduli first)
    orderings.append(sorted(divs))
    # 2. Descending
    orderings.append(sorted(divs, reverse=True))
    # 3. Random orderings (more of them)
    for _ in range(10):
        orderings.append(sorted(divs, key=lambda x: random.random()))
    # 4. Hybrid: small first, then random
    for _ in range(5):
        order = list(divs)
        random.shuffle(order)
        half = len(order) // 2
        order[:half] = sorted(order[:half])
        order[half:] = sorted(order[half:], reverse=True)
        orderings.append(order)
    # 5. Random with small bias: sort by (random, -m) so small moduli tend to go first
    for _ in range(5):
        orderings.append(sorted(divs, key=lambda m: (random.random(), -m)))

    for order in orderings:
        if time_left() < 10:
            break

        classes, score, cover_count = greedy_build(order)
        if score < best_score:
            best_score = score
            best_classes = classes
            best_cover_count = cover_count.copy()

        # Local search
        if time_left() > 10:
            classes, score = local_search(classes, cover_count, score)
            if score < best_score:
                best_score = score
                best_classes = classes
                best_cover_count = cover_count.copy()

        # Polish pass
        if time_left() > 5 and best_classes is not None:
            classes, score = polish_pass(classes, cover_count, score)
            if score < best_score:
                best_score = score
                best_classes = classes
                best_cover_count = cover_count.copy()

    # Final global polish on best solution
    if best_classes is not None and time_left() > 3:
        best_classes, best_score = polish_pass(best_classes, best_cover_count, best_score)

    return best_classes if best_classes is not None else [[3, 0]]
