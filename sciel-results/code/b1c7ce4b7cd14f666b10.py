import time
import random
import math

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 85

    # Precompute comparability bitsets for all masks
    # comp_bits[mask] is a bitset where bit s is set if mask and s are comparable
    comp_bits = [0] * total_masks
    for mask in range(total_masks):
        bits = 0
        for s in range(total_masks):
            if (mask & ~s) == 0 or (s & ~mask) == 0:
                bits |= (1 << s)
        comp_bits[mask] = bits

    def is_comparable(a, b):
        return (a & ~b) == 0 or (b & ~a) == 0

    def greedy_init():
        family = []
        fam_set = set()
        scores = [0] * total_masks

        family.append(0)
        fam_set.add(0)
        # Update scores: all sets comparable to 0 (which is all sets since 0 is subset of everything)
        for s in range(total_masks):
            scores[s] += 1

        while len(family) < m:
            best_score = -1
            best_mask = -1
            for s in range(total_masks):
                if s in fam_set:
                    continue
                if scores[s] > best_score:
                    best_score = scores[s]
                    best_mask = s
            if best_mask == -1:
                break
            family.append(best_mask)
            fam_set.add(best_mask)
            for s in range(total_masks):
                if (s & ~best_mask) == 0 or (best_mask & ~s) == 0:
                    scores[s] += 1
        return family, fam_set, scores

    def compute_score(family, scores):
        # scores[s] for s in family counts comparable sets in family including self
        # Sum over family of (scores[s] - 1) / 2 = total comparable pairs
        total = 0
        for s in family:
            total += scores[s] - 1
        return total // 2

    def local_search(family, fam_set, scores, T_start, T_end, cooling, time_limit):
        T = T_start
        best_family = family[:]
        best_score_val = compute_score(family, scores)

        end_time = time.time() + time_limit

        while time.time() < end_time and T > T_end:
            if random.random() < 0.6:
                # 1-swap
                i = random.randint(0, m - 1)
                A = family[i]
                while True:
                    B = random.randint(0, total_masks - 1)
                    if B not in fam_set:
                        break

                # Gain calculation using precomputed comp_bits
                # comp(A, family\{A}) = popcount(comp_bits[A] & fam_mask) - 1
                # comp(B, family\{A}) = popcount(comp_bits[B] & (fam_mask without A))
                # But we track scores incrementally

                comp_AB = 1 if is_comparable(A, B) else 0
                # scores[A] includes A itself and all comparable in family
                # scores[B] is count of comparable in family (B not in family)
                # After removing A: comp(B, R) = scores[B] - comp_AB
                # comp(A, R) = scores[A] - 1
                gain = (scores[B] - comp_AB) - (scores[A] - 1)

                if gain > 0 or random.random() < math.exp(gain / T):
                    fam_set.remove(A)
                    fam_set.add(B)
                    family[i] = B

                    # Update scores
                    for s in range(total_masks):
                        if (s & ~A) == 0 or (A & ~s) == 0:
                            scores[s] -= 1
                        if (s & ~B) == 0 or (B & ~s) == 0:
                            scores[s] += 1

                    new_score = compute_score(family, scores)
                    if new_score > best_score_val:
                        best_score_val = new_score
                        best_family = family[:]
            else:
                # 2-swap
                i, j = random.sample(range(m), 2)
                A, B = family[i], family[j]

                while True:
                    C = random.randint(0, total_masks - 1)
                    D = random.randint(0, total_masks - 1)
                    if C not in fam_set and D not in fam_set and C != D:
                        break

                comp_AB = 1 if is_comparable(A, B) else 0
                comp_CA = 1 if is_comparable(C, A) else 0
                comp_CB = 1 if is_comparable(C, B) else 0
                comp_DA = 1 if is_comparable(D, A) else 0
                comp_DB = 1 if is_comparable(D, B) else 0
                comp_CD = 1 if is_comparable(C, D) else 0

                old_val = scores[A] + scores[B] - 2 - comp_AB
                new_val = scores[C] + scores[D] - comp_CA - comp_CB - comp_DA - comp_DB + comp_CD

                gain = new_val - old_val

                if gain > 0 or random.random() < math.exp(gain / T):
                    fam_set.remove(A)
                    fam_set.remove(B)
                    fam_set.add(C)
                    fam_set.add(D)
                    family[i] = C
                    family[j] = D

                    for s in range(total_masks):
                        if (s & ~A) == 0 or (A & ~s) == 0:
                            scores[s] -= 1
                        if (s & ~B) == 0 or (B & ~s) == 0:
                            scores[s] -= 1
                        if (s & ~C) == 0 or (C & ~s) == 0:
                            scores[s] += 1
                        if (s & ~D) == 0 or (D & ~s) == 0:
                            scores[s] += 1

                    new_score = compute_score(family, scores)
                    if new_score > best_score_val:
                        best_score_val = new_score
                        best_family = family[:]

            T *= cooling

        return best_family, best_score_val

    # Multi-start with different seeds
    best_overall_family = None
    best_overall_score = -1

    # Start 1: Greedy
    random.seed(42)
    family1, fam_set1, scores1 = greedy_init()
    fam1, score1 = local_search(family1, fam_set1, scores1, 50.0, 0.01, 0.99, 25)
    if score1 > best_overall_score:
        best_overall_score = score1
        best_overall_family = fam1

    if time.time() > deadline:
        return _convert(best_overall_family)

    # Start 2: Different random seed
    random.seed(123)
    family2, fam_set2, scores2 = greedy_init()
    fam2, score2 = local_search(family2, fam_set2, scores2, 40.0, 0.01, 0.99, 20)
    if score2 > best_overall_score:
        best_overall_score = score2
        best_overall_family = fam2

    if time.time() > deadline:
        return _convert(best_overall_family)

    # Start 3: Polish the best found with lower temperature
    # Reconstruct scores for best_overall_family
    fam_set3 = set(best_overall_family)
    scores3 = [0] * total_masks
    for s in range(total_masks):
        count = 0
        for f in best_overall_family:
            if (s & ~f) == 0 or (f & ~s) == 0:
                count += 1
        scores3[s] = count

    # If s is in family, it counts itself
    # For s not in family, it doesn't count itself
    # But our compute_score assumes scores[s] includes self if s in family
    # Let me fix: rebuild scores properly
    scores3 = [0] * total_masks
    for f in best_overall_family:
        for s in range(total_masks):
            if (s & ~f) == 0 or (f & ~s) == 0:
                scores3[s] += 1

    fam3, score3 = local_search(best_overall_family[:], fam_set3, scores3, 10.0, 0.001, 0.995, 15)
    if score3 > best_overall_score:
        best_overall_score = score3
        best_overall_family = fam3

    return _convert(best_overall_family)

def _convert(family_masks):
    result = []
    for mask in family_masks:
        s = []
        for b in range(12):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    return result
