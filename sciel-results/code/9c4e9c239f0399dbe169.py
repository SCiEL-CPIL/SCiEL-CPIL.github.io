"""Backtracking CSP over 3x3 sum grid with difference-structure constraints."""
import math
import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    # Precompute small primes up to 256 (max sum = 128+128)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
                    113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
                    181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241,
                    251]

    # Precompute prime mask for each number 1..256
    prime_mask = [0] * 257
    for n in range(2, 257):
        m = 0
        for i, p in enumerate(small_primes):
            if p > n:
                break
            if n % p == 0:
                m |= (1 << i)
        prime_mask[n] = m

    def pairwise_coprime(sums_list):
        """Check if all values in sums_list are pairwise coprime using bitmask AND."""
        n = len(sums_list)
        for i in range(n):
            mi = prime_mask[sums_list[i]]
            for j in range(i + 1, n):
                if mi & prime_mask[sums_list[j]]:
                    return False
        return True

    def evaluate(A, B):
        """Return (cardinality, is_valid) for a pair of 3-element lists."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        card = len(distinct)
        if card < 5:
            return card, False
        return card, pairwise_coprime(distinct)

    # --- CSP Backtracking over the 3x3 sum grid ---
    # Key insight: If S[i][j] = a_i + b_j, then:
    #   a_i - a_k = S[i][0] - S[k][0]  (for any fixed column j=0)
    #   b_j - b_l = S[0][j] - S[0][l]  (for any fixed row i=0)
    # So the entire matrix is determined by S[0][0], the row differences, and column differences.
    # We can enumerate valid 9-element coprime sets and check which form a valid 3x3 grid.

    # Precompute: for each pair of values (x, y) in [2, 256], whether gcd(x,y)==1
    # Using bitmask: gcd(x,y)==1 iff prime_mask[x] & prime_mask[y] == 0

    # Strategy: Enumerate all 9-tuples of pairwise-coprime values in [2, 256],
    # then check if they can be arranged as a 3x3 grid S[i][j] = a_i + b_j
    # with a_i, b_j in [1, 128] and distinct.

    # But C(255, 9) is too large. Instead, use the structure:
    # S[i][j] - S[i][0] = b_j - b_0  (same for all i)
    # S[i][0] - S[0][0] = a_i - a_0  (same for all j)
    # So: S[i][j] = S[0][0] + (a_i - a_0) + (b_j - b_0)
    # Let d_i = a_i - a_0 (i=1,2), e_j = b_j - b_0 (j=1,2)
    # Then S[i][j] = S[0][0] + d_i + e_j where d_0=e_0=0

    # We need to find: S00, d1, d2, e1, e2 such that:
    # 1. All 9 values S00 + d_i + e_j are in [2, 256] and pairwise coprime
    # 2. a_0 = S00 - b_0, a_1 = a_0 + d1, a_2 = a_0 + d2 are in [1,128] and distinct
    # 3. b_0, b_1 = b_0 + e1, b_2 = b_0 + e2 are in [1,128] and distinct
    # 4. All a_i distinct, all b_j distinct

    # Let's enumerate d1, d2, e1, e2 (differences) and S00, then solve for b_0.
    # From constraints:
    #   1 <= b_0 <= 128
    #   1 <= S00 - b_0 <= 128  =>  S00 - 128 <= b_0 <= S00 - 1
    #   1 <= S00 - b_0 + d1 <= 128
    #   1 <= S00 - b_0 + d2 <= 128
    #   1 <= b_0 + e1 <= 128
    #   1 <= b_0 + e2 <= 128
    #   2 <= S00 + d_i + e_j <= 256 for all i,j

    # Range of differences: d_i, e_j in [-127, 127] (since a_i, b_j in [1,128])
    # But we need distinctness, so d1 != d2, d1 != 0, d2 != 0, e1 != e2, e1 != 0, e2 != 0

    # Precompute valid difference pairs
    # d1, d2: distinct nonzero values in [-127, 127]
    # e1, e2: distinct nonzero values in [-127, 127]

    # For efficiency, limit the search space:
    # The 9 sums must be in [2, 256]. If S00 is around 128, then d_i + e_j should be small.
    # Let's focus on |d_i|, |e_j| <= 50 initially, then expand if needed.

    def find_solution_with_csp():
        """Backtracking search for a valid 9-sum coprime configuration."""
        # Try different ranges of differences
        for max_diff in [10, 20, 30, 50, 80, 127]:
            if time.time() > deadline - 10:
                break

            diffs = list(range(-max_diff, max_diff + 1))
            diffs.remove(0)

            # Enumerate (d1, d2, e1, e2) with d1 != d2, e1 != e2
            # To reduce symmetry, fix d1 < d2 and e1 < e2
            for d1 in diffs:
                for d2 in diffs:
                    if d2 <= d1:
                        continue
                    for e1 in diffs:
                        for e2 in diffs:
                            if e2 <= e1:
                                continue

                            # Check: all 9 sums S00 + d_i + e_j must be in [2, 256]
                            # and pairwise coprime.
                            # The 9 values are: S00 + di + ej for di in [0,d1,d2], ej in [0,e1,e2]

                            # For a given (d1,d2,e1,e2), the 9 offsets are fixed:
                            offsets = [0, d1, d2]
                            eoffsets = [0, e1, e2]
                            sum_offsets = [di + ej for di in offsets for ej in eoffsets]

                            # We need S00 + s in [2, 256] for all s in sum_offsets
                            # => S00 >= 2 - min(sum_offsets) and S00 <= 256 - max(sum_offsets)
                            min_off = min(sum_offsets)
                            max_off = max(sum_offsets)
                            s00_min = max(2 - min_off, 1)  # also need a_0 = S00 - b_0 >= 1
                            s00_max = min(256 - max_off, 255)

                            if s00_min > s00_max:
                                continue

                            # Also need: a_i = S00 - b_0 + d_i in [1, 128]
                            # and b_j = b_0 + e_j in [1, 128]
                            # Let b_0 = t. Then:
                            #   1 <= t <= 128
                            #   1 <= S00 - t <= 128  =>  S00 - 128 <= t <= S00 - 1
                            #   1 <= S00 - t + d1 <= 128
                            #   1 <= S00 - t + d2 <= 128
                            #   1 <= t + e1 <= 128
                            #   1 <= t + e2 <= 128

                            # So t must satisfy:
                            #   t >= max(1, S00 - 128, S00 - 128 - d1, S00 - 128 - d2, 1 - e1, 1 - e2)
                            #   t <= min(128, S00 - 1, S00 - 1 - d1, S00 - 1 - d2, 128 - e1, 128 - e2)

                            # For each valid S00, check if there exists t in the valid range
                            # such that all 9 sums are pairwise coprime.

                            # To speed up, pre-check: the 9 sums are S00 + sum_offsets[k]
                            # They must be pairwise coprime. This depends only on S00 and the offsets.

                            # Try a few S00 values in the valid range
                            s00_range = range(s00_min, s00_max + 1)
                            if len(s00_range) > 200:
                                # Sample S00 values
                                step = len(s00_range) // 200 + 1
                                s00_candidates = list(s00_range)[::step]
                            else:
                                s00_candidates = list(s00_range)

                            random.shuffle(s00_candidates)

                            for S00 in s00_candidates:
                                if time.time() > deadline - 5:
                                    return None

                                # Compute the 9 sums
                                nine_sums = [S00 + di + ej for di in offsets for ej in eoffsets]

                                # Check all in [2, 256]
                                if any(s < 2 or s > 256 for s in nine_sums):
                                    continue

                                # Check pairwise coprime
                                if not pairwise_coprime(nine_sums):
                                    continue

                                # Now find valid b_0 = t
                                t_min = max(1, S00 - 128, S00 - 128 - d1, S00 - 128 - d2,
                                            1 - e1, 1 - e2)
                                t_max = min(128, S00 - 1, S00 - 1 - d1, S00 - 1 - d2,
                                            128 - e1, 128 - e2)

                                if t_min > t_max:
                                    continue

                                # Find a valid t
                                for t in range(t_min, t_max + 1):
                                    # Compute A and B
                                    b0 = t
                                    b1 = t + e1
                                    b2 = t + e2
                                    a0 = S00 - t
                                    a1 = a0 + d1
                                    a2 = a0 + d2

                                    A = [a0, a1, a2]
                                    B = [b0, b1, b2]

                                    # Check all in [1, 128] and distinct
                                    if not all(1 <= x <= 128 for x in A + B):
                                        continue
                                    if len(set(A)) != 3 or len(set(B)) != 3:
                                        continue

                                    # Verify
                                    sums = [a + b for a in A for b in B]
                                    distinct = list(set(sums))
                                    if len(distinct) != 9:
                                        continue
                                    if not pairwise_coprime(distinct):
                                        continue

                                    return [A, B]

        return None

    # --- Phase 1: CSP backtracking ---
    result = find_solution_with_csp()
    if result is not None:
        return result

    # --- Phase 2: Fallback to random + local search ---
    odds = [i for i in range(1, 129) if i % 2 != 0]
    evens = [i for i in range(1, 129) if i % 2 == 0]

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_card = 5

    # Random search
    for _ in range(5000):
        if time.time() > deadline - 10:
            break
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        card, valid = evaluate(A, B)
        if valid and card > best_card:
            best_A, best_B = A, B
            best_card = card
            if card == 9:
                return [best_A, best_B]

    # Local search
    def local_search(A, B, remaining_time):
        current_A = list(A)
        current_B = list(B)
        curr_card, curr_valid = evaluate(current_A, current_B)
        if not curr_valid:
            return list(A), list(B), curr_card

        all_nums = list(range(1, 129))
        no_improve = 0
        max_no_improve = 15

        while time.time() < start_time + remaining_time:
            if curr_card == 9:
                return list(current_A), list(current_B), curr_card

            improved = False

            for _ in range(80):
                idx = random.randint(0, 2)
                old_val = current_A[idx]
                new_val = random.choice(all_nums)
                if new_val == old_val:
                    continue
                test_A = list(current_A)
                test_A[idx] = new_val
                if len(set(test_A)) != 3:
                    continue
                card, valid = evaluate(test_A, current_B)
                if valid and card > curr_card:
                    current_A = test_A
                    curr_card, curr_valid = card, True
                    improved = True
                    no_improve = 0
                    break
                elif valid and card == curr_card and random.random() < 0.05:
                    current_A = test_A
                    curr_card, curr_valid = card, True
                    improved = True
                    no_improve = 0
                    break

            for _ in range(80):
                idx = random.randint(0, 2)
                old_val = current_B[idx]
                new_val = random.choice(all_nums)
                if new_val == old_val:
                    continue
                test_B = list(current_B)
                test_B[idx] = new_val
                if len(set(test_B)) != 3:
                    continue
                card, valid = evaluate(current_A, test_B)
                if valid and card > curr_card:
                    current_B = test_B
                    curr_card, curr_valid = card, True
                    improved = True
                    no_improve = 0
                    break
                elif valid and card == curr_card and random.random() < 0.05:
                    current_B = test_B
                    curr_card, curr_valid = card, True
                    improved = True
                    no_improve = 0
                    break

            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    break

        return list(current_A), list(current_B), curr_card

    # Try local search from best
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        la, lb, card = local_search(best_A, best_B, deadline - start_time)
        if card > best_card:
            best_A, best_B = la, lb
            best_card = card
            if card == 9:
                break

    return [best_A, best_B]
