import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit

    # Sieve primes up to 256 (max sum = 128+128)
    max_sum = 256
    is_prime_arr = [True] * (max_sum + 1)
    is_prime_arr[0] = False
    is_prime_arr[1] = False
    for i in range(2, int(math.isqrt(max_sum)) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, max_sum + 1, i):
                is_prime_arr[j] = False

    def is_prime(n):
        return 0 <= n <= max_sum and is_prime_arr[n]

    def is_pairwise_coprime(nums):
        if len(nums) < 2:
            return True
        nums = list(nums)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True

    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) == 0:
            return -1
        if not is_pairwise_coprime(sums):
            return -1
        return len(sums)

    # --- Phase 1: Fast prime-sieve search (kept from current design) ---
    # Key insight: if all 9 sums are prime, they're automatically pairwise coprime.
    # For a fixed A = {a1, a2, a3}, a valid b must satisfy:
    #   is_prime(a1+b) and is_prime(a2+b) and is_prime(a3+b).
    # Iterate over all A (C(128,3) ≈ 343k), collect valid b's, pick 3.
    # Cost: ~343k * 128 ≈ 44M prime lookups — fast.

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = evaluate(best_A, best_B)

    all_A = list(combinations(range(1, 129), 3))

    for A in all_A:
        if time.time() > deadline:
            break

        a1, a2, a3 = A
        valid_bs = []
        for b in range(1, 129):
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if s1 <= max_sum and s2 <= max_sum and s3 <= max_sum:
                if is_prime_arr[s1] and is_prime_arr[s2] and is_prime_arr[s3]:
                    valid_bs.append(b)

        if len(valid_bs) >= 3:
            # All 9 sums are prime => pairwise coprime and distinct
            B = [valid_bs[0], valid_bs[1], valid_bs[2]]
            score = evaluate(A, B)
            if score == 9:
                return [list(A), B]
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = B

    # --- Phase 2: Simulated annealing over difference-vector parameterization ---
    # The 9 sums are s_{ij} = s00 + i*u + j*v for i,j in {0,1,2}
    # where s00 = A[0]+B[0], u = A[1]-A[0] (step in A), v = B[1]-B[0] (step in B).
    # But A and B each have 3 elements, not necessarily in arithmetic progression.
    # General parameterization:
    #   A = [a0, a0+da1, a0+da1+da2]  (da1, da2 are the two gaps)
    #   B = [b0, b0+db1, b0+db1+db2]
    #   s_{ij} = a_i + b_j
    # So the 9 sums are determined by (a0, da1, da2, b0, db1, db2).
    #
    # Objective: maximize distinct sums that are pairwise coprime.
    # Surrogate: minimize total GCD-violation penalty while rewarding distinctness.
    #   penalty = sum over pairs (i,j) of (gcd(s_i, s_j) - 1)  [0 if coprime, >0 otherwise]
    #   distinctness = 9 - len(set(sums))
    #   total_cost = penalty + 10 * distinctness  (heavy penalty for duplicates)
    #
    # SA moves: perturb one of the 6 parameters by a small random amount,
    # keeping all values in [1, 128] and all 3 elements distinct.

    random.seed(42)

    def params_to_AB(params):
        """Convert (a0, da1, da2, b0, db1, db2) to sorted A, B lists."""
        a0, da1, da2, b0, db1, db2 = params
        A = [a0, a0 + da1, a0 + da1 + da2]
        B = [b0, b0 + db1, b0 + db1 + db2]
        # Check all in [1, 128]
        for x in A + B:
            if x < 1 or x > 128:
                return None, None
        # Check distinctness
        if len(set(A)) < 3 or len(set(B)) < 3:
            return None, None
        return sorted(A), sorted(B)

    def compute_sums(A, B):
        """Compute the 9 sums as a flat list."""
        return [a + b for a in A for b in B]

    def sa_cost(A, B):
        """Compute the SA objective: GCD violations + duplicate penalty."""
        sums = compute_sums(A, B)
        # GCD violation count
        n = len(sums)
        gcd_viol = 0
        for i in range(n):
            for j in range(i + 1, n):
                g = math.gcd(sums[i], sums[j])
                if g > 1:
                    gcd_viol += g - 1  # heavier penalty for larger GCDs
        # Duplicate penalty
        distinct_count = len(set(sums))
        dup_penalty = (9 - distinct_count) * 10
        return gcd_viol + dup_penalty

    def is_valid_AB(A, B):
        """Check if A, B are valid (3 distinct each, in [1,128])."""
        if len(A) != 3 or len(B) != 3:
            return False
        if len(set(A)) < 3 or len(set(B)) < 3:
            return False
        for x in A + B:
            if x < 1 or x > 128:
                return False
        return True

    # Initialize SA with the best found so far, or a random start
    if best_score >= 9:
        return [best_A, best_B]

    # Start from a random valid configuration
    cur_A = list(random.sample(range(1, 129), 3))
    cur_B = list(random.sample(range(1, 129), 3))
    cur_cost = sa_cost(cur_A, cur_B)

    # Also try starting from best_A, best_B
    if best_score > 0:
        alt_cost = sa_cost(best_A, best_B)
        if alt_cost < cur_cost:
            cur_A = list(best_A)
            cur_B = list(best_B)
            cur_cost = alt_cost

    best_sa_A = list(cur_A)
    best_sa_B = list(cur_B)
    best_sa_cost = cur_cost

    # SA parameters
    T_start = 50.0
    T_end = 0.1
    n_iter = 500000
    # We'll run for as many iterations as time allows

    for it in range(n_iter):
        if time.time() > deadline - 2.0:
            break

        # Cooling schedule
        frac = it / n_iter
        T = T_start * (T_end / T_start) ** frac

        # Propose a move: perturb one element of A or B
        if random.random() < 0.5:
            idx = random.randint(0, 2)
            delta = random.randint(-15, 15)
            new_val = cur_A[idx] + delta
            if 1 <= new_val <= 128:
                new_A = list(cur_A)
                new_A[idx] = new_val
                if len(set(new_A)) == 3:
                    new_B = cur_B
                else:
                    continue
            else:
                continue
        else:
            idx = random.randint(0, 2)
            delta = random.randint(-15, 15)
            new_val = cur_B[idx] + delta
            if 1 <= new_val <= 128:
                new_B = list(cur_B)
                new_B[idx] = new_val
                if len(set(new_B)) == 3:
                    new_A = cur_A
                else:
                    continue
            else:
                continue

        new_cost = sa_cost(new_A, new_B)
        delta_cost = new_cost - cur_cost

        # Accept if better, or with probability exp(-delta/T)
        if delta_cost < 0 or (delta_cost > 0 and random.random() < math.exp(-delta_cost / max(T, 1e-10))):
            cur_A = new_A
            cur_B = new_B
            cur_cost = new_cost

            if cur_cost < best_sa_cost:
                best_sa_cost = cur_cost
                best_sa_A = list(cur_A)
                best_sa_B = list(cur_B)

            # Check if we found a perfect solution
            if cur_cost == 0:
                # Verify: all 9 sums pairwise coprime and distinct
                sums = compute_sums(cur_A, cur_B)
                if len(set(sums)) == 9 and is_pairwise_coprime(sums):
                    return [sorted(cur_A), sorted(cur_B)]

    # After SA, check best_sa result
    if best_sa_cost == 0:
        sums = compute_sums(best_sa_A, best_sa_B)
        if len(set(sums)) == 9 and is_pairwise_coprime(sums):
            return [sorted(best_sa_A), sorted(best_sa_B)]

    # Compare with phase-1 best
    sa_score = evaluate(best_sa_A, best_sa_B)
    if sa_score > best_score:
        best_A = list(best_sa_A)
        best_B = list(best_sa_B)
        best_score = sa_score

    # --- Phase 3: Local refinement around the best SA solution ---
    # Try all single-element perturbations of the best solution to see if we can
    # reduce cost further (hill-climbing polish).
    if best_score < 9:
        polish_A = list(best_A)
        polish_B = list(best_B)
        polish_cost = sa_cost(polish_A, polish_B)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for target in ['A', 'B']:
                for idx in range(3):
                    for delta in range(-20, 21):
                        if delta == 0:
                            continue
                        new_val = (polish_A if target == 'A' else polish_B)[idx] + delta
                        if not (1 <= new_val <= 128):
                            continue
                        if target == 'A':
                            new_A = list(polish_A)
                            new_A[idx] = new_val
                            if len(set(new_A)) != 3:
                                continue
                            new_B = polish_B
                        else:
                            new_B = list(polish_B)
                            new_B[idx] = new_val
                            if len(set(new_B)) != 3:
                                continue
                            new_A = polish_A
                        new_cost = sa_cost(new_A, new_B)
                        if new_cost < polish_cost:
                            polish_A, polish_B = new_A, new_B
                            polish_cost = new_cost
                            improved = True
                            if polish_cost == 0:
                                sums = compute_sums(polish_A, polish_B)
                                if len(set(sums)) == 9 and is_pairwise_coprime(sums):
                                    return [sorted(polish_A), sorted(polish_B)]

        ps_score = evaluate(polish_A, polish_B)
        if ps_score > best_score:
            best_A = list(polish_A)
            best_B = list(polish_B)
            best_score = ps_score

    return [best_A, best_B]
