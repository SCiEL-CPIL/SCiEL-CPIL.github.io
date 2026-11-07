import time
import sys

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    # Strategy: For a given span S, find if there exists a window [L, L+S]
    # that contains distinct multiples for each divisor.
    # 
    # Key optimization: Instead of scanning all L, we can observe that
    # a valid L must be such that for each divisor d, there's a multiple in [L, L+S].
    # This means L must be in [m_d - S, m_d] for some multiple m_d of d.
    # 
    # Even better: We can use the fact that the pattern repeats with period lcm of divisors,
    # but that's too large. Instead, for each S, we scan L but with smart stepping.
    #
    # Actually, the most efficient approach for small S:
    # For each S from 1 upward, scan L from 1 to some bound.
    # For each L, build candidate sets and check for perfect matching via backtracking.
    #
    # To speed up the L scan: note that if we shift L by 1, the candidate sets change
    # predictably. But for S < 50, the number of candidates per divisor is small (1-3),
    # so backtracking is very fast.
    #
    # Critical optimization: Precompute for each possible L (up to a reasonable bound)
    # the candidate multiples for each divisor. But L can be large.
    #
    # Alternative: For each S, the valid L values are those where for every divisor d,
    # ceil(L/d) * d <= L + S, i.e., L mod d <= S (when considering the next multiple).
    # Actually: the smallest multiple of d >= L is m = ceil(L/d)*d.
    # We need m <= L + S, which means L - m <= -S... wait.
    # m - L <= S, where m = ceil(L/d)*d.
    # Let r = L mod d. Then m = L + (d - r) if r != 0, else m = L.
    # So m - L = (d - r) % d. We need (d - r) % d <= S.
    # If r = 0, m - L = 0 <= S (always true for S >= 0).
    # If r > 0, we need d - r <= S, i.e., r >= d - S.
    # So for divisor d, L is valid iff L mod d = 0 or L mod d >= d - S.
    # Equivalently: L mod d NOT IN [1, d-S-1] (when S < d).
    #
    # This gives us a way to quickly filter L values!
    # For a given S, L must satisfy for ALL divisors d: L mod d == 0 or L mod d >= d - S.
    #
    # We can use this to skip many L values. Let's implement this.

    max_S = 50  # We expect to find a solution with S <= 50
    max_L = 50000  # Upper bound for L scan per S

    for S in range(1, max_S + 1):
        if time.time() > deadline:
            break

        # For this S, scan L and check the modular condition first
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break

            if L + S > MAXV:
                break

            # Quick filter: check modular conditions
            valid_mod = True
            for d in DIVISORS:
                r = L % d
                if r != 0 and r < d - S:
                    valid_mod = False
                    break
            if not valid_mod:
                continue

            # Build candidates for each divisor
            candidates = []
            for d in DIVISORS:
                k = (L + d - 1) // d
                m = k * d
                cands = []
                while m <= L + S:
                    cands.append(m)
                    m += d
                if not cands:
                    valid_mod = False
                    break
                candidates.append(cands)

            if not valid_mod:
                continue

            # Backtracking to find distinct assignment
            # Sort by number of candidates for better pruning
            indices = sorted(range(N), key=lambda i: len(candidates[i]))
            assignment = [None] * N
            used = set()

            def backtrack(idx):
                if idx == len(indices):
                    return True
                i = indices[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[i] = None
                return False

            if backtrack(0):
                result = [int(assignment[i]) for i in range(N)]
                return result

    # Fallback: return divisors themselves (not optimal but valid)
    return DIVISORS[:]
