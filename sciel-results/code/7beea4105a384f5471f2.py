"""Find worst interval start for distinct-multiple matching with n=12."""
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_lcm_table(n):
    L = 1
    for i in range(2, n + 1):
        L = _lcm(L, i)
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit

    n = 12
    LCM = _build_lcm_table(n)  # 27720
    max_L = 144  # f(12, m) is at most 144

    # Precompute: for each i in 1..n and each residue rem in 0..i-1,
    # the list of offsets j in 1..max_L such that j % i == rem
    # This lets us quickly get candidates for a given m mod i
    valid_offsets = {}
    for i in range(1, n + 1):
        for rem in range(i):
            valid_offsets[(i, rem)] = [j for j in range(1, max_L + 1) if j % i == rem]

    def compute_f(r, n, max_L, valid_offsets):
        """Compute f(n, r): least positive L such that (r, r+L] has distinct b_i with i|b_i.
        
        b_i = r + j_i where j_i is in [1, L] and j_i ≡ -r (mod i), all j_i distinct.
        """
        for L in range(1, max_L + 1):
            # Build candidate lists for each i
            # For i, candidates are j in [1, L] with j % i == (-r) % i
            items = []
            for i in range(1, n + 1):
                rem = (-r) % i
                cands = [j for j in valid_offsets[(i, rem)] if j <= L]
                if not cands:
                    return L + 1  # not feasible at this L, continue
                items.append((i, cands))

            # Sort by number of candidates (ascending) for better pruning
            items.sort(key=lambda x: len(x[1]))

            # Backtracking search for system of distinct representatives
            used = [False] * (L + 1)

            def dfs(idx):
                if idx == n:
                    return True
                i, cands = items[idx]
                for j in cands:
                    if not used[j]:
                        used[j] = True
                        if dfs(idx + 1):
                            return True
                        used[j] = False
                return False

            if dfs(0):
                return L

        return max_L  # should not happen

    # Compute f(r) for all residues r mod LCM
    best_m = 0
    best_f = 0

    for r in range(LCM):
        if time.time() > deadline:
            break
        f_val = compute_f(r, n, max_L, valid_offsets)
        if f_val > best_f:
            best_f = f_val
            best_m = r

    # best_m is in [0, LCM-1] which is within [0, 1000000]
    result = int(best_m)
    return result

construct()
