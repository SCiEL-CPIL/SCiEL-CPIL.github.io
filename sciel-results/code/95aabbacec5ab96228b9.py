import math
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill

    def check_pair(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0

    def check_all(classes):
        n = len(classes)
        for i in range(n):
            for j in range(i + 1, n):
                if not check_pair(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                    return False
        return True

    # Seed solution: powers of 2
    seed = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    best = seed
    best_count = len(seed)

    # Strategy: Use all even moduli (64 candidates from 2 to 128)
    # All pairs have gcd >= 2, so we need to assign residues carefully
    even_moduli = [m for m in range(2, 129) if m % 2 == 0]
    n_mod = len(even_moduli)

    # Precompute gcd matrix
    gcd_mat = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            gcd_mat[i][ j] = math.gcd(even_moduli[i], even_moduli[j])
            gcd_mat[j][i] = gcd_mat[i][j]

    # Variable ordering: moduli with higher max gcd with others are more constrained
    def max_gcd_score(idx):
        return max(gcd_mat[idx][j] for j in range(n_mod) if j != idx)

    var_order = sorted(range(n_mod), key=max_gcd_score, reverse=True)

    # Backtracking search with MRV and forward checking
    def backtrack_search(time_limit=30.0):
        t0 = time.time()
        n = n_mod
        assignment = [None] * n
        # For each variable, track which residues are still possible
        # Actually, let's do simple backtracking with pruning

        # Precompute for each pair (i,j) the forbidden residue differences mod gcd
        # For variable i with modulus m_i, when assigning residue r_i:
        # For each already-assigned j: need (r_i - r_j) % gcd(i,j) != 0

        # Use recursive backtracking
        # Order variables by constraint
        order = var_order  # most constrained first

        def bt(pos, current_count):
            if time.time() - t0 > time_limit:
                return False
            if pos == n:
                return True
            idx = order[pos]
            m = even_moduli[idx]
            
            # Try residues in a smart order
            # Residues that are "different" from already assigned ones mod their gcds
            # Try 0, 1, 2, ... but skip those that conflict
            for r in range(m):
                ok = True
                for prev_pos in range(pos):
                    pidx = order[prev_pos]
                    if assignment[pidx] is not None:
                        g = gcd_mat[idx][pidx]
                        if (r - assignment[pidx]) % g == 0:
                            ok = False
                            break
                if ok:
                    assignment[idx] = r
                    if bt(pos + 1, current_count + 1):
                        return True
                    assignment[idx] = None
            
            # Try skipping this modulus
            assignment[idx] = None
            return bt(pos + 1, current_count)

        bt(0, 0)
        
        # Collect results
        result = []
        for i in range(n):
            if assignment[i] is not None:
                result.append([assignment[i], even_moduli[i]])
        return result

    # Try backtracking on all even moduli
    if time.time() < deadline - 15:
        result = backtrack_search(time_limit=min(25.0, deadline - time.time() - 5))
        if len(result) > best_count:
            best = result
            best_count = len(result)

    # If backtracking didn't find a full solution, try greedy + local search
    if best_count < n_mod and time.time() < deadline - 10:
        # Greedy with multiple orderings
        def greedy_build(order, time_limit=5.0):
            t0 = time.time()
            selected = []
            for idx in order:
                if time.time() - t0 > time_limit:
                    break
                m = even_moduli[idx]
                for r in range(m):
                    ok = True
                    for (r_s, m_s) in selected:
                        g = math.gcd(m, m_s)
                        if (r - r_s) % g == 0:
                            ok = False
                            break
                    if ok:
                        selected.append([r, m])
                        break
            return selected

        # Try different orderings
        orderings = [
            var_order,
            list(range(n_mod)),  # ascending
            list(range(n_mod - 1, -1, -1)),  # descending
            sorted(range(n_mod), key=lambda i: even_moduli[i] // 2, reverse=True),  # by odd part
        ]

        for order in orderings:
            if time.time() > deadline - 5:
                break
            result = greedy_build(order, time_limit=min(3.0, deadline - time.time() - 3))
            if len(result) > best_count:
                best = result
                best_count = len(result)

        # Random restarts
        random.seed(42)
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
            shuffled = list(range(n_mod))
            random.shuffle(shuffled)
            result = greedy_build(shuffled, time_limit=1.0)
            if len(result) > best_count:
                best = result
                best_count = len(result)

    # Try odd moduli too (multiples of 3, 5, etc.)
    # But even moduli give 64 candidates, which is likely the max
    # Let's also try a mixed approach if we haven't filled all even moduli

    # Final verification
    if not check_all(best):
        # Fallback to seed
        best = seed
        best_count = len(seed)

    return best
