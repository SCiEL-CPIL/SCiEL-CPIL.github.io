"""Rewritten program: faster incremental validity + smarter search for max cardinality."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 84.0  # safety margin before 90s kill

    N = 192
    MAX_K = 48

    # Precompute for each x in 1..N, a list of residues of all values 1..N
    # residues[x] = dict mapping r -> list of values v in 1..N with v % x == r
    residues = {}
    for x in range(1, N + 1):
        residues[x] = {}
        for v in range(1, N + 1):
            r = v % x
            if r not in residues[x]:
                residues[x][r] = []
            residues[x][r].append(v)

    def can_add_fast(a, selected):
        """
        Check if 'a' can be added to 'selected'.
        Two conditions:
        1. a does not divide sum of any 3 distinct elements in selected.
        2. For each x in selected, x does not divide (a + b + c) for distinct b,c in selected\{x}.
        """
        sel = selected
        n = len(sel)

        # Condition 1: a does not divide sum of any 3 distinct elements in selected
        if n >= 3:
            # Build residue lookup for selected elements mod a
            res_lookup = {}
            for idx, v in enumerate(sel):
                r = v % a
                if r not in res_lookup:
                    res_lookup[r] = []
                res_lookup[r].append(idx)

            for i in range(n):
                b = sel[i]
                for j in range(i + 1, n):
                    c = sel[j]
                    need_res = (-(b + c)) % a
                    if need_res in res_lookup:
                        # Check if there's an element with that residue that's not b or c
                        lst = res_lookup[need_res]
                        for k in lst:
                            if k != i and k != j:
                                return False

        # Condition 2: For each x in sel, x does not divide (a + b + c) for distinct b,c in sel\{x}
        for idx_x, x in enumerate(sel):
            # We need: for no distinct b, c in sel\{x} does (a + b + c) % x == 0
            # i.e., for no distinct b, c in sel\{x} does (b + c) % x == (-a) % x
            need_res = (-a) % x

            # Build residue lookup for sel\{x}
            res_lookup = {}
            for idx, v in enumerate(sel):
                if idx == idx_x:
                    continue
                r = v % x
                if r not in res_lookup:
                    res_lookup[r] = []
                res_lookup[r].append(idx)

            # For each b in sel\{x}, check if there exists c in sel\{x,b} with (b+c) % x == need_res
            # i.e., c % x == (need_res - b) % x
            for idx_b, b in enumerate(sel):
                if idx_b == idx_x:
                    continue
                need_c_res = (need_res - b) % x
                if need_c_res in res_lookup:
                    lst = res_lookup[need_c_res]
                    for idx_c in lst:
                        if idx_c != idx_b:
                            return False

        return True

    def full_check(candidate_set):
        """Full validation of a set."""
        n = len(candidate_set)
        for i, x in enumerate(candidate_set):
            others = [candidate_set[j] for j in range(n) if j != i]
            if len(others) < 3:
                continue
            for combo in combinations(others, 3):
                if sum(combo) % x == 0:
                    return False
        return True

    # Phase 1: Greedy from large to small
    selected = []
    for x in range(N, 0, -1):
        if time.time() > deadline - 5.0:
            break
        if can_add_fast(x, selected):
            selected.append(x)

    best_selected = list(selected)
    best_size = len(selected)

    # Phase 2: Try to add more elements in different orders
    if time.time() < deadline - 10.0:
        for x in range(1, N + 1):
            if time.time() > deadline - 10.0:
                break
            if x not in best_selected:
                if can_add_fast(x, best_selected):
                    best_selected.append(x)
                    best_size = len(best_selected)

    # Phase 3: Local search with swaps
    if time.time() < deadline - 15.0:
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            for i in range(len(best_selected)):
                if time.time() > deadline - 15.0:
                    break
                for replacement in range(N, 0, -1):
                    if time.time() > deadline - 15.0:
                        break
                    if replacement in best_selected:
                        continue
                    new_selected = best_selected[:i] + best_selected[i + 1:] + [replacement]
                    if full_check(new_selected):
                        best_selected = new_selected
                        best_size = len(best_selected)
                        improved = True
                        # Try to add more after swap
                        for x in range(N, 0, -1):
                            if time.time() > deadline - 15.0:
                                break
                            if x not in best_selected:
                                if can_add_fast(x, best_selected):
                                    best_selected.append(x)
                                    best_size = len(best_selected)
                        break

    # Phase 4: Multi-start with random perturbations
    if time.time() < deadline - 5.0:
        random.seed(42)
        num_restarts = 0
        while time.time() < deadline - 5.0 and num_restarts < 30:
            num_restarts += 1

            # Create a perturbed version of the best solution
            perturbed = list(best_selected)

            # Remove 2-4 random elements
            remove_count = random.randint(2, 4)
            indices_to_remove = random.sample(range(len(perturbed)), min(remove_count, len(perturbed)))
            for idx in sorted(indices_to_remove, reverse=True):
                perturbed.pop(idx)

            # Try to greedily fill back up
            for x in range(N, 0, -1):
                if time.time() > deadline - 5.0:
                    break
                if x not in perturbed:
                    if can_add_fast(x, perturbed):
                        perturbed.append(x)

            if len(perturbed) > best_size:
                best_selected = perturbed
                best_size = len(best_selected)

    # Phase 5: Final attempt to add any remaining elements
    if time.time() < deadline:
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if x not in best_selected:
                if can_add_fast(x, best_selected):
                    best_selected.append(x)
                    best_size = len(best_selected)

    return best_selected
# EVOLVE-BLOCK-END
