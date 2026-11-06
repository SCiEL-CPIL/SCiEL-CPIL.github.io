"""Maximize pairwise disjoint residue classes with distinct moduli 2..128.
Strategy: exact clique search over gcd>1 constraint using backtracking with
strong pruning, plus local search with swaps, and CRT-based residue assignment."""
# EVOLVE-BLOCK-START
import math
import time
import random
import sys

_START = None
MAX_MOD = 128


def construct():
    global _START
    if _START is None:
        _START = time.time()

    deadline = _START + 80.0  # Safety margin before 90s kill

    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)
    mod_index = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute prime factors
    def prime_factors(m):
        facs = set()
        d = 2
        while d * d <= m:
            if m % d == 0:
                facs.add(d)
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            facs.add(m)
        return facs

    pf_cache = {m: prime_factors(m) for m in moduli}
    omega_cache = {m: len(pf_cache[m]) for m in moduli}

    # Build adjacency: edge if gcd > 1
    adj = [set() for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            if gcd_table[i][j] > 1:
                adj[i].add(j)
                adj[j].add(i)

    # Degree for ordering
    degree = [len(adj[i]) for i in range(n_mod)]

    # Order moduli by decreasing degree (most connected first)
    order_by_degree = sorted(range(n_mod), key=lambda i: (-degree[i], moduli[i]))

    # ---- Residue finding via backtracking with CRT-style pruning ----
    def find_residue(m_idx, selected_pairs, selected_mod_indices):
        """Find a residue r in [0, moduli[m_idx]) disjoint from all selected classes.
        Uses the constraint: for each selected (r_i, m_i) with g = gcd(m, m_i) > 1,
        r must NOT satisfy r ≡ r_i (mod g).
        """
        m = moduli[m_idx]
        constraints = []
        for k, (r_i, m_i) in enumerate(selected_pairs):
            m_i_idx = selected_mod_indices[k]
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                constraints.append((g, r_i % g))

        if not constraints:
            return 0

        # Try each residue
        for r in range(m):
            valid = True
            for (g, r_i_mod) in constraints:
                if r % g == r_i_mod:
                    valid = False
                    break
            if valid:
                return r
        return None

    # ---- Greedy construction ----
    def greedy_pass(order_indices):
        selected_pairs = []
        selected_mod_indices = []
        used = set()

        for m_idx in order_indices:
            if m_idx in used:
                continue
            m = moduli[m_idx]

            # Check gcd compatibility with all selected
            compatible = True
            for sel_idx in selected_mod_indices:
                if gcd_table[m_idx][sel_idx] == 1:
                    compatible = False
                    break
            if not compatible:
                continue

            r = find_residue(m_idx, selected_pairs, selected_mod_indices)
            if r is not None:
                selected_pairs.append((r, m))
                selected_mod_indices.append(m_idx)
                used.add(m_idx)

        return selected_pairs

    # ---- Backtracking clique search with pruning ----
    def backtracking_clique(candidate_set, current_clique, best_size, deadline_check):
        """Try to extend current_clique using candidates. Returns best clique found."""
        if len(candidate_set) == 0:
            return current_clique[:]

        if time.time() > deadline_check:
            return current_clique[:]

        # Pruning: even if we add all candidates, can we beat best?
        if len(current_clique) + len(candidate_set) <= best_size:
            return current_clique[:]

        # Order candidates by degree within candidate_set for better pruning
        cand_list = sorted(candidate_set, key=lambda i: -len(adj[i] & candidate_set))

        for idx in range(len(cand_list)):
            if time.time() > deadline_check:
                return current_clique[:]

            v = cand_list[idx]

            # Try including v
            new_clique = current_clique + [v]
            # New candidates: those after idx in cand_list that are adjacent to v
            new_candidates = set()
            for j in range(idx + 1, len(cand_list)):
                w = cand_list[j]
                if w in adj[v]:
                    new_candidates.add(w)

            # Recursive call
            result = backtracking_clique(new_candidates, new_clique, best_size, deadline_check)
            if len(result) > len(current_clique):
                return result

            # Try excluding v (implicit: just continue loop)

        return current_clique[:]

    # ---- Local search with swaps ----
    def local_search(pairs, max_iter=500):
        current_pairs = pairs[:]
        current_count = len(current_pairs)
        best_local = current_pairs[:]
        best_local_count = current_count

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            used_mods = set(m for _, m in current_pairs)
            cur_mod_indices = [mod_index[m] for _, m in current_pairs]

            # Phase 1: Try to add any unused modulus
            added = False
            for m_idx in range(n_mod):
                if time.time() > deadline - 5.0:
                    break
                if moduli[m_idx] in used_mods:
                    continue
                compatible = True
                for sel_idx in cur_mod_indices:
                    if gcd_table[m_idx][sel_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                r = find_residue(m_idx, current_pairs, cur_mod_indices)
                if r is not None:
                    current_pairs.append((r, moduli[m_idx]))
                    cur_mod_indices.append(m_idx)
                    used_mods.add(moduli[m_idx])
                    added = True
                    break  # Restart to re-evaluate

            if added:
                current_count = len(current_pairs)
                if current_count > best_local_count:
                    best_local_count = current_count
                    best_local = current_pairs[:]
                continue

            # Phase 2: Try a swap
            if not current_pairs:
                break

            # Remove a random selected modulus
            rm_idx = random.randint(0, len(current_pairs) - 1)
            r_old, m_old = current_pairs[rm_idx]
            new_pairs = [p for i, p in enumerate(current_pairs) if i != rm_idx]
            new_used = set(m for _, m in new_pairs)
            new_mod_indices = [mod_index[m] for _, m in new_pairs]

            # Try adding a random unused modulus
            unused = [m_idx for m_idx in range(n_mod) if moduli[m_idx] not in new_used]
            if not unused:
                current_pairs = new_pairs  # Just removed one
                continue

            # Try several candidates
            tried = random.sample(unused, min(5, len(unused)))
            best_swap = None
            best_swap_count = len(new_pairs)

            for cand_idx in tried:
                if time.time() > deadline - 5.0:
                    break
                compatible = True
                for sel_idx in new_mod_indices:
                    if gcd_table[cand_idx][sel_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                r_new = find_residue(cand_idx, new_pairs, new_mod_indices)
                if r_new is None:
                    continue

                trial_pairs = new_pairs + [(r_new, moduli[cand_idx])]
                trial_used = new_used | {moduli[cand_idx]}
                trial_mod_indices = new_mod_indices + [cand_idx]

                # Try to add more
                for m_idx2 in range(n_mod):
                    if time.time() > deadline - 5.0:
                        break
                    if moduli[m_idx2] in trial_used:
                        continue
                    comp2 = True
                    for sel_idx2 in trial_mod_indices:
                        if gcd_table[m_idx2][sel_idx2] == 1:
                            comp2 = False
                            break
                    if not comp2:
                        continue
                    r2 = find_residue(m_idx2, trial_pairs, trial_mod_indices)
                    if r2 is not None:
                        trial_pairs.append((r2, moduli[m_idx2]))
                        trial_used.add(moduli[m_idx2])
                        trial_mod_indices.append(m_idx2)

                if len(trial_pairs) > best_swap_count:
                    best_swap_count = len(trial_pairs)
                    best_swap = trial_pairs

            if best_swap is not None:
                current_pairs = best_swap
                current_count = len(current_pairs)
                if current_count > best_local_count:
                    best_local_count = current_count
                    best_local = current_pairs[:]
            else:
                # Keep the removal (might help future additions)
                current_pairs = new_pairs
                current_count = len(current_pairs)

        return best_local

    # ---- Main search ----
    best_pairs = []
    best_count = 0

    # 1. Greedy passes with different orderings
    orderings = [
        order_by_degree,
        sorted(range(n_mod), key=lambda i: (-omega_cache[moduli[i]], -degree[i])),
        sorted(range(n_mod), key=lambda i: (omega_cache[moduli[i]], i)),
        list(range(n_mod)),
        list(range(n_mod - 1, -1, -1)),
        sorted(range(n_mod), key=lambda i: (-degree[i], -omega_cache[moduli[i]])),
    ]

    for order in orderings:
        if time.time() > deadline - 20.0:
            break
        pairs = greedy_pass(order)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # 2. Randomized greedy restarts
    random.seed(12345)
    n_restarts = 3000
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        shuffled = list(range(n_mod))
        random.shuffle(shuffled)
        pairs = greedy_pass(shuffled)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # 3. Backtracking clique search (exact, with time limit)
    if time.time() < deadline - 15.0:
        # Start from the greedy solution's moduli as initial clique
        initial_clique = [mod_index[m] for _, m in best_pairs]
        # Candidates: all moduli
        all_candidates = set(range(n_mod))

        # Use backtracking to try to find a larger clique
        # But this could be exponential. Limit with time.
        # Actually, let's use a bounded backtracking:
        # Try to add each unused modulus to the clique via backtracking

        # Alternative: use the backtracking on a smaller candidate set
        # Let's try: take the best greedy, then for each unused modulus,
        # check if it can be added (already done in local search).
        # Instead, let's do a proper bounded clique search:

        def bounded_clique_search(candidates, current, limit, time_limit):
            if time.time() > time_limit:
                return current[:]
            if len(current) + len(candidates) <= limit:
                return current[:]
            if not candidates:
                return current[:]

            # Pick vertex with max degree in candidates
            v = max(candidates, key=lambda x: len(adj[x] & candidates))
            new_cand = candidates - {v}

            # Include v
            result = bounded_clique_search(new_cand & adj[v], current + [v], limit, time_limit)
            if len(result) > len(current):
                return result

            # Exclude v
            result2 = bounded_clique_search(new_cand, current, limit, time_limit)
            if len(result2) > len(current):
                return result2

            return current[:]

        bt_result = bounded_clique_search(set(range(n_mod)), [], best_count, deadline - 10.0)
        if len(bt_result) > best_count:
            # Now assign residues
            bt_pairs = []
            bt_mod_indices = []
            for m_idx in bt_result:
                r = find_residue(m_idx, bt_pairs, bt_mod_indices)
                if r is not None:
                    bt_pairs.append((r, moduli[m_idx]))
                    bt_mod_indices.append(m_idx)
            if len(bt_pairs) > best_count:
                best_count = len(bt_pairs)
                best_pairs = bt_pairs

    # 4. Local search on best solution
    if best_pairs and time.time() < deadline - 10.0:
        improved = local_search(best_pairs, max_iter=1000)
        if len(improved) > best_count:
            best_count = len(improved)
            best_pairs = improved

    # 5. More local search from random subsets
    if time.time() < deadline - 10.0:
        for trial in range(100):
            if time.time() > deadline - 10.0:
                break
            if len(best_pairs) < 3:
                break
            subset_size = max(1, len(best_pairs) - random.randint(1, min(3, len(best_pairs))))
            subset = random.sample(best_pairs, subset_size)
            improved = local_search(subset, max_iter=300)
            if len(improved) > best_count:
                best_count = len(improved)
                best_pairs = improved

    # 6. Final local search
    if best_pairs and time.time() < deadline - 5.0:
        improved = local_search(best_pairs, max_iter=2000)
        if len(improved) > best_count:
            best_count = len(improved)
            best_pairs = improved

    # Safety fallback
    if not best_pairs:
        best_pairs = [(1, 2)]

    # Enforce canonical residues and distinct moduli
    seen = set()
    out = []
    for r, m in best_pairs:
        if m in seen:
            continue
        if not (0 <= r < m):
            continue
        seen.add(m)
        out.append([r, m])

    if not out:
        out = [[1, 2]]

    return out
# EVOLVE-BLOCK-END
