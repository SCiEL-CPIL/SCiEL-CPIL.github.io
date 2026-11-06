"""Combined: gcd>1 clique constraint + fast residue finder + randomized restarts +
branch-and-bound extension. Exploits that two classes are disjoint iff
(r1-r2) % gcd(m1,m2) != 0, which forces gcd(m1,m2) > 1 for every pair."""
# EVOLVE-BLOCK-START
import math
import time
import random

_START = None
MAX_MOD = 128


def _prime_factors(m):
    """Distinct prime factors of m."""
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


def _omega(m):
    count = 0
    d = 2
    while d * d <= m:
        if m % d == 0:
            count += 1
            while m % d == 0:
                m //= d
            d += 1
        else:
            d += 1
    if m > 1:
        count += 1
    return count


def construct():
    global _START
    if _START is None:
        _START = time.time()

    deadline = _START + 80.0  # safety margin before 90s kill

    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)
    mod_index = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table (indexed by modulus index)
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    omega_cache = {m: _omega(m) for m in moduli}

    # For each modulus index, precompute the list of other indices it shares gcd>1 with
    # (these are the only ones that impose a residue constraint).
    shared = [[] for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(n_mod):
            if i != j and gcd_table[i][j] > 1:
                shared[i].append(j)

    # ---------- Residue finding (fast) ----------
    def _find_residue(m_idx, selected_pairs, selected_mod_indices):
        """Find smallest r in [0, m) disjoint from all selected classes.
        Uses a 'blocked' boolean array: a residue is blocked if it lies in
        (r_i mod g) for any selected class with gcd g>1."""
        m = moduli[m_idx]
        blocked = [False] * m
        for k, (r_i, _) in enumerate(selected_pairs):
            g = gcd_table[m_idx][selected_mod_indices[k]]
            if g > 1:
                start = r_i % g
                for r in range(start, m, g):
                    blocked[r] = True
        for r in range(m):
            if not blocked[r]:
                return r
        return None

    def _find_residues(m_idx, selected_pairs, selected_mod_indices, limit=6):
        """Find up to `limit` valid residues for branch-and-bound."""
        m = moduli[m_idx]
        blocked = [False] * m
        for k, (r_i, _) in enumerate(selected_pairs):
            g = gcd_table[m_idx][selected_mod_indices[k]]
            if g > 1:
                start = r_i % g
                for r in range(start, m, g):
                    blocked[r] = True
        out = []
        for r in range(m):
            if not blocked[r]:
                out.append(r)
                if len(out) >= limit:
                    break
        return out

    # ---------- Greedy pass ----------
    def greedy_pass(order):
        selected_pairs = []
        selected_mod_indices = []
        used = set()
        for m in order:
            if m in used:
                continue
            m_idx = mod_index[m]
            # gcd must be > 1 with ALL selected (clique constraint)
            compatible = True
            for m_sel_idx in selected_mod_indices:
                if gcd_table[m_idx][m_sel_idx] == 1:
                    compatible = False
                    break
            if not compatible:
                continue
            r = _find_residue(m_idx, selected_pairs, selected_mod_indices)
            if r is not None:
                selected_pairs.append((r, m))
                selected_mod_indices.append(m_idx)
                used.add(m)
        return selected_pairs

    best_pairs = []
    best_count = 0

    # ---------- Deterministic orderings ----------
    orderings = [
        sorted(moduli, key=lambda m: (-omega_cache[m], m)),
        sorted(moduli, key=lambda m: (omega_cache[m], m)),
        sorted(moduli),
        sorted(moduli, reverse=True),
        sorted(moduli, key=lambda m: (-omega_cache[m], -m)),
    ]
    for order in orderings:
        if time.time() > deadline:
            break
        pairs = greedy_pass(order)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # ---------- Randomized restarts ----------
    random.seed(42)
    while time.time() < deadline - 20.0:
        shuffled = moduli[:]
        random.shuffle(shuffled)
        pairs = greedy_pass(shuffled)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # ---------- Local search: add unused moduli ----------
    def _try_add_all(pairs, mod_indices):
        """Greedily add every compatible unused modulus. Returns (pairs, mod_indices)."""
        used = set(m for _, m in pairs)
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for m in moduli:
                if m in used:
                    continue
                m_idx = mod_index[m]
                compatible = True
                for m_sel_idx in mod_indices:
                    if gcd_table[m_idx][m_sel_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                r = _find_residue(m_idx, pairs, mod_indices)
                if r is not None:
                    pairs.append((r, m))
                    mod_indices.append(m_idx)
                    used.add(m)
                    changed = True
        return pairs, mod_indices

    if best_pairs and time.time() < deadline - 15.0:
        cur = best_pairs[:]
        cur_idx = [mod_index[m] for _, m in cur]
        cur, cur_idx = _try_add_all(cur, cur_idx)
        if len(cur) > best_count:
            best_count = len(cur)
            best_pairs = cur[:]

    # ---------- Branch-and-bound extension: explore multiple residues ----------
    def _bnb_extend(pairs, mod_indices, depth=0):
        """Try to extend by picking a compatible unused modulus and exploring
        multiple residue choices recursively (bounded)."""
        if time.time() > deadline - 2.0 or depth > 4:
            return pairs, mod_indices
        used = set(m for _, m in pairs)
        # candidate moduli: compatible with all selected
        cands = []
        for m in moduli:
            if m in used:
                continue
            m_idx = mod_index[m]
            ok = True
            for m_sel_idx in mod_indices:
                if gcd_table[m_idx][m_sel_idx] == 1:
                    ok = False
                    break
            if ok:
                cands.append(m)
        if not cands:
            return pairs, mod_indices
        # pick the most constrained candidate (largest omega, smallest m)
        cands.sort(key=lambda m: (-omega_cache[m], m))
        m = cands[0]
        m_idx = mod_index[m]
        residues = _find_residues(m_idx, pairs, mod_indices, limit=4)
        best_local = (pairs[:], mod_indices[:])
        for r in residues:
            new_pairs = pairs[:] + [(r, m)]
            new_idx = mod_indices[:] + [m_idx]
            ext, ext_idx = _bnb_extend(new_pairs, new_idx, depth + 1)
            if len(ext) > len(best_local[0]):
                best_local = (ext, ext_idx)
        return best_local

    if best_pairs and time.time() < deadline - 10.0:
        cur = best_pairs[:]
        cur_idx = [mod_index[m] for _, m in cur]
        ext, ext_idx = _bnb_extend(cur, cur_idx)
        if len(ext) > best_count:
            best_count = len(ext)
            best_pairs = ext[:]

    # ---------- Randomized subset restarts + extend ----------
    if time.time() < deadline - 5.0 and best_pairs:
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            subset_size = max(1, len(best_pairs) // 2)
            subset = random.sample(best_pairs, min(subset_size, len(best_pairs)))
            cur = subset[:]
            cur_idx = [mod_index[m] for _, m in cur]
            cur, cur_idx = _try_add_all(cur, cur_idx)
            if len(cur) > best_count:
                best_count = len(cur)
                best_pairs = cur[:]

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
