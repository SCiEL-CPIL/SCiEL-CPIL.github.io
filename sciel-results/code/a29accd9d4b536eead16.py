import numpy as np
import itertools
import time

D = 11

def construct():
    start = time.time()
    deadline = start + 135  # safety margin before the 150 s kill

    # ---- D11 lattice roots: vectors with exactly two non-zero entries in {+1,-1} ----
    # squared norm = 2.  For two such u,v: ||u-v||^2 = 4 - 2*(u.v), and u.v in {-1,0,1}.
    # We need ||u-v||^2 >= 2 (max norm = sqrt(2))  <=>  u.v <= 1  <=>  u.v != 1.
    # u.v == 1 happens exactly when u and v share a coordinate with the SAME sign.
    # So the constraint is: across the whole set, each coordinate may carry at most
    # one +1 and at most one -1.

    coords = list(range(D))

    # All candidate roots as (i, sign_i, j, sign_j) with i < j
    candidates = []
    for i in range(D):
        for si in (1, -1):
            for j in range(i + 1, D):
                for sj in (1, -1):
                    candidates.append((i, si, j, sj))

    # Build the conflict graph: two roots conflict if they share a coordinate with same sign.
    # Represent each "signed coordinate" as an atom: (coord, sign).  A root uses two atoms.
    # Two roots conflict iff they share an atom.
    N = len(candidates)
    atom_of = [[(i, si), (j, sj)] for (i, si, j, sj) in candidates]

    # Map atom -> list of candidate indices containing it
    atom_to_idx = {}
    for idx, atoms in enumerate(atom_of):
        for a in atoms:
            atom_to_idx.setdefault(a, []).append(idx)

    # Conflict matrix (symmetric, no diagonal)
    Conflict = np.zeros((N, N), dtype=bool)
    for idx, atoms in enumerate(atom_of):
        for a in atoms:
            for other in atom_to_idx[a]:
                if other != idx:
                    Conflict[idx, other] = True

    # ---- Deterministic symmetry-aware greedy: maximize total pairwise angular separation ----
    # Equivalent to: pick roots so that no atom is reused.  Greedy by "most constrained first".
    def run_greedy(order):
        used_atoms = set()
        selected = []
        for idx in order:
            if any(a in used_atoms for a in atom_of[idx]):
                continue
            for a in atom_of[idx]:
                used_atoms.add(a)
            selected.append(idx)
        return selected

    # Order candidates to encourage a balanced, symmetric selection.
    # Strategy: process in a round-robin over coordinate pairs so each coordinate
    # gets used roughly equally, and alternate signs for balance.
    # Build a deterministic ordering that pairs up coordinates cyclically.
    def cyclic_order():
        order = []
        # For each "offset" pairing (i, i+offset mod D), add all sign combos.
        for offset in range(1, D):
            for i in range(D):
                j = (i + offset) % D
                if j <= i:
                    continue
                for si in (1, -1):
                    for sj in (1, -1):
                        order.append((i, si, j, sj))
        return order

    cand_to_idx = {c: idx for idx, c in enumerate(candidates)}
    order1 = [cand_to_idx[c] for c in cyclic_order() if c in cand_to_idx]

    sel1 = run_greedy(order1)

    # Second ordering: reverse sign preference for diversity
    def cyclic_order2():
        order = []
        for offset in range(1, D):
            for i in range(D):
                j = (i + offset) % D
                if j <= i:
                    continue
                for si in (-1, 1):
                    for sj in (-1, 1):
                        order.append((i, si, j, sj))
        return order

    order2 = [cand_to_idx[c] for c in cyclic_order2() if c in cand_to_idx]
    sel2 = run_greedy(order2)

    # Third ordering: reverse coordinate direction
    def cyclic_order3():
        order = []
        for offset in range(1, D):
            for i in range(D - 1, -1, -1):
                j = (i + offset) % D
                if j <= i:
                    continue
                for si in (1, -1):
                    for sj in (1, -1):
                        order.append((i, si, j, sj))
        return order

    order3 = [cand_to_idx[c] for c in cyclic_order3() if c in cand_to_idx]
    sel3 = run_greedy(order3)

    best = max([sel1, sel2, sel3], key=len)

    # ---- Local search: 1-for-2 swap to try to improve ----
    def local_search(selected):
        if time.time() > deadline:
            return list(selected)
        selected_set = set(selected)

        # Build a quick lookup: for a candidate, which atoms it uses
        # and whether it's compatible with the current set.
        def compatible(idx):
            return not any(a in used for a in atom_of[idx])

        # Track used atoms as a set for fast checks
        def used_atom_set(s):
            s2 = set()
            for idx in s:
                for a in atom_of[idx]:
                    s2.add(a)
            return s2

        used = used_atom_set(selected_set)

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to add any compatible candidate
            for idx in range(N):
                if idx in selected_set:
                    continue
                if compatible(idx):
                    selected_set.add(idx)
                    for a in atom_of[idx]:
                        used.add(a)
                    improved = True
            if not improved:
                break

        # 1-for-2 swap: remove one, try to add two that were blocked by it
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in list(selected_set):
                # atoms freed by removing idx
                freed = set(atom_of[idx])
                # find candidates compatible with (selected - {idx})
                # i.e. candidates whose atoms don't intersect (used - freed)
                used_minus = used - freed
                adds = []
                for j in range(N):
                    if j in selected_set or j == idx:
                        continue
                    if any(a in used_minus for a in atom_of[j]):
                        continue
                    adds.append(j)
                # try to add two from 'adds' that are mutually compatible
                for a_i in range(len(adds)):
                    if time.time() > deadline - 2:
                        break
                    j1 = adds[a_i]
                    for b_i in range(a_i + 1, len(adds)):
                        j2 = adds[b_i]
                        if any(a in atom_of[j1] for a in atom_of[j2]):
                            continue
                        # swap
                        selected_set.remove(idx)
                        for a in atom_of[idx]:
                            used.discard(a)
                        selected_set.add(j1)
                        selected_set.add(j2)
                        for a in atom_of[j1] + atom_of[j2]:
                            used.add(a)
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break

        return list(selected_set)

    if time.time() < deadline - 5:
        best = local_search(best)

    # ---- Build the integer vector array ----
    if len(best) == 0:
        # Fallback: trivially valid small set
        vecs = []
        for i in range(D):
            for si in (1, -1):
                for j in range(i + 1, D):
                    v = [0] * D
                    v[i] = si
                    v[j] = -si
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)

    result = np.zeros((len(best), D), dtype=np.int64)
    for k, idx in enumerate(best):
        i, si, j, sj = candidates[idx]
        result[k, i] = si
        result[k, j] = sj

    return result
