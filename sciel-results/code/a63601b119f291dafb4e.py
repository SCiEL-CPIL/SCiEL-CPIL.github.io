"""XOR-circuit synthesis: greedy common-subexpression extraction with deadline guard.

Model: wires are GF(2) vectors over the n inputs, represented as n-bit ints.
Target j = row j as a mask. We build a pool of "atoms" (wires) closed under XOR-gates,
starting from the n unit vectors and the zero wire. We greedily add new atoms c=a^b
that reduce the total cost (gates + per-target decomposition size), absorbing c into
target decompositions via small subset-XOR searches. Finally each target is realized
as a wire via a binary tree of gates that reuses existing atoms.
"""
import time

_DEADLINE = 85.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _rows_to_masks(matrix, n, m):
    rows = []
    for j in range(m):
        v = 0
        row = matrix[j]
        for i in range(n):
            if row[i]:
                v |= (1 << i)
        rows.append(v)
    return rows

def _subset_xor_to(target, atoms):
    """Return list of indices (each used once) of atoms whose XOR == target, else None."""
    if not atoms:
        return None
    if target == 0:
        return []
    dp = {0: (None, None)}
    for idx, a in enumerate(atoms):
        for val in list(dp.keys()):
            nv = val ^ a
            if nv not in dp:
                dp[nv] = (val, idx)
    if target not in dp:
        return None
    res = []
    cur = target
    while cur != 0:
        prev, idx = dp[cur]
        if prev is None:
            break
        res.append(idx)
        cur = prev
    x = 0
    for i in res:
        x ^= atoms[i]
    if x != target:
        return None
    return res

def synthesize(matrix, depth_limit):
    t0 = time.time()
    n = len(matrix[0])
    m = len(matrix)
    masks = _rows_to_masks(matrix, n, m)

    # atoms list; atom_wire: atom->wire id; gates list of [a,b]
    atoms = [0] + [(1 << i) for i in range(n)]
    atom_wire = {}
    for i in range(n):
        atom_wire[1 << i] = i
    atom_wire[0] = n
    next_gate_id = n + 1
    gates = []

    # decomp[j]: list of atom indices whose XOR == masks[j]
    decomp = []
    for t in masks:
        lst = []
        for i in range(n):
            if t & (1 << i):
                lst.append(1 + i)
        decomp.append(lst)

    def total_cost():
        c = len(gates)
        for lst in decomp:
            c += max(0, len(lst) - 1)
        return c

    max_atoms = 220
    improved = True
    while improved and _elapsed() < _DEADLINE - 2 and len(atoms) < max_atoms:
        improved = False
        A = len(atoms)
        # candidate new atoms
        cand = {}
        for i in range(A):
            ai = atoms[i]
            if ai == 0:
                continue
            for j in range(i + 1, A):
                aj = atoms[j]
                if aj == 0:
                    continue
                c = ai ^ aj
                if c == 0 or c in atom_wire:
                    continue
                cand[c] = (i, j)
        if not cand:
            break
        base_cost = total_cost()
        best_save = 0
        best_c = None
        best_pair = None
        for c, (i, j) in cand.items():
            if _elapsed() > _DEADLINE - 2:
                break
            # simulate adding c
            new_idx = A
            new_decomp = []
            saving = 0
            for lst in decomp:
                sub = _subset_xor_to(c, [atoms[k] for k in lst])
                if sub is not None and len(sub) >= 2:
                    subset = set(sub)
                    newlst = []
                    for pos, k in enumerate(lst):
                        if pos not in subset:
                            newlst.append(k)
                    newlst.append(new_idx)
                    new_decomp.append(newlst)
                    saving += len(sub) - 1
                else:
                    new_decomp.append(list(lst))
            new_cost = (len(gates) + 1) + sum(max(0, len(x) - 1) for x in new_decomp)
            save = base_cost - new_cost
            if save > best_save:
                best_save = save
                best_c = c
                best_pair = (i, j)
        if best_c is not None and best_save > 0:
            i, j = best_pair
            c = best_c
            atoms.append(c)
            atom_wire[c] = next_gate_id
            gates.append([atom_wire[atoms[i]], atom_wire[atoms[j]]])
            next_gate_id += 1
            new_idx = len(atoms) - 1
            new_decomp = []
            for lst in decomp:
                sub = _subset_xor_to(c, [atoms[k] for k in lst])
                if sub is not None and len(sub) >= 2:
                    subset = set(sub)
                    newlst = []
                    for pos, k in enumerate(lst):
                        if pos not in subset:
                            newlst.append(k)
                    newlst.append(new_idx)
                    new_decomp.append(newlst)
                else:
                    new_decomp.append(list(lst))
            decomp = new_decomp
            improved = True

    # Realize each target as a wire, reusing atoms and adding gates as needed.
    outputs = []
    for j in range(m):
        t = masks[j]
        lst = decomp[j]
        # verify lst XOR == t; else fall back to units
        x = 0
        for k in lst:
            x ^= atoms[k]
        if x != t:
            lst = [1 + i for i in range(n) if t & (1 << i)]
        if not lst:
            outputs.append(n)
            continue
        # build wire for XOR of atoms in lst
        full = 0
        for k in lst:
            full ^= atoms[k]
        if full in atom_wire:
            outputs.append(atom_wire[full])
            continue
        items = [(atoms[k], atom_wire[atoms[k]]) for k in lst]
        while len(items) > 1:
            ni = []
            for i in range(0, len(items), 2):
                if i + 1 < len(items):
                    v1, w1 = items[i]
                    v2, w2 = items[i + 1]
                    v = v1 ^ v2
                    if v in atom_wire:
                        w = atom_wire[v]
                    else:
                        w = next_gate_id
                        gates.append([w1, w2])
                        atom_wire[v] = w
                        next_gate_id += 1
                    ni.append((v, w))
                else:
                    ni.append(items[i])
            items = ni
        outputs.append(items[0][1])

    return {'gates': gates, 'outputs': outputs}
