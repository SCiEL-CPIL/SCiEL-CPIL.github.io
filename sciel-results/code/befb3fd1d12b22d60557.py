"""XOR-circuit synthesis: shared-subexpression search with exact GF(2) verify."""
# EVOLVE-BLOCK-START
import time
import random

def synthesize(matrix, depth_limit):
    t0 = time.time()
    deadline = t0 + 78.0  # explicit safety margin under the 90s wall-clock kill

    # ---------- parse ----------
    rows = [list(r) for r in matrix]
    m = len(rows)
    n = (len(rows[0]) if m else 0)
    zero_wire = n
    masks = []
    for r in rows:
        v = 0
        for j, bit in enumerate(r):
            if bit:
                v |= (1 << j)
        masks.append(v)

    def pc(x):
        return x.bit_count()

    baseline = sum(max(0, pc(v) - 1) for v in masks)

    # ---------- baseline (balanced, no sharing) circuit ----------
    def build_baseline():
        gates = []
        depth = {}
        outs = []
        for v in masks:
            bits = [j for j in range(n) if (v >> j) & 1]
            if not bits:
                outs.append(zero_wire)
                continue
            if len(bits) == 1:
                outs.append(bits[0])
                continue
            stack = []
            # balanced build via recursion
            def bal(bs):
                if len(bs) == 1:
                    return bs[0]
                mid = len(bs) // 2
                wl = bal(bs[:mid])
                wr = bal(bs[mid:])
                gw = n + len(gates)
                gates.append([wl, wr])
                depth[gw] = max(depth.get(wl, 0), depth.get(wr, 0)) + 1
                return gw
            outs.append(bal(bits))
        return gates, outs, depth

    base_gates, base_outs, base_depth = build_baseline()

    # ---------- precompute submask frequency (sharing potential) ----------
    freq = {}
    for T in masks:
        s = T
        # enumerate proper nonempty submasks (cap work)
        if pc(T) <= 16:
            sub = T
            while sub:
                if sub != T:
                    freq[sub] = freq.get(sub, 0) + 1
                sub = (sub - 1) & T
    # also count the full targets themselves as highly shared
    for T in masks:
        freq[T] = freq.get(T, 0) + 1

    # ---------- shared builder ----------
    def run_once(rng, max_full_k, depth_limit):
        mask_to_wire = {}
        mask_to_depth = {}
        gates = []
        nodes = [0]

        def new_gate(wl, wr):
            gw = n + len(gates)
            gates.append([wl, wr])
            mask_to_depth[gw] = max(mask_to_depth.get(wl, 0), mask_to_depth.get(wr, 0)) + 1
            return gw

        def direct_balanced(mask):
            bits = [j for j in range(n) if (mask >> j) & 1]
            if not bits:
                return zero_wire
            if len(bits) == 1:
                return bits[0]
            def rec(bs):
                if len(bs) == 1:
                    return bs[0]
                mid = len(bs) // 2
                wl = rec(bs[:mid])
                wr = rec(bs[mid:])
                return new_gate(wl, wr)
            return rec(bits)

        def candidates(mask):
            k = pc(mask)
            cands = []
            if k <= max_full_k:
                sub = mask
                seen = set()
                while sub:
                    q = mask ^ sub
                    if q and sub != q and (sub < q):
                        if sub not in seen:
                            seen.add(sub)
                            cands.append((sub, q))
                    sub = (sub - 1) & mask
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                # limited balanced-ish splits
                for size in (1, 2, k // 2, k // 2 + 1):
                    if 0 < size < k:
                        # a few random subsets of that size
                        for _ in range(3):
                            sel = rng.sample(bits, size)
                            p = 0
                            for b in sel:
                                p |= (1 << b)
                            q = mask ^ p
                            if p and q and p < q:
                                cands.append((p, q))
            return cands

        def score_split(p, q):
            sp = freq.get(p, 0)
            sq = freq.get(q, 0)
            bal = -abs(pc(p) - pc(q))
            return (sp + sq, bal, -pc(p) - pc(q), rng.random())

        def produce(mask):
            if mask in mask_to_wire:
                return mask_to_wire[mask]
            if mask == 0:
                mask_to_wire[0] = zero_wire
                mask_to_depth[zero_wire] = 0
                return zero_wire
            if pc(mask) == 1:
                w = (mask & -mask).bit_length() - 1
                mask_to_wire[mask] = w
                mask_to_depth[w] = 0
                return w
            nodes[0] += 1
            if nodes[0] > 400000 or time.time() > deadline:
                w = direct_balanced(mask)
                mask_to_wire[mask] = w
                return w
            cands = candidates(mask)
            if not cands:
                w = direct_balanced(mask)
                mask_to_wire[mask] = w
                return w
            # filter by depth feasibility
            feasible = []
            for (p, q) in cands:
                # optimistic depth check
                feasible.append((p, q))
            best = None
            best_key = None
            for (p, q) in feasible:
                key = score_split(p, q)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (p, q)
            p, q = best
            wp = produce(p)
            wq = produce(q)
            d = max(mask_to_depth.get(wp, 0), mask_to_depth.get(wq, 0)) + 1
            if d > depth_limit:
                # fall back to balanced direct for this mask
                w = direct_balanced(mask)
                mask_to_wire[mask] = w
                return w
            w = new_gate(wp, wq)
            mask_to_wire[mask] = w
            return w

        outs = []
        for v in masks:
            outs.append(produce(v))
        # total gates = number of gate entries
        total = len(gates)
        maxd = max(mask_to_depth.values()) if mask_to_depth else 0
        return gates, outs, total, maxd

    # ---------- symbolic verification ----------
    def verify(gates, outs, masks):
        func = {}
        for j in range(n):
            func[j] = (1 << j)
        func[zero_wire] = 0
        for k, (a, b) in enumerate(gates):
            gw = n + 1 + k
            func[gw] = func.get(a, 0) ^ func.get(b, 0)
        for w, target in zip(outs, masks):
            if func.get(w, -1) != target:
                return False
        return True

    # ---------- ensemble ----------
    best = (baseline, base_gates, base_outs)  # (total, gates, outs)
    variants = [
        (0, 12), (1, 12), (2, 10), (3, 14), (4, 12), (5, 8), (6, 16), (7, 12),
    ]
    for seed, mk in variants:
        if time.time() > deadline - 3.0:
            break
        rng = random.Random(seed)
        try:
            g, o, total, maxd = run_once(rng, mk, depth_limit)
        except Exception:
            continue
        if maxd > depth_limit:
            continue
        if verify(g, o, masks):
            if total < best[0]:
                best = (total, g, o)

    gates, outs = best[1], best[2]

    return {'gates': gates, 'outputs': list(outs)}
# EVOLVE-BLOCK-END
