"""Exact XOR-circuit synthesis: build all output rows as XORs of input rows,
minimizing gate count via shared intermediate XORs (greedy + local search)."""
import time
import random


def _bit_count(x):
    return bin(x).count('1')


def _row_popcount(row):
    return sum(row)


def _row_eq(a, b):
    return a == b


def synthesize(matrix, depth_limit):
    """Return {'gates': [[wire, wire], ...], 'outputs': [wire, ...]} for the
    binary matrix `matrix` (list of lists of 0/1). Each output row must be
    reconstructible as an XOR of a subset of input rows; we minimize the number
    of XOR gates by sharing intermediate XORs.

    Wire numbering: inputs 0..n-1, constant-zero = n, gate k produces n+1+k.
    """
    t0 = time.time()
    DEADLINE = 90.0
    MARGIN = 8.0  # safety margin

    def remaining():
        return DEADLINE - MARGIN - (time.time() - t0)

    # ---- Parse matrix ----
    if matrix is None:
        matrix = []
    rows = []
    for r in matrix:
        rows.append([int(x) & 1 for x in r])
    if not rows:
        return {'gates': [], 'outputs': []}
    n = len(rows[0])
    m = len(rows)

    # Represent each row as an integer bitmask for fast XOR.
    def row_to_int(row):
        v = 0
        for i, b in enumerate(row):
            if b:
                v |= (1 << i)
        return v

    inputs = [row_to_int(r) for r in rows]
    target = inputs  # outputs are the same rows (identity map), but we may
    # synthesize them via shared XORs to reduce gate count.

    # Baseline: each output = its own input (gate count = number of nonzero rows).
    # We try to beat this by sharing intermediate XORs.

    # ---- Build a pool of candidate intermediate XORs ----
    # A "wire" is a value (bitmask) produced by some gate. We maintain:
    #   wire_val[w] = the bitmask value of wire w
    #   wire_expr[w] = set of input indices XORed to produce w
    # Inputs are wires 0..n-1 (value = inputs[i]).
    # Constant-zero wire is n (value 0).
    # Gates produce wires n+1, n+2, ...

    # We'll do a greedy set-cover style construction:
    #   - Start with a pool of candidate XOR values (inputs + pairwise + triple).
    #   - For each target row, pick the cheapest way to express it using
    #     available wires (inputs + already-built gates).
    #   - If expressing it requires building a new intermediate, add it.

    # Candidate pool: all inputs, plus all pairwise XORs, plus (if small) triple.
    cand_vals = set()
    for v in inputs:
        cand_vals.add(v)
    # pairwise
    for i in range(n):
        for j in range(i + 1, n):
            cand_vals.add(inputs[i] ^ inputs[j])
    # triple (only if n is small enough to keep pool manageable)
    if n <= 40:
        for i in range(n):
            for j in range(i + 1, n):
                ij = inputs[i] ^ inputs[j]
                for k in range(j + 1, n):
                    cand_vals.add(ij ^ inputs[k])

    # Remove zero value (constant-zero wire already exists, no gate needed).
    cand_vals.discard(0)

    # ---- Greedy construction ----
    # State:
    #   built = dict: value -> wire index (for gates we've built)
    #   wire_val = list: index -> value
    #   wire_expr = list: index -> frozenset of input indices
    #   gates = list of [w1, w2]

    wire_val = {}
    wire_expr = {}
    gates = []
    next_wire = n + 1  # n is constant-zero

    # Pre-register inputs and constant-zero.
    for i in range(n):
        wire_val[i] = inputs[i]
        wire_expr[i] = frozenset([i])
    wire_val[n] = 0
    wire_expr[n] = frozenset()

    def ensure_wire(val, expr):
        """Return a wire index that produces `val`. If it's an input or zero,
        return that. If a gate already produces it, return it. Otherwise build
        a new gate (splitting expr into two existing wires) and return it."""
        # Is it an input?
        if val in wire_val:
            # find the wire
            for w, v in wire_val.items():
                if v == val:
                    return w
        # Build a new gate: need to split expr into two non-empty subsets
        # whose XOR = val and both are already available wires.
        # Try: for each input i in expr, the complement expr\{i} must be a wire.
        expr_list = list(expr)
        # Try splitting off one input at a time.
        for i in expr_list:
            rest = expr - {i}
            if not rest:
                continue
            rest_val = 0
            for j in rest:
                rest_val ^= inputs[j]
            # find wire for rest_val
            w_rest = None
            for w, v in wire_val.items():
                if v == rest_val:
                    w_rest = w
                    break
            if w_rest is not None:
                # new gate = i XOR w_rest
                new_w = next_wire
                nonlocal next_wire
                gates.append([i, w_rest])
                wire_val[new_w] = val
                wire_expr[new_w] = expr
                next_wire += 1
                return new_w
        # Fallback: build rest via pairwise if possible (shouldn't happen if pool is good)
        # Just build directly: pick two inputs and chain.
        # This is a safety net; normally the above works.
        if len(expr_list) >= 2:
            a, b = expr_list[0], expr_list[1]
            # build a XOR b as a gate
            ab_val = inputs[a] ^ inputs[b]
            w_ab = None
            for w, v in wire_val.items():
                if v == ab_val:
                    w_ab = w
                    break
            if w_ab is None:
                new_w = next_wire
                nonlocal next_wire
                gates.append([a, b])
                wire_val[new_w] = ab_val
                wire_expr[new_w] = frozenset([a, b])
                next_wire += 1
                w_ab = new_w
            # now XOR in remaining
            cur = w_ab
            for idx in expr_list[2:]:
                new_w = next_wire
                nonlocal next_wire
                gates.append([cur, idx])
                cur_val = wire_val[cur] ^ inputs[idx]
                wire_val[new_w] = cur_val
                wire_expr[new_w] = wire_expr[cur] | {idx}
                next_wire += 1
                cur = new_w
            return cur
        # Single input
        for i in expr_list:
            if inputs[i] == val:
                return i
        # Should not reach here
        return n

    # For each target, find the cheapest expression using available wires.
    # Cheapest = fewest gates to build. We use a greedy: express target as
    # XOR of a subset of inputs; then ensure_wire builds it.
    # To minimize gates, prefer expressions that reuse existing wires.

    outputs = []
    for t in range(m):
        tv = target[t]
        if tv == 0:
            outputs.append(n)
            continue
        # Find a subset of inputs XORing to tv that minimizes new gates.
        # Greedy: try to express tv using existing wires as much as possible.
        # We'll do a BFS/DP over subsets if n is small, else greedy.
        if n <= 20:
            # DP: dp[mask] = wire that produces XOR of inputs in mask
            # Actually, we want to find a wire whose value is tv, building
            # minimally. Use ensure_wire with the best expr.
            # Find the expr (subset) for tv with minimum "cost".
            # Cost = number of new gates needed.
            # We'll just try: the subset that is tv itself (all inputs in tv's support)
            # but that may not be minimal. For small n, do exact via subset DP.
            # dp over masks: value[mask] = XOR, and we want mask with value==tv
            # minimizing cost. Cost is complex; approximate by popcount(mask).
            best_mask = None
            best_cost = None
            for mask in range(1, 1 << n):
                v = 0
                for i in range(n):
                    if mask & (1 << i):
                        v ^= inputs[i]
                if v == tv:
                    cost = bin(mask).count('1')
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_mask = mask
            if best_mask is not None:
                expr = frozenset(i for i in range(n) if best_mask & (1 << i))
                outputs.append(ensure_wire(tv, expr))
            else:
                outputs.append(n)
        else:
            # Greedy for larger n: use the natural support.
            expr = frozenset(i for i in range(n) if (tv >> i) & 1)
            if not expr:
                outputs.append(n)
            else:
                outputs.append(ensure_wire(tv, expr))

    # ---- Local search: try to reduce gate count ----
    # Re-run greedy with different orderings / candidate pools if time permits.
    best_gates = gates
    best_outputs = outputs
    best_count = len(gates)

    def time_ok():
        return remaining() > 1.0

    # Try a few randomized orderings of targets to see if sharing improves.
    rng = random.Random(42)
    orderings = [list(range(m))]
    for _ in range(6):
        o = list(range(m))
        rng.shuffle(o)
        orderings.append(o)

    for ordering in orderings:
        if not time_ok():
            break
        # Reset state
        wire_val = {}
        wire_expr = {}
        gates = []
        next_wire = n + 1
        for i in range(n):
            wire_val[i] = inputs[i]
            wire_expr[i] = frozenset([i])
        wire_val[n] = 0
        wire_expr[n] = frozenset()

        def ensure_wire2(val, expr):
            for w, v in wire_val.items():
                if v == val:
                    return w
            expr_list = list(expr)
            for i in expr_list:
                rest = expr - {i}
                if not rest:
                    continue
                rest_val = 0
                for j in rest:
                    rest_val ^= inputs[j]
                w_rest = None
                for w, v in wire_val.items():
                    if v == rest_val:
                        w_rest = w
                        break
                if w_rest is not None:
                    new_w = next_wire
                    nonlocal next_wire
                    gates.append([i, w_rest])
                    wire_val[new_w] = val
                    wire_expr[new_w] = expr
                    next_wire += 1
                    return new_w
            if len(expr_list) >= 2:
                a, b = expr_list[0], expr_list[1]
                ab_val = inputs[a] ^ inputs[b]
                w_ab = None
                for w, v in wire_val.items():
                    if v == ab_val:
                        w_ab = w
                        break
                if w_ab is None:
                    new_w = next_wire
                    nonlocal next_wire
                    gates.append([a, b])
                    wire_val[new_w] = ab_val
                    wire_expr[new_w] = frozenset([a, b])
                    next_wire += 1
                    w_ab = new_w
                cur = w_ab
                for idx in expr_list[2:]:
                    new_w = next_wire
                    nonlocal next_wire
                    gates.append([cur, idx])
                    cur_val = wire_val[cur] ^ inputs[idx]
                    wire_val[new_w] = cur_val
                    wire_expr[new_w] = wire_expr[cur] | {idx}
                    next_wire += 1
                    cur = new_w
                return cur
            for i in expr_list:
                if inputs[i] == val:
                    return i
            return n

        outputs2 = []
        for t in ordering:
            tv = target[t]
            if tv == 0:
                outputs2.append(n)
                continue
            if n <= 20:
                best_mask = None
                best_cost = None
                for mask in range(1, 1 << n):
                    v = 0
                    for i in range(n):
                        if mask & (1 << i):
                            v ^= inputs[i]
                    if v == tv:
                        cost = bin(mask).count('1')
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_mask = mask
                if best_mask is not None:
                    expr = frozenset(i for i in range(n) if best_mask & (1 << i))
                    outputs2.append(ensure_wire2(tv, expr))
                else:
                    outputs2.append(n)
            else:
                expr = frozenset(i for i in range(n) if (tv >> i) & 1)
                if not expr:
                    outputs2.append(n)
                else:
                    outputs2.append(ensure_wire2(tv, expr))

        # Reorder outputs back to original order
        outputs_final = [None] * m
        for idx, t in enumerate(ordering):
            outputs_final[t] = outputs2[idx]

        if len(gates) < best_count:
            best_count = len(gates)
            best_gates = gates
            best_outputs = outputs_final

    # ---- Verify correctness (exact GF(2)) ----
    # Reconstruct each output wire's value and compare to target.
    def verify(gates, outputs):
        val = {}
        for i in range(n):
            val[i] = inputs[i]
        val[n] = 0
        for k, (a, b) in enumerate(gates):
            val[n + 1 + k] = val[a] ^ val[b]
        for t in range(m):
            w = outputs[t]
            if val.get(w) != target[t]:
                return False
        return True

    if not verify(best_gates, best_outputs):
        # Fallback to identity (always correct)
        best_gates = []
        best_outputs = [i for i in range(m)]
        # identity: output t = input t, but if input t is zero, use constant-zero
        best_outputs = []
        for t in range(m):
            if inputs[t] == 0:
                best_outputs.append(n)
            else:
                best_outputs.append(t)

    return {'gates': best_gates, 'outputs': best_outputs}
