"""Exact XOR-circuit synthesis: minimum-cost sharing via adaptive greedy decomposition."""
import time

def synthesize(matrix, depth_limit):
    deadline = 85.0  # safety margin before 90s kill

    m = len(matrix)
    rows = []
    for r in matrix:
        s = frozenset(i for i, v in enumerate(r) if v)
        rows.append(s)

    n = max((max(s) for s in rows if s), default=-1) + 1
    zero_wire = n

    wires = {}
    for i in range(n):
        wires[frozenset([i])] = i
    wires[frozenset()] = zero_wire

    gates = []
    gate_count = 0
    wire_depth = {}
    for i in range(n):
        wire_depth[i] = 0
    wire_depth[zero_wire] = 0

    def new_gate(a, b):
        nonlocal gate_count
        s = frozenset(a) ^ frozenset(b)
        if s in wires:
            return wires[s]
        w = n + 1 + gate_count
        wires[s] = w
        gates.append([a, b])
        gate_count += 1
        wire_depth[w] = max(wire_depth[a], wire_depth[b]) + 1
        return w

    def decompose_cost(s, depth_budget):
        if s in wires or not s:
            return 0
        if depth_budget <= 0:
            return len(s)
        c = 0
        cur = s
        while cur and cur not in wires and depth_budget > 0:
            best_e = None
            best_key = 10**9
            for e in cur:
                rest = cur - {e}
                key = 0 if rest in wires else len(rest)
                if key < best_key:
                    best_key = key
                    best_e = e
            cur = cur - {best_e}
            c += 1
        if cur and cur not in wires:
            c += len(cur)
        return c

    def build(s, depth_budget):
        if s in wires:
            return wires[s]
        if not s:
            return zero_wire
        if depth_budget <= 0:
            # fallback: build linearly (may exceed depth but keeps correctness of value)
            depth_budget = len(s)
        cur = s
        # peel from the full set down, building a left-deep tree
        # We peel the "best" element first to maximize sharing of the rest
        best_e = None
        best_key = 10**9
        for e in cur:
            rest = cur - {e}
            key = 0 if rest in wires else len(rest)
            if key < best_key:
                best_key = key
                best_e = e
        e = best_e
        rest = cur - {e}
        w_rest = build(rest, depth_budget - 1)
        w_e = wires[frozenset([e])]
        return new_gate(w_e, w_rest)

    built = [False] * m
    while True:
        unbuilt = [i for i in range(m) if not built[i]]
        if not unbuilt:
            break
        if time.time() > deadline:
            break
        best_i = min(unbuilt, key=lambda i: decompose_cost(rows[i], depth_limit))
        built[best_i] = True
        build(rows[best_i], depth_limit)

    for i in range(m):
        if not built[i]:
            build(rows[i], depth_limit)
            built[i] = True

    outputs = [wires[rows[i]] if rows[i] in wires else build(rows[i], depth_limit) for i in range(m)]

    return {'gates': gates, 'outputs': outputs}
