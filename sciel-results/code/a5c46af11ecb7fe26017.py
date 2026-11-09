import time
import heapq

def synthesize(matrix, depth_limit):
    t_start = time.time()
    DEADLINE = 90.0
    MARGIN = 3.0

    # matrix: list of rows, each row is a binary vector of length n (inputs)
    # Represent each vector as int bitmask
    rows = []
    n = None
    for r in matrix:
        v = 0
        for i, b in enumerate(r):
            if b:
                v |= (1 << i)
        rows.append(v)
        if n is None:
            n = len(r)
    if n is None:
        n = 0
    m = len(rows)

    # wires: 0..n-1 inputs, n = zero
    # gate k -> wire n+1+k
    # We'll build a list `wires` of int bitmasks; wires[i] is the value of wire i.
    # Initially wires[0..n-1] = 1<<i, wires[n] = 0.
    wires = [1 << i for i in range(n)] + [0]
    # gate list: each gate is [in_a, in_b]
    gates = []

    # targets: each output row must be realized by some wire index
    # target_value -> set of wire indices that realize it (we track which wire is "assigned")
    # We'll maintain for each target its current wire (or None)
    target_wire = [None] * m

    def time_ok():
        return (time.time() - t_start) < (DEADLINE - MARGIN)

    def realize_targets():
        # assign each target to an existing wire if possible
        for i in range(m):
            tv = rows[i]
            if tv == 0:
                target_wire[i] = n  # zero wire
            else:
                # find a wire with this value
                for w in range(len(wires)):
                    if wires[w] == tv:
                        target_wire[i] = w
                        break
                else:
                    target_wire[i] = None

    realize_targets()

    def all_realized():
        return all(w is not None for w in target_wire)

    # cost model: for a target value t, given current wires, estimate min gates
    # Use popcount-based lower bound: a target with popcount p needs at least p-1 gates
    # if no sharing; with sharing, less. We approximate the "remaining cost" of target t
    # as the minimum over available wires w of (1 if t != w else 0) + popcount(t ^ w) - 1
    # i.e., produce w (free, already have it) then XOR with (t^w) which needs popcount(t^w)-1 gates.
    def target_remaining_cost(t):
        if t == 0:
            return 0
        best = popcount(t) - 1  # naive: XOR all set bits
        # try using each available wire as a building block
        # but that's O(|wires|) per target; do a cheap version: only consider wires that are "useful"
        # We'll consider a small set: inputs and zero are always useful; also recently added gates.
        # For speed, sample: consider wires with popcount <= popcount(t)
        for w in wires:
            if w == t:
                return 0
            d = t ^ w
            c = popcount(d) - 1
            if c < best:
                best = c
        return best

    def popcount(x):
        return bin(x).count('1')

    # Greedy: while not all realized and time ok:
    #   for each pair (a,b) of distinct wires, g = a^b
    #   compute gain = sum over not-realized targets of (cost_before - cost_after)
    #   pick max gain gate
    while not all_realized() and time_ok():
        num_wires = len(wires)
        best_gain = 0
        best_gate = None
        # not realized targets
        nrt = [i for i in range(m) if target_wire[i] is None]
        if not nrt:
            break
        # precompute before costs
        before = {i: target_remaining_cost(rows[i]) for i in nrt}
        # iterate candidate gates
        # To limit cost, we consider pairs; num_wires up to ~ n + gates
        # For n=16, m=16, gates maybe ~30, num_wires ~ 46, pairs ~ 1000, times m=16, times popcount -> fine
        for a in range(num_wires):
            wa = wires[a]
            for b in range(a + 1, num_wires):
                g = wa ^ wires[b]
                if g == 0:
                    continue
                # skip if g already exists as a wire (no new info)
                # (it might, but adding duplicate is useless)
                # compute after costs: now g is available
                gain = 0
                for i in nrt:
                    t = rows[i]
                    if t == g:
                        gain += before[i]  # now cost 0
                        continue
                    d = t ^ g
                    c = popcount(d) - 1
                    if c < before[i]:
                        gain += before[i] - c
                if gain > best_gain:
                    best_gain = gain
                    best_gate = (a, b)
        if best_gate is None:
            # no beneficial gate; fallback: realize each remaining target naively
            for i in range(m):
                if target_wire[i] is None:
                    t = rows[i]
                    # build t from inputs: XOR set bits
                    # We'll add gates to build t
                    acc = 0
                    acc_wire = n  # zero
                    for bit in range(n):
                        if t & (1 << bit):
                            if acc_wire is None:
                                acc_wire = bit
                                acc = 1 << bit
                            else:
                                neww = acc ^ (1 << bit)
                                gates.append([acc_wire, bit])
                                wires.append(neww)
                                acc_wire = len(wires) - 1
                                acc = neww
                    target_wire[i] = acc_wire if acc_wire is not None else n
            break
        a, b = best_gate
        g = wires[a] ^ wires[b]
        gates.append([a, b])
        wires.append(g)
        # update target assignments: any target equal to g now realized
        for i in range(m):
            if target_wire[i] is None and rows[i] == g:
                target_wire[i] = len(wires) - 1
        # also re-realize any target that might now be available (in case g equals some target)
        # (covered above)
        realize_targets()

    # Local search: gate-drop. Try removing each gate; if all targets still realizable (recompute), keep removed.
    # We'll do a few passes.
    def build_circuit_from_gates(gate_list):
        # returns (wires, target_wire) 
        ws = [1 << i for i in range(n)] + [0]
        for (a, b) in gate_list:
            ws.append(ws[a] ^ ws[b])
        tw = [None] * m
        for i in range(m):
            tv = rows[i]
            for w in range(len(ws)):
                if ws[w] == tv:
                    tw[i] = w
                    break
        return ws, tw

    # Try dropping gates (one at a time, restart greedily)
    improved = True
    passes = 0
    while improved and time_ok() and passes < 4:
        improved = False
        passes += 1
        cur_len = len(gates)
        for idx in range(cur_len):
            if not time_ok():
                break
            trial = gates[:idx] + gates[idx+1:]
            ws, tw = build_circuit_from_gates(trial)
            if all(w is not None for w in tw):
                gates = trial
                improved = True
                break  # restart scan

    # Final: ensure all targets realized; if any not, realize naively
    realize_targets()
    for i in range(m):
        if target_wire[i] is None:
            t = rows[i]
            acc_wire = None
            for bit in range(n):
                if t & (1 << bit):
                    if acc_wire is None:
                        acc_wire = bit
                    else:
                        gates.append([acc_wire, bit])
                        wires.append(wires[acc_wire] ^ (1 << bit))
                        acc_wire = len(wires) - 1
            target_wire[i] = acc_wire if acc_wire is not None else n

    # depth check: recompute depths; if > depth_limit, we may need to restructure.
    # For our sizes greedy depth is small; but ensure correctness of depth <= 16.
    # Compute depth of each wire.
    depth = [0] * len(wires)
    # inputs and zero depth 0
    for k, (a, b) in enumerate(gates):
        gw = n + 1 + k
        depth[gw] = 1 + max(depth[a], depth[b])
    max_depth = max(depth[target_wire[i]] for i in range(m)) if m else 0
    # If depth exceeds limit, fall back to a depth-bounded naive construction (popcount <= n so depth <= n-1 <= 15 for n<=16)
    if max_depth > depth_limit:
        # rebuild naively with depth = popcount-1 <= n-1
        gates = []
        wires = [1 << i for i in range(n)] + [0]
        target_wire = [None] * m
        for i in range(m):
            t = rows[i]
            acc_wire = None
            for bit in range(n):
                if t & (1 << bit):
                    if acc_wire is None:
                        acc_wire = bit
                    else:
                        gates.append([acc_wire, bit])
                        wires.append(wires[acc_wire] ^ (1 << bit))
                        acc_wire = len(wires) - 1
            target_wire[i] = acc_wire if acc_wire is not None else n

    # Build output
    outputs = [target_wire[i] for i in range(m)]

    # Exact verification (symbolic GF(2))
    def verify():
        ws = [1 << i for i in range(n)] + [0]
        for (a, b) in gates:
            ws.append(ws[a] ^ ws[b])
        for i in range(m):
            if ws[outputs[i]] != rows[i]:
                return False
        return True

    if not verify():
        # safety: rebuild naive
        gates = []
        wires = [1 << i for i in range(n)] + [0]
        outputs = []
        for i in range(m):
            t = rows[i]
            acc_wire = None
            for bit in range(n):
                if t & (1 << bit):
                    if acc_wire is None:
                        acc_wire = bit
                    else:
                        gates.append([acc_wire, bit])
                        wires.append(wires[acc_wire] ^ (1 << bit))
                        acc_wire = len(wires) - 1
            outputs.append(acc_wire if acc_wire is not None else n)

    return {'gates': gates, 'outputs': outputs}
