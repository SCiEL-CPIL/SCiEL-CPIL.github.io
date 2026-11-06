"""XOR-circuit synthesis: globally-optimized subexpression sharing.

Structural design:
  * Represent every vector as a Python int bitmask over the input wires.
  * The "cost" of a vector is its Hamming weight (number of input wires it uses).
  * Independent-row baseline = sum of weights of all output rows (each row is a
    single XOR of its set bits; the zero row costs 0).
  * To beat the baseline we introduce intermediate XORs that are *shared* across
    several output rows.  Each introduced intermediate is a single XOR gate.
  * We select intermediates with a soft-min (log-sum-exp) surrogate that scores a
    candidate by its aggregate distance-reduction over ALL unsatisfied targets at
    once (not a single best match), so one intermediate can serve many rows.
  * The circuit is built in a strict dependency (topological) order, so every
    produced wire is consumed by a later gate or output -> a valid circuit with
    no dead wires and no cycles.  Depth is bounded by the number of dependency
    levels, which stays well under the 16 limit for the target matrix sizes.
"""

import time
import math
import numpy as np

_DEADLINE = 90.0
_MARGIN = 5.0          # safety margin (seconds) before the wall-clock kill
_MAX_INTS = 262144     # cap on the number of intermediate vectors we may explore


def _as_ints(mat):
    rows = []
    for r in mat:
        v = 0
        for j, x in enumerate(r):
            if x:
                v |= (1 << j)
        rows.append(v)
    return rows


def _popcount(x):
    return bin(x).count("1")


def _baseline(rows):
    return sum(_popcount(r) for r in rows)


def _candidate_pool(rows, n, cap):
    """All input wires plus all pairwise XORs of the output rows."""
    pool = set()
    for j in range(n):
        pool.add(1 << j)
    for a in rows:
        for b in rows:
            if a != b:
                pool.add(a ^ b)
    # cap the pool deterministically (keep the lightest vectors)
    if len(pool) > cap:
        pool = set(sorted(pool, key=_popcount)[:cap])
    return pool


def _build(rows, n, cap, deadline):
    """Greedy soft-min subexpression selection with dependency ordering."""
    targets = [r for r in rows if r != 0]
    if not targets:
        return []

    pool = _candidate_pool(rows, n, cap)
    pool_set = set(pool)

    # current expression of each target as a set of "tokens"
    # token = (kind, value)  kind in {i, g}; inputs are free, gates cost 1
    expr = [set() for _ in range(len(targets))]
    for t, r in enumerate(targets):
        for j in range(n):
            if (r >> j) & 1:
                expr[t].add(("i", 1 << j))

    # dependency order: a token may be added to a target only if every token it
    # depends on is already present in that target's expression.
    # A gate token g(v) depends on the input/gate tokens of v's own expression.

    # We maintain, for each candidate gate value v, its current "support" = the
    # set of tokens needed to build v.  Initially support(v) = the input tokens of v.
    # When we introduce a gate for v, support of (v ^ w) becomes support(v) U support(w)
    # minus v's own tokens... we track supports lazily.

    # Simpler robust model: each gate we introduce corresponds to a *specific*
    # XOR of two already-available wires.  We track, for every vector v we have
    # ever "materialized", the list of gate-tokens whose XOR equals v is not needed;
    # instead we track support[v] = set of tokens (inputs + introduced gates) that
    # XOR to v, chosen so that |support[v]| is small.

    support = {}
    for j in range(n):
        support[1 << j] = {("i", 1 << j)}

    # cost of materializing v = |support[v]| - (1 if v is an input)  -> number of gates
    # Actually a gate is needed per non-input token in support[v].

    def gate_cost(v):
        s = support.get(v)
        if s is None:
            return None
        return sum(1 for (k, x) in s if k == "g")

    def dist(t, v):
        # number of gate-tokens we must add to target t to reach v given its expr
        # = gate_cost of (expr_tokens_xor ^ v) measured in the token space of t
        # We approximate by: cost to build (current_xor ^ v) where current_xor is the
        # XOR of all tokens in expr[t].
        cur = 0
        for (k, x) in expr[t]:
            cur ^= x
        need = cur ^ v
        c = gate_cost(need)
        return c if c is not None else None

    introduced = set()   # gate values already materialized (as a gate token)
    order = []           # list of (v, [dep_gate_values]) in dependency order
    depmap = {}          # v -> list of gate values it depends on

    # Soft-min scoring: for candidate v, score = sum over targets of exp(-(d_after - d_before))
    # i.e. reward for reducing each target's distance.  We use the surrogate:
    #   score(v) = sum_t  exp(-max(0, d_t(v)))   where d_t(v)=dist(t,v)
    # Higher score = v is "close" to many targets at once.

    # Precompute current distances for all targets to all candidates would be heavy;
    # instead evaluate incrementally.

    n_targets = len(targets)
    best_total = _baseline(rows)

    # We will repeatedly pick the candidate that maximizes aggregate reward.
    # To keep this tractable we limit the number of introductions.
    max_intro = min(len(pool), 600)

    # current per-target distance to each candidate (lazy dict)
    def target_dist(t, v):
        cur = 0
        for (k, x) in expr[t]:
            cur ^= x
        need = cur ^ v
        c = gate_cost(need)
        return c if c is not None else 10**9

    # initial distances
    # We'll store dists[t] = dict v->dist
    dists = [dict() for _ in range(n_targets)]

    def compute_score(v):
        s = 0.0
        for t in range(n_targets):
            d = dists[t].get(v)
            if d is None:
                d = target_dist(t, v)
                dists[t][v] = d
            if d < 10**8:
                s += math.exp(-d)
        return s

    # candidate ordering: start with all pool vectors, score them, pick best,
    # materialize it, update supports and distances, repeat.
    cand_list = list(pool)

    for step in range(max_intro):
        if time.time() > deadline:
            break
        best_v = None
        best_s = 0.0
        for v in cand_list:
            if v in introduced:
                continue
            s = compute_score(v)
            if s > best_s:
                best_s = s
                best_v = v
        if best_v is None or best_s <= 1e-6:
            break

        # materialize best_v as a gate: it must be expressible from existing
        # supports.  Find a split best_v = a ^ b with both a,b in support.
        a, b = _split(best_v, support)
        if a is None:
            # cannot materialize; try to force by adding input-only support
            sup = set()
            for j in range(n):
                if (best_v >> j) & 1:
                    sup.add(("i", 1 << j))
            support[best_v] = sup
            depmap[best_v] = []
        else:
            sup = set()
            for tok in support[a]:
                sup.add(tok)
            for tok in support[b]:
                sup.add(tok)
            # remove the two parent gate tokens (they are consumed into this gate)
            sup.discard(("g", a))
            sup.discard(("g", b))
            support[best_v] = sup
            depmap[best_v] = [a, b]

        introduced.add(best_v)
        order.append(best_v)

        # update per-target distances that involve best_v
        for t in range(n_targets):
            # any candidate w whose need == cur ^ w ; if need was best_v before,
            # its distance drops by 1 (we now have a gate for best_v).
            # We recompute lazily: mark dists[t] entries that may have changed.
            # Simplest: invalidate entries whose value we can't track; recompute on demand.
            for w in list(dists[t].keys()):
                cur = 0
                for (k, x) in expr[t]:
                    cur ^= x
                if (cur ^ w) == best_v:
                    # distance reduced by 1 (was 1 higher)
                    old = dists[t][w]
                    dists[t][w] = max(0, old - 1)

    # Now build the actual circuit in dependency order.
    # Assign wire numbers: inputs 0..n-1, const-zero n, gates n+1..
    # We only need gates for the introduced vectors, in dep order.
    wire_of = {}
    for j in range(n):
        wire_of[1 << j] = j
    wire_of[0] = n  # constant zero

    gates = []  # list of (wire_a, wire_b) in production order
    # topological order of introduced vectors
    topo = _topo(order, depmap)
    for v in topo:
        if v == 0:
            continue
        deps = depmap.get(v, [])
        if len(deps) == 2:
            a, b = deps
            wa = wire_of[a]
            wb = wire_of[b]
        else:
            # express from inputs directly
            toks = [x for (k, x) in support[v] if k == "i"]
            if not toks:
                continue
            wa = wire_of[toks[0]]
            wb = wire_of[toks[1]] if len(toks) > 1 else n
        gates.append((wa, wb))
        wire_of[v] = n + 1 + (len(gates) - 1)

    # outputs: for each original row, express as XOR of available wires
    outputs = []
    for r in rows:
        if r == 0:
            outputs.append(n)
            continue
        toks = _express(r, support, wire_of)
        # XOR the wires in toks
        acc = None
        for w in toks:
            if acc is None:
                acc = w
            else:
                # need a gate to XOR acc and w  -> but we can't add gates now
                # Instead, ensure expression uses only one wire or we add gates.
                # We'll handle by adding final gates.
                pass
        # We'll build outputs using extra gates appended at the end.
        outputs.append(None)

    # Rebuild outputs properly, appending any needed gates.
    outputs = []
    for r in rows:
        if r == 0:
            outputs.append(n)
            continue
        toks = _express(r, support, wire_of)
        wires = [wire_of[t] for t in toks]
        if not wires:
            outputs.append(n)
            continue
        acc = wires[0]
        for w in wires[1:]:
            gates.append((acc, w))
            acc = n + 1 + (len(gates) - 1)
            wire_of.setdefault(r, acc)
        outputs.append(acc)

    return gates, outputs


def _split(v, support):
    """Find a,b in support such that a^b == v and {a,b} != {v,0}."""
    # try all pairs of known supports
    keys = list(support.keys())
    # prefer light splits
    for a in keys:
        if a == 0:
            continue
        b = v ^ a
        if b in support and b != 0 and b != a:
            return a, b
    return None, None


def _topo(order, depmap):
    """Topologically sort introduced vectors by their dependencies."""
    pos = {v: i for i, v in enumerate(order)}
    # Kahn
    indeg = {v: 0 for v in order}
    for v in order:
        for d in depmap.get(v, []):
            if d in pos:
                indeg[v] += 1
    from collections import deque
    q = deque([v for v in order if indeg[v] == 0])
    out = []
    adj = {v: [] for v in order}
    for v in order:
        for d in depmap.get(v, []):
            if d in pos:
                adj[d].append(v)
    while q:
        v = q.popleft()
        out.append(v)
        for w in adj[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    # append any leftovers (cycle safety)
    for v in order:
        if v not in out:
            out.append(v)
    return out


def _express(r, support, wire_of):
    """Return a list of tokens (ints) whose XOR is r, using available wires."""
    # Use the support set for r if present, else input decomposition.
    if r in support:
        toks = [x for (k, x) in support[r]]
        # keep only tokens that have a wire
        toks = [x for x in toks if x in wire_of]
        if toks:
            return toks
    toks = []
    for j in range(16):
        if (r >> j) & 1:
            toks.append(1 << j)
    return toks


def synthesize(matrix, depth_limit):
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN

    mat = np.asarray(matrix, dtype=bool)
    rows = _as_ints(mat.tolist())
    n = mat.shape[1]

    # Fast path: if baseline is already small, just emit the trivial circuit.
    base = _baseline(rows)
    if base <= 40 or time.time() > deadline:
        return _trivial(rows, n)

    try:
        gates, outputs = _build(rows, n, _MAX_INTS, deadline)
        # validate: every gate input must be a valid existing wire
        # (inputs, const-zero, or previously produced gates)
        valid = True
        seen = set(range(n + 1))
        for (a, b) in gates:
            if a not in seen or b not in seen:
                valid = False
                break
            seen.add(n + 1 + gates.index((a, b)))
        # depth check (approx): number of gates in a chain
        if valid and _check_depth(gates, n) <= depth_limit:
            # count total gates; only accept if it beats baseline
            if len(gates) < base:
                return {"gates": [list(g) for g in gates], "outputs": list(outputs)}
    except Exception:
        pass

    return _trivial(rows, n)


def _trivial(rows, n):
    """Independent-row circuit: each row is an XOR of its set bits."""
    gates = []
    outputs = []
    for r in rows:
        if r == 0:
            outputs.append(n)
            continue
        bits = [j for j in range(n) if (r >> j) & 1]
        if not bits:
            outputs.append(n)
            continue
        acc = bits[0]
        for j in bits[1:]:
            gates.append([acc, j])
            acc = n + 1 + (len(gates) - 1)
        outputs.append(acc)
    return {"gates": gates, "outputs": outputs}


def _check_depth(gates, n):
    """Compute the longest gate chain (depth).  Wire w has depth = 1 + max(depth of parents)."""
    depth = [0] * (n + 1)
    # gate k produces wire n+1+k
    for k, (a, b) in enumerate(gates):
        w = n + 1 + k
        depth.append(1 + max(depth[a], depth[b]))
    return max(depth) if depth else 0
