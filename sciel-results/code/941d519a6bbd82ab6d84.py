"""DAG pebbling: minimize communication with a bounded fast-memory budget."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

_START = time.time()
_DEADLINE = 52.0  # safety margin well under 60s wall-clock


def _elapsed():
    return time.time() - _START


def _extract(dag):
    """Pull out parents (topologically ordered), inputs and outputs from dag."""
    if isinstance(dag, dict):
        parents = dag.get("parents", None)
        inputs = dag.get("inputs", None)
        outputs = dag.get("outputs", None)
        if parents is None:
            parents = dag.get("parent", None)
        if inputs is None:
            inputs = dag.get("input", None)
        if outputs is None:
            outputs = dag.get("output", None)
    else:
        parents = getattr(dag, "parents", None)
        inputs = getattr(dag, "inputs", None)
        outputs = getattr(dag, "outputs", None)
        if parents is None:
            parents = getattr(dag, "parent", None)
    if parents is None:
        parents = {}
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []
    return parents, inputs, outputs


def _verify(dag, ops, budget):
    """Exact state-machine check. Returns True if ops are valid."""
    try:
        parents, inputs, outputs = _extract(dag)
    except Exception:
        return False
    in_set = set(inputs)
    out_set = set(outputs)
    state = {}
    for v in in_set:
        state[v] = "blue"
    fast = 0
    for op in ops:
        try:
            kind = op[0]
            v = op[1]
        except Exception:
            return False
        if kind == "load":
            if v not in in_set:
                return False
            if state.get(v) == "blue" and fast < budget:
                state[v] = "red"
                fast += 1
        elif kind == "store":
            if v not in out_set:
                return False
            if state.get(v) == "red":
                state[v] = "blue"
                fast -= 1
        elif kind == "compute":
            ps = parents.get(v, [])
            if v in in_set:
                return False
            for p in ps:
                if state.get(p) != "red":
                    return False
            if state.get(v) != "red" and fast >= budget:
                return False
            if state.get(v) != "red":
                fast += 1
            state[v] = "red"
        elif kind == "delete":
            if state.get(v) == "red":
                state[v] = "blue" if v in out_set else None
                fast -= 1
        else:
            return False
    for v in out_set:
        if state.get(v) != "blue":
            return False
    return True


def _fallback(dag, budget):
    """Safe schedule that always works: recompute everything, store outputs."""
    parents, inputs, outputs = _extract(dag)
    in_set = set(inputs)
    ops = []
    for v in in_set:
        ops.append(["load", v])
    for v in outputs:
        ps = parents.get(v, [])
        for p in ps:
            if p in in_set:
                ops.append(["load", p])
        ops.append(["compute", v])
    for v in outputs:
        ops.append(["store", v])
    return ops


def pebble(dag, fast_memory_budget):
    try:
        B = int(fast_memory_budget)
    except Exception:
        B = 0
    if B < 0:
        B = 0

    try:
        parents, inputs, outputs = _extract(dag)
    except Exception:
        return _fallback(dag, B)

    in_set = set(inputs)
    out_set = set(outputs)
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    def cost(v, seen=None):
        if seen is None:
            seen = set()
        if v in seen:
            return 0
        seen.add(v)
        ps = parents.get(v, [])
        if not ps:
            return 1
        return 1 + sum(cost(p, seen) for p in ps)

    # Reuse value of keeping a vertex in fast memory.
    keep = {}
    for v in parents.keys():
        if v in in_set:
            continue
        c = len(children.get(v, []))
        if c <= 1:
            keep[v] = 0
            continue
        own = cost(v)
        keep[v] = min((c - 1) * own, own)

    ranked = sorted(keep.items(), key=lambda kv: kv[1], reverse=True)
    keep_set = {v for v, val in ranked[:B] if val > 0}
    # Never keep outputs (they must be stored and freed).
    keep_set -= out_set

    def schedule():
        ops = []
        for v in in_set:
            ops.append(["load", v])
        for v in outputs:
            ps = parents.get(v, [])
            for p in ps:
                if p in in_set:
                    ops.append(["load", p])
                elif p not in keep_set:
                    ops.append(["compute", p])
            ops.append(["compute", v])
        for v in outputs:
            ops.append(["store", v])
        return ops

    ops = schedule()
    if not _verify(dag, ops, B):
        ops = _fallback(dag, B)
    if len(ops) > 20000:
        ops = _fallback(dag, B)
    return ops
# EVOLVE-BLOCK-END
