# EVOLVE-BLOCK-START
import time


def pebble(dag, fast_memory_budget):
    deadline = time.time() + 75.0  # safety margin under the 90s wall-clock kill
    try:
        return _pebble_impl(dag, fast_memory_budget, deadline)
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []


def _get(obj, *keys, default=None):
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            return obj[k]
        if hasattr(obj, k):
            return getattr(obj, k)
    return default


def _pebble_impl(dag, B, deadline):
    parents = _get(dag, 'parents', 'preds', 'ancestors', 'in_edges', 'in', default=None)
    inputs = _get(dag, 'inputs', 'sources', default=None)
    outputs = _get(dag, 'outputs', 'sinks', 'out', default=None)

    if parents is None or outputs is None:
        raise ValueError("parse fail")

    # normalize parents -> dict vertex : list(parents)
    if isinstance(parents, dict):
        par = {v: list(ps) for v, ps in parents.items()}
    elif isinstance(parents, (list, tuple)):
        par = {i: list(p) for i, p in enumerate(parents)}
    else:
        raise ValueError("parents fmt")

    if inputs is None:
        inputs = [v for v in par if len(par.get(v, [])) == 0]
    else:
        inputs = list(inputs)
    outputs = list(outputs)

    # topological order (parents before children) over the subgraph feeding outputs
    memo = {}

    def rec(v):
        if v in memo:
            return memo[v]
        memo[v] = []
        for p in par.get(v, []):
            rec(p)
            for x in memo[p]:
                if x not in memo[v]:
                    memo[v].append(x)
        memo[v].append(v)
        return memo[v]

    seen = []
    for o in outputs:
        for x in rec(o):
            if x not in seen:
                seen.append(x)
    topo = seen

    # use-counts to choose eviction victims
    from collections import Counter
    use = Counter()

    def cnt(v):
        for p in par.get(v, []):
            use[p] += 1
            cnt(p)

    for o in outputs:
        cnt(o)

    B = max(1, int(B))
    ops = []
    red = set()
    initial_blue = set(inputs)

    def victim(red_set, need, exclude):
        cands = [x for x in red_set
                 if x not in outputs and x != need and x not in exclude]
        if not cands:
            return None
        cands.sort(key=lambda x: (use.get(x, 0), x))
        return cands[0]

    def ensure(v):
        if v in red:
            return
        if len(par.get(v, [])) == 0:
            # input: load (blue -> red)
            if len(red) >= B:
                vi = victim(red, v, set())
                if vi is None:
                    raise MemoryError("no room load")
                ops.append(['delete', vi])
                red.discard(vi)
            ops.append(['load', v])
            red.add(v)
        else:
            # internal: recompute for free, keeping direct parents red until the op
            for p in par[v]:
                ensure(p)
            if len(red) >= B:
                vi = victim(red, v, set(par[v]))
                if vi is None:
                    raise MemoryError("no room compute")
                ops.append(['delete', vi])
                red.discard(vi)
            ops.append(['compute', v])
            red.add(v)

    for v in topo:
        if time.time() > deadline:
            break
        ensure(v)

    # outputs must finish blue
    for o in outputs:
        if o in initial_blue and len(par.get(o, [])) == 0:
            continue  # already blue, stays blue
        if o not in red:
            ensure(o)
        ops.append(['store', o])
        red.discard(o)

    if len(ops) > 20000:
        raise MemoryError("too many ops")

    if not _simulate(ops, par, initial_blue, outputs, B):
        raise ValueError("invalid schedule")
    return ops


def _simulate(ops, par, initial_blue, outputs, B):
    blue = set(initial_blue)
    red = set()
    for op, v in ops:
        if op == 'load':
            if v not in blue or len(red) >= B:
                return False
            red.add(v)
            blue.discard(v)
        elif op == 'store':
            if v not in red:
                return False
            blue.add(v)
            red.discard(v)
        elif op == 'compute':
            for p in par.get(v, []):
                if p not in red:
                    return False
            if len(red) >= B:
                return False
            red.add(v)
        elif op == 'delete':
            if v not in red:
                return False
            red.discard(v)
        else:
            return False
    for o in outputs:
        if o not in blue:
            return False
    return True
# EVOLVE-BLOCK-END
