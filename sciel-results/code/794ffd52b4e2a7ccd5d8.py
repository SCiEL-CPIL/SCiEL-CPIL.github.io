"""Rewritten pebbling: topological schedule + dead-value deletion + self-verify fallback."""
import time
import bisect
import sys

sys.setrecursionlimit(100000)

_DEADLINE = 85.0  # safety margin under the 90s wall-clock kill


def _extract(dag):
    parents = getattr(dag, "parents", None)
    if parents is None:
        parents = dag["parents"]
    n = getattr(dag, "n", None)
    if n is None:
        n = len(parents)
    outputs = getattr(dag, "outputs", None)
    if outputs is None:
        outputs = dag.get("outputs", [])
    parents = [list(p) for p in parents]
    outputs = list(outputs)
    return n, parents, outputs


def _build(n, parents):
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    for v in range(n):
        children[v].sort()
    # transitive ancestors
    anc = [set() for _ in range(n)]
    for v in range(n):
        s = set()
        for p in parents[v]:
            s.add(p)
            s |= anc[p]
        anc[v] = s
    return children, anc


def _gen(n, parents, outputs, children, anc, budget, allow_delete, t0):
    ops = []
    out_set = set(outputs)
    state = [0] * n  # 0 white, 1 blue, 2 red
    for v in range(n):
        if not parents[v]:
            state[v] = 1
    fast = set()

    def next_use(u, cur):
        ch = children[u]
        i = bisect.bisect_left(ch, cur)
        if i < len(ch):
            return ch[i]
        return None

    def evict(protected, cur):
        best = None
        best_key = None
        for u in list(fast):
            if u in protected:
                continue
            nu = next_use(u, cur)
            if nu is None and u not in out_set:
                key = (1, 0)  # dead -> delete (free)
            else:
                key = (0, -(nu if nu is not None else 10 ** 9))
            if best_key is None or key > best_key:
                best_key = key
                best = u
        if best is None:
            return False
        nu = next_use(best, cur)
        if allow_delete and nu is None and best not in out_set:
            ops.append(["delete", best])
            state[best] = 0
        else:
            ops.append(["store", best])
            state[best] = 1
        fast.discard(best)
        return True

    def ensure_red(v, protected, cur):
        if state[v] == 2:
            return True
        if state[v] == 1:
            while len(fast) >= budget:
                if len(ops) >= 20000 or time.time() - t0 > _DEADLINE:
                    return False
                if not evict(protected, cur):
                    return False
            ops.append(["load", v])
            state[v] = 2
            fast.add(v)
            return True
        # white -> recompute
        for p in parents[v]:
            if not ensure_red(p, protected, cur):
                return False
        while len(fast) >= budget:
            if len(ops) >= 20000 or time.time() - t0 > _DEADLINE:
                return False
            if not evict(protected, cur):
                return False
        ops.append(["compute", v])
        state[v] = 2
        fast.add(v)
        return True

    for v in range(n):
        if time.time() - t0 > _DEADLINE or len(ops) >= 20000:
            return None
        if parents[v]:
            protected = anc[v] | {v}
            if not ensure_red(v, protected, v):
                return None
            while len(fast) > budget:
                if not evict(set(), v):
                    break
    # final: outputs blue
    for v in outputs:
        if time.time() - t0 > _DEADLINE or len(ops) >= 20000:
            return None
        if state[v] == 2:
            ops.append(["store", v])
            state[v] = 1
            fast.discard(v)
        elif state[v] == 0:
            protected = anc[v] | {v}
            if not ensure_red(v, protected, n):
                return None
            while len(fast) > budget:
                if not evict(set(), n):
                    break
            ops.append(["store", v])
            state[v] = 1
            fast.discard(v)
    return ops


def _verify(ops, n, parents, outputs, budget):
    if ops is None or len(ops) > 20000:
        return False
    state = [0] * n
    for v in range(n):
        if not parents[v]:
            state[v] = 1
    fast = 0
    for op, v in ops:
        if op == "load":
            if state[v] != 1:
                return False
            state[v] = 2
            fast += 1
        elif op == "store":
            if state[v] != 2:
                return False
            state[v] = 1
            fast -= 1
        elif op == "delete":
            if state[v] != 2:
                return False
            state[v] = 0
            fast -= 1
        elif op == "compute":
            for p in parents[v]:
                if state[p] != 2:
                    return False
            state[v] = 2
            fast += 1
        else:
            return False
        if fast > budget:
            return False
    for v in outputs:
        if state[v] != 1:
            return False
    return True


def pebble(dag, fast_memory_budget):
    t0 = time.time()
    try:
        n, parents, outputs = _extract(dag)
        children, anc = _build(n, parents)

        # Optimized: allow free deletion of dead values
        cand = _gen(n, parents, outputs, children, anc, fast_memory_budget,
                    allow_delete=True, t0=t0)
        if _verify(cand, n, parents, outputs, fast_memory_budget):
            return cand

        # Fallback: store-only eviction (conservative, still correct)
        safe = _gen(n, parents, outputs, children, anc, fast_memory_budget,
                    allow_delete=False, t0=t0)
        if _verify(safe, n, parents, outputs, fast_memory_budget):
            return safe

        # Last resort: minimal correct schedule (load on demand, store outputs)
        # Build a very conservative schedule that is correct by construction.
        ops = []
        state = [0] * n
        for v in range(n):
            if not parents[v]:
                state[v] = 1
        fast = set()

        def evict_any(protected):
            for u in list(fast):
                if u in protected:
                    continue
                ops.append(["store", u])
                state[u] = 1
                fast.discard(u)
                return True
            return False

        def ensure(v, protected):
            if state[v] == 2:
                return True
            if state[v] == 1:
                while len(fast) >= fast_memory_budget and not evict_any(protected | {v}):
                    return False
                ops.append(["load", v])
                state[v] = 2
                fast.add(v)
                return True
            for p in parents[v]:
                if not ensure(p, protected | {v}):
                    return False
            while len(fast) >= fast_memory_budget and not evict_any(protected | {v}):
                return False
            ops.append(["compute", v])
            state[v] = 2
            fast.add(v)
            return True

        for v in range(n):
            if parents[v]:
                if not ensure(v, set(parents[v]) | {v}):
                    break
                while len(fast) > fast_memory_budget and evict_any(set()):
                    pass
        for v in outputs:
            if state[v] == 2:
                ops.append(["store", v])
                state[v] = 1
                fast.discard(v)
            elif state[v] == 0:
                if not ensure(v, set(parents[v]) | {v}):
                    break
                while len(fast) > fast_memory_budget and evict_any(set()):
                    pass
                ops.append(["store", v])
                state[v] = 1
                fast.discard(v)
        if _verify(ops, n, parents, outputs, fast_memory_budget):
            return ops
        return []
    except Exception:
        return []
