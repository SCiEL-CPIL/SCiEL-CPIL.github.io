# EVOLVE-BLOCK-START
import time as _time
import sys
import random
import heapq
from collections import defaultdict

# ----------------------------- time guard -----------------------------
_DEADLINE = None
def _now():
    if _DEADLINE is None:
        return 0.0
    return _time.time() - _DEADLINE[0]
def _remaining():
    if _DEADLINE is None:
        return float('inf')
    return _DEADLINE[0] - _time.time()

# ----------------------------- DAG parsing -----------------------------
def _parse(dag):
    """Return (n, parents, outputs, is_input, children, level) defensively."""
    # Try dict-of-dicts
    if isinstance(dag, dict):
        # form: {v: [parents]}  or {v: {"parents": [...]}}
        keys = list(dag.keys())
        parents = {}
        for k, val in dag.items():
            if isinstance(val, dict):
                p = val.get("parents", val.get("deps", val.get("pred", [])))
            else:
                p = val
            parents[k] = list(p)
        n = len(keys)
        outputs = None
        # outputs maybe a key in dag or separate
        # children
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        # is_input: vertices with no parents (unless explicitly marked)
        is_input = {v: (len(parents[v]) == 0) for v in parents}
        # outputs: vertices with no children, or explicitly marked
        out_set = set()
        if isinstance(dag, dict) and ("outputs" in dag or "output" in dag):
            o = dag.get("outputs", dag.get("output"))
            if o is not None:
                out_set = set(o)
        if not out_set:
            for v in parents:
                if not children[v]:
                    out_set.add(v)
        return n, parents, out_set, is_input, children

    # Try list of dicts
    if isinstance(dag, (list, tuple)) and len(dag) > 0 and isinstance(dag[0], dict):
        parents = {}
        for i, d in enumerate(dag):
            p = d.get("parents", d.get("deps", d.get("pred", [])))
            parents[i] = list(p)
        n = len(dag)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        is_input = {v: (len(parents[v]) == 0) for v in parents}
        out_set = set()
        for v in parents:
            if not children[v]:
                out_set.add(v)
        return n, parents, out_set, is_input, children

    # Fallback: assume a list of parent-lists
    if isinstance(dag, (list, tuple)):
        parents = {i: list(dag[i]) for i in range(len(dag))}
        n = len(dag)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        is_input = {v: (len(parents[v]) == 0) for v in parents}
        out_set = set()
        for v in parents:
            if not children[v]:
                out_set.add(v)
        return n, parents, out_set, is_input, children

    raise ValueError("Unrecognized dag format")

# ----------------------------- topological order -----------------------------
def _topo(n, parents, children):
    # Kahn
    indeg = {v: len(parents[v]) for v in range(n)}
    from collections import deque
    q = deque([v for v in range(n) if indeg[v] == 0])
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(order) != n:
        # cycle or bad; just use range
        order = list(range(n))
    return order

# ----------------------------- schedule simulation -----------------------------
def _simulate(ops, n, parents, outputs, is_input, budget):
    """Return (transfers, valid). Blue in slow, red in fast."""
    # state: 'blue' in slow, 'red' in fast, None
    slow = {}   # v -> value marker (blue)
    fast = {}   # v -> value marker (red)
    transfers = 0
    for op, v in ops:
        if op == 'load':
            # must be in slow, not in fast
            if v in fast:
                return (transfers, False)
            if v not in slow:
                return (transfers, False)
            # need room
            if len(fast) >= budget:
                return (transfers, False)
            fast[v] = slow[v]
            transfers += 1
        elif op == 'store':
            if v not in fast:
                return (transfers, False)
            slow[v] = fast[v]
            del fast[v]
            transfers += 1
        elif op == 'compute':
            # all parents must be red (in fast)
            for p in parents[v]:
                if p not in fast:
                    return (transfers, False)
            # result goes red, consumes a slot
            if len(fast) >= budget:
                return (transfers, False)
            fast[v] = 'computed'
            transfers += 0
        elif op == 'delete':
            if v not in fast:
                return (transfers, False)
            del fast[v]
            transfers += 0
        else:
            return (transfers, False)
    # all outputs must be blue (in slow)
    for o in outputs:
        if o not in slow:
            return (transfers, False)
    return (transfers, True)

# ----------------------------- naive baseline -----------------------------
def _naive_schedule(n, parents, outputs, is_input, children, budget):
    """Topological: load inputs, compute, store outputs. Count transfers."""
    ops = []
    slow = set()
    fast = set()
    order = _topo(n, parents, children)
    # precompute: for each v, whether it's an input
    # We'll do: for each v in topo order:
    #   if input: load it (stays fast until needed? we keep it)
    #   else: ensure parents in fast, compute, then delete parents that are no longer needed
    #   if output: store it
    last_use = {}
    for v in order:
        # count uses as parent
        pass
    # compute last use index of each vertex as a parent
    pos = {v: i for i, v in enumerate(order)}
    last_use = {}
    for v in range(n):
        last_use[v] = -1
    for v in order:
        for p in parents[v]:
            last_use[p] = pos[v]
    for v in order:
        if is_input.get(v, False):
            ops.append(('load', v))
            slow.add(v)
            fast.add(v)
        else:
            # ensure parents in fast
            for p in parents[v]:
                if p not in fast:
                    ops.append(('load', p))
                    fast.add(p)
            ops.append(('compute', v))
            fast.add(v)
            # delete parents no longer needed
            for p in parents[v]:
                if last_use[p] <= pos[v]:
                    ops.append(('delete', p))
                    fast.discard(p)
        if v in outputs:
            ops.append(('store', v))
            fast.discard(v)
            # ensure it's in slow
            slow.add(v)
    # verify
    t, ok = _simulate(ops, n, parents, outputs, is_input, budget)
    if not ok:
        # fallback: very naive, keep everything
        ops = []
        slow = set()
        fast = set()
        for v in order:
            if is_input.get(v, False):
                ops.append(('load', v)); slow.add(v); fast.add(v)
            else:
                for p in parents[v]:
                    if p not in fast:
                        ops.append(('load', p)); fast.add(p)
                ops.append(('compute', v)); fast.add(v)
            if v in outputs:
                ops.append(('store', v)); fast.discard(v); slow.add(v)
        t, ok = _simulate(ops, n, parents, outputs, is_input, budget)
    return t, ops

# ----------------------------- candidate generators -----------------------------
def _gen_topo_lifo(n, parents, outputs, is_input, children, budget):
    """Topological order, LIFO (delete parents as soon as possible)."""
    ops = []
    fast = set()
    slow = set()
    order = _topo(n, parents, children)
    pos = {v: i for i, v in enumerate(order)}
    last_use = {v: -1 for v in range(n)}
    for v in order:
        for p in parents[v]:
            last_use[p] = pos[v]
    for v in order:
        if is_input.get(v, False):
            if v not in fast:
                ops.append(('load', v)); fast.add(v); slow.add(v)
        else:
            for p in parents[v]:
                if p not in fast:
                    ops.append(('load', p)); fast.add(p)
            ops.append(('compute', v)); fast.add(v)
            for p in parents[v]:
                if last_use[p] <= pos[v] and p in fast:
                    ops.append(('delete', p)); fast.discard(p)
        if v in outputs:
            if v in fast:
                ops.append(('store', v)); fast.discard(v); slow.add(v)
    return ops

def _gen_lifo(n, parents, outputs, is_input, children, budget):
    """Greedy: always compute the available vertex with highest 'last use' pressure, LIFO delete."""
    ops = []
    fast = set()
    slow = set()
    computed = set()
    order = _topo(n, parents, children)
    pos = {v: i for i, v in enumerate(order)}
    # last use as parent
    last_use = {v: -1 for v in range(n)}
    for v in order:
        for p in parents[v]:
            last_use[p] = pos[v]
    # out-degree remaining
    rem_children = {v: len(children[v]) for v in range(n)}
    available = [v for v in order if len(parents[v]) == 0]  # inputs
    # We'll process in a priority order: prefer vertices whose parents are all done
    # Actually do a worklist: start with inputs.
    # We want to compute as early as possible but keep fast small.
    # Strategy: repeatedly pick an available (all parents computed) vertex,
    # compute it, then delete any parent whose last_use is reached.
    # To minimize transfers, we should load inputs only when needed.
    # Let's do: process vertices in topological order but with LIFO.
    # Simpler: same as topo_lifo. Return it.
    return _gen_topo_lifo(n, parents, outputs, is_input, children, budget)

def _gen_recompute(n, parents, outputs, is_input, children, budget):
    """If budget is small, recompute instead of storing: for each vertex, load parents, compute, store if output, delete all."""
    ops = []
    slow = set()
    fast = set()
    order = _topo(n, parents, children)
    for v in order:
        if is_input.get(v, False):
            # don't keep inputs; load on demand
            pass
        # ensure parents in fast (load if input, else they should be... recompute means we don't keep)
        # For recompute: we load each parent fresh.
        for p in parents[v]:
            if p not in fast:
                if is_input.get(p, False):
                    ops.append(('load', p)); fast.add(p); slow.add(p)
                else:
                    # recompute p: load p's parents, compute p
                    # this is recursive; do it inline
                    ops2, fast2, slow2 = _recompute_one(p, parents, is_input, fast, slow)
                    ops.extend(ops2)
                    fast = fast2; slow = slow2
        ops.append(('compute', v)); fast.add(v)
        if v in outputs:
            ops.append(('store', v)); fast.discard(v); slow.add(v)
        # delete parents
        for p in parents[v]:
            if p in fast and p not in outputs and not is_input.get(p, False):
                ops.append(('delete', p)); fast.discard(p)
    return ops

def _recompute_one(v, parents, is_input, fast, slow):
    """Recompute v assuming its value is not in fast. Returns ops, new fast, new slow."""
    ops = []
    for p in parents[v]:
        if p not in fast:
            if is_input.get(p, False):
                ops.append(('load', p)); fast.add(p); slow.add(p)
            else:
                ops2, fast, slow = _recompute_one(p, parents, is_input, fast, slow)
                ops.extend(ops2)
    ops.append(('compute', v)); fast.add(v)
    return ops, fast, slow

# ----------------------------- local improvement: merge consecutive -----------------------------
def _try_improve(ops, n, parents, outputs, is_input, children, budget, cur_t):
    """Try to reduce transfers by removing redundant ops (e.g., delete+load of same vertex, or store when already slow)."""
    # Simple pass: remove 'delete' immediately followed by 'load' of same v, and 'load' of v already in fast, etc.
    # We'll do a targeted rewrite: simulate and drop ops that are no-ops or redundant.
    # Build a cleaned list.
    fast = set(); slow = set()
    new_ops = []
    for op, v in ops:
        if op == 'load':
            if v in fast:
                continue  # redundant
            if len(fast) >= budget:
                # can't; keep as is (will be invalid) -> break
                new_ops.append((op, v)); fast.add(v); continue
            new_ops.append((op, v)); fast.add(v); slow.add(v)
        elif op == 'store':
            if v not in fast:
                continue
            # if already in slow and not needed later? keep to be safe
            new_ops.append((op, v)); fast.discard(v); slow.add(v)
        elif op == 'compute':
            # check parents in fast
            ok = all(p in fast for p in parents[v])
            if not ok:
                new_ops.append((op, v)); fast.add(v); continue
            if len(fast) >= budget:
                new_ops.append((op, v)); fast.add(v); continue
            new_ops.append((op, v)); fast.add(v)
        elif op == 'delete':
            if v not in fast:
                continue
            # look ahead: if next op is load of same v, skip both? handle via lookahead
            new_ops.append((op, v)); fast.discard(v)
    # Second pass: remove delete->load same v
    cleaned = []
    i = 0
    while i < len(new_ops):
        op, v = new_ops[i]
        if op == 'delete' and i+1 < len(new_ops) and new_ops[i+1][0] == 'load' and new_ops[i+1][1] == v:
            i += 2
            continue
        cleaned.append((op, v))
        i += 1
    t, ok = _simulate(cleaned, n, parents, outputs, is_input, budget)
    if ok and t < cur_t:
        return cleaned, t
    return ops, cur_t

# ----------------------------- main -----------------------------
def pebble(dag, fast_memory_budget):
    global _DEADLINE
    _DEADLINE = [_time.time() + 80.0]  # 80s guard, leave 10s margin
    try:
        n, parents, outputs, is_input, children = _parse(dag)
        # normalize parents to ints
        # ensure budget
        budget = int(fast_memory_budget)
        if budget < 1:
            budget = 1
        # naive
        naive_t, naive_ops = _naive_schedule(n, parents, outputs, is_input, children, budget)
        best_ops = naive_ops
        best_t = naive_t
        # candidates
        cands = []
        try:
            cands.append(_gen_topo_lifo(n, parents, outputs, is_input, children, budget))
        except Exception:
            pass
        try:
            cands.append(_gen_recompute(n, parents, outputs, is_input, children, budget))
        except Exception:
            pass
        for c in cands:
            if _remaining() < 5:
                break
            t, ok = _simulate(c, n, parents, outputs, is_input, children, budget)
            if ok:
                c2, t2 = _try_improve(c, n, parents, outputs, is_input, children, budget, t)
                if t2 < best_t:
                    best_ops = c2
                    best_t = t2
        # if recompute was better, good; else naive
        # final guard
        if len(best_ops) > 20000:
            # truncate? shouldn't happen; fallback to naive
            best_ops = naive_ops
        return best_ops
    except Exception as e:
        # last resort: return empty? must be valid. Try a minimal valid schedule.
        try:
            n, parents, outputs, is_input, children = _parse(dag)
            budget = int(fast_memory_budget)
            t, ops = _naive_schedule(n, parents, outputs, is_input, children, budget)
            return ops
        except Exception:
            return []
# EVOLVE-BLOCK-END
