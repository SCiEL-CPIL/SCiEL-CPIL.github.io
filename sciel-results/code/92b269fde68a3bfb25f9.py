"""DAG pebbling: minimize load/store transfers under a fast-memory budget."""

def pebble(dag, fast_memory_budget):
    import collections
    from collections import deque

    # ---------- Parse DAG (defensive across common formats) ----------
    nodes = []
    parents = {}
    outputs = []
    inputs = []

    def _get(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    if isinstance(dag, dict):
        v = _get(dag, ('nodes', 'node', 'vertices', 'ids'))
        if v is not None:
            nodes = list(v)
        v = _get(dag, ('parents', 'preds', 'predecessors', 'inputs', 'in_edges'))
        if v is not None and isinstance(v, dict):
            for k, pv in v.items():
                parents[k] = list(pv)
        v = _get(dag, ('outputs', 'output', 'sinks', 'targets', 'result'))
        if v is not None:
            outputs = list(v)
        v = _get(dag, ('inputs', 'input', 'sources', 'roots'))
        if v is not None:
            inputs = list(v)
    elif isinstance(dag, (list, tuple)):
        for item in dag:
            if isinstance(item, dict):
                nid = _get(item, ('id', 'node', 'name', 'idx', 'i', 'v'))
                if nid is None:
                    nid = item
                nodes.append(nid)
                p = _get(item, ('parents', 'preds', 'predecessors', 'inputs'))
                parents[nid] = list(p) if p else []
                if item.get('output') or item.get('is_output') or item.get('sink'):
                    outputs.append(nid)
            else:
                nodes.append(item)

    node_set = set(nodes)
    for n in nodes:
        if n not in parents:
            parents[n] = []
        parents[n] = [p for p in parents[n] if p in node_set]

    if not inputs:
        inputs = [n for n in nodes if not parents[n]]
    input_set = set(inputs)

    children = collections.defaultdict(list)
    for n in nodes:
        for p in set(parents[n]):
            children[p].append(n)

    if not outputs:
        outputs = [n for n in nodes if not children[n]]
    output_set = set(outputs)

    # ---------- Topological order ----------
    indeg = {n: len(set(parents[n])) for n in nodes}
    q = deque([n for n in nodes if indeg[n] == 0])
    topo = []
    d = dict(indeg)
    while q:
        n = q.popleft()
        topo.append(n)
        for c in children[n]:
            d[c] -= 1
            if d[c] == 0:
                q.append(c)
    if len(topo) != len(nodes):
        topo = list(nodes)
    topo_pos = {n: i for i, n in enumerate(topo)}

    budget = int(fast_memory_budget)
    if budget < 1:
        budget = 1

    # ---------- Pebbling with furthest-next-use eviction ----------
    fast = set()          # red: in fast memory (counts toward budget)
    slow = set(inputs)    # blue: in slow memory, loadable
    ops = []

    children_topo = {n: sorted(children[n], key=lambda c: topo_pos.get(c, 10**9))
                     for n in nodes}
    nxt = {n: 0 for n in nodes}
    END = len(topo)
    INF = END + 10**9

    def next_use(n):
        lst = children_topo[n]
        i = nxt[n]
        if i < len(lst):
            return topo_pos[lst[i]]
        if n in output_set:
            return END
        return INF

    def evict_pick(cands):
        best, bv = None, -1
        for n in cands:
            nu = next_use(n)
            if nu > bv:
                bv, best = nu, n
        return best

    for n in topo:
        if n in input_set:
            continue
        par = list(set(parents[n]))
        # bring parents into fast memory
        for p in par:
            if p not in fast and p in slow:
                ops.append(['load', p])
                slow.discard(p)
                fast.add(p)
        # free a slot for n's result
        if len(fast) >= budget:
            cands = [b for b in fast if b not in set(par)]
            if cands:
                ev = evict_pick(cands)
                if ev is not None:
                    ops.append(['store', ev])
                    fast.discard(ev)
                    slow.add(ev)
            else:
                # recovery: spill everything, then reload exactly the parents
                for b in list(fast):
                    ops.append(['store', b])
                    fast.discard(b)
                    slow.add(b)
                for p in par:
                    if p in slow:
                        ops.append(['load', p])
                        slow.discard(p)
                        fast.add(p)
        # compute n (result lands in fast memory)
        ops.append(['compute', n])
        fast.add(n)
        # advance next-use pointers for n's parents
        for p in par:
            lst = children_topo[p]
            i = nxt[p]
            while i < len(lst) and lst[i] != n:
                i += 1
            if i < len(lst):
                i += 1
            nxt[p] = i

    # outputs must finish blue (in slow memory)
    for o in outputs:
        if o in fast:
            ops.append(['store', o])
            fast.discard(o)
            slow.add(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
