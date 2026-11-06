"""Communication-minimizing layered DAG pebbling."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    dag: dict with keys:
      'vertices': list of vertex ids (topologically ordered)
      'parents': dict vertex -> list of parent vertex ids (already topo ordered)
      'outputs': list of output vertex ids
      (possibly 'inputs' list; inputs start blue)
    Returns list of [op, vertex] with op in {load, store, compute, delete}.
    """
    t0 = time.time()
    DEADLINE = 60.0  # safety margin before 90s kill

    # ---- normalize dag structure ----
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        parents = dict(dag.get('parents', {}))
        outputs = list(dag.get('outputs', []))
        inputs = list(dag.get('inputs', []))
    else:
        raise ValueError("dag must be a dict")

    if not vertices:
        return []

    # topological order
    topo = list(vertices)
    # ensure parents are ordered before children
    # (they are given topo ordered; but re-derive to be safe)
    indeg = {v: 0 for v in topo}
    children = {v: [] for v in topo}
    for v in topo:
        ps = parents.get(v, [])
        for p in ps:
            if p not in indeg:
                indeg[p] = 0
                children[p] = []
            children[p].append(v)
            indeg[v] = indeg.get(v, 0) + 1
    from collections import deque
    q = deque([v for v in topo if indeg.get(v, 0) == 0])
    topo2 = []
    indeg2 = dict(indeg)
    while q:
        u = q.popleft()
        topo2.append(u)
        for c in children.get(u, []):
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    if len(topo2) != len(topo):
        # fallback to given order
        topo2 = topo
    else:
        topo = topo2

    # input vertices = those with no parents
    input_set = set(v for v in topo if not parents.get(v, []))
    if not inputs:
        inputs = [v for v in topo if v in input_set]
    input_set = set(inputs)

    # outputs
    if not outputs:
        outputs = [v for v in topo if v not in children or not children[v]]
    output_set = set(outputs)

    # children count (how many times a value is needed as a parent)
    use_count = {v: 0 for v in topo}
    for v in topo:
        for p in parents.get(v, []):
            use_count[p] += 1

    # ---- layered decomposition ----
    # assign each vertex a layer = longest path from an input
    layer = {}
    for v in topo:
        ps = parents.get(v, [])
        if not ps:
            layer[v] = 0
        else:
            layer[v] = 1 + max(layer[p] for p in ps)
    max_layer = max(layer.values()) if layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in topo:
        layers[layer[v]].append(v)

    # ---- pebbling schedule ----
    # We process layer by layer. A vertex is "red" (in fast memory) after compute
    # until all its children have been computed (i.e., until the next layer is done).
    # With bounded fast memory, we keep the current layer's values red while computing
    # the next layer, then release.
    #
    # Schedule:
    #   for each layer L (0..max_layer):
    #     for each v in layer L:
    #        for each parent p of v: ensure p is red (load if not)
    #        compute v
    #     after finishing layer L, the values of layer L are no longer needed
    #     (their only consumers are in layer L+1 which we just computed) -> delete them
    #   outputs must finish blue: store them at the end.
    #
    # But we must respect fast_memory_budget: at any time, number of red values <= budget.
    # We'll track red set and evict (delete) values whose consumers are all done.

    ops = []
    red = set()          # currently in fast memory (red)
    blue = set()         # currently in slow memory (stored / originally blue inputs)
    # inputs start blue
    for v in inputs:
        blue.add(v)

    # A value is "done" once all its children have been computed.
    pending_children = {v: len(children.get(v, [])) for v in topo}
    computed = set()

    def is_dead(v):
        # no remaining consumers
        return pending_children[v] <= 0

    def load(v):
        if v in red:
            return
        ops.append(["load", v])
        red.add(v)
        blue.discard(v)

    def store(v):
        if v in blue:
            return
        ops.append(["store", v])
        blue.add(v)
        red.discard(v)

    def delete(v):
        if v in red:
            ops.append(["delete", v])
            red.discard(v)

    def compute(v):
        ops.append(["compute", v])
        red.add(v)
        blue.discard(v)
        computed.add(v)
        for c in children.get(v, []):
            pending_children[c] -= 1

    # capacity check helper: if red would exceed budget, evict dead values
    def make_room(need=1):
        if len(red) < fast_memory_budget:
            return True
        # evict dead red values
        for v in list(red):
            if is_dead(v):
                delete(v)
                if len(red) < fast_memory_budget:
                    return True
        # if still over and no dead values, we must evict a live value -> store it
        # (this causes a later reload; acceptable)
        for v in list(red):
            if not is_dead(v):
                store(v)
                if len(red) < fast_memory_budget:
                    return True
        return len(red) < fast_memory_budget

    # main loop
    for L in range(max_layer + 1):
        for v in layers[L]:
            if time.time() - t0 > DEADLINE - 5:
                break
            # ensure parents are red
            for p in parents.get(v, []):
                if p not in red:
                    make_room(1)
                    load(p)
            # compute v
            make_room(1)
            compute(v)
        # after layer L, its values' consumers (layer L+1) are done -> they are dead
        # release them to free memory (except outputs which must stay until stored)
        for v in layers[L]:
            if is_dead(v) and v not in output_set:
                delete(v)

    # ensure outputs are blue
    for v in outputs:
        if v in red:
            store(v)

    # cap operations
    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

# EVOLVE-BLOCK-END
