"""DAG pebbling: communication-minimizing scheduler."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 85.0

    vertices = list(dag['vertices'])
    parents = dag['parents']
    outputs = list(dag['outputs'])
    B = max(0, int(fast_memory_budget))

    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)

    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    output_set = set(outputs)

    if B == 0:
        ops = []
        for v in vertices:
            for p in parents.get(v, []):
                ops.append(['load', p])
            ops.append(['compute', v])
            if in_degree[v] > 0:
                ops.append(['store', v])
        for o in outputs:
            ops.append(['load', o])
        return ops

    fast_mem = set()
    slow_stored = set()
    ops = []
    unprocessed_parents = {v: in_degree[v] for v in vertices}

    def do_load(v):
        if v not in fast_mem:
            fast_mem.add(v)
            ops.append(['load', v])

    def do_store(v):
        if v not in slow_stored:
            slow_stored.add(v)
            ops.append(['store', v])

    def do_compute(v):
        ops.append(['compute', v])
        fast_mem.add(v)

    def do_delete(v):
        if v in fast_mem:
            fast_mem.discard(v)
            ops.append(['delete', v])

    for v in vertices:
        if len(ops) >= 19500:
            break
        if time.time() - t0 > DEADLINE:
            break

        for p in parents.get(v, []):
            if p not in fast_mem:
                do_load(p)

        do_compute(v)

        for c in children[v]:
            unprocessed_parents[c] -= 1

        while len(fast_mem) > B:
            best = None
            best_score = None
            for u in fast_mem:
                if u == v or u in output_set:
                    continue
                needed = 0
                for c in children[u]:
                    if unprocessed_parents[c] > 0:
                        needed += 1
                score = (0 if needed == 0 else 1, needed)
                if best_score is None or score < best_score:
                    best_score = score
                    best = u
            if best is None:
                break
            do_store(best)
            do_delete(best)

    for o in outputs:
        if o not in fast_mem:
            do_load(o)

    return ops
# EVOLVE-BLOCK-END
