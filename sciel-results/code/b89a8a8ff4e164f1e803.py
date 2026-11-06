import time

def pebble(dag, fast_memory_budget):
    start = time.time()
    deadline = start + 85

    n = len(dag['parents'])
    parents = [list(p) for p in dag['parents']]
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])

    ops = []
    fast = set()
    slow = set(inputs)

    def load(v):
        if v in slow and v not in fast:
            ops.append(['load', v])
            fast.add(v)

    def store(v):
        if v in fast:
            ops.append(['store', v])
            slow.add(v)
            fast.discard(v)

    def compute(v):
        ops.append(['compute', v])
        fast.add(v)

    def evict_one():
        for v in sorted(fast):
            if v not in outputs:
                store(v)
                return

    for v in range(n):
        if time.time() > deadline:
            break
        if v in inputs:
            load(v)
        else:
            for p in parents[v]:
                if p not in fast:
                    if p in slow:
                        load(p)
                    else:
                        for pp in parents[p]:
                            if pp not in fast and pp in slow:
                                load(pp)
                        if len(fast) >= fast_memory_budget:
                            evict_one()
                        compute(p)
            if len(fast) >= fast_memory_budget:
                evict_one()
            compute(v)

    for v in outputs:
        if v not in fast and v in slow:
            load(v)

    return ops
