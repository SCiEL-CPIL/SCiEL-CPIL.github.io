"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    import time
    _start = time.time()
    _DEADLINE = 80.0

    # --- Parse DAG structure ---
    if isinstance(dag, dict):
        n = dag.get('vertices', dag.get('n', 0))
        parents = dag.get('parents', [])
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
    else:
        n = len(dag) if hasattr(dag, '__len__') else 0
        parents = []
        outputs = []
        inputs = []

    if n == 0:
        return []

    # Build children
    children = [[] for _ in range(n)]
    for i in range(n):
        if i < len(parents):
            for p in parents[i]:
                children[p].append(i)

    if not inputs:
        inputs = [i for i in range(n) if (i >= len(parents) or len(parents[i]) == 0)]
    if not outputs:
        outputs = [i for i in range(n) if len(children[i]) == 0]
    output_set = set(outputs)
    input_set = set(inputs)

    # State
    in_fast = set()
    in_slow_blue = set(inputs)

    ops = []
    ops_count = 0

    def add_op(op, v):
        nonlocal ops_count
        if ops_count >= 19999:
            return False
        ops.append([op, v])
        ops_count += 1
        return True

    remaining_children = [len(children[i]) for i in range(n)]

    for v in range(n):
        if not time.time() - _start < _DEADLINE:
            break

        if v in input_set:
            continue

        # Load parents not in fast memory
        for p in parents[v]:
            if p not in in_fast:
                if add_op('load', p):
                    in_fast.add(p)
                    in_slow_blue.discard(p)

        # Compute
        if add_op('compute', v):
            in_fast.add(v)

        # Update parents
        for p in parents[v]:
            if p in in_fast:
                remaining_children[p] -= 1
                if remaining_children[p] == 0:
                    if p in output_set:
                        if add_op('store', p):
                            in_fast.discard(p)
                            in_slow_blue.add(p)
                    else:
                        if add_op('delete', p):
                            in_fast.discard(p)

        # Handle v
        if v in output_set:
            if add_op('store', v):
                in_fast.discard(v)
                in_slow_blue.add(v)
        elif remaining_children[v] > 0:
            if len(in_fast) > fast_memory_budget:
                # Over budget, store v
                if add_op('store', v):
                    in_fast.discard(v)
                    in_slow_blue.add(v)
        else:
            if add_op('delete', v):
                in_fast.discard(v)

    return ops
# EVOLVE-BLOCK-END

