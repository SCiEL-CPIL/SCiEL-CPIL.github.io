"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin out of 60s

    try:
        return _pebble_optimized(dag, fast_memory_budget, deadline)
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

def _pebble_optimized(dag, fast_memory_budget, deadline):
    """
    Optimized pebbling strategy using LRU-like eviction with next-use analysis.
    """
    vertices = list(dag.get_vertices())
    n = len(vertices)
    if n == 0:
        return []

    parents = {v: dag.get_parents(v) for v in vertices}
    inputs = set(dag.get_inputs())
    outputs = set(dag.get_outputs())

    v_to_idx = {v: i for i, v in enumerate(vertices)}

    # Build dependents list
    dependents = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            dependents[p].append(v_to_idx[v])

    for p in vertices:
        dependents[p].sort()

    # Pointers to next dependent
    ptr = {v: 0 for v in vertices}

    in_fast = set(inputs)
    in_slow = set()
    operations = []
    budget = fast_memory_budget

    def get_next_use(v):
        if ptr[v] < len(dependents[v]):
            return dependents[v][ptr[v]]
        return None

    for i, v in enumerate(vertices):
        if time.time() > deadline:
            break

        if v in inputs:
            continue

        # Load parents if not in fast memory
        for p in parents[v]:
            if p not in in_fast:
                if p in in_slow:
                    operations.append(['load', p])
                    in_slow.remove(p)
                    in_fast.add(p)

        # Compute v
        operations.append(['compute', v])
        in_fast.add(v)

        # Update pointers for parents of v
        for p in parents[v]:
            while ptr[p] < len(dependents[p]) and dependents[p][ptr[p]] <= i:
                ptr[p] += 1

        # Proactive deletion of nodes no longer needed
        to_delete = []
        for c in in_fast:
            if get_next_use(c) is None and c not in outputs:
                to_delete.append(c)

        for c in to_delete:
            operations.append(['delete', c])
            in_fast.remove(c)
            if c in in_slow:
                in_slow.remove(c)

        # Eviction phase
        while len(in_fast) > budget:
            best_candidate = None
            best_priority = float('-inf')
            best_action = None

            for c in in_fast:
                next_use = get_next_use(c)

                if next_use is None:
                    if c in outputs:
                        priority = 50
                        action = 'store'
                    else:
                        priority = 100
                        action = 'delete'
                else:
                    distance = next_use - i
                    remaining_uses = len(dependents[c]) - ptr[c]
                    priority = distance * 10000 - remaining_uses
                    action = 'store'

                if priority > best_priority:
                    best_priority = priority
                    best_candidate = c
                    best_action = action

            if best_candidate is None:
                break

            if best_action == 'delete':
                operations.append(['delete', best_candidate])
                in_fast.remove(best_candidate)
                if best_candidate in in_slow:
                    in_slow.remove(best_candidate)
            else:
                operations.append(['store', best_candidate])
                in_fast.remove(best_candidate)
                in_slow.add(best_candidate)

    # Ensure all outputs are in fast memory
    for v in outputs:
        if v not in in_fast:
            if v in in_slow:
                operations.append(['load', v])
                in_slow.remove(v)
                in_fast.add(v)
            while len(in_fast) > budget:
                deleted = False
                for candidate in list(in_fast):
                    if candidate not in outputs:
                        operations.append(['delete', candidate])
                        in_fast.remove(candidate)
                        if candidate in in_slow:
                            in_slow.remove(candidate)
                        deleted = True
                        break
                if not deleted:
                    break

    if len(operations) > 20000:
        operations = operations[:20000]

    return operations
# EVOLVE-BLOCK-END
