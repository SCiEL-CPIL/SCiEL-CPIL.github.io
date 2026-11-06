"""Communication-minimizing DAG pebbling solver."""
import time
import sys
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

_DEADLINE = None
_TIME_MARGIN = 0.5

def _time_ok():
    global _DEADLINE
    if _DEADLINE is None:
        return True
    return (time.time() - _DEADLINE) < _TIME_MARGIN

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize total memory transfers.
    
    dag: dict with keys 'vertices', 'parents', 'outputs', 'inputs', optionally 'n'
    fast_memory_budget: int, max fast memory slots
    Returns: list of [operation, vertex]
    """
    global _DEADLINE
    _DEADLINE = time.time() + 89.0
    
    # Parse DAG
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        n = dag.get('n', None)
    else:
        vertices = list(range(len(dag))) if hasattr(dag, '__len__') else []
        parents = {}
        outputs = []
        inputs = []
        n = len(vertices) if vertices else 0
    
    if vertices is None:
        all_verts = set()
        for v, ps in parents.items():
            all_verts.add(v)
            for p in ps:
                all_verts.add(p)
        vertices = sorted(all_verts)
        n = len(vertices)
    
    if n is None:
        n = len(vertices)
    
    vert_to_idx = {v: i for i, v in enumerate(vertices)}
    
    parents_idx = {}
    for v, ps in parents.items():
        parents_idx[v] = [vert_to_idx[p] for p in ps]
    
    outputs_idx = [vert_to_idx[v] for v in outputs]
    inputs_idx = [vert_to_idx[v] for v in inputs]
    
    for v in vertices:
        if v not in parents_idx:
            parents_idx[v] = []
    
    # Build children
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents_idx.get(vertices[v], []):
            children[p].append(v)
    
    is_input = [False] * n
    for v in inputs_idx:
        is_input[v] = True
    for v in range(n):
        if not parents_idx.get(vertices[v], []):
            is_input[v] = True
    
    is_output = [False] * n
    for v in outputs_idx:
        is_output[v] = True
    
    budget = max(1, fast_memory_budget)
    
    # For each vertex, compute how many times it will be "needed" in the future
    # This is the number of children that depend on it
    # We'll use this to decide what to keep in fast memory
    
    # Strategy: Process vertices in topological order.
    # Maintain fast memory with the most valuable vertices (highest future demand).
    # When we need to free space, delete the vertex with lowest future demand.
    
    operations = []
    fast_mem = set()
    slow_mem = set(inputs_idx)
    budget_used = 0
    
    # Future demand: how many more times this vertex will be needed
    # Initially, demand[v] = number of children of v
    # As we process children, we decrement the demand of their parents
    future_demand = [len(children[v]) for v in range(n)]
    
    for v in range(n):
        if not _time_ok():
            break
        
        if is_input[v]:
            continue
        
        ps = parents_idx.get(vertices[v], [])
        
        # Ensure all parents are in fast memory
        for p in ps:
            if p not in fast_mem:
                if p in slow_mem:
                    if budget_used >= budget:
                        # Free space: delete vertex with lowest future demand
                        candidates = list(fast_mem)
                        if candidates:
                            # Never delete outputs
                            def val(x):
                                if is_output[x]:
                                    return -10**9
                                return future_demand[x]
                            del_v = min(candidates, key=val)
                            operations.append(['delete', vertices[del_v]])
                            fast_mem.remove(del_v)
                            budget_used -= 1
                            future_demand[del_v] = 0
                    operations.append(['load', vertices[p]])
                    fast_mem.add(p)
                    slow_mem.discard(p)
                    budget_used += 1
                else:
                    # Shouldn't happen in valid DAG
                    pass
        
        # Compute vertex v
        if v not in fast_mem:
            if budget_used >= budget:
                candidates = list(fast_mem)
                if candidates:
                    def val(x):
                        if is_output[x]:
                            return -10**9
                        return future_demand[x]
                    del_v = min(candidates, key=val)
                    operations.append(['delete', vertices[del_v]])
                    fast_mem.remove(del_v)
                    budget_used -= 1
                    future_demand[del_v] = 0
            
            operations.append(['compute', vertices[v]])
            fast_mem.add(v)
            budget_used += 1
        
        # Decrement future demand of parents
        for p in ps:
            if p in fast_mem:
                future_demand[p] = max(0, future_demand[p] - 1)
    
    # Ensure outputs end in slow memory
    for v in outputs_idx:
        if v in fast_mem:
            operations.append(['store', vertices[v]])
            slow_mem.add(v)
            fast_mem.remove(v)
            budget_used = max(0, budget_used - 1)
        elif v not in slow_mem:
            pass
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
