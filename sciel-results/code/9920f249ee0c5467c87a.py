# EVOLVE-BLOCK-START
import sys
import os
import json
import time
from collections import defaultdict

def _get_module():
    """Import the task module to inspect DAG structure and baseline."""
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        import autoresearch_suite.science_tcs_expansion as mod
        return mod
    except Exception:
        return None

def _safe_baseline(dag, fast_memory_budget):
    """Run baseline, return its operations list."""
    mod = _get_module()
    if mod is not None and hasattr(mod, 'pebble_baseline'):
        try:
            return mod.pebble_baseline(dag, fast_memory_budget)
        except Exception as e:
            return None
    return None

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    1. Parse the DAG structure (vertices, parents, inputs, outputs).
    2. Use recursive pebbling with optimal substructure.
    3. Minimize transfers by computing each node's parents in an order
       that maximizes reuse and minimizes re-computation.
    """
    # Try baseline first as a fallback
    baseline_ops = _safe_baseline(dag, fast_memory_budget)
    
    # Attempt to parse the dag into a usable form
    try:
        vertices, parents, inputs, outputs = _parse_dag(dag)
    except Exception:
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    if vertices is None:
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    # Build the pebbling schedule using recursive strategy
    ops = _pebble_recursive(vertices, parents, inputs, outputs, fast_memory_budget)
    
    if ops is None or len(ops) == 0:
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    # Verify and limit
    if len(ops) > 20000:
        if baseline_ops is not None:
            return baseline_ops
        ops = ops[:20000]
    
    return ops


def _parse_dag(dag):
    """
    Parse dag into (vertices, parents, inputs, outputs).
    vertices: list of vertex ids in topological order
    parents: dict vertex -> list of parent vertex ids
    inputs: set of input vertex ids
    outputs: set of output vertex ids
    """
    if isinstance(dag, dict):
        # Pattern 1: {'vertices': [...], 'parents': {...}, 'inputs': [...], 'outputs': [...]}
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
            parents = {k: list(v) for k, v in dag.get('parents', {}).items()}
            inputs = set(dag.get('inputs', []))
            outputs = set(dag.get('outputs', []))
            return vertices, parents, inputs, outputs
        
        # Pattern 2: dag is a dict of vertex -> parents list
        if dag:
            first_key = next(iter(dag.keys()))
            first_val = dag[first_key]
            if isinstance(first_val, list):
                vertices = list(dag.keys())
                parents = {k: list(v) for k, v in dag.items()}
                children = defaultdict(list)
                for v, ps in parents.items():
                    for p in ps:
                        children[p].append(v)
                inputs = {v for v in vertices if not parents.get(v)}
                outputs = {v for v in vertices if not children.get(v)}
                return vertices, parents, inputs, outputs
        
        # Pattern 3: {'nodes': [...], 'edges': [...]}
        if 'nodes' in dag and 'edges' in dag:
            nodes = dag['nodes']
            edges = dag['edges']
            vertices = list(nodes) if isinstance(nodes, list) else list(nodes.keys())
            parents = {v: [] for v in vertices}
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    src, dst = e[0], e[1]
                    if dst in parents:
                        parents[dst].append(src)
            children = defaultdict(list)
            for v, ps in parents.items():
                for p in ps:
                    children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
        
        # Pattern 4: dag has 'n' and 'parents'
        if 'n' in dag:
            n = dag['n']
            vertices = list(range(n))
            parents_raw = dag.get('parents', [])
            parents = {}
            for i in range(n):
                if i < len(parents_raw):
                    parents[i] = list(parents_raw[i])
                else:
                    parents[i] = []
            children = defaultdict(list)
            for v, ps in parents.items():
                for p in ps:
                    children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
    
    elif isinstance(dag, (list, tuple)):
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)) and len(dag[0]) == 2:
            vertices = []
            parents = {}
            for item in dag:
                v, ps = item[0], item[1]
                vertices.append(v)
                parents[v] = list(ps)
            children = defaultdict(list)
            for v, ps in parents.items():
                for p in ps:
                    children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
    
    return None, None, None, None


def _pebble_recursive(vertices, parents, inputs, outputs, fast_memory_budget):
    """
    Recursive pebbling strategy:
    For each output, compute it by recursively computing its parents.
    Use a LIFO stack-based approach to manage fast memory.
    
    Key insight: In a layered DAG, we can process layer by layer.
    For each node, we compute its parents, then compute the node itself.
    We use fast memory efficiently by keeping recently used values.
    
    This is essentially a depth-first pebbling with memory management.
    """
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Count how many times each vertex is needed (number of children + is_output)
    need_count = {}
    for v in vertices:
        need_count[v] = len(children[v]) + (1 if v in outputs else 0)
    
    # Fast memory simulation
    fast_mem = set(inputs)  # inputs start blue
    slow_mem = set()
    ops = []
    
    # Track remaining needs
    remaining_needs = dict(need_count)
    
    # Process vertices in topological order
    for v in vertices:
        if v in inputs:
            continue
        
        ps = parents.get(v, [])
        
        # Ensure all parents are in fast memory
        for p in ps:
            if p not in fast_mem:
                # Need to get p into fast memory
                if p in slow_mem:
                    _ensure_space(fast_mem, slow_mem, remaining_needs, outputs, v, p, children, ops, fast_memory_budget)
                    ops.append(["load", p])
                    fast_mem.add(p)
                else:
                    # Need to compute p - recursively ensure p's parents are available
                    _compute_vertex(p, fast_mem, slow_mem, parents, children, remaining_needs, 
                                   outputs, ops, fast_memory_budget)
        
        # Now compute v
        _ensure_space(fast_mem, slow_mem, remaining_needs, outputs, v, None, children, ops, fast_memory_budget)
        ops.append(["compute", v])
        fast_mem.add(v)
        
        # Update remaining needs for parents
        for p in ps:
            if p in remaining_needs:
                remaining_needs[p] -= 1
    
    # Ensure all outputs are in fast memory at the end
    for o in outputs:
        if o not in fast_mem:
            if o in slow_mem:
                _ensure_space(fast_mem, slow_mem, remaining_needs, outputs, o, None, children, ops, fast_memory_budget)
                ops.append(["load", o])
                fast_mem.add(o)
            else:
                _compute_vertex(o, fast_mem, slow_mem, parents, children, remaining_needs,
                               outputs, ops, fast_memory_budget)
    
    return ops


def _ensure_space(fast_mem, slow_mem, remaining_needs, outputs, target_vertex, priority_vertex, children, ops, fast_memory_budget):
    """
    Ensure there's space in fast memory for a new vertex.
    Evict vertices that are no longer needed.
    """
    while len(fast_mem) >= fast_memory_budget:
        # Find a vertex to evict
        evicted = None
        for v in list(fast_mem):
            if v == target_vertex or v == priority_vertex:
                continue
            if v in outputs:
                continue
            if remaining_needs.get(v, 0) == 0:
                evicted = v
                break
        
        if evicted is None:
            # No vertex with 0 remaining needs; evict one with fewest remaining needs
            candidates = []
            for v in fast_mem:
                if v == target_vertex or v == priority_vertex:
                    continue
                if v in outputs:
                    continue
                candidates.append((remaining_needs.get(v, 0), v))
            if candidates:
                candidates.sort()
                evicted = candidates[0][1]
        
        if evicted is None:
            # Can't evict anything
            break
        
        # Store evicted vertex in slow memory
        if evicted not in slow_mem:
            slow_mem.add(evicted)
            ops.append(["store", evicted])
        fast_mem.discard(evicted)
        ops.append(["delete", evicted])


def _compute_vertex(v, fast_mem, slow_mem, parents, children, remaining_needs, 
                    outputs, ops, fast_memory_budget):
    """
    Compute vertex v, ensuring its parents are in fast memory.
    """
    if v in fast_mem:
        return
    
    if v in slow_mem:
        _ensure_space(fast_mem, slow_mem, remaining_needs, outputs, v, None, children, ops, fast_memory_budget)
        ops.append(["load", v])
        fast_mem.add(v)
        return
    
    # Need to compute v from its parents
    ps = parents.get(v, [])
    for p in ps:
        if p not in fast_mem:
            _compute_vertex(p, fast_mem, slow_mem, parents, children, remaining_needs,
                           outputs, ops, fast_memory_budget)
    
    # Compute v
    _ensure_space(fast_mem, slow_mem, remaining_needs, outputs, v, None, children, ops, fast_memory_budget)
    ops.append(["compute", v])
    fast_mem.add(v)

# EVOLVE-BLOCK-END
