# EVOLVE-BLOCK-START
import sys
import os
import json
import time
from collections import defaultdict, deque

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

def _parse_dag(dag):
    """
    Parse dag into (vertices, parents, inputs, outputs).
    vertices: list of vertex ids in topological order
    parents: dict vertex -> list of parent vertex ids
    inputs: set of input vertex ids
    outputs: set of output vertex ids
    """
    if isinstance(dag, dict):
        # Pattern 1: {'vertices': [...], 'parents': {...}, ...}
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
            parents = {}
            raw_parents = dag.get('parents', {})
            if isinstance(raw_parents, dict):
                for k in raw_parents:
                    parents[k] = list(raw_parents[k])
            else:
                # Maybe it's a list of lists
                for i, v in enumerate(vertices):
                    if i < len(raw_parents):
                        parents[v] = list(raw_parents[i])
                    else:
                        parents[v] = []
            inputs = dag.get('inputs', set())
            outputs = dag.get('outputs', set())
            if isinstance(inputs, list):
                inputs = set(inputs)
            if isinstance(outputs, list):
                outputs = set(outputs)
            return vertices, parents, inputs, outputs
        
        # Pattern 2: dag is a dict of vertex -> parents
        first_val = next(iter(dag.values())) if dag else None
        if isinstance(first_val, list):
            vertices = list(dag.keys())
            parents = {k: list(v) for k, v in dag.items()}
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
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
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
        
        # Pattern 4: dag has 'n' and 'parents' as arrays
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
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
    
    elif isinstance(dag, (list, tuple)):
        # Maybe it's a list of (vertex, parents) tuples
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)) and len(dag[0]) == 2:
            vertices = []
            parents = {}
            for item in dag:
                v, ps = item[0], item[1]
                vertices.append(v)
                parents[v] = list(ps)
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
    
    # Fallback: try to use the module's helper if available
    mod = _get_module()
    if mod is not None:
        for attr in ['parse_dag', 'get_dag_info', 'extract_dag']:
            if hasattr(mod, attr):
                try:
                    result = getattr(mod, attr)(dag)
                    if isinstance(result, tuple) and len(result) == 4:
                        return result
                except Exception:
                    pass
    
    return None, None, None, None


def _topological_sort(vertices, parents):
    """
    Perform topological sort on the DAG.
    Returns a list of vertices in topological order.
    """
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    queue = deque()
    for v in vertices:
        if in_degree[v] == 0:
            queue.append(v)
    
    topo_order = []
    while queue:
        # Use a priority queue to get deterministic order
        # But for performance, just use regular queue
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If not all vertices are in topo_order, there's a cycle
    if len(topo_order) != len(vertices):
        # Fallback to original order
        return list(vertices)
    
    return topo_order


def _compute_remaining_uses(vertices, parents, children):
    """
    Compute how many times each vertex is needed by its children.
    """
    remaining_uses = {}
    for v in vertices:
        remaining_uses[v] = len(children.get(v, []))
    return remaining_uses


def _pebble_optimized(vertices, parents, inputs, outputs, fast_memory_budget):
    """
    Optimized pebbling using a better eviction strategy.
    
    Key insight: We should evict vertices that are:
    1. No longer needed (remaining_uses == 0)
    2. Not outputs
    3. Not parents of the current vertex
    
    Additionally, we should store values to slow memory before evicting
    if they might be needed again.
    """
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Determine topological order
    topo_order = _topological_sort(vertices, parents)
    
    # Count remaining uses
    remaining_uses = _compute_remaining_uses(vertices, parents, children)
    
    # Initial state: inputs are in fast memory (blue)
    fast_mem = set(inputs)
    slow_mem = set()
    computed = set(inputs)
    
    ops = []
    
    # Process in topological order
    for v in topo_order:
        if v in inputs:
            # Already in fast memory
            continue
        
        # Ensure all parents are in fast memory
        ps = parents.get(v, [])
        for p in ps:
            if p not in fast_mem:
                if p in slow_mem:
                    # Need to load
                    if len(fast_mem) >= fast_memory_budget:
                        evicted = _evict_optimized(fast_mem, slow_mem, remaining_uses, outputs, v, p, children, topo_order)
                        if evicted is not None:
                            # Store to slow memory before deleting if it was computed
                            if evicted in computed and evicted not in inputs:
                                ops.append(["store", evicted])
                                slow_mem.add(evicted)
                            fast_mem.discard(evicted)
                            ops.append(["delete", evicted])
                    ops.append(["load", p])
                    fast_mem.add(p)
                else:
                    # Parent not in slow either - need to recompute
                    _ensure_computed_optimized(p, fast_mem, slow_mem, computed, parents, children, 
                                             remaining_uses, outputs, fast_memory_budget, ops, topo_order, inputs)
        
        # Now compute v
        # Check space
        if len(fast_mem) >= fast_memory_budget:
            evicted = _evict_optimized(fast_mem, slow_mem, remaining_uses, outputs, v, None, children, topo_order)
            if evicted is not None:
                # Store to slow memory before deleting if it was computed
                if evicted in computed and evicted not in inputs:
                    ops.append(["store", evicted])
                    slow_mem.add(evicted)
                fast_mem.discard(evicted)
                ops.append(["delete", evicted])
        
        ops.append(["compute", v])
        fast_mem.add(v)
        computed.add(v)
        
        # Update remaining uses for parents
        for p in ps:
            if p in remaining_uses:
                remaining_uses[p] -= 1
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_mem:
            if o in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evicted = _evict_optimized(fast_mem, slow_mem, remaining_uses, outputs, o, None, children, topo_order)
                    if evicted is not None:
                        # Store to slow memory before deleting if it was computed
                        if evicted in computed and evicted not in inputs:
                            ops.append(["store", evicted])
                            slow_mem.add(evicted)
                        fast_mem.discard(evicted)
                        ops.append(["delete", evicted])
                ops.append(["load", o])
                fast_mem.add(o)
            else:
                # Need to recompute
                _ensure_computed_optimized(o, fast_mem, slow_mem, computed, parents, children,
                                         remaining_uses, outputs, fast_memory_budget, ops, topo_order, inputs)
    
    return ops


def _evict_optimized(fast_mem, slow_mem, remaining_uses, outputs, current_vertex, priority_parent, children, topo_order):
    """
    Choose a vertex to evict from fast memory.
    Prefer vertices with 0 remaining uses that are not outputs.
    Then prefer vertices that are not needed by current computation.
    """
    candidates = []
    for v in fast_mem:
        if v == current_vertex:
            continue
        if v == priority_parent:
            continue
        if v in outputs:
            continue
        uses = remaining_uses.get(v, 0)
        candidates.append((uses, v))
    
    if not candidates:
        # Last resort: evict anything that's not an output and not current
        for v in fast_mem:
            if v != current_vertex and v != priority_parent and v not in outputs:
                return v
        return None
    
    # Sort by remaining uses (fewest first)
    candidates.sort()
    return candidates[0][1]


def _ensure_computed_optimized(v, fast_mem, slow_mem, computed, parents, children, 
                               remaining_uses, outputs, fast_memory_budget, ops, topo_order, inputs):
    """
    Ensure vertex v is computed and in fast memory.
    Recursively compute parents if needed.
    """
    if v in fast_mem:
        return
    
    if v in slow_mem:
        # Load it
        if len(fast_mem) >= fast_memory_budget:
            evicted = _evict_optimized(fast_mem, slow_mem, remaining_uses, outputs, v, None, children, topo_order)
            if evicted is not None:
                # Store to slow memory before deleting if it was computed
                if evicted in computed and evicted not in inputs:
                    ops.append(["store", evicted])
                    slow_mem.add(evicted)
                fast_mem.discard(evicted)
                ops.append(["delete", evicted])
        ops.append(["load", v])
        fast_mem.add(v)
        return
    
    # Need to compute v from scratch
    ps = parents.get(v, [])
    for p in ps:
        _ensure_computed_optimized(p, fast_mem, slow_mem, computed, parents, children,
                                  remaining_uses, outputs, fast_memory_budget, ops, topo_order, inputs)
    
    # Compute v
    if len(fast_mem) >= fast_memory_budget:
        evicted = _evict_optimized(fast_mem, slow_mem, remaining_uses, outputs, v, None, children, topo_order)
        if evicted is not None:
            # Store to slow memory before deleting if it was computed
            if evicted in computed and evicted not in inputs:
                ops.append(["store", evicted])
                slow_mem.add(evicted)
            fast_mem.discard(evicted)
            ops.append(["delete", evicted])
    
    ops.append(["compute", v])
    fast_mem.add(v)
    computed.add(v)


def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    1. Parse the DAG structure (vertices, parents, inputs, outputs).
    2. Use an optimized topological pebbling with a memory-aware eviction policy.
    3. Minimize transfers by keeping values in fast memory when possible,
       and only storing/reloading when memory is full.
    """
    # Attempt to parse the dag into a usable form
    try:
        vertices, parents, inputs, outputs = _parse_dag(dag)
    except Exception:
        # Fall back to baseline
        baseline_ops = _safe_baseline(dag, fast_memory_budget)
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    if vertices is None:
        baseline_ops = _safe_baseline(dag, fast_memory_budget)
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    # Build the pebbling schedule
    ops = _pebble_optimized(vertices, parents, inputs, outputs, fast_memory_budget)
    
    if ops is None or len(ops) == 0:
        baseline_ops = _safe_baseline(dag, fast_memory_budget)
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    # Verify and limit
    if len(ops) > 20000:
        baseline_ops = _safe_baseline(dag, fast_memory_budget)
        if baseline_ops is not None:
            return baseline_ops
        ops = ops[:20000]
    
    return ops

# EVOLVE-BLOCK-END
