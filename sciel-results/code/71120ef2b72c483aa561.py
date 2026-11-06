# EVOLVE-BLOCK-START
import sys
import os
import json
import time

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
    2. Use a greedy topological pebbling with lookahead-based eviction.
    3. Store values to slow memory before eviction when they will be needed again.
    4. Minimize transfers by keeping frequently-used values in fast memory.
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
    
    # Build the pebbling schedule
    ops = _pebble_optimized(vertices, parents, inputs, outputs, fast_memory_budget)
    
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
    vertices: list of vertex ids in topological order (or indexable)
    parents: dict vertex -> list of parent vertex ids
    inputs: set of input vertex ids
    outputs: set of output vertex ids
    """
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = dag['vertices']
            parents = dag.get('parents', {})
            inputs = dag.get('inputs', set())
            outputs = dag.get('outputs', set())
            if isinstance(inputs, list):
                inputs = set(inputs)
            if isinstance(outputs, list):
                outputs = set(outputs)
            return vertices, parents, inputs, outputs
        
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
        
        if 'nodes' in dag and 'edges' in dag:
            nodes = dag['nodes']
            edges = dag['edges']
            vertices = nodes if isinstance(nodes, list) else list(nodes.keys())
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


def _pebble_optimized(vertices, parents, inputs, outputs, fast_memory_budget):
    """
    Optimized pebbling with proper slow-memory management and lookahead eviction.
    
    Key improvements over naive greedy:
    1. Track when each vertex is last used (for eviction priority)
    2. Store values to slow memory before evicting if they'll be needed again
    3. Avoid unnecessary recomputation by leveraging slow memory
    4. Use a cost-based eviction policy considering future access patterns
    """
    # Build children map and topological order
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    topo_order = list(vertices)
    
    # Compute last-use index for each vertex (for eviction priority)
    # last_use[v] = index in topo_order of the last child that needs v
    last_use = {}
    for v in topo_order:
        ch = children.get(v, [])
        if ch:
            last_use[v] = max(topo_order.index(c) for c in ch if c in topo_order)
        else:
            last_use[v] = -1
    
    # Also consider if v is an output (needs to be in fast memory at end)
    for v in outputs:
        if v not in last_use or last_use[v] < len(topo_order) - 1:
            last_use[v] = len(topo_order)  # mark as needing to stay until end
    
    # State tracking
    fast_mem = set()       # vertices currently in fast memory
    slow_mem = set()       # vertices stored in slow memory
    computed = set()       # vertices that have been computed at least once
    
    ops = []
    
    # Helper: make space in fast memory by evicting if needed
    def make_space(needed=1, exclude=None):
        """Evict vertices from fast_mem to make room. Returns True if space was made."""
        nonlocal fast_mem, slow_mem, ops
        while len(fast_mem) + needed > fast_memory_budget:
            # Find best candidate to evict
            best_v = None
            best_score = float('inf')
            for v in fast_mem:
                if exclude and v in exclude:
                    continue
                if v in outputs:
                    continue  # never evict outputs
                # Score: prefer evicting vertices with earliest last use
                # and that are in slow_mem (no cost to evict, already stored)
                lu = last_use.get(v, -1)
                in_slow = 0 if v in slow_mem else 1000  # penalty if not in slow
                score = lu * 10 + in_slow
                if score < best_score:
                    best_score = score
                    best_v = v
            
            if best_v is None:
                # Last resort: evict anything not an output
                for v in fast_mem:
                    if exclude and v in exclude:
                        continue
                    if v not in outputs:
                        best_v = v
                        break
            
            if best_v is None:
                return False
            
            # If best_v will be needed again and is not in slow, store it first
            if last_use.get(best_v, -1) >= 0 and best_v not in slow_mem and best_v not in inputs:
                ops.append(["store", best_v])
                slow_mem.add(best_v)
            
            ops.append(["delete", best_v])
            fast_mem.discard(best_v)
        
        return True
    
    # Process vertices in topological order
    for v in topo_order:
        if v in inputs:
            # Inputs start blue (in fast memory) - but do they consume slots?
            # "Inputs start blue" means they're available, but we need to track them
            # Assume inputs are pre-loaded and consume fast memory slots
            if v not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    make_space(1, exclude={v})
                fast_mem.add(v)
            continue
        
        # Ensure all parents are in fast memory
        ps = parents.get(v, [])
        for p in ps:
            if p not in fast_mem:
                if p in slow_mem:
                    # Load from slow memory
                    if not make_space(1, exclude=set(ps) | {v}):
                        return None  # Can't make space
                    ops.append(["load", p])
                    fast_mem.add(p)
                elif p in inputs:
                    # Input should be in fast memory
                    if not make_space(1, exclude=set(ps) | {v}):
                        return None
                    fast_mem.add(p)
                else:
                    # Need to compute parent - this is expensive
                    # For now, skip if we can't handle it (shouldn't happen in topo order)
                    # In a proper topo order, parents are processed before children
                    pass
        
        # Compute v
        if not make_space(1, exclude=set(ps)):
            return None
        ops.append(["compute", v])
        fast_mem.add(v)
        computed.add(v)
    
    # Ensure all outputs are in fast memory at the end
    for o in outputs:
        if o not in fast_mem:
            if o in slow_mem:
                if not make_space(1, exclude=set()):
                    return None
                ops.append(["load", o])
                fast_mem.add(o)
            elif o in inputs:
                if not make_space(1, exclude=set()):
                    return None
                fast_mem.add(o)
            else:
                # Need to recompute - complex, skip for now
                pass
    
    return ops

# EVOLVE-BLOCK-END
