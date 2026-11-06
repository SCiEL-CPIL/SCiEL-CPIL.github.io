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

def _inspect_dag_structure(dag):
    """
    Inspect the dag object to understand its structure.
    Returns a dict with keys: type, keys, sample
    """
    info = {"type": type(dag).__name__}
    if isinstance(dag, dict):
        info["keys"] = list(dag.keys())[:20]
        for k in list(dag.keys())[:5]:
            v = dag[k]
            info[f"sample_{k}"] = (type(v).__name__, str(v)[:200])
    elif isinstance(dag, (list, tuple)):
        info["len"] = len(dag)
        for i in range(min(3, len(dag))):
            info[f"item_{i}"] = (type(dag[i]).__name__, str(dag[i])[:200])
    else:
        # Try common attributes
        for attr in ['vertices', 'nodes', 'edges', 'parents', 'children', 'inputs', 'outputs', 'n', 'num_vertices']:
            if hasattr(dag, attr):
                v = getattr(dag, attr)
                info[attr] = (type(v).__name__, str(v)[:200])
    return info

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
    2. Use a greedy topological pebbling with a memory-aware eviction policy.
    3. Minimize transfers by keeping values in fast memory when possible,
       and only storing/reloading when memory is full.
    """
    # First, try to understand the dag structure
    dag_info = _inspect_dag_structure(dag)
    
    # Try baseline first as a fallback
    baseline_ops = _safe_baseline(dag, fast_memory_budget)
    
    # Attempt to parse the dag into a usable form
    try:
        vertices, parents, inputs, outputs = _parse_dag(dag)
    except Exception:
        # Fall back to baseline
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    if vertices is None:
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    # Build the pebbling schedule
    ops = _pebble_greedy(vertices, parents, inputs, outputs, fast_memory_budget)
    
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
        # Try common key patterns
        # Pattern 1: {'vertices': [...], 'parents': {...}, ...}
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
        
        # Pattern 2: dag is a dict of vertex -> parents
        # Check if values are lists (parents)
        first_val = next(iter(dag.values())) if dag else None
        if isinstance(first_val, list):
            vertices = list(dag.keys())
            parents = {k: list(v) for k, v in dag.items()}
            # Determine inputs (no parents) and outputs (no children)
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


def _pebble_greedy(vertices, parents, inputs, outputs, fast_memory_budget):
    """
    Greedy pebbling with memory management.
    
    State:
    - fast_memory: dict vertex -> True (value is in fast memory, red)
    - slow_memory: dict vertex -> True (value is stored in slow memory)
    - computed: set of vertices that have been computed at least once
    
    Operations:
    - load(vertex): bring from slow to fast (if in slow)
    - store(vertex): put from fast to slow
    - compute(vertex): compute from parents (parents must be in fast)
    - delete(vertex): remove from fast
    
    Goal: all outputs must finish blue (in fast memory).
    Inputs start blue (in fast memory initially? or need to be loaded?)
    
    "Inputs start blue" - blue means in fast memory. So inputs are initially in fast memory.
    But they consume fast memory slots.
    
    Strategy:
    1. Start with inputs in fast memory.
    2. Process vertices in topological order.
    3. For each vertex, ensure parents are in fast memory (load if needed).
    4. Compute the vertex.
    5. If memory is full, evict a vertex that is no longer needed.
    6. At the end, ensure all outputs are in fast memory.
    """
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Determine topological order (vertices should already be in topo order, but verify)
    # Use the given order as topo order
    topo_order = list(vertices)
    
    # Count remaining uses: how many children still need this vertex
    remaining_uses = {}
    for v in topo_order:
        remaining_uses[v] = len(children.get(v, []))
    
    # Initial state: inputs are in fast memory (blue)
    fast_mem = set(inputs)
    slow_mem = set()
    computed = set(inputs)
    
    ops = []
    
    # Track which vertices are still needed (have unprocessed children or are outputs)
    # A vertex is "needed" if it has remaining children that haven't been processed, 
    # or if it's an output
    
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
                    # Check if we have space
                    if len(fast_mem) >= fast_memory_budget:
                        # Need to evict something
                        evicted = _evict(fast_mem, slow_mem, remaining_uses, outputs, v, p, children, topo_order)
                        if evicted is not None:
                            _perform_eviction(evicted, fast_mem, slow_mem, computed, inputs, ops)
                    ops.append(["load", p])
                    fast_mem.add(p)
                else:
                    # Parent not in slow either - need to recompute
                    # This is complex; for now, try to compute parent
                    # Recursively ensure parent's parents are in fast, then compute parent
                    _ensure_computed(p, fast_mem, slow_mem, computed, parents, children, 
                                    remaining_uses, outputs, fast_memory_budget, ops, topo_order)
        
        # Now compute v
        # Check space
        if len(fast_mem) >= fast_memory_budget:
            evicted = _evict(fast_mem, slow_mem, remaining_uses, outputs, v, None, children, topo_order)
            if evicted is not None:
                _perform_eviction(evicted, fast_mem, slow_mem, computed, inputs, ops)
        
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
                    evicted = _evict(fast_mem, slow_mem, remaining_uses, outputs, o, None, children, topo_order)
                    if evicted is not None:
                        fast_mem.discard(evicted)
                        ops.append(["delete", evicted])
                ops.append(["load", o])
                fast_mem.add(o)
            else:
                # Need to recompute
                _ensure_computed(o, fast_mem, slow_mem, computed, parents, children,
                                remaining_uses, outputs, fast_memory_budget, ops, topo_order)
    
    return ops


def _perform_eviction(evicted, fast_mem, slow_mem, computed, inputs, ops):
    """
    Perform the eviction of a vertex from fast memory.
    If the vertex is a computed value (not an input), store it to slow memory.
    If it is an input, just delete it (assuming inputs can be reloaded from source).
    """
    if evicted in computed and evicted not in inputs:
        ops.append(["store", evicted])
        slow_mem.add(evicted)
    else:
        # For inputs, we mark them as available in slow memory (virtual)
        slow_mem.add(evicted)
    
    fast_mem.discard(evicted)
    ops.append(["delete", evicted])

def _evict(fast_mem, slow_mem, remaining_uses, outputs, current_vertex, priority_parent, children, topo_order):
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


def _ensure_computed(v, fast_mem, slow_mem, computed, parents, children, 
                     remaining_uses, outputs, fast_memory_budget, ops, topo_order):
    """
    Ensure vertex v is computed and in fast memory.
    Recursively compute parents if needed.
    """
    if v in fast_mem:
        return
    
    if v in slow_mem:
        # Load it
        if len(fast_mem) >= fast_memory_budget:
            evicted = _evict(fast_mem, slow_mem, remaining_uses, outputs, v, None, children, topo_order)
            if evicted is not None:
                _perform_eviction(evicted, fast_mem, slow_mem, computed, inputs, ops)
        ops.append(["load", v])
        fast_mem.add(v)
        return
    
    # Need to compute v from scratch
    ps = parents.get(v, [])
    for p in ps:
        _ensure_computed(p, fast_mem, slow_mem, computed, parents, children,
                        remaining_uses, outputs, fast_memory_budget, ops, topo_order)
    
    # Compute v
    if len(fast_mem) >= fast_memory_budget:
        evicted = _evict(fast_mem, slow_mem, remaining_uses, outputs, v, None, children, topo_order)
        if evicted is not None:
            _perform_eviction(evicted, fast_mem, slow_mem, computed, inputs, ops)
    
    ops.append(["compute", v])
    fast_mem.add(v)
    computed.add(v)

# EVOLVE-BLOCK-END
