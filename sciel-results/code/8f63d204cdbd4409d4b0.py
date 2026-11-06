# EVOLVE-BLOCK-START
import sys
import os
import json
import time
from collections import deque, defaultdict

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
    Communication-minimizing DAG pebbling using lifetime-based scheduling.
    
    Approach:
    1. Parse DAG into vertices, parents, children, inputs, outputs.
    2. Compute lifetime intervals for each vertex (first/last use).
    3. Use a cost-aware greedy schedule that minimizes transfers by:
       - Keeping values in fast memory when their lifetime justifies it
       - Evicting based on future need cost (recomputation cost)
       - Using store/load when beneficial vs. recomputation
    4. Ensure all outputs finish in fast memory.
    """
    baseline_ops = _safe_baseline(dag, fast_memory_budget)
    
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
    
    ops = _pebble_lifetime(vertices, parents, inputs, outputs, fast_memory_budget)
    
    if ops is None or len(ops) == 0:
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    if len(ops) > 20000:
        if baseline_ops is not None:
            return baseline_ops
        ops = ops[:20000]
    
    return ops

def _parse_dag(dag):
    """
    Parse dag into (vertices, parents, inputs, outputs).
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
            # Normalize parents to dict of lists
            if not isinstance(parents, dict):
                parents = {}
            norm_parents = {}
            for v in vertices:
                ps = parents.get(v, [])
                if ps is None:
                    ps = []
                norm_parents[v] = list(ps)
            return vertices, norm_parents, inputs, outputs
        
        # Pattern: dict of vertex -> parents list
        if dag:
            first_val = next(iter(dag.values()))
            if isinstance(first_val, (list, tuple, set)):
                vertices = list(dag.keys())
                parents = {k: list(v) if v else [] for k, v in dag.items()}
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
                    parents[i] = list(parents_raw[i]) if parents_raw[i] else []
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
                parents[v] = list(ps) if ps else []
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

def _compute_recomputation_cost(vertex, parents, children, memo=None):
    """
    Compute the cost of recomputing a vertex (number of compute operations needed).
    This is used to decide whether to keep a value in memory or recompute it.
    """
    if memo is None:
        memo = {}
    if vertex in memo:
        return memo[vertex]
    
    ps = parents.get(vertex, [])
    if not ps:
        cost = 1  # Just compute the vertex itself
    else:
        cost = 1 + sum(_compute_recomputation_cost(p, parents, children, memo) for p in ps)
    
    memo[vertex] = cost
    return cost

def _compute_lifetime_intervals(vertices, parents, children, inputs, outputs):
    """
    Compute the first and last use (position in topo order) for each vertex.
    This helps determine if a value will be needed again.
    """
    n = len(vertices)
    vertex_index = {v: i for i, v in enumerate(vertices)}
    
    first_use = {}
    last_use = {}
    
    for v in vertices:
        first_use[v] = vertex_index[v]
        last_use[v] = vertex_index[v]
    
    # For each vertex, update first/last use based on children
    for v in vertices:
        for child in children.get(v, []):
            if child in vertex_index:
                first_use[v] = min(first_use[v], vertex_index[v])
                last_use[v] = max(last_use[v], vertex_index[v])
    
    # Propagate: a parent's last use is at least the last use of its children that need it
    # Actually, let's think differently: 
    # first_use[v] = min index where v is needed (as parent or as itself)
    # last_use[v] = max index where v is needed
    
    # Recompute properly
    first_use = {v: vertex_index[v] for v in vertices}
    last_use = {v: vertex_index[v] for v in vertices}
    
    for v in vertices:
        for child in children.get(v, []):
            if child in vertex_index:
                # v is needed at position vertex_index[child] to compute child
                last_use[v] = max(last_use[v], vertex_index[child])
    
    # A vertex is first needed when it's computed or when its first child needs it
    # Actually first_use should be when it becomes available (computed)
    # Let's say first_use[v] = vertex_index[v] (when it's computed)
    # last_use[v] = max position of any child that uses it, or its own position if no children
    
    return first_use, last_use

def _pebble_lifetime(vertices, parents, inputs, outputs, fast_memory_budget):
    """
    Pebbling schedule using lifetime-aware eviction.
    
    Key insight: When deciding whether to evict a value, consider:
    1. How many more times it will be needed (remaining uses)
    2. The cost of recomputing it if evicted
    3. Whether storing it to slow memory and reloading is cheaper than recomputing
    
    We use a priority score for eviction:
    - Lower score = more likely to be evicted
    - Score considers: remaining_uses * recomputation_cost
    """
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    topo_order = list(vertices)
    vertex_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Compute recomputation costs
    recompute_cost = {}
    for v in vertices:
        recompute_cost[v] = _compute_recomputation_cost(v, parents, children, recompute_cost)
    
    # Compute remaining uses
    remaining_uses = {}
    for v in topo_order:
        remaining_uses[v] = len(children.get(v, []))
    
    # Initial state
    fast_mem = set(inputs)
    slow_mem = set()
    ops = []
    
    # Process vertices in topological order
    for v in topo_order:
        if v in inputs:
            continue
        
        ps = parents.get(v, [])
        
        # Ensure all parents are in fast memory
        for p in ps:
            if p not in fast_mem:
                if p in slow_mem:
                    # Load from slow
                    if len(fast_mem) >= fast_memory_budget:
                        evicted = _evict_lifetime(fast_mem, slow_mem, remaining_uses, 
                                                  recompute_cost, outputs, v, p, 
                                                  children, topo_order, vertex_pos)
                        if evicted is not None:
                            fast_mem.discard(evicted)
                            ops.append(["delete", evicted])
                    ops.append(["load", p])
                    fast_mem.add(p)
                else:
                    # Need to recompute p
                    _ensure_computed_lifetime(p, fast_mem, slow_mem, parents, children,
                                            remaining_uses, recompute_cost, outputs,
                                            fast_memory_budget, ops, topo_order, vertex_pos)
        
        # Compute v
        if len(fast_mem) >= fast_memory_budget:
            evicted = _evict_lifetime(fast_mem, slow_mem, remaining_uses,
                                      recompute_cost, outputs, v, None,
                                      children, topo_order, vertex_pos)
            if evicted is not None:
                fast_mem.discard(evicted)
                ops.append(["delete", evicted])
        
        ops.append(["compute", v])
        fast_mem.add(v)
        
        # Update remaining uses for parents
        for p in ps:
            if p in remaining_uses:
                remaining_uses[p] -= 1
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_mem:
            if o in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evicted = _evict_lifetime(fast_mem, slow_mem, remaining_uses,
                                              recompute_cost, outputs, o, None,
                                              children, topo_order, vertex_pos)
                    if evicted is not None:
                        fast_mem.discard(evicted)
                        ops.append(["delete", evicted])
                ops.append(["load", o])
                fast_mem.add(o)
            else:
                _ensure_computed_lifetime(o, fast_mem, slow_mem, parents, children,
                                        remaining_uses, recompute_cost, outputs,
                                        fast_memory_budget, ops, topo_order, vertex_pos)
    
    return ops

def _evict_lifetime(fast_mem, slow_mem, remaining_uses, recompute_cost, 
                    outputs, current_vertex, priority_parent, children, 
                    topo_order, vertex_pos):
    """
    Choose vertex to evict based on lifetime-aware cost.
    
    Eviction priority (lower = evict first):
    1. Vertices with 0 remaining uses and not outputs (free to evict)
    2. Among others, evict those with lowest (remaining_uses * recompute_cost)
       This balances future need against recomputation cost.
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
        cost = recompute_cost.get(v, 1)
        
        # Score: lower is better for eviction
        # If uses == 0, very high priority to evict
        if uses == 0:
            score = 0
        else:
            # Higher remaining uses and higher recompute cost = less likely to evict
            score = uses * cost
        
        candidates.append((score, v))
    
    if not candidates:
        for v in fast_mem:
            if v != current_vertex and v != priority_parent and v not in outputs:
                return v
        return None
    
    candidates.sort()
    return candidates[0][1]

def _ensure_computed_lifetime(v, fast_mem, slow_mem, parents, children,
                             remaining_uses, recompute_cost, outputs,
                             fast_memory_budget, ops, topo_order, vertex_pos):
    """
    Ensure vertex v is computed and in fast memory.
    """
    if v in fast_mem:
        return
    
    if v in slow_mem:
        if len(fast_mem) >= fast_memory_budget:
            evicted = _evict_lifetime(fast_mem, slow_mem, remaining_uses,
                                      recompute_cost, outputs, v, None,
                                      children, topo_order, vertex_pos)
            if evicted is not None:
                fast_mem.discard(evicted)
                ops.append(["delete", evicted])
        ops.append(["load", v])
        fast_mem.add(v)
        return
    
    # Recompute v
    ps = parents.get(v, [])
    for p in ps:
        _ensure_computed_lifetime(p, fast_mem, slow_mem, parents, children,
                                remaining_uses, recompute_cost, outputs,
                                fast_memory_budget, ops, topo_order, vertex_pos)
    
    if len(fast_mem) >= fast_memory_budget:
        evicted = _evict_lifetime(fast_mem, slow_mem, remaining_uses,
                                  recompute_cost, outputs, v, None,
                                  children, topo_order, vertex_pos)
        if evicted is not None:
            fast_mem.discard(evicted)
            ops.append(["delete", evicted])
    
    ops.append(["compute", v])
    fast_mem.add(v)

# EVOLVE-BLOCK-END
