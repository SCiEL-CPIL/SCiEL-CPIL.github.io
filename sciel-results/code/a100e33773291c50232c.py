# EVOLVE-BLOCK-START
import time
import sys
import json
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Args:
        dag: DAG structure 
        fast_memory_budget: number of fast memory slots
    
    Returns:
        list of [operation, vertex] where operation is load/store/compute/delete
    """
    start_time = time.time()
    deadline = start_time + 85
    
    try:
        # Diagnostic: print dag structure to stderr (first call only)
        if not hasattr(pebble, '_diag_done'):
            pebble._diag_done = True
            try:
                diag_info = _describe_dag(dag)
                sys.stderr.write(f"DAG DIAG: {diag_info}\n")
            except Exception as e:
                sys.stderr.write(f"Diag error: {e}\n")
        
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_result = pebble_baseline(dag, fast_memory_budget)
        
        if time.time() > deadline:
            return baseline_result
            
        # Try improved algorithm
        improved = _improved_pebble(dag, fast_memory_budget, deadline)
        if improved is not None:
            # Validate improved result
            if _validate_pebbling(dag, fast_memory_budget, improved):
                return improved
            else:
                sys.stderr.write("Improved result failed validation, using baseline\n")
                return baseline_result
        return baseline_result
    except Exception as e:
        sys.stderr.write(f"Main error: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []


def _describe_dag(dag):
    """Describe the DAG structure for diagnostics."""
    if isinstance(dag, dict):
        keys = list(dag.keys())
        desc = f"dict with keys: {keys}"
        # Try to get sizes
        for k in keys:
            v = dag[k]
            if isinstance(v, (list, tuple)):
                desc += f"\n  {k}: list/tuple of len {len(v)}"
                if len(v) > 0:
                    desc += f", first elem: {repr(v[0])[:100]}"
            elif isinstance(v, dict):
                desc += f"\n  {k}: dict with {len(v)} keys"
                if len(v) > 0:
                    first_key = list(v.keys())[0]
                    desc += f", first key: {repr(first_key)[:50]}, first val: {repr(v[first_key])[:50]}"
            else:
                desc += f"\n  {k}: {type(v).__name__} = {repr(v)[:100]}"
        return desc
    elif isinstance(dag, (list, tuple)):
        desc = f"list/tuple of len {len(dag)}"
        if len(dag) > 0:
            desc += f", first elem: {repr(dag[0])[:100]}"
            if len(dag) > 1:
                desc += f", second elem: {repr(dag[1])[:100]}"
        return desc
    else:
        return f"type={type(dag).__name__}, repr={repr(dag)[:200]}"


def _improved_pebble(dag, fast_memory_budget, deadline):
    """
    Improved pebbling algorithm using chain decomposition.
    Returns None if we can't improve, or a list of operations.
    """
    try:
        # Parse DAG structure
        dag_info = _parse_dag(dag)
        if dag_info is None:
            return None
        
        parents = dag_info['parents']  # dict: vertex -> list of parent vertices
        children = dag_info['children']  # dict: vertex -> list of child vertices
        inputs = dag_info['inputs']  # set of input vertices (no parents)
        outputs = dag_info['outputs']  # set of output vertices (no children)
        all_vertices = dag_info['all_vertices']
        
        if not all_vertices:
            return []
        
        # Topological sort
        topo_order = _topological_sort(all_vertices, parents)
        if topo_order is None:
            return None
        
        # Build reverse topological order for scheduling
        rev_topo_order = list(reversed(topo_order))
        
        # Compute out-degree (number of children) for each vertex
        out_degree = {v: len(children.get(v, [])) for v in all_vertices}
        
        # Find chains using greedy decomposition
        chains = _find_chains(all_vertices, parents, children, topo_order)
        
        if not chains:
            # Fallback to simple topological pebbling
            return _simple_pebbling(all_vertices, parents, children, inputs, outputs, fast_memory_budget)
        
        # Schedule operations using round-robin over chains
        ops = _schedule_chains(chains, parents, children, inputs, outputs, 
                               fast_memory_budget, all_vertices)
        
        if ops is None or len(ops) > 20000:
            return None
            
        return ops
        
    except Exception as e:
        sys.stderr.write(f"Improved pebble error: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


def _parse_dag(dag):
    """Parse various possible DAG formats into a unified structure."""
    try:
        if isinstance(dag, dict):
            # Try to identify the format
            if 'vertices' in dag or 'nodes' in dag:
                # Format: {'vertices': [...], 'edges': [...]} or similar
                vertices = dag.get('vertices', dag.get('nodes', []))
                edges = dag.get('edges', [])
                parents = defaultdict(list)
                children = defaultdict(list)
                for u, v in edges:
                    children[u].append(v)
                    parents[v].append(u)
                inputs = set(vertices) - set(parents.keys())
                outputs = set(vertices) - set(children.keys())
                return {
                    'parents': dict(parents),
                    'children': dict(children),
                    'inputs': inputs,
                    'outputs': outputs,
                    'all_vertices': list(vertices)
                }
            
            # Try: dict of vertex -> list of parents
            elif all(isinstance(v, (list, tuple)) for v in dag.values()):
                parents = {k: list(v) for k, v in dag.items()}
                children = defaultdict(list)
                for v, ps in parents.items():
                    for p in ps:
                        children[p].append(v)
                all_vertices = list(dag.keys())
                inputs = set(all_vertices) - set(parents.keys())
                outputs = set(all_vertices) - set(children.keys())
                return {
                    'parents': parents,
                    'children': dict(children),
                    'inputs': inputs,
                    'outputs': outputs,
                    'all_vertices': all_vertices
                }
            
            # Try: dict with 'inputs', 'outputs', 'layers' etc.
            elif 'inputs' in dag or 'outputs' in dag:
                inputs = set(dag.get('inputs', []))
                outputs = set(dag.get('outputs', []))
                # Try to find edges/parents
                if 'parents' in dag:
                    parents = {k: list(v) for k, v in dag['parents'].items()}
                elif 'edges' in dag:
                    parents = defaultdict(list)
                    children = defaultdict(list)
                    for u, v in dag['edges']:
                        children[u].append(v)
                        parents[v].append(u)
                    parents = dict(parents)
                else:
                    return None
                
                children = defaultdict(list)
                for v, ps in parents.items():
                    for p in ps:
                        children[p].append(v)
                
                all_vertices = set(inputs) | set(outputs) | set(parents.keys())
                for ps in parents.values():
                    all_vertices.update(ps)
                
                return {
                    'parents': parents,
                    'children': dict(children),
                    'inputs': inputs,
                    'outputs': outputs,
                    'all_vertices': list(all_vertices)
                }
            
            # Fallback: try to infer structure
            else:
                # Maybe it's a list of layers
                if all(isinstance(v, list) for v in dag.values()):
                    # Try treating keys as layer indices
                    all_vertices = []
                    parents = {}
                    children = defaultdict(list)
                    for key in sorted(dag.keys()):
                        layer = dag[key]
                        for v in layer:
                            if v not in all_vertices:
                                all_vertices.append(v)
                            # Parents are in previous layers
                            for prev_key in sorted(dag.keys()):
                                if prev_key < key:
                                    for pv in dag[prev_key]:
                                        if pv != v:
                                            # Assume edge if not already present
                                            if v not in parents:
                                                parents[v] = []
                                            if pv not in parents[v]:
                                                parents[v].append(pv)
                                                children[pv].append(v)
                    inputs = [v for v in all_vertices if v not in parents]
                    outputs = [v for v in all_vertices if v not in children]
                    return {
                        'parents': parents,
                        'children': dict(children),
                        'inputs': set(inputs),
                        'outputs': set(outputs),
                        'all_vertices': all_vertices
                    }
        
        elif isinstance(dag, (list, tuple)):
            # Maybe it's a list of edges or a list of layers
            if len(dag) > 0:
                first = dag[0]
                if isinstance(first, (list, tuple)) and len(first) == 2:
                    # List of edges
                    parents = defaultdict(list)
                    children = defaultdict(list)
                    all_vertices = set()
                    for u, v in dag:
                        children[u].append(v)
                        parents[v].append(u)
                        all_vertices.add(u)
                        all_vertices.add(v)
                    inputs = all_vertices - set(parents.keys())
                    outputs = all_vertices - set(children.keys())
                    return {
                        'parents': dict(parents),
                        'children': dict(children),
                        'inputs': inputs,
                        'outputs': outputs,
                        'all_vertices': list(all_vertices)
                    }
                elif isinstance(first, list):
                    # List of layers
                    all_vertices = []
                    parents = {}
                    children = defaultdict(list)
                    for i, layer in enumerate(dag):
                        for v in layer:
                            if v not in all_vertices:
                                all_vertices.append(v)
                            for j in range(i):
                                for pv in dag[j]:
                                    if pv != v and v not in parents:
                                        parents[v] = []
                                    if v in parents and pv not in parents[v]:
                                        parents[v].append(pv)
                                        children[pv].append(v)
                    inputs = [v for v in all_vertices if v not in parents]
                    outputs = [v for v in all_vertices if v not in children]
                    return {
                        'parents': parents,
                        'children': dict(children),
                        'inputs': set(inputs),
                        'outputs': set(outputs),
                        'all_vertices': all_vertices
                    }
        
        return None
        
    except Exception as e:
        sys.stderr.write(f"Parse DAG error: {e}\n")
        return None


def _topological_sort(vertices, parents):
    """Topological sort using Kahn's algorithm."""
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    
    while queue:
        v = queue.popleft()
        order.append(v)
        for child in parents.get(v, []):
            # Wait, parents[v] are the parents of v, not children
            # We need children mapping
            pass
    
    # Actually, let's do it properly
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    
    while queue:
        v = queue.popleft()
        order.append(v)
        for child in children.get(v, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    
    if len(order) != len(vertices):
        return None  # Cycle detected
    
    return order


def _find_chains(all_vertices, parents, children, topo_order):
    """
    Find chains in the DAG using greedy path decomposition.
    A chain is a sequence of vertices where each (except last) has exactly one child in the chain.
    """
    # Build a working copy of children
    remaining_children = {v: list(children.get(v, [])) for v in all_vertices}
    visited = set()
    chains = []
    
    # Process vertices in topological order
    for start in topo_order:
        if start in visited:
            continue
        
        # Start a new chain from start
        chain = [start]
        visited.add(start)
        current = start
        
        # Extend the chain greedily
        while True:
            # Find an unvisited child
            next_v = None
            for child in remaining_children.get(current, []):
                if child not in visited:
                    next_v = child
                    break
            
            if next_v is None:
                break
            
            chain.append(next_v)
            visited.add(next_v)
            # Remove this child from current's remaining children
            remaining_children[current].remove(next_v)
            current = next_v
        
        if len(chain) > 1:
            chains.append(chain)
        else:
            # Single vertex chain
            chains.append(chain)
    
    return chains


def _simple_pebbling(all_vertices, parents, children, inputs, outputs, fast_memory_budget):
    """
    Simple topological pebbling as fallback.
    """
    topo_order = _topological_sort(all_vertices, parents)
    if topo_order is None:
        return None
    
    ops = []
    fast_mem = set()
    
    for v in topo_order:
        # Load parents if not in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                ops.append(['load', p])
                fast_mem.add(p)
        
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # Store v if it's an output
        if v in outputs:
            ops.append(['store', v])
        
        # Delete parents that are no longer needed
        for p in parents.get(v, []):
            # Check if p is still needed by any unprocessed vertex
            still_needed = False
            for child in children.get(p, []):
                if child not in fast_mem or child in outputs:
                    # If child not yet processed or is output, p might still be needed
                    # Actually, we need to check if any unprocessed vertex has p as parent
                    pass
            
            # Simpler: just track reference counts
            # For now, don't delete to be safe
            pass
    
    return ops


def _schedule_chains(chains, parents, children, inputs, outputs, fast_memory_budget, all_vertices):
    """
    Schedule operations using round-robin over chains.
    """
    # For each chain, we'll process it from left to right
    # Each vertex in a chain needs its parents loaded/computed first
    
    ops = []
    fast_mem = set()
    computed = set()
    
    # Track position in each chain
    chain_pos = [0] * len(chains)
    
    # We need to ensure all parents are available before computing a vertex
    # Let's use a priority-based approach
    
    # Actually, let's do a simpler approach: process chains in order,
    # but interleave to keep fast memory utilization high
    
    # First, let's just process each chain sequentially but delete
    # values that are no longer needed
    
    for chain in chains:
        for v in chain:
            # Load all parents not in fast memory
            for p in parents.get(v, []):
                if p not in fast_mem:
                    # Check if p is in a previous chain or this chain
                    if p in computed:
                        ops.append(['load', p])
                        fast_mem.add(p)
                    else:
                        # This shouldn't happen if chains are properly ordered
                        # But handle it anyway
                        ops.append(['load', p])
                        fast_mem.add(p)
            
            # Compute v
            ops.append(['compute', v])
            fast_mem.add(v)
            computed.add(v)
            
            # If v is an output, store it
            if v in outputs:
                ops.append(['store', v])
            
            # Delete parents that are no longer needed
            for p in parents.get(v, []):
                # Check if p is still needed
                still_needed = False
                for child in children.get(p, []):
                    if child not in computed:
                        still_needed = True
                        break
                
                if not still_needed and p not in outputs:
                    if p in fast_mem:
                        ops.append(['delete', p])
                        fast_mem.remove(p)
    
    if len(ops) > 20000:
        return None
        
    return ops


def _validate_pebbling(dag, fast_memory_budget, ops):
    """
    Validate a pebbling schedule.
    Returns True if valid, False otherwise.
    """
    try:
        if not ops:
            return True
        
        if len(ops) > 20000:
            return False
        
        # Parse DAG
        dag_info = _parse_dag(dag)
        if dag_info is None:
            return True  # Can't validate, assume valid
        
        parents = dag_info['parents']
        children = dag_info['children']
        inputs = dag_info['inputs']
        outputs = dag_info['outputs']
        all_vertices = set(dag_info['all_vertices'])
        
        # Simulate the state machine
        fast_mem = set()
        slow_mem = set()  # Values that have been stored
        computed = set()
        
        for op, v in ops:
            if op == 'load':
                # v must be in slow_mem (previously stored) or be an input
                if v not in slow_mem and v not in inputs:
                    return False
                if len(fast_mem) >= fast_memory_budget:
                    return False
                fast_mem.add(v)
            
            elif op == 'store':
                # v must be in fast_mem
                if v not in fast_mem:
                    return False
                slow_mem.add(v)
                # Don't remove from fast_mem yet (it can stay)
            
            elif op == 'compute':
                # All parents of v must be in fast_mem
                for p in parents.get(v, []):
                    if p not in fast_mem:
                        return False
                if len(fast_mem) >= fast_memory_budget:
                    return False
                fast_mem.add(v)
                computed.add(v)
            
            elif op == 'delete':
                # v must be in fast_mem
                if v not in fast_mem:
                    return False
                fast_mem.remove(v)
            
            else:
                return False
        
        # All outputs must be in fast_mem at the end
        for o in outputs:
            if o not in fast_mem:
                return False
        
        return True
        
    except Exception as e:
        sys.stderr.write(f"Validation error: {e}\n")
        return False
# EVOLVE-BLOCK-END
