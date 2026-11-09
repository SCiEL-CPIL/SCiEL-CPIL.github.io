"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill

    def time_left():
        return deadline - (time.time() - t0)

    # Fallback to baseline if available
    baseline_ops = None
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_ops = pebble_baseline(dag, fast_memory_budget)
    except Exception:
        baseline_ops = None

    # Parse DAG
    nodes = None
    parents = None
    outputs = None
    
    try:
        if isinstance(dag, dict):
            if 'parents' in dag:
                parents = dag['parents']
            if 'outputs' in dag:
                outputs = dag['outputs']
            
            if isinstance(parents, list):
                nodes = list(range(len(parents)))
                parents = [list(p) for p in parents]
            elif isinstance(parents, dict):
                nodes = list(parents.keys())
                # Ensure keys are 0..n-1 or similar
                if nodes:
                    max_node = max(nodes)
                    if max_node == len(nodes) - 1 and min(nodes) == 0:
                        parents = [list(parents.get(i, [])) for i in range(len(nodes))]
                    else:
                        # Remap
                        node_map = {node: i for i, node in enumerate(nodes)}
                        new_parents = []
                        for node in nodes:
                            p_list = parents.get(node, [])
                            new_parents.append([node_map[p] for p in p_list])
                        parents = new_parents
                else:
                    parents = []
            else:
                parents = []
            
            if outputs is None:
                # Infer outputs: nodes with no children
                n = len(nodes) if nodes else 0
                children = [[] for _ in range(n)]
                for i in range(n):
                    for p in parents[i]:
                        children[p].append(i)
                outputs = [i for i in range(n) if not children[i]]
                if not outputs and n > 0:
                    outputs = [n-1]
            else:
                if not isinstance(outputs, list):
                    outputs = [outputs]
                # Remap outputs if nodes were remapped
                if isinstance(dag.get('parents'), dict) and nodes:
                    node_map = {node: i for i, node in enumerate(nodes)}
                    outputs = [node_map[o] for o in outputs if o in node_map]

        elif isinstance(dag, list):
            nodes = list(range(len(dag)))
            parents = []
            for i, node in enumerate(dag):
                if isinstance(node, dict):
                    p = node.get('parents', [])
                    parents.append(list(p))
                elif isinstance(node, (list, tuple)):
                    if len(node) >= 2 and isinstance(node[1], (list, tuple)):
                        parents.append(list(node[1]))
                    else:
                        parents.append([])
                else:
                    parents.append([])
            outputs = []
            for i, node in enumerate(dag):
                if isinstance(node, dict):
                    if node.get('is_output') or node.get('output'):
                        outputs.append(i)
            if not outputs:
                # Infer outputs
                n = len(nodes)
                children = [[] for _ in range(n)]
                for i in range(n):
                    for p in parents[i]:
                        children[p].append(i)
                outputs = [i for i in range(n) if not children[i]]
                if not outputs and n > 0:
                    outputs = [n-1]
    except Exception:
        pass

    if baseline_ops is not None and (nodes is None or parents is None):
        return baseline_ops

    if nodes is None or parents is None:
        return []

    n = len(nodes)
    budget = fast_memory_budget
    if budget <= 0:
        return []

    # Build children and compute depths
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)

    # Topological sort
    in_degree = [len(parents[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in children[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
        
    if len(topo) != n:
        # Cycle detected, fallback
        if baseline_ops is not None:
            return baseline_ops
        return []

    # Calculate depth (distance from source) and future usage
    depth = [0] * n
    for u in topo:
        for v in children[u]:
            depth[v] = max(depth[v], depth[u] + 1)
    
    # Count future usages: for each node, how many times is it needed in the future?
    # This is just the number of children, but we need to know when they are processed.
    # We will use a greedy strategy:
    # 1. Process in topological order.
    # 2. Keep a set `in_fast`.
    # 3. When `in_fast` is full and we need to add a new node, evict the node with the **lowest future demand**.
    #    Future demand is approximated by the number of children that are NOT yet processed.
    
    ops = []
    in_fast = set()
    
    # Pre-compute the position of each node in the topological order
    pos_in_topo = {v: i for i, v in enumerate(topo)}
    
    # To decide eviction, we need to know for each node in `in_fast`, how many of its children are still "pending" (not yet processed).
    # We can maintain a count `pending_children_count` for each node.
    pending_children = [0] * n
    for u in topo:
        for v in children[u]:
            pending_children[u] += 1 # Initially all children are pending
    
    # Wait, this static count doesn't update as we process.
    # Let's maintain a dynamic count.
    # When we process a node `v`, we decrement `pending_children` for all its parents.
    
    # Actually, simpler:
    # We process `v`. Before computing `v`, we ensure parents are in fast memory.
    # If a parent is not in fast memory, we MUST recompute it?
    # No, if we follow topological order, we compute parents first.
    # If we evicted a parent, we can't use it.
    # So, the strategy MUST be: **Never evict a node that is a parent of a node that is going to be processed in the immediate future** unless we are willing to recompute.
    # Recomputation is expensive.
    
    # Alternative: **Strict Topological Order with Memory Management**
    # We compute nodes in topo order.
    # We load inputs.
    # We compute nodes.
    # We store outputs.
    # We delete nodes that are no longer needed (all children processed).
    # If memory is full, we must evict a node that IS still needed.
    # Which one? The one that is "furthest" from its next use?
    # Or the one with the **smallest sub-DAG** (cheapest to recompute)?
    
    # Let's try: Evict the node with the **minimum depth** among those that are still needed.
    # Rationale: Shallow nodes are cheaper to recompute (fewer ancestors).
    
    # We need to know which nodes are "needed".
    # A node is needed if it has any children that have not yet been processed.
    
    # Let's simulate.
    processed = [False] * n
    # pending_children_count[u] = number of children of u that are NOT yet processed.
    # Initialize: pending_children_count[u] = len(children[u])
    
    for v in topo:
        # 1. Handle Inputs
        if len(parents[v]) == 0:
            if v not in in_fast:
                # Check memory
                if len(in_fast) >= budget:
                    # Evict
                    # Find candidate to evict: 
                    # Candidates are nodes in in_fast that are NOT v (v is new)
                    # We should evict a node that is still needed (pending_children > 0) or not needed?
                    # If a node is not needed (pending_children == 0), we should have deleted it already.
                    # So we only have nodes in in_fast that are either:
                    # a) Just computed and needed for future
                    # b) Inputs loaded and needed for future
                    # c) Outputs loaded? No, we store outputs immediately.
                    
                    # Let's maintain the invariant: 
                    # - Inputs are loaded when first needed.
                    # - Intermediate nodes are computed when reached.
                    # - Nodes are deleted when all their children are processed.
                    # - If memory is full, we evict the node with the **lowest "recomputation cost"**.
                    # Recomputation cost is approximated by the size of the sub-DAG rooted at that node? No, the sub-DAG *feeding* it.
                    # Let's use **Depth** as a proxy for recomputation cost. Shallow = cheap.
                    
                    best_evict = -1
                    best_depth = float('inf')
                    for u in in_fast:
                        # Only evict if u is not v (v is not in in_fast yet)
                        # And u is still needed? 
                        # If u is not needed, we should have deleted it.
                        # But maybe we didn't.
                        # Let's delete any node that is not needed first.
                        if pending_children[u] == 0:
                            best_evict = u
                            break
                    if best_evict == -1:
                        for u in in_fast:
                            if depth[u] < best_depth:
                                best_depth = depth[u]
                                best_evict = u
                    if best_evict != -1:
                        ops.append(['delete', best_evict])
                        in_fast.discard(best_evict)
                
                ops.append(['load', v])
                in_fast.add(v)
        
        # 2. Compute Intermediate/Output
        else:
            # Ensure parents are in fast memory
            # In topo order, parents are processed before v.
            # If a parent was evicted, we must recompute it.
            # This is the hard part.
            # To avoid this, we should **never evict a parent of a node that is about to be processed**.
            # How do we know? 
            # We can check: for each parent p of v, if p is not in in_fast, we must recompute p.
            # Recomputing p requires its parents, etc.
            
            # Simple Heuristic: 
            # Before processing v, check parents.
            # If any parent p is not in in_fast:
            #   We must recompute p.
            #   To recompute p, we need p's parents.
            #   This cascades.
            
            # Given the complexity, let's assume the budget is large enough or the DAG is layered such that we can keep all parents of the current layer.
            # If not, we will fall back to a simpler strategy: **Recompute from scratch if needed**.
            
            # For now, let's just assume parents are there. If not, we insert a recomputation block.
            # But implementing full recomputation logic is complex.
            
            # Let's try a different approach: **Layered Scheduling**.
            # Process by depth.
            # This is safer for memory management.
            
            pass # We will rewrite the main loop below.

    # REWRITE: Layered Scheduling
    # Group nodes by depth.
    max_depth = max(depth) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        layers[depth[v]].append(v)
    
    ops = []
    in_fast = set()
    processed = [False] * n
    
    # pending_children: how many children of u are not yet processed
    pending_children = [len(children[u]) for u in range(n)]
    
    # We process layer by layer.
    # For each layer, we compute all nodes.
    # To compute a node, its parents (in previous layer) must be in fast memory.
    # So, after processing a layer, we must ensure that all nodes in the current layer that are parents of the NEXT layer are kept in fast memory.
    # Nodes that are NOT parents of the next layer can be deleted.
    
    # Determine which nodes in current layer are needed for next layer.
    # needed_in_next = set of nodes in current layer that have at least one child in next layer.
    
    for d in range(max_depth + 1):
        current_layer = layers[d]
        if not current_layer:
            continue
            
        # Determine which nodes in current layer are needed for the next layer
        next_layer = layers[d+1] if d + 1 <= max_depth else []
        next_layer_set = set(next_layer)
        
        needed_in_next = set()
        for v in current_layer:
            for c in children[v]:
                if c in next_layer_set:
                    needed_in_next.add(v)
                    break
        
        # Compute nodes in current layer
        for v in current_layer:
            # If v is an input, load it
            if len(parents[v]) == 0:
                if v not in in_fast:
                    # Check memory
                    if len(in_fast) >= budget:
                        # Evict a node that is NOT needed_in_next
                        evicted = False
                        for u in in_fast:
                            if u not in needed_in_next:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            # Evict any node (shouldn't happen if budget >= width of critical path)
                            # Evict the one with lowest depth
                            best_evict = min(in_fast, key=lambda x: depth[x])
                            ops.append(['delete', best_evict])
                            in_fast.discard(best_evict)
                    ops.append(['load', v])
                    in_fast.add(v)
            else:
                # Compute v
                # Parents must be in in_fast
                # Since we process layer by layer, parents are in previous layer.
                # We ensured previous layer nodes needed for this layer were kept.
                # So parents should be in in_fast.
                ops.append(['compute', v])
                in_fast.add(v)
                processed[v] = True
                
                # Update pending_children for parents
                for p in parents[v]:
                    pending_children[p] -= 1
                    
        # After processing current layer, delete nodes that are NOT needed for next layer
        # And are not outputs (outputs must be stored)
        for v in current_layer:
            if v in needed_in_next:
                continue
            # v is not needed for next layer
            # If v is an output, store it
            if v in outputs:
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
            else:
                # Delete it
                if v in in_fast:
                    ops.append(['delete', v])
                    in_fast.discard(v)
                    
    # Check if all outputs are stored
    # If some outputs were not processed (e.g., if they were in a layer that was skipped?), they should be.
    # The loop covers all layers.
    
    # Validate
    if len(ops) > 20000:
        if baseline_ops is not None:
            return baseline_ops
        return []
        
    return ops

# EVOLVE-BLOCK-END
