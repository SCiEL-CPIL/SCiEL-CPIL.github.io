import sys
import time
from collections import defaultdict, deque

sys.setrecursionlimit(10000)

def _topo_sort(parents):
    # Kahn's algorithm
    vertices = set(parents.keys())
    for ps in parents.values():
        for p in ps:
            vertices.add(p)
    
    in_degree = {v: 0 for v in vertices}
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
            in_degree[v] += 1
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo) != len(vertices):
        raise ValueError("Cycle detected")
        
    return topo, vertices, children

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 55.0 # Safety margin
    
    if isinstance(dag, dict):
        parents = dag['parents']
        outputs = set(dag['outputs'])
    elif isinstance(dag, list):
        parents = {}
        outputs = set()
        for i, v in enumerate(dag):
            # Handle potential ID mismatch if 'id' is present
            vid = v.get('id', i)
            # Parents key might vary, try 'parents', 'preds', 'inputs'
            p_list = v.get('parents', v.get('preds', v.get('inputs', [])))
            parents[vid] = p_list
            # Output key might vary, try 'is_output', 'output', 'out'
            if v.get('is_output', v.get('output', v.get('out', False))):
                outputs.add(vid)
    else:
        raise ValueError("Unsupported dag type")
    
    topo, vertices, children = _topo_sort(parents)
    topo_index = {v: i for i, v in enumerate(topo)}
    
    # Inputs are vertices with no parents
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    # State
    state = {v: 'Gone' for v in vertices}
    for v in inputs:
        state[v] = 'Blue'
        
    red_set = set()
    ops = []
    transfers = 0
    
    # remaining_children: how many children of v are not yet processed
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # next_use: min topo index of unprocessed children
    # Initially, for each v, min topo index of children[v]
    # If no children, infinity
    INF = float('inf')
    next_use = {}
    for v in vertices:
        if not children[v]:
            next_use[v] = INF
        else:
            next_use[v] = min(topo_index[c] for c in children[v])
            
    def evict(protected):
        nonlocal state, red_set, ops, transfers, remaining_children, next_use
        # Find candidate in red_set not in protected
        candidates = [v for v in red_set if v not in protected]
        if not candidates:
            # If we cannot evict, we might be stuck.
            # Raise error to signal failure, but maybe try to find any red node?
            # No, if we evict protected, we break computation.
            # Let's just raise.
            raise RuntimeError("No room to evict")
        
        # Choose candidate with max next_use
        # If next_use is INF, it can be deleted (0 cost)
        # If next_use is finite, it must be stored (1 cost)
        # We prefer to delete nodes with INF next_use if possible?
        # Actually, if we delete a node with finite next_use, we recompute later.
        # If we store a node with finite next_use, we load later.
        # Load is cheaper than recompute.
        # So, among candidates:
        # 1. If any candidate has remaining_children == 0, delete one (0 transfer).
        # 2. Else, store the one with max next_use (1 transfer).
        
        # Check for deletable candidates
        deletable = [v for v in candidates if remaining_children[v] == 0]
        if deletable:
            # Pick one (arbitrary or least recently used? doesn't matter much if 0 cost)
            v = deletable[0]
            state[v] = 'Gone'
            red_set.remove(v)
            ops.append(['delete', v])
            return
        
        # Must store
        # Pick v with max next_use
        v = max(candidates, key=lambda x: next_use[x])
        state[v] = 'Blue'
        red_set.remove(v)
        ops.append(['store', v])
        transfers += 1
        return v

    def ensure(v, protected_stack):
        nonlocal state, red_set, ops, transfers, remaining_children, next_use
        
        if state[v] == 'Red':
            return
        
        # Protect v and its ancestors in the current ensure chain
        protected = set(protected_stack)
        protected.add(v)
        
        if state[v] == 'Blue':
            # Need to load
            if len(red_set) >= fast_memory_budget:
                evict(protected)
            state[v] = 'Red'
            red_set.add(v)
            ops.append(['load', v])
            transfers += 1
        else: # Gone
            # Need to compute
            # Ensure parents are red
            for p in parents.get(v, []):
                ensure(p, protected_stack + [v])
            
            if len(red_set) >= fast_memory_budget:
                # After ensuring parents, v is not red yet.
                # We need space for v.
                # v is in protected.
                evict(protected)
            
            state[v] = 'Red'
            red_set.add(v)
            ops.append(['compute', v])
            
    # Process nodes in topological order
    for v in topo:
        if time.time() > deadline:
            break
            
        # Ensure v is computed and red
        ensure(v, [])
        
        # v is now Red.
        # We are "processing" v.
        # Update remaining_children for parents of v?
        # No, remaining_children counts how many children are *not yet processed*.
        # Since we are processing v now, v is no longer "unprocessed".
        # So for each parent p of v, remaining_children[p] decreases.
        for p in parents.get(v, []):
            remaining_children[p] -= 1
            # Update next_use for p
            # next_use[p] is min topo index of unprocessed children.
            # We need to recompute min of children[p] where remaining_children is not the only indicator.
            # Actually, remaining_children is just a count.
            # We need to know which children are still unprocessed.
            # Since we process in topo order, a child c is unprocessed if topo_index[c] > topo_index[v].
            # So next_use[p] should be the min topo_index[c] for c in children[p] such that topo_index[c] > topo_index[v].
            # We can update this efficiently or just recompute if small.
            # Given constraints, recompute is safe if N is small.
            unprocessed_children = [c for c in children[p] if topo_index[c] > topo_index[v]]
            if unprocessed_children:
                next_use[p] = min(topo_index[c] for c in unprocessed_children)
            else:
                next_use[p] = INF

    # After processing all nodes, all outputs must be Blue.
    # Currently, they might be Red.
    for v in outputs:
        if state[v] == 'Red':
            # Store to Blue
            # No budget constraint for storing? "Every red value consumes one fast-memory slot".
            # Storing frees a slot.
            state[v] = 'Blue'
            red_set.remove(v)
            ops.append(['store', v])
            transfers += 1
            
    # Any other Red nodes?
    # The problem doesn't explicitly say other nodes must be Blue, but "Inputs start blue... all outputs must finish blue".
    # Usually, we clean up.
    for v in list(red_set):
        state[v] = 'Blue'
        red_set.remove(v)
        ops.append(['store', v])
        transfers += 1

    return ops
