"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Minimize transfers (load+store) for DAG pebbling.
    
    dag: dict with keys 'vertices' (list), 'parents' (dict vertex->list of parent vertices),
         'outputs' (list of output vertices). Vertices are topologically ordered in 'vertices'.
    fast_memory_budget: int, max number of red pebbles (fast memory slots).
    
    Returns: list of [operation, vertex] where operation in {'load','store','compute','delete'}.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    n = len(vertices)
    if n == 0:
        return []
    
    vidx = {v: i for i, v in enumerate(vertices)}
    
    # Build children and fanout
    children = defaultdict(list)
    fanout = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
            fanout[p] += 1
    
    # State: 0=not computed, 1=blue, 2=red
    state = [0] * n
    for v in vertices:
        if not parents.get(v, []):
            state[vidx[v]] = 1
    
    ops = []
    red = set()
    
    def load(v):
        i = vidx[v]
        if state[i] == 2:
            return
        if state[i] == 1:
            ops.append(['load', v])
            state[i] = 2
            red.add(v)
        else:
            raise ValueError(f"load {v}: not blue")
    
    def store(v):
        i = vidx[v]
        if state[i] == 1:
            return
        if state[i] == 2:
            ops.append(['store', v])
            state[i] = 1
            red.discard(v)
    
    def delete(v):
        i = vidx[v]
        if state[i] == 2:
            ops.append(['delete', v])
            state[i] = 0
            red.discard(v)
    
    def compute(v):
        i = vidx[v]
        for p in parents.get(v, []):
            if state[vidx[p]] != 2:
                raise ValueError(f"compute {v}: parent {p} not red")
        ops.append(['compute', v])
        state[i] = 2
        red.add(v)
    
    # Last use: for each vertex, the max topo index of any child
    last_use = [0] * n
    for v in vertices:
        i = vidx[v]
        for c in children.get(v, []):
            if vidx[c] > last_use[i]:
                last_use[i] = vidx[c]
    
    # Process in topological order
    for v in vertices:
        i = vidx[v]
        if state[i] == 2:
            continue
        if state[i] == 1:
            # Blue. If it has children, we'll need it red later.
            # If no children and not output, we can delete it.
            # For now, leave it blue.
            continue
        
        # Need to compute v. Ensure parents are red.
        for p in parents.get(v, []):
            pi = vidx[p]
            if state[pi] != 2:
                if state[pi] == 1:
                    # Need to load p
                    if len(red) >= fast_memory_budget:
                        # Evict something
                        # Prefer evicting: not a parent of v, low fanout, early last_use
                        cands = []
                        for rv in red:
                            if rv in parents.get(v, []):
                                continue
                            cands.append((fanout[rv], last_use[vidx[rv]], rv))
                        if cands:
                            cands.sort(key=lambda x: (x[0], x[1]))
                            ev = cands[0][2]
                            if last_use[vidx[ev]] > i:
                                store(ev)
                            else:
                                delete(ev)
                        else:
                            # Evict a parent
                            for p2 in parents.get(v, []):
                                if state[vidx[p2]] == 2:
                                    if last_use[vidx[p2]] > i:
                                        store(p2)
                                    else:
                                        delete(p2)
                                    break
                    load(p)
                else:
                    raise ValueError(f"parent {p} of {v} not ready")
        
        compute(v)
        
        # Manage fast memory: if over budget, evict
        if len(red) > fast_memory_budget:
            # Evict the vertex with lowest future utility
            cands = []
            for rv in red:
                ri = vidx[rv]
                cands.append((fanout[rv], last_use[ri], ri, rv))
            cands.sort(key=lambda x: (x[0], x[1]))
            ev = cands[0][3]
            if last_use[vidx[ev]] > i:
                store(ev)
            else:
                delete(ev)
    
    # Ensure outputs are blue
    for v in outputs:
        i = vidx[v]
        if state[i] == 2:
            store(v)
        elif state[i] == 0:
            # Compute it
            for p in parents.get(v, []):
                pi = vidx[p]
                if state[pi] != 2:
                    if state[pi] == 1:
                        if len(red) >= fast_memory_budget:
                            for rv in list(red):
                                if vidx[rv] != i:
                                    store(rv)
                                    break
                        load(p)
                    else:
                        raise ValueError(f"parent {p} of output {v} not ready")
            compute(v)
            store(v)
    
    return ops

# EVOLVE-BLOCK-END
