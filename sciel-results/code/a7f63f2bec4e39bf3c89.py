"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
        elif 'nodes' in dag:
            vertices = list(dag['nodes'])
        else:
            parents_dict = dag.get('parents', {})
            vertices = list(parents_dict.keys())
            all_p = set()
            for pl in parents_dict.values():
                all_p.update(pl)
            for v in all_p:
                if v not in vertices:
                    vertices.append(v)
        parents_dict = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        inputs = set(dag.get('inputs', []))
        if not inputs:
            inputs = {v for v in vertices if not parents_dict.get(v, [])}
        if not outputs:
            has_child = set()
            for v in vertices:
                for p in parents_dict.get(v, []):
                    has_child.add(p)
            outputs = {v for v in vertices if v not in has_child}
    else:
        vertices = []
        parents_dict = {}
        for item in dag:
            if isinstance(item, tuple) and len(item) == 2:
                v, ps = item
                vertices.append(v)
                parents_dict[v] = list(ps)
            elif isinstance(item, dict):
                v = item.get('vertex', item.get('id'))
                ps = item.get('parents', [])
                vertices.append(v)
                parents_dict[v] = list(ps)
        has_child = set()
        for v in vertices:
            for p in parents_dict.get(v, []):
                has_child.add(p)
        inputs = {v for v in vertices if not parents_dict.get(v, [])}
        outputs = {v for v in vertices if v not in has_child}
    
    if not vertices:
        return []
    
    children_dict = defaultdict(list)
    for v in vertices:
        for p in parents_dict.get(v, []):
            children_dict[p].append(v)
    
    # Topological sort
    in_deg = {v: len(parents_dict.get(v, [])) for v in vertices}
    q = deque([v for v in vertices if in_deg[v] == 0])
    topo = []
    idg = dict(in_deg)
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children_dict[v]:
            idg[c] -= 1
            if idg[c] == 0:
                q.append(c)
    if len(topo) < len(vertices):
        topo = list(vertices)
    
    # Determine relevant vertices (lead to outputs)
    leads_out = {}
    def _lo(v, vis=None):
        if v in leads_out:
            return leads_out[v]
        if vis is None:
            vis = set()
        if v in vis:
            return False
        vis.add(v)
        if v in outputs:
            leads_out[v] = True
            return True
        for c in children_dict[v]:
            if _lo(c, vis):
                leads_out[v] = True
                return True
        leads_out[v] = False
        return False
    for v in vertices:
        _lo(v)
    
    relevant = [v for v in topo if leads_out.get(v, False)]
    
    # Count how many relevant children each vertex has
    demand = {v: len([c for c in children_dict[v] if leads_out.get(c, False)]) for v in relevant}
    
    # Build schedule
    ops = []
    blue = set(inputs)
    red = set()
    computed = set()
    remaining = {v: demand.get(v, 0) for v in relevant}
    
    for v in topo:
        if time.time() - start_time > DEADLINE:
            break
        if not leads_out.get(v, False):
            continue
        
        parents_v = parents_dict.get(v, [])
        
        if not parents_v:
            # Input vertex
            if v not in red:
                if v in blue:
                    ops.append(['load', v])
                    red.add(v)
                    blue.discard(v)
            computed.add(v)
        else:
            # Compute vertex: ensure parents in red
            for p in parents_v:
                if p not in red:
                    if p in blue:
                        ops.append(['load', p])
                        red.add(p)
                        blue.discard(p)
            ops.append(['compute', v])
            red.add(v)
            computed.add(v)
            
            # Decrement remaining for parents
            for p in parents_v:
                remaining[p] = remaining.get(p, 0) - 1
        
        # Delete parents that are no longer needed (not outputs)
        for p in parents_v:
            if p in red and remaining.get(p, 0) <= 0 and p not in outputs:
                ops.append(['delete', p])
                red.discard(p)
        
        # Enforce fast memory budget
        while len(red) > fast_memory_budget:
            cands = [x for x in red if x not in outputs]
            if not cands:
                break
            mv = min(cands, key=lambda x: remaining.get(x, 0))
            ops.append(['delete', mv])
            red.discard(mv)
    
    # Store outputs
    for v in outputs:
        if v in red:
            ops.append(['store', v])
            red.discard(v)
            blue.add(v)
    
    return ops
# EVOLVE-BLOCK-END
