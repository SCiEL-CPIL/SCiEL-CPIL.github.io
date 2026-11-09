"""Communication-minimizing DAG pebbling with caching and budget-aware eviction."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 85.0
    MAX_OPS = 20000

    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    topo_order = list(vertices)
    
    remaining_use = {}
    for v in topo_order:
        remaining_use[v] = len(children.get(v, []))
    
    state = {}
    for v in topo_order:
        if v in inputs:
            state[v] = 'blue'
        else:
            state[v] = 'none'
    
    red_set = set()
    ops = []
    
    def add_op(op, vertex):
        if len(ops) < MAX_OPS:
            ops.append([op, vertex])
    
    def load(v):
        add_op('load', v)
        state[v] = 'red'
        red_set.add(v)
    
    def store(v):
        add_op('store', v)
        state[v] = 'blue'
        red_set.discard(v)
    
    def delete_v(v):
        add_op('delete', v)
        state[v] = 'gone'
        red_set.discard(v)
    
    def compute(v):
        add_op('compute', v)
        state[v] = 'red'
        red_set.add(v)
    
    def evict_if_needed():
        while len(red_set) > fast_memory_budget and len(ops) < MAX_OPS:
            candidates = [v for v in red_set if v not in outputs]
            if not candidates:
                candidates = list(red_set)
            if not candidates:
                break
            victim = min(candidates, key=lambda v: remaining_use.get(v, 0))
            store(victim)
    
    for v in topo_order:
        if time.time() > DEADLINE or len(ops) >= MAX_OPS:
            break
        
        for p in parents.get(v, []):
            if state.get(p) != 'red':
                if state.get(p) == 'blue':
                    load(p)
                # If 'gone' or 'none', we can't easily recompute here. 
                # This should be rare with our strategy.
            evict_if_needed()
        
        compute(v)
        evict_if_needed()
        
        for p in parents.get(v, []):
            remaining_use[p] = remaining_use.get(p, 0) - 1
            if remaining_use.get(p, 0) == 0 and state.get(p) == 'red':
                if p in outputs:
                    store(p)
                else:
                    delete_v(p)
    
    for v in topo_order:
        if v in outputs and state.get(v) == 'red' and len(ops) < MAX_OPS:
            store(v)
    
    return ops

# EVOLVE-BLOCK-END
