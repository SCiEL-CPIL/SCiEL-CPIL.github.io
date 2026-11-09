"""Improved DAG pebbling with memory-aware scheduling."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimize communication for DAG pebbling.
    Returns list of [operation, vertex].
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        outputs = list(dag.get('outputs', []))
        inputs = list(dag.get('inputs', []))
        
        if 'parents' in dag:
            parents_map = {v: list(ps) for v, ps in dag['parents'].items()}
        elif 'edges' in dag:
            parents_map = {v: [] for v in vertices}
            for p, c in dag['edges']:
                parents_map.setdefault(c, []).append(p)
        else:
            parents_map = {v: [] for v in vertices}
        
        if 'children' in dag:
            children_map = {v: list(cs) for v, cs in dag['children'].items()}
        elif 'edges' in dag:
            children_map = {v: [] for v in vertices}
            for p, c in dag['edges']:
                children_map.setdefault(p, []).append(c)
        else:
            children_map = {v: [] for v in vertices}
        
        for v in vertices:
            parents_map.setdefault(v, [])
            children_map.setdefault(v, [])
    else:
        vertices = []
        outputs = []
        inputs = []
        parents_map = {}
        children_map = {}
    
    if not vertices and parents_map:
        vertices = list(parents_map.keys())
    
    if not inputs:
        inputs = [v for v in vertices if not parents_map.get(v, [])]
    
    # Topological sort
    in_deg = {v: len(parents_map.get(v, [])) for v in vertices}
    topo = []
    q = deque([v for v in vertices if in_deg[v] == 0])
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children_map.get(v, []):
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    if not topo:
        topo = vertices
    
    # Ref count: how many times each vertex is used as a parent
    use_count = defaultdict(int)
    for v in vertices:
        for c in children_map.get(v, []):
            use_count[v] += 1
    
    # Greedy: process in topo order, keep values in fast mem if they'll be reused
    schedule = []
    fast = set()
    done = set()
    
    def evict():
        """Evict least useful vertex from fast memory"""
        if not fast:
            return
        # Evict vertex with fewest remaining uses
        best = None
        best_uses = float('inf')
        for v in fast:
            # Count uncomputed children
            rem = 0
            for c in children_map.get(v, []):
                if c not in done:
                    rem += 1
            if rem < best_uses:
                best_uses = rem
                best = v
        if best is not None:
            schedule.append(['store', best])
            fast.discard(best)
    
    for v in topo:
        if time.time() - start_time > DEADLINE:
            break
        if v in done:
            continue
        
        # Load parents into fast memory
        for p in parents_map.get(v, []):
            if p not in fast:
                # Make room
                while len(fast) >= fast_memory_budget:
                    evict()
                    if len(fast) < fast_memory_budget:
                        break
                schedule.append(['load', p])
                fast.add(p)
        
        # Compute
        schedule.append(['compute', v])
        done.add(v)
        fast.add(v)
        
        # If over budget, evict
        while len(fast) > fast_memory_budget:
            evict()
    
    # Store all outputs
    for v in outputs:
        if time.time() - start_time > DEADLINE:
            break
        if v in fast:
            schedule.append(['store', v])
            fast.discard(v)
    
    if len(schedule) > 20000:
        schedule = schedule[:20000]
    
    return schedule
# EVOLVE-BLOCK-END
