"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

def _topological_sort(vertices, parents):
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in in_degree:
                    in_degree[v] += 1
                    children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(order) != len(vertices):
        return vertices
    return order

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
    try:
        vertices = dag['vertices']
        parents = dag['parents']
        outputs = set(dag['outputs'])
        
        order = _topological_sort(vertices, parents)
        
        # Pre-compute ancestors for each vertex
        # ancestors[v] is the set of all ancestors of v
        ancestors = {v: set() for v in vertices}
        for v in order:
            for p in parents.get(v, []):
                ancestors[v].add(p)
                ancestors[v].update(ancestors[p])
        
        in_memory = set()
        stored = set()
        ops = []
        
        def check_limits():
            if len(ops) > 20000:
                raise Exception("Ops limit exceeded")
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
                
        def make_space(protected):
            # protected: set of vertices that must remain in memory
            # Find a vertex u in in_memory not in protected
            # Heuristic: Pick the one with the largest topological index (furthest back)
            # or just the first one.
            candidates = [u for u in in_memory if u not in protected]
            if not candidates:
                return False
            
            # Sort candidates by topological index?
            # We can use a map for index
            # For speed, just pick the first one
            u = candidates[0]
            
            if u in outputs:
                if u not in stored:
                    ops.append(['store', u])
                    stored.add(u)
                in_memory.remove(u)
            else:
                ops.append(['delete', u])
                in_memory.remove(u)
            
            return True

        def compute(v, protected):
            # protected: set of vertices that must remain in memory
            check_limits()
            
            if v in in_memory:
                return
            
            # If we need to add v to memory, we need space
            if len(in_memory) >= fast_memory_budget:
                if not make_space(protected):
                    raise Exception("Cannot free space")
            
            if v in stored:
                ops.append(['load', v])
                in_memory.add(v)
                return
                
            if not parents.get(v, []):
                # Input
                ops.append(['load', v])
                in_memory.add(v)
                return
                
            # Intermediate
            # Ensure parents are in memory
            for p in parents[v]:
                # The protected set for p should include p's ancestors that are in memory
                # And also v? No, v is not in memory.
                # But p is an ancestor of v.
                # So we protect p's ancestors.
                compute(p, ancestors[p] & in_memory)
                
            # Now parents are in memory
            ops.append(['compute', v])
            in_memory.add(v)
            
        for v in order:
            # Compute v
            # Protected: ancestors of v that are in memory
            compute(v, ancestors[v] & in_memory)
            
            if v in outputs:
                # Store v
                # v is in memory
                ops.append(['store', v])
                stored.add(v)
                in_memory.remove(v)
                
        return ops
        
    except Exception as e:
        # Fallback to baseline if my logic fails
        # We don't have access to pebble_baseline directly in this scope unless we import it
        # The current code imports it in the except block.
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

# EVOLVE-BLOCK-END
