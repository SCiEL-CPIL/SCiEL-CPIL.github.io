"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Generates a pebbling schedule for the given DAG.
    
    Args:
        dag: A dictionary or object representing the DAG. 
             Expected structure: 
             - dag['vertices']: list of vertex ids
             - dag['edges']: list of (u, v) where u is parent of v
             - dag['inputs']: list of input vertex ids
             - dag['outputs']: list of output vertex ids
             Or similar. We need to be robust to structure.
        fast_memory_budget: Integer, max number of vertices in fast memory.
    
    Returns:
        list of [operation, vertex]
    """
    start_time = time.time()
    deadline = start_time + 45.0 # Safety margin for 60s limit

    # 1. Parse DAG structure
    # Assuming standard structure from previous tasks in this suite:
    # dag is a dict with 'vertices', 'parents', 'children', 'inputs', 'outputs'
    # Or 'edges'. Let's inspect common patterns.
    # The prompt says "dag has topologically ordered parents and output vertices".
    # This implies dag might be a list of parents for each vertex?
    # "dag has topologically ordered parents" -> dag[i] is list of parents of vertex i?
    # Let's assume dag is a list of lists of parents, or a dict.
    
    # Robust parsing
    if isinstance(dag, dict):
        if 'parents' in dag:
            parents = dag['parents']
            vertices = list(range(len(parents))) if 'vertices' not in dag else dag['vertices']
        elif 'edges' in dag:
            edges = dag['edges']
            vertices = dag.get('vertices', list(range(max([u for u,v in edges] + [v for u,v in edges]) + 1) if edges else [0]))
            parents = defaultdict(list)
            for u, v in edges:
                parents[v].append(u)
            # Ensure all vertices have an entry
            for v in vertices:
                if v not in parents:
                    parents[v] = []
            parents = [parents[v] for v in vertices]
        else:
            # Fallback: try to guess
            raise ValueError("Unrecognized DAG format")
    elif isinstance(dag, (list, tuple)):
        # Assume list of parents
        parents = dag
        vertices = list(range(len(parents)))
    else:
        raise ValueError("Unrecognized DAG type")
        
    num_vertices = len(vertices)
    if num_vertices == 0:
        return []
        
    # Identify inputs and outputs
    inputs = set()
    outputs = set()
    if isinstance(dag, dict):
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        # Inputs are vertices with no parents
        for i, p_list in enumerate(parents):
            if not p_list:
                inputs.add(i)
        # Outputs are vertices with no children? Or specified?
        # If not specified, assume all vertices with no children are outputs?
        # Or maybe the last layer?
        # Let's compute children to find sinks
        children = defaultdict(list)
        for v in range(num_vertices):
            for p in parents[v]:
                children[p].append(v)
        for v in range(num_vertices):
            if not children[v]:
                outputs.add(v)
                
    if not outputs:
        # If no outputs specified, assume all vertices are outputs? Or just the last ones?
        # For safety, if no outputs, we might just need to compute everything?
        # But the contract says "all outputs must finish blue".
        # If outputs is empty, maybe we don't need to do anything? 
        # But usually there are outputs.
        pass

    # Topological Sort
    # Kahn's algorithm
    in_degree = [len(p_list) for p_list in parents]
    # Wait, in_degree for topological sort is number of incoming edges (parents).
    # So in_degree[v] = len(parents[v])
    
    queue = []
    for v in range(num_vertices):
        if in_degree[v] == 0:
            queue.append(v)
            
    topo_order = []
    # Use a list as a queue
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        topo_order.append(v)
        for u in children[v]:
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append(u)
                
    if len(topo_order) != num_vertices:
        # Cycle detected, but DAG should be acyclic
        # Fallback to sorted order?
        topo_order = list(range(num_vertices))

    # Simulation of Pebbling
    # State:
    # in_memory: set of vertices currently in fast memory (red)
    # stored_in_slow: set of vertices that have been computed and stored in slow memory (blue/available)
    # We need to generate the sequence of operations.
    
    operations = []
    in_memory = set()
    stored_in_slow = set(inputs) # Inputs are available in slow memory initially? 
    # "Inputs start blue". Blue likely means available in slow memory.
    
    # We need to compute all vertices in topo_order.
    # For each vertex v:
    # 1. Ensure all parents p of v are in in_memory.
    #    - If p is in in_memory, do nothing.
    #    - If p is in stored_in_slow, load(p).
    #    - If p is neither, we have a problem. We must have computed it before.
    #      Since we process in topo order, parents are always processed before children.
    #      So parents must be in stored_in_slow or in_memory.
    # 2. compute(v). v becomes in_memory.
    # 3. store(v) to stored_in_slow. (This makes it available for future loads if evicted).
    # 4. Manage memory:
    #    - If len(in_memory) > fast_memory_budget, we must evict.
    #    - We should evict a vertex that is least likely to be needed.
    #    - Heuristic: Lookahead. Find the next vertex in topo_order that needs a parent.
    #    - Keep parents of upcoming nodes.
    
    # Precompute "next use" for better eviction?
    # For each vertex v, find the indices in topo_order where v is a parent.
    # parent_usage = {v: [indices in topo_order where v is a parent]}
    
    parent_usage = defaultdict(list)
    for i, v in enumerate(topo_order):
        for p in parents[v]:
            parent_usage[p].append(i)
            
    # For efficient lookup, we can maintain a pointer for each vertex to its next usage.
    # next_use_ptr = {v: 0}
    # When we pass index i in topo_order, we update next_use for parents of topo_order[i].
    
    # Simpler: Just use LRU. It's often good enough.
    # But let's try a simple lookahead: 
    # When evicting, pick the vertex with the furthest next use.
    
    # We need to track the current index in topo_order to know "future".
    # Since we process sequentially, we can pre-compute the list of parents for each step.
    
    # Let's use a simple LRU implementation with a counter for last access.
    last_access = {}
    access_counter = 0
    
    # To make eviction smarter, we can check the next few vertices in topo_order.
    # Let's stick to LRU first, as it's simple and effective.
    # If LRU is not enough, we can switch to "Furthest Next Use".
    
    # Implementation of Furthest Next Use:
    # For each vertex in memory, we need to know its next use index.
    # We can maintain a heap of (next_use_index, vertex) for vertices in memory.
    # When we access a vertex, we update its next use.
    
    # Let's pre-compute the list of next uses for each vertex.
    # next_uses[v] = sorted list of indices i in topo_order such that v is a parent of topo_order[i].
    # Since topo_order is fixed, we can just use the lists from parent_usage.
    # We need to know which one is the "next" one given the current position in topo_order.
    
    # To avoid O(N^2) or complex pointer management, let's use a simpler heuristic:
    # Keep parents of the next K vertices in memory.
    # K = fast_memory_budget
    
    # Actually, let's just implement standard LRU. It's robust.
    
    for v in topo_order:
        # Check deadline
        if time.time() > deadline:
            break
            
        # 1. Load parents
        for p in parents[v]:
            if p not in in_memory:
                # p must be in stored_in_slow
                if p in stored_in_slow:
                    operations.append(["load", p])
                    in_memory.add(p)
                    # Update last access
                    last_access[p] = access_counter
                    access_counter += 1
                else:
                    # This shouldn't happen if topo order is correct and we store all computed nodes
                    # But if budget is 0?
                    pass
        
        # 2. Compute v
        operations.append(["compute", v])
        in_memory.add(v)
        last_access[v] = access_counter
        access_counter += 1
        
        # 3. Store v to slow memory
        operations.append(["store", v])
        stored_in_slow.add(v)
        
        # 4. Evict if necessary
        if len(in_memory) > fast_memory_budget:
            # Find LRU
            # We need to find the vertex in in_memory with the smallest last_access
            # To make this efficient, we could use a heap, but for small B, linear scan is fine.
            # If B is large, we need efficiency.
            # Let's assume B is small enough for linear scan, or use a heap.
            # For now, linear scan.
            lru_vertex = None
            min_access = float('inf')
            for u in in_memory:
                if last_access.get(u, -1) < min_access:
                    min_access = last_access.get(u, -1)
                    lru_vertex = u
                    
            if lru_vertex is not None:
                operations.append(["delete", lru_vertex])
                in_memory.remove(lru_vertex)
                # We don't remove from stored_in_slow, as it's still there.

    # Ensure outputs are in memory? "All outputs must finish blue".
    # If "blue" means in slow memory, we are done.
    # If "blue" means in fast memory, we need to load them back?
    # "Inputs start blue... all outputs must finish blue".
    # Usually, "blue" is the stable state (slow memory). "Red" is working.
    # So we just need them to be computed and stored.
    
    return operations

# EVOLVE-BLOCK-END
