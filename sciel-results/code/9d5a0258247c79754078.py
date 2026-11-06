"""Optimized DAG pebbling: greedy with local-search refinement under a time budget."""
import time
import heapq
import random
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Produce a pebbling schedule for the given DAG that minimizes memory transfers.
    
    dag: dict or object with keys:
        - 'vertices': list of vertex ids
        - 'parents': dict mapping vertex -> list of parent vertices (topologically ordered)
        - 'inputs': list of input vertex ids (start blue)
        - 'outputs': list of output vertex ids (must finish blue)
        Or similar structure; we'll handle common formats.
    
    fast_memory_budget: int, max number of red pebbles allowed simultaneously.
    
    Returns: list of [operation, vertex] where operation in {'load','store','compute','delete'}.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # --- Parse DAG structure ---
    # Try to handle multiple common formats
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        parents = dict(dag.get('parents', {}))
        inputs = list(dag.get('inputs', []))
        outputs = list(dag.get('outputs', []))
    else:
        # Try attribute access
        vertices = list(getattr(dag, 'vertices', []))
        parents = dict(getattr(dag, 'parents', {}))
        inputs = list(getattr(dag, 'inputs', []))
        outputs = list(getattr(dag, 'outputs', []))
    
    if not vertices:
        return []
    
    # Ensure all vertices that appear in parents are in vertices
    all_verts = set(vertices)
    for v, ps in parents.items():
        all_verts.add(v)
        for p in ps:
            all_verts.add(p)
    vertices = list(all_verts)
    
    # Topological order (parents should already be ordered, but verify)
    # Build children map
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    # Compute topological order via Kahn's algorithm
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topo_order doesn't cover all, add remaining (shouldn't happen for valid DAG)
    if len(topo_order) < len(vertices):
        for v in vertices:
            if v not in topo_order:
                topo_order.append(v)
    
    # Determine which vertices are inputs and outputs
    input_set = set(inputs) if inputs else set()
    output_set = set(outputs) if outputs else set()
    
    # If no explicit inputs/outputs, infer:
    # Inputs: vertices with no parents
    # Outputs: vertices with no children
    if not input_set:
        input_set = set(v for v in vertices if not parents.get(v, []))
    if not output_set:
        output_set = set(v for v in vertices if not children.get(v, []))
    
    # Vertices that need to be computed (have parents)
    computed_vertices = [v for v in topo_order if parents.get(v, [])]
    
    # --- State representation ---
    # State: set of vertices currently in fast memory (red pebbles)
    # We track:
    # - pebbled: set of vertices with red pebbles
    # - value: for each pebbled vertex, whether its value is valid
    # 
    # Operations:
    # - load(v): move v from slow to fast (1 transfer). v must have been stored or is an input.
    # - store(v): move v from fast to slow (1 transfer). v must be pebbled.
    # - compute(v): compute v from its parents (0 transfers if all parents pebbled, 
    #                or requires loading parents). Result goes to fast memory.
    # - delete(v): remove v from fast memory (0 transfers). v must be pebbled.
    
    # Actually, let me re-read the problem:
    # "Inputs start blue" - so inputs are initially in slow memory (blue)
    # "all outputs must finish blue" - outputs must be stored to slow memory at the end
    # "every red value consumes one fast-memory slot"
    # Operations: load/store/compute/delete
    
    # Let me reconsider the model:
    # - Blue = in slow memory (persistent, free)
    # - Red = in fast memory (limited by budget)
    # - load(v): blue -> red (1 transfer)
    # - store(v): red -> blue (1 transfer)  
    # - compute(v): requires all parents to be red (or blue and loaded), produces v as red
    #   Actually, compute should require parents to be available (red), and produces v in red
    # - delete(v): red -> gone (0 transfers), but value lost unless stored
    
    # Wait, "delete" likely means removing a pebble without storing, freeing a slot.
    # But if you need the value later, you'd have to recompute or reload.
    
    # Let me define the state machine more carefully:
    # State: (pebbled_set, stored_set)
    # - pebbled_set: vertices currently in fast memory (red)
    # - stored_set: vertices currently in slow memory (blue) that have been computed/stored
    # 
    # Initially: stored_set = input_set (inputs start blue), pebbled_set = {}
    # Goal: all outputs in stored_set, pebbled_set can be anything (but we minimize transfers)
    
    # Transitions:
    # - load(v): v in stored_set, v not in pebbled_set -> pebbled_set += {v}, cost=1
    # - store(v): v in pebbled_set -> stored_set += {v}, pebbled_set -= {v}, cost=1
    #   (v remains in stored_set after being pebbled again? No, store moves it to slow)
    #   Actually: store(v) makes v blue (in slow memory). If v was pebbled, it becomes unpebbled.
    #   stored_set tracks what's in slow memory. A vertex can be both pebbled and stored?
    #   Typically in pebbling, a pebble is either on the vertex (red) or off (blue/slow).
    #   Let me assume: each vertex is either red (in fast mem) or blue (in slow mem) or absent.
    #   For vertices that have been computed, they exist in slow memory until deleted.
    
    # Simpler model:
    # - Each vertex has a status: 'absent', 'blue' (in slow mem), 'red' (in fast mem)
    # - Inputs start 'blue'
    # - To compute v: all parents must be 'red' (or we load them first). After compute, v becomes 'red'.
    #   But wait, if a parent is 'blue', we can load it to make it 'red', compute, then maybe store/delete.
    # - load(v): v must be 'blue' -> v becomes 'red', cost 1
    # - store(v): v must be 'red' -> v becomes 'blue', cost 1  
    # - compute(v): all parents must be 'red' -> v becomes 'red', cost 0 (but uses a fast mem slot)
    #   Actually, compute might not cost a transfer but does use a fast memory slot for the result.
    # - delete(v): v must be 'red' or 'blue' -> v becomes 'absent', cost 0 (if red) or 1 (if blue? no, delete from slow is free?)
    #   Let's say delete(v) removes v from fast memory if red (cost 0), or from slow memory if blue (cost 0).
    
    # The key constraint: |{v : status[v] == 'red'}| <= fast_memory_budget
    
    # This is a complex planning problem. Let me implement a greedy heuristic with local search.
    
    # Greedy strategy:
    # Process vertices in topological order. For each vertex to compute:
    # 1. Ensure all parents are in fast memory (load if needed)
    # 2. Compute the vertex (it goes into fast memory)
    # 3. If fast memory is full, evict some vertex (store it to free a slot)
    # 4. At the end, store all outputs to slow memory
    
    # Better strategy: Look ahead to minimize loads/stores.
    
    # Let me implement a simple greedy first, then try to improve.
    
    def greedy_schedule():
        """Simple greedy: process in topo order, load parents as needed, evict LRU."""
        status = {v: 'absent' for v in vertices}
        for v in input_set:
            status[v] = 'blue'
        
        ops = []
        fast_mem = set()  # vertices currently red
        
        # Track last access time for LRU eviction
        last_access = {}
        time_counter = 0
        
        def load(v):
            nonlocal time_counter
            assert status[v] == 'blue'
            status[v] = 'red'
            fast_mem.add(v)
            last_access[v] = time_counter
            time_counter += 1
            ops.append(['load', v])
        
        def store(v):
            nonlocal time_counter
            assert status[v] == 'red'
            status[v] = 'blue'
            fast_mem.discard(v)
            last_access.pop(v, None)
            time_counter += 1
            ops.append(['store', v])
        
        def compute(v):
            nonlocal time_counter
            # All parents must be red
            for p in parents.get(v, []):
                assert status[p] == 'red', f"Parent {p} not red when computing {v}"
            status[v] = 'red'
            fast_mem.add(v)
            last_access[v] = time_counter
            time_counter += 1
            ops.append(['compute', v])
        
        def delete(v):
            nonlocal time_counter
            if status[v] == 'red':
                status[v] = 'absent'
                fast_mem.discard(v)
                last_access.pop(v, None)
            elif status[v] == 'blue':
                status[v] = 'absent'
            time_counter += 1
            ops.append(['delete', v])
        
        def ensure_red(v):
            """Make sure v is in fast memory (red)."""
            if status[v] == 'red':
                last_access[v] = time_counter
                time_counter += 1
            elif status[v] == 'blue':
                load(v)
            else:
                raise ValueError(f"Vertex {v} is absent, cannot make red")
        
        # Process vertices in topological order
        for v in topo_order:
            if not parents.get(v, []):
                continue  # Input, already blue
            
            # Ensure all parents are red
            for p in parents[v]:
                ensure_red(p)
            
            # If fast memory is full, evict
            while len(fast_mem) >= fast_memory_budget:
                # Find least recently used vertex that is not an output we still need
                # Actually, we should evict something we won't need soon
                # Simple: evict LRU among non-essential vertices
                candidates = [x for x in fast_mem if x not in output_set or status[x] == 'blue']
                # Actually, all in fast_mem are red. Let's just evict LRU.
                if not candidates:
                    candidates = list(fast_mem)
                # Evict the one with oldest last_access
                lru_v = min(candidates, key=lambda x: last_access.get(x, 0))
                store(lru_v)
            
            # Now compute v
            compute(v)
        
        # At the end, store all outputs to slow memory
        for v in output_set:
            if status[v] == 'red':
                store(v)
            elif status[v] == 'absent':
                # Need to recompute? This shouldn't happen if we're careful
                pass
        
        return ops
    
    # Run greedy to get a baseline
    baseline_ops = greedy_schedule()
    baseline_cost = sum(1 for op, v in baseline_ops if op in ('load', 'store'))
    
    # Check time
    if time.time() > deadline:
        return baseline_ops
    
    # --- Local search improvement ---
    # Try to improve by reordering operations or using different eviction strategies
    
    def count_transfers(ops):
        return sum(1 for op, v in ops if op in ('load', 'store'))
    
    def validate_ops(ops, dag_data):
        """Check if ops are valid and produce correct final state."""
        vertices, parents, input_set, output_set = dag_data
        status = {v: 'absent' for v in vertices}
        for v in input_set:
            status[v] = 'blue'
        fast_mem = set()
        
        for op, v in ops:
            if op == 'load':
                if status[v] != 'blue':
                    return False
                if len(fast_mem) >= fast_memory_budget:
                    return False
                status[v] = 'red'
                fast_mem.add(v)
            elif op == 'store':
                if status[v] != 'red':
                    return False
                status[v] = 'blue'
                fast_mem.discard(v)
            elif op == 'compute':
                for p in parents.get(v, []):
                    if status[p] != 'red':
                        return False
                if len(fast_mem) >= fast_memory_budget:
                    return False
                status[v] = 'red'
                fast_mem.add(v)
            elif op == 'delete':
                if status[v] == 'red':
                    fast_mem.discard(v)
                status[v] = 'absent'
            else:
                return False
        
        # Check outputs are blue
        for v in output_set:
            if status[v] != 'blue':
                return False
        return True
    
    dag_data = (vertices, parents, input_set, output_set)
    
    # Try several heuristics and pick the best
    best_ops = baseline_ops
    best_cost = baseline_cost
    
    # Heuristic 2: Process in reverse topological order for outputs, forward for inputs
    # Actually, let's try a different eviction policy: evict vertices that are not needed soon
    
    def greedy_with_lookahead():
        """Greedy with look-ahead: evict vertices that are least likely to be needed."""
        status = {v: 'absent' for v in vertices}
        for v in input_set:
            status[v] = 'blue'
        
        ops = []
        fast_mem = set()
        time_counter = 0
        
        # Precompute: for each vertex, which future vertices depend on it
        dependents = defaultdict(set)
        for v in vertices:
            for c in children.get(v, []):
                dependents[v].add(c)
                # Transitively
                for cc in children.get(c, []):
                    dependents[v].add(cc)
        
        def load(v):
            nonlocal time_counter
            assert status[v] == 'blue'
            status[v] = 'red'
            fast_mem.add(v)
            time_counter += 1
            ops.append(['load', v])
        
        def store(v):
            nonlocal time_counter
            assert status[v] == 'red'
            status[v] = 'blue'
            fast_mem.discard(v)
            time_counter += 1
            ops.append(['store', v])
        
        def compute(v):
            nonlocal time_counter
            for p in parents.get(v, []):
                assert status[p] == 'red'
            status[v] = 'red'
            fast_mem.add(v)
            time_counter += 1
            ops.append(['compute', v])
        
        def ensure_red(v):
            if status[v] == 'red':
                pass
            elif status[v] == 'blue':
                load(v)
            else:
                raise ValueError(f"Vertex {v} absent")
        
        for v in topo_order:
            if not parents.get(v, []):
                continue
            
            for p in parents[v]:
                ensure_red(p)
            
            while len(fast_mem) >= fast_memory_budget:
                # Evict vertex with fewest future dependents
                candidates = list(fast_mem)
                if not candidates:
                    break
                # Score: number of future vertices that depend on this
                def eviction_score(x):
                    future_dep = len(dependents.get(x, set()) & set(topo_order[topo_order.index(v):]))
                    return future_dep
                lru_v = min(candidates, key=eviction_score)
                store(lru_v)
            
            compute(v)
        
        for v in output_set:
            if status[v] == 'red':
                store(v)
        
        return ops
    
    if time.time() < deadline:
        ops2 = greedy_with_lookahead()
        cost2 = count_transfers(ops2)
        if validate_ops(ops2, dag_data) and cost2 < best_cost:
            best_ops = ops2
            best_cost = cost2
    
    # Heuristic 3: Try to minimize by keeping frequently-used vertices in fast memory
    def greedy_frequency():
        """Keep vertices with high fan-out in fast memory."""
        status = {v: 'absent' for v in vertices}
        for v in input_set:
            status[v] = 'blue'
        
        ops = []
        fast_mem = set()
        time_counter = 0
        
        # Precompute fan-out (number of children)
        fanout = {v: len(children.get(v, [])) for v in vertices}
        
        def load(v):
            nonlocal time_counter
            status[v] = 'red'
            fast_mem.add(v)
            time_counter += 1
            ops.append(['load', v])
        
        def store(v):
            nonlocal time_counter
            status[v] = 'blue'
            fast_mem.discard(v)
            time_counter += 1
            ops.append(['store', v])
        
        def compute(v):
            nonlocal time_counter
            status[v] = 'red'
            fast_mem.add(v)
            time_counter += 1
            ops.append(['compute', v])
        
        def ensure_red(v):
            if status[v] != 'red':
                if status[v] == 'blue':
                    load(v)
                else:
                    raise ValueError(f"Vertex {v} absent")
        
        for v in topo_order:
            if not parents.get(v, []):
                continue
            
            for p in parents[v]:
                ensure_red(p)
            
            while len(fast_mem) >= fast_memory_budget:
                candidates = list(fast_mem)
                if not candidates:
                    break
                # Evict vertex with lowest fan-out
                lru_v = min(candidates, key=lambda x: fanout.get(x, 0))
                store(lru_v)
            
            compute(v)
        
        for v in output_set:
            if status[v] == 'red':
                store(v)
        
        return ops
    
    if time.time() < deadline:
        ops3 = greedy_frequency()
        cost3 = count_transfers(ops3)
        if validate_ops(ops3, dag_data) and cost3 < best_cost:
            best_ops = ops3
            best_cost = cost3
    
    # Try random restarts with different eviction orders
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Randomized greedy
        status = {v: 'absent' for v in vertices}
        for v in input_set:
            status[v] = 'blue'
        
        ops = []
        fast_mem = set()
        
        def load(v):
            status[v] = 'red'
            fast_mem.add(v)
            ops.append(['load', v])
        
        def store(v):
            status[v] = 'blue'
            fast_mem.discard(v)
            ops.append(['store', v])
        
        def compute(v):
            status[v] = 'red'
            fast_mem.add(v)
            ops.append(['compute', v])
        
        def ensure_red(v):
            if status[v] != 'red':
                if status[v] == 'blue':
                    load(v)
                else:
                    raise ValueError(f"Vertex {v} absent")
        
        success = True
        for v in topo_order:
            if not parents.get(v, []):
                continue
            
            for p in parents[v]:
                try:
                    ensure_red(p)
                except:
                    success = False
                    break
            
            if not success:
                break
            
            while len(fast_mem) >= fast_memory_budget:
                candidates = list(fast_mem)
                if not candidates:
                    break
                # Random eviction
                evict_v = random.choice(candidates)
                store(evict_v)
            
            compute(v)
        
        if success:
            for v in output_set:
                if status[v] == 'red':
                    store(v)
            
            cost = count_transfers(ops)
            if validate_ops(ops, dag_data) and cost < best_cost:
                best_ops = ops
                best_cost = cost
    
    return best_ops
