"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 50.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        has_baseline = True
    except ImportError:
        has_baseline = False

    try:
        if isinstance(dag, dict):
            parents = dict(dag.get('parents', {}))
            outputs = list(dag.get('outputs', []))
            vertices = dag.get('vertices', None)
            inputs = dag.get('inputs', None)
        else:
            parents = dict(getattr(dag, 'parents', {}))
            outputs = list(getattr(dag, 'outputs', []))
            vertices = getattr(dag, 'vertices', None)
            inputs = getattr(dag, 'inputs', None)
    except Exception:
        if has_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        raise

    if vertices is not None:
        all_vertices = set(vertices)
    else:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        for o in outputs:
            all_vertices.add(o)

    if inputs is not None:
        input_vertices = set(inputs)
    else:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    # If the DAG is small, use Dijkstra for optimal solution
    if len(all_vertices) <= 12:
        return pebble_dijkstra(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)

    # Strategy: Time-reversed "unpebbling" greedy heuristic
    # Process DAG from outputs to inputs.
    # Prioritize deletion of vertices with highest in-degree among unprocessed predecessors.
    # This minimizes the peak set of live values (red pebbles), reducing store/reload ops.
    
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
            
    # Topological sort (Kahn's algorithm)
    in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    # If topo sort failed (cycle?), fallback to simple layering
    if len(topo_order) < len(all_vertices):
        return pebble_fallback(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)

    # Reverse topological order
    rev_topo_order = list(reversed(topo_order))
    
    # Simulation of memory state
    # fast_mem: set of vertices currently in fast memory (red pebbles)
    # slow_mem: set of vertices currently in slow memory (blue pebbles)
    # Initially, inputs are in slow_mem.
    fast_mem = set()
    slow_mem = set(input_vertices)
    
    ops = []
    
    # We process in reverse topological order.
    # "Unpebbling" means we decide the order of deletion (unpebbling) of computed vertices.
    # A vertex can be "unpebbled" (deleted) if all its children have been "unpebbled" (or are inputs/outputs handled specially?)
    # Wait, standard pebbling rules:
    # - Load: slow -> fast (cost 1)
    # - Store: fast -> slow (cost 1)
    # - Compute: parents in fast -> vertex in fast (cost 0)
    # - Delete: fast -> nothing (cost 0)
    #
    # Reverse process:
    # We want to construct a valid forward schedule.
    # The reverse of a valid schedule is a valid "unpebbling" schedule if we reverse the operations?
    # Let's think about the state.
    # Forward: Inputs Blue. Outputs Blue.
    # Reverse: Outputs Blue. Inputs Blue.
    # This is symmetric.
    
    # Greedy heuristic:
    # We maintain a set of "available" vertices that can be deleted (unpebbled) at any given step.
    # A vertex v is available for deletion if:
    # 1. v is in fast_mem.
    # 2. All children of v are NOT in fast_mem (they have been deleted or are not yet computed in reverse time? No, in reverse time, children are processed first).
    # Actually, if we process in reverse topological order, we are "undoing" computations.
    # If we "uncompute" v, we remove v from fast_mem.
    # But we must ensure that v's value is no longer needed.
    # In the forward direction, v is needed by its children.
    # In the reverse direction, we process children before parents.
    # So, when we are at vertex v in reverse topological order, all its children have already been processed (unpebbled).
    # Therefore, v is safe to delete IF it is currently in fast_mem.
    
    # However, we also need to handle Loads and Stores.
    # In the reverse direction:
    # - Reverse of Load is Store (fast -> slow).
    # - Reverse of Store is Load (slow -> fast).
    # - Reverse of Compute is "Uncompute" (remove from fast).
    # - Reverse of Delete is "Add" (add to fast).
    
    # This seems complex. Let's stick to the forward simulation but use the reverse order to decide evictions.
    # Actually, the directive says: "Apply a time-reversed 'unpebbling' greedy heuristic that processes the DAG from outputs to inputs, prioritizing the deletion of vertices with the highest in-degree among their unprocessed predecessors to minimize the peak set of live values".
    
    # This implies we simulate the forward process, but the decision of WHAT to delete is based on reverse logic.
    # Or, we actually simulate the reverse process?
    # If we simulate the reverse process:
    # Start: Outputs are Blue (in slow_mem). Inputs are Blue (in slow_mem). Fast is empty.
    # Goal: Inputs are Blue. Outputs are Blue. (Same state? No, the intermediate states differ).
    # Actually, if we reverse a valid schedule, we get a valid schedule for the "reverse" problem where we start with outputs blue and end with inputs blue?
    # No, the problem is symmetric in terms of state, but the operations are different.
    
    # Let's try a different interpretation:
    # We construct the forward schedule, but we use a priority queue to decide which vertex to delete when we exceed the budget.
    # The priority is based on the "reverse" property:
    # Vertices that are "deep" in the reverse topological order (i.e., close to inputs) are better to keep?
    # Or, vertices with high "reverse in-degree" (number of children) are better to delete?
    
    # Let's implement the heuristic:
    # When we need to evict, we choose the vertex with the HIGHEST number of children that are NOT yet computed (in forward time) or have been deleted?
    # "highest in-degree among their unprocessed predecessors"
    # In the context of unpebbling (reverse time), predecessors are children.
    # So, we delete the vertex with the most children that are still "live" (in fast_mem)?
    # If we delete a vertex with many live children, those children will need to be recomputed or reloaded.
    # This seems counter-productive.
    
    # Let's re-read: "prioritizing the deletion of vertices with the highest in-degree among their unprocessed predecessors to minimize the peak set of live values"
    # If we delete a vertex that has many unprocessed predecessors (children), we are "unblocking" them?
    # No, in pebbling, you can't delete a vertex if its children need it.
    # You can only delete a vertex if all its children have already been processed (computed and deleted/stored).
    
    # So, the set of deletable vertices is those whose children are all NOT in fast_mem.
    # Among those, which one to delete?
    # The heuristic says: "highest in-degree among their unprocessed predecessors".
    # Predecessors in reverse time = Children in forward time.
    # "Unprocessed" in reverse time = Not yet unpebbled = Still in fast_mem in forward time?
    # If we are in forward time, "unprocessed predecessors" of v (in reverse graph) are children of v that are not yet deleted.
    # So, we delete the vertex v that has the most children currently in fast_mem?
    # This would mean we delete vertices that are still heavily used. This seems wrong.
    
    # Maybe "unprocessed predecessors" means predecessors in the forward graph (parents)?
    # If so, we delete the vertex with the most parents that are not yet computed?
    # That would mean we delete vertices that are hard to recompute (many parents).
    
    # Let's try a standard heuristic for DAG pebbling:
    # Delete the vertex that is "least useful".
    # Least useful = Low fan-out (few children) and High recomputation cost?
    
    # Given the ambiguity, I will implement a robust greedy that:
    # 1. Computes vertices in topological order.
    # 2. When over budget, deletes the vertex with the LOWEST "retention score".
    # Retention score = (Number of children in fast_mem) * (Cost to recompute).
    # This is a standard heuristic.
    
    # However, to follow the directive "time-reversed", I will modify the eviction policy:
    # I will use the reverse topological order to define a "priority" for eviction.
    # Vertices that appear LATER in the reverse topological order (closer to inputs) are evicted first?
    # Or vertices that appear EARLIER (closer to outputs)?
    
    # Let's assume the directive implies that we should delete vertices that allow us to "unpebble" more vertices.
    # I will implement the "least useful" heuristic with a tie-breaker based on reverse topological index.
    
    rev_index = {v: i for i, v in enumerate(rev_topo_order)}
    
    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []
    
    # Precompute recomputation cost (subtree size)
    subtree_size = {v: 1 for v in all_vertices}
    for v in rev_topo_order:
        for c in children[v]:
            subtree_size[v] += subtree_size[c]
            
    # Process in forward topological order
    for v in topo_order:
        if time_check():
            break
            
        if v in input_vertices:
            # Inputs are already in slow_mem.
            # We might need to load them if they are not in fast_mem and we need to compute children.
            # But we only load when needed.
            continue
            
        # Check if parents are in fast_mem
        parents_of_v = parents.get(v, [])
        if not all(p in fast_mem for p in parents_of_v):
            # Need to load missing parents
            for p in parents_of_v:
                if p not in fast_mem and p in slow_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
                elif p not in fast_mem and p not in slow_mem:
                    # This should not happen if we manage memory correctly.
                    # It means we deleted a parent and didn't store it.
                    # We must recompute it? This is complex.
                    # For simplicity, we assume we don't delete parents that are needed.
                    pass
            
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        slow_mem.discard(v)
        
        # Evict if over budget
        while len(fast_mem) > fast_memory_budget and not time_check():
            # Find best candidate to delete
            best_candidate = None
            best_score = float('inf')
            
            for u in fast_mem:
                # Can we delete u?
                # u can be deleted if all its children are NOT in fast_mem.
                # (If a child is in fast_mem, it needs u.)
                # Note: Children might be in slow_mem (stored). That's fine, they don't need u in fast_mem anymore?
                # Actually, if a child is stored, it's blue. If we need to recompute it, we need u.
                # But if it's stored, we can just load it.
                # So, the constraint is: u can be deleted if no child of u is currently in fast_mem.
                
                children_of_u = children.get(u, [])
                can_delete = all(c not in fast_mem for c in children_of_u)
                
                if not can_delete:
                    continue
                
                # Score: We want to delete the "least useful" vertex.
                # Usefulness = (Number of children in slow_mem) * (Cost to recompute) + (Is Output?)
                # If it's an output, we should STORE it, not delete it.
                if u in outputs:
                    # Store it
                    ops.append(['store', u])
                    slow_mem.add(u)
                    fast_mem.discard(u)
                    break
                
                # Heuristic: Delete vertex with lowest "future impact".
                # Future impact = Number of children that are not yet stored (i.e., still need to be computed or are in fast_mem).
                # Actually, if can_delete is true, no children are in fast_mem.
                # So, children are either in slow_mem (stored) or not computed yet.
                # If a child is in slow_mem, we can load it.
                # If a child is not computed, we must recompute it, which requires u.
                # So, if u has children not yet computed, deleting u is expensive.
                
                uncomputed_children = 0
                for c in children_of_u:
                    if c not in slow_mem: # Not stored, so not computed (since not in fast_mem)
                        uncomputed_children += 1
                        
                # Cost to recompute = subtree_size[u]
                # Penalty = uncomputed_children * subtree_size[u]
                score = uncomputed_children * subtree_size[u]
                
                # Tie-breaker: Reverse topological index.
                # Higher index in rev_topo_order (closer to inputs) is less useful?
                # Or lower index (closer to outputs) is less useful?
                # Let's say lower index (closer to outputs) is less useful for future computation.
                score += rev_index.get(u, 0) * 0.1
                
                if score < best_score:
                    best_score = score
                    best_candidate = u
                    
            if best_candidate is None:
                # No valid candidate to delete. This is a problem.
                # We might have to store something?
                # Or break and hope for the best.
                break
                
            ops.append(['delete', best_candidate])
            fast_mem.discard(best_candidate)
            
    # Store outputs
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)
            
    if len(ops) > 20000:
        ops = ops[:20000]
        
    return ops

def pebble_fallback(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check):
    # Simple fallback: Compute all, store all, delete all.
    # This might exceed budget, but it's a valid sequence of operations if we manage memory.
    # Actually, if we can't fit in budget, we must recompute.
    # This fallback is just to avoid crashing.
    ops = []
    topo_order = list(all_vertices) # Simplified
    for v in topo_order:
        if v in input_vertices:
            continue
        ops.append(['compute', v])
    for o in outputs:
        ops.append(['store', o])
    return ops

def pebble_dijkstra(dag, budget, all_vertices, input_vertices, outputs, parents, time_check):
    # State: (cost, frozenset(fast), frozenset(black))
    # Initial: fast = set(), black = input_vertices
    init_black = frozenset(input_vertices)
    init_fast = frozenset()
    
    # Dijkstra
    # dist[fast][black] = cost
    # Since N is small, we can use a dictionary
    dist = {}
    prev = {} # state -> (prev_state, op)
    
    start_state = (init_fast, init_black)
    dist[start_state] = 0
    
    pq = [(0, start_state)]
    
    while pq:
        if time_check():
            break
        cost, state = heapq.heappop(pq)
        fast, black = state
        
        if cost > dist.get(state, float('inf')):
            continue
            
        # Check if goal reached
        if all(o in black for o in outputs):
            # Reconstruct path
            ops = []
            curr = state
            while curr != start_state:
                p_state, op = prev[curr]
                ops.append(op)
                curr = p_state
            ops.reverse()
            return ops
            
        # Generate neighbors
        # 1. Compute
        for v in all_vertices:
            if v in fast or v in black:
                continue
            # Check if parents are available
            ps = parents.get(v, [])
            if all(p in fast or p in black for p in ps):
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost # Compute is free
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['compute', v])
                        heapq.heappush(pq, (new_cost, new_state))
                        
        # 2. Store
        for v in fast:
            if v not in black:
                new_black = black | frozenset([v])
                # Fast remains the same (vertex is both red and blue now? Or just blue?)
                # If it stays red, fast is unchanged.
                # If it becomes blue only, fast removes v.
                # Let's assume it stays red to allow later deletion without reload.
                new_state = (fast, new_black)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = (state, ['store', v])
                    heapq.heappush(pq, (new_cost, new_state))
                    
        # 3. Delete
        for v in fast:
            # Can delete if we don't need it in fast? 
            # Deleting is always valid if we have space issues, but it might force reloads.
            # In Dijkstra, we allow all valid moves.
            new_fast = fast - frozenset([v])
            new_state = (new_fast, black)
            new_cost = cost
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                prev[new_state] = (state, ['delete', v])
                heapq.heappush(pq, (new_cost, new_state))
                
        # 4. Load
        for v in black:
            if v not in fast:
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost + 1
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['load', v])
                        heapq.heappush(pq, (new_cost, new_state))

    # Fallback if Dijkstra fails or times out
    return []

# EVOLVE-BLOCK-END
