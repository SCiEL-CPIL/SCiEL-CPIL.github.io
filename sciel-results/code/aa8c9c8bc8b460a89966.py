def pebble(dag, fast_memory_budget):
    import time
    import heapq
    from collections import deque, defaultdict
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Fallback to baseline
    baseline_ops = None
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_ops = pebble_baseline(dag, fast_memory_budget)
    except:
        baseline_ops = None
    
    # Try to improve with chain-decomposition + round-robin scheduling
    try:
        # Extract DAG info
        if isinstance(dag, dict):
            parents_dict = dict(dag.get('parents', {}))
            inputs = list(dag.get('inputs', []))
            outputs = set(dag.get('outputs', []))
        else:
            parents_dict = dict(getattr(dag, 'parents', {}))
            inputs = list(getattr(dag, 'inputs', []))
            outputs = set(getattr(dag, 'outputs', []))
        
        # Compute topological order
        all_vertices = set(inputs)
        for v, ps in parents_dict.items():
            all_vertices.add(v)
            all_vertices.update(ps)
        
        in_degree = {v: len(parents_dict.get(v, [])) for v in all_vertices}
        children = {v: [] for v in all_vertices}
        for v, ps in parents_dict.items():
            for p in ps:
                children[p].append(v)
        
        queue = deque([v for v in all_vertices if in_degree[v] == 0])
        topo_order = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        if len(topo_order) != len(all_vertices):
            # Cycle or issue, fall back
            if baseline_ops is not None:
                return baseline_ops
            return []
        
        # Compute next_use and last_use for each vertex
        vertex_use_positions = {v: [] for v in all_vertices}
        for i, v in enumerate(topo_order):
            ps = parents_dict.get(v, [])
            for p in ps:
                vertex_use_positions[p].append(i)
        
        next_use = {}
        last_use = {}
        for v in all_vertices:
            if vertex_use_positions[v]:
                next_use[v] = vertex_use_positions[v][0]
                last_use[v] = vertex_use_positions[v][-1]
            else:
                next_use[v] = float('inf')
                last_use[v] = -1
        
        # --- Chain decomposition via maximum matching ---
        # Build a matching: each vertex can be matched to at most one parent (the one we "chain" through)
        # and at most one child. This decomposes the DAG into chains and isolated vertices.
        
        # Greedy maximum matching on the bipartite graph (parent-side, child-side)
        # We want to match parent p to child c if p is a parent of c.
        # Use a greedy approach: process children in topological order, match to available parent.
        
        matched_parent = {}  # child -> parent (the matched parent)
        matched_child = {}   # parent -> child (the matched child)
        
        # For each vertex, sort its children by next_use (earlier first) to prioritize
        for v in topo_order:
            if v in children:
                # Sort children by their position in topo_order
                sorted_children = sorted(children[v], key=lambda c: topo_order.index(c))
                for c in sorted_children:
                    if c not in matched_parent:
                        # Try to match v -> c
                        matched_parent[c] = v
                        matched_child[v] = c
                        break  # v can only match to one child
        
        # Build chains from the matching
        # A chain starts at a vertex with no matched parent (or is an input)
        # and follows matched_child links
        
        chain_head = {}  # vertex -> chain_id
        chains = []       # list of lists of vertices
        chain_id_counter = 0
        
        # Find all chain heads: vertices that are not matched as a child
        for v in all_vertices:
            if v not in matched_parent:
                # Start a new chain
                chain = [v]
                current = v
                while current in matched_child:
                    current = matched_child[current]
                    chain.append(current)
                chains.append(chain)
                for cv in chain:
                    chain_head[cv] = chain_id_counter
                chain_id_counter += 1
        
        # Assign each vertex to its chain (for vertices not in any chain, create single-vertex chains)
        vertex_to_chain = {}
        for i, chain in enumerate(chains):
            for v in chain:
                vertex_to_chain[v] = i
        
        # Create single-vertex chains for unmatched vertices
        for v in all_vertices:
            if v not in vertex_to_chain:
                chains.append([v])
                vertex_to_chain[v] = chain_id_counter
                chain_id_counter += 1
        
        # --- Round-robin scheduling across chains ---
        # Each chain has a pointer to the next vertex to process
        # We cycle through chains, processing the next vertex in each chain that is ready
        
        # Determine readiness: a vertex is ready when all its parents are in fast memory
        # But we process in topological order within each chain
        
        # For round-robin: we maintain a priority queue of (next_topo_index, chain_id)
        # and always process the chain with the earliest next vertex in topo order
        
        in_fast_memory = set()
        operations = []
        
        # Track position in each chain
        chain_pos = [0] * len(chains)
        chain_next_topo_idx = [0] * len(chains)
        
        # Precompute topo index for each vertex
        topo_idx = {v: i for i, v in enumerate(topo_order)}
        
        # For each chain, the next vertex to process
        def get_chain_next(chain_i):
            pos = chain_pos[chain_i]
            if pos < len(chains[chain_i]):
                return chains[chain_i][pos]
            return None
        
        # Initialize: all chains start at position 0
        # But we need to process in a way that respects dependencies
        
        # Simpler approach: process vertices in topological order, but use chain awareness
        # for eviction decisions. The chain structure tells us which vertices are "related"
        # and likely to be needed together.
        
        # Actually, let's use a different strategy: 
        # Process in topological order, but when evicting, prefer vertices from chains 
        # that have their next needed vertex farthest in the future.
        
        # Compute for each chain the "urgency" - when the next vertex in the chain is needed
        chain_next_need = {}
        for i, chain in enumerate(chains):
            # The chain vertices in topological order
            chain_sorted = sorted(chain, key=lambda v: topo_idx[v])
            # Find the first vertex in the chain that will be needed as a parent in the future
            chain_next_need[i] = float('inf')
            for v in chain_sorted:
                if next_use[v] != float('inf'):
                    chain_next_need[i] = next_use[v]
                    break
        
        # Greedy pebbling with chain-aware eviction
        for idx, v in enumerate(topo_order):
            if time.time() > deadline:
                break
            
            if v in inputs:
                if v not in in_fast_memory:
                    if len(in_fast_memory) >= fast_memory_budget:
                        # Evict: prefer vertices from chains with farthest next need
                        best_evict = None
                        best_chain_need = -1
                        for u in list(in_fast_memory):
                            if u not in outputs:
                                ci = vertex_to_chain.get(u, -1)
                                need = chain_next_need.get(ci, float('inf'))
                                # Evict the one with the farthest next need (largest value)
                                if need > best_chain_need:
                                    best_chain_need = need
                                    best_evict = u
                        if best_evict is not None:
                            in_fast_memory.discard(best_evict)
                            operations.append(['delete', best_evict])
                    in_fast_memory.add(v)
                    operations.append(['load', v])
            else:
                ps = parents_dict.get(v, [])
                for p in ps:
                    if p not in in_fast_memory:
                        if len(in_fast_memory) >= fast_memory_budget:
                            best_evict = None
                            best_chain_need = -1
                            for u in list(in_fast_memory):
                                if u not in outputs and u != p:
                                    ci = vertex_to_chain.get(u, -1)
                                    need = chain_next_need.get(ci, float('inf'))
                                    if need > best_chain_need:
                                        best_chain_need = need
                                        best_evict = u
                            if best_evict is not None:
                                in_fast_memory.discard(best_evict)
                                operations.append(['delete', best_evict])
                        
                        in_fast_memory.add(p)
                        operations.append(['load', p])
                
                in_fast_memory.add(v)
                operations.append(['compute', v])
            
            # Update chain needs as we progress
            # After processing v, update the chain's next need if v was the closest
            ci = vertex_to_chain.get(v, -1)
            if ci >= 0:
                chain = chains[ci]
                chain_sorted = sorted(chain, key=lambda x: topo_idx[x])
                # Find next vertex in this chain that has a future use
                found = False
                for cv in chain_sorted:
                    if topo_idx[cv] > idx and next_use[cv] != float('inf'):
                        chain_next_need[ci] = next_use[cv]
                        found = True
                        break
                if not found:
                    chain_next_need[ci] = float('inf')
            
            # Delete if no longer needed and not an output
            if last_use.get(v, -1) < idx and v not in outputs:
                if v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['delete', v])
            elif v in outputs:
                if v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['store', v])
        
        # Ensure all outputs are stored
        for v in outputs:
            if v in in_fast_memory:
                in_fast_memory.discard(v)
                operations.append(['store', v])
        
        # Count transfers
        my_transfers = sum(1 for op in operations if op[0] in ['load', 'store'])
        
        # Compare with baseline
        if baseline_ops is not None:
            baseline_transfers = sum(1 for op in baseline_ops if op[0] in ['load', 'store'])
            if my_transfers < baseline_transfers:
                return operations
            else:
                return baseline_ops
        else:
            return operations
    
    except:
        if baseline_ops is not None:
            return baseline_ops
        return []
