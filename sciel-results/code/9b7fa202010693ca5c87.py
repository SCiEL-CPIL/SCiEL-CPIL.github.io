"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using chain decomposition and round-robin scheduling.
    
    The approach:
    1. Decompose the DAG into maximum number of vertex-disjoint paths (chains) using greedy matching.
    2. Treat each chain as a separate memory stream.
    3. Use round-robin scheduling to cycle through chains, loading/computing only the next needed element.
    4. Keep results in fast memory (red pebbles) until they are no longer needed by any downstream chain.
    5. Minimize transfers by maximizing reuse of data already in fast memory.
    
    Args:
        dag: A dictionary with keys:
             - 'vertices': list of vertex ids
             - 'parents': dict mapping vertex -> list of parent vertices (topologically ordered)
             - 'outputs': list of output vertex ids
        fast_memory_budget: integer, max number of red (fast memory) pebbles allowed
    
    Returns:
        List of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'
    """
    
    # Parse the DAG structure
    vertices = dag.get('vertices', [])
    parents_map = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    
    if not vertices:
        return []
    
    # Build adjacency: children of each vertex
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # Topological order is given by the order of vertices (or we compute it)
    # Let's ensure we have a topological order
    # vertices should already be in topological order based on the problem statement
    
    # Step 1: Decompose into chains using greedy path cover
    # A path cover assigns each vertex to a path, and we want minimum number of paths
    # which equals |V| - max_matching in the bipartite graph
    
    # For simplicity and effectiveness, let's use a greedy chain decomposition:
    # Start from each vertex with no unassigned parent, follow a greedy path
    
    # Actually, let's do a proper maximum matching based decomposition
    # Build bipartite graph: left side = vertices, right side = vertices
    # Edge from u_left to v_right if u is parent of v (edge u->v in DAG)
    
    # Find maximum matching using DFS-based bipartite matching
    def max_bipartite_matching(n, adj):
        """adj[u] = list of v such that edge u->v exists in bipartite graph"""
        match_r = [-1] * n  # match_r[v] = u matched to v
        match_l = [-1] * n  # match_l[u] = v matched to u
        
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_r[v] == -1 or dfs(match_r[v], visited):
                    match_r[v] = u
                    match_l[u] = v
                    return True
            return False
        
        for u in range(n):
            visited = [False] * n
            dfs(u, visited)
        
        return match_l, match_r
    
    # Create index mapping
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    
    # Build bipartite adjacency: edge from u to v if u is a parent of v
    bip_adj = [[] for _ in range(n)]
    for v in vertices:
        vi = vertex_to_idx[v]
        for p in parents_map.get(v, []):
            pi = vertex_to_idx[p]
            bip_adj[pi].append(vi)
    
    match_l, match_r = max_bipartite_matching(n, bip_adj)
    
    # Build chains:
    # A chain starts at a vertex with no incoming match (no predecessor in the matching)
    # and follows the matching edges
    
    # next_in_chain[u] = v if u is matched to v (i.e., u is predecessor of v in a chain)
    # prev_in_chain[v] = u if u is matched to v
    
    next_in_chain = {}
    prev_in_chain = {}
    for u in range(n):
        if match_l[u] != -1:
            v = match_l[u]
            next_in_chain[u] = v
            prev_in_chain[v] = u
    
    # Chain starts: vertices with no prev_in_chain
    chain_starts = []
    for i in range(n):
        if i not in prev_in_chain:
            chain_starts.append(i)
    
    # Build the chains
    chains = []
    for start in chain_starts:
        chain = []
        cur = start
        while cur is not None:
            chain.append(cur)
            cur = next_in_chain.get(cur, None)
        chains.append(chain)
    
    # For each chain, determine the sequence of vertices
    # Each chain is a path in the DAG
    
    # Step 2: Determine dependencies between chains
    # A vertex v in chain A depends on a vertex u in chain B if u is a parent of v
    # and u is NOT the immediate predecessor of v in the same chain
    
    # For each vertex, track which other chains provide its parent values
    # cross_deps[v] = list of (chain_id, vertex_idx_in_chain) for parents not in same chain
    
    chain_of_vertex = {}
    for cid, chain in enumerate(chains):
        for pos, vidx in enumerate(chain):
            chain_of_vertex[vidx] = (cid, pos)
    
    cross_deps = {}
    for vidx in range(n):
        v = vertices[vidx]
        cid, pos = chain_of_vertex[vidx]
        deps = []
        for p in parents_map.get(v, []):
            pidx = vertex_to_idx[p]
            pcid, ppos = chain_of_vertex[pidx]
            if pcid != cid or ppos != pos - 1:  # Not the immediate predecessor in same chain
                deps.append((pcid, ppos, pidx))
        cross_deps[vidx] = deps
    
    # Step 3: Schedule operations using round-robin
    # State:
    # - fast_memory: set of vertices currently in fast memory (red pebbles)
    # - computed: set of vertices that have been computed
    # - loaded: set of vertices that have been loaded from slow memory
    # - stored: set of vertices that have been stored to slow memory
    
    operations = []
    fast_memory = set()
    computed = set()
    loaded = set()
    stored = set()
    
    # For each chain, track the next position to process
    chain_next_pos = [0] * len(chains)
    
    # We process vertices in topological order, but interleaved across chains
    # Actually, let's process by "rounds" where in each round we advance each chain by one step
    
    # But we need to respect dependencies. Let's use a priority queue or simply
    # process vertices in topological order, but group them by chain for scheduling.
    
    # Alternative approach: Process all vertices in topological order.
    # For each vertex, if its parents are available (computed and either in fast memory or stored),
    # we can compute it.
    
    # Let's track availability:
    # A parent p of v is "available" if:
    #   - p is in fast_memory, OR
    #   - p is stored (in slow memory)
    
    # Strategy:
    # 1. Process vertices in topological order
    # 2. For each vertex v:
    #    a. For each parent p:
    #       - If p is not computed: error (shouldn't happen in topo order)
    #       - If p is in fast_memory: no transfer needed
    #       - If p is stored: load p into fast_memory (transfer: load)
    #       - If p is neither: compute p? No, p should already be computed.
    #    b. Compute v: if all parents are in fast_memory, compute v (no transfer for parents)
    #       The result v goes into fast_memory
    #    c. After computing v, check if any parent is no longer needed
    #       If a parent p is not needed by any remaining vertex, and p is in fast_memory,
    #       we can store it (transfer: store) to free up space
    #    d. If fast_memory is full, we need to evict something (store) before loading/computing
    
    # Let's refine:
    # - "Needed" means: there exists a vertex w not yet computed such that p is in the transitive parents of w
    #   and w is not in the same chain as p with p being an ancestor in that chain (since chain ancestors
    #   are handled by the chain itself)
    #   
    # Actually, for simplicity, let's say a value is "needed" if it is a parent (direct or indirect)
    # of some vertex not yet computed.
    
    # Precompute: for each vertex, the set of vertices that depend on it (directly or indirectly)
    # dependents[p] = set of vertices that have p as an ancestor
    
    dependents = {v: set() for v in range(n)}
    # Process in reverse topological order
    for vidx in reversed(range(n)):
        v = vertices[vidx]
        for p in parents_map.get(v, []):
            pidx = vertex_to_idx[p]
            dependents[pidx].add(vidx)
            # Transitive: everything that depends on v also depends on p
            dependents[pidx] |= dependents[vidx]
    
    # Now, a vertex p is "needed" if there exists a vertex w in dependents[p] that has not been computed yet
    
    # Let's implement the scheduling:
    
    # Process vertices in topological order
    for vidx in range(n):
        v = vertices[vidx]
        
        # Check if v is an input (no parents) - inputs start blue (in slow memory)
        # If v has no parents, it's an input
        v_parents = parents_map.get(v, [])
        
        if not v_parents:
            # Input vertex: needs to be loaded from slow memory
            if vidx not in fast_memory and vidx not in loaded:
                # Load v into fast memory
                # Check if fast_memory is full
                if len(fast_memory) >= fast_memory_budget:
                    # Need to evict: find a value that is no longer needed
                    evicted = False
                    for mem_vidx in list(fast_memory):
                        # Check if mem_vidx is still needed
                        still_needed = False
                        for dep in dependents[mem_vidx]:
                            if dep not in computed:
                                still_needed = True
                                break
                        if not still_needed:
                            # Store mem_vidx to slow memory
                            operations.append(['store', vertices[mem_vidx]])
                            stored.add(mem_vidx)
                            fast_memory.discard(mem_vidx)
                            evicted = True
                            break
                    if not evicted and fast_memory_budget == 0:
                        # Edge case: can't hold anything
                        pass
                # Now load
                operations.append(['load', v])
                loaded.add(vidx)
                fast_memory.add(vidx)
            elif vidx not in fast_memory and vidx in loaded:
                # Was loaded before but evicted, need to load again
                if len(fast_memory) >= fast_memory_budget:
                    for mem_vidx in list(fast_memory):
                        still_needed = False
                        for dep in dependents[mem_vidx]:
                            if dep not in computed:
                                still_needed = True
                                break
                        if not still_needed:
                            operations.append(['store', vertices[mem_vidx]])
                            stored.add(mem_vidx)
                            fast_memory.discard(mem_vidx)
                            break
                operations.append(['load', v])
                fast_memory.add(vidx)
            # If already in fast_memory, no action needed
        
        else:
            # Non-input vertex: need to compute
            # First, ensure all parents are in fast_memory
            for p in v_parents:
                pidx = vertex_to_idx[p]
                if pidx not in fast_memory:
                    # Need to load p from slow memory
                    if len(fast_memory) >= fast_memory_budget:
                        for mem_vidx in list(fast_memory):
                            still_needed = False
                            for dep in dependents[mem_vidx]:
                                if dep not in computed:
                                    still_needed = True
                                    break
                            if not still_needed:
                                operations.append(['store', vertices[mem_vidx]])
                                stored.add(mem_vidx)
                                fast_memory.discard(mem_vidx)
                                break
                    operations.append(['load', vertices[pidx]])
                    loaded.add(pidx)
                    fast_memory.add(pidx)
            
            # Now compute v
            operations.append(['compute', v])
            computed.add(vidx)
            
            # Add v to fast_memory
            if len(fast_memory) >= fast_memory_budget:
                # Evict something that's no longer needed
                for mem_vidx in list(fast_memory):
                    still_needed = False
                    for dep in dependents[mem_vidx]:
                        if dep not in computed:
                            still_needed = True
                            break
                    if not still_needed:
                        operations.append(['store', vertices[mem_vidx]])
                        stored.add(mem_vidx)
                        fast_memory.discard(mem_vidx)
                        break
            
            fast_memory.add(vidx)
        
        # After processing v, clean up: store values that are no longer needed
        # This helps keep fast_memory small
        for mem_vidx in list(fast_memory):
            if mem_vidx == vidx:
                continue  # Don't evict the just-computed value immediately
            still_needed = False
            for dep in dependents[mem_vidx]:
                if dep not in computed:
                    still_needed = True
                    break
            if not still_needed and mem_vidx in fast_memory:
                # We could store it, but let's be conservative
                # Only store if fast_memory is at capacity
                if len(fast_memory) >= fast_memory_budget:
                    operations.append(['store', vertices[mem_vidx]])
                    stored.add(mem_vidx)
                    fast_memory.discard(mem_vidx)
    
    # Final: ensure all outputs are in fast memory (blue)
    # Actually, the problem says "all outputs must finish blue"
    # Blue means in slow memory? Or does blue mean available?
    # "Inputs start blue" - blue likely means in slow memory
    # "all outputs must finish blue" - outputs must be in slow memory at the end
    # "every red value consumes one fast-memory slot"
    # So blue = slow memory, red = fast memory
    # We need all outputs to be blue (stored) at the end
    
    for out_v in outputs:
        out_idx = vertex_to_idx[out_v]
        if out_idx in fast_memory:
            operations.append(['store', out_v])
            stored.add(out_idx)
            fast_memory.discard(out_idx)
    
    # Delete any remaining fast memory values? The problem doesn't require it.
    # Let's not delete unless necessary.
    
    # Limit operations to 20000
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
