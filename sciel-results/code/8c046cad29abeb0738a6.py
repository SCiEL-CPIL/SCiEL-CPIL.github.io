import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    def decompose_graph(n, edges, time_limit):
        if not edges:
            return []

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # 1. Identify odd degree vertices
        odd_vertices = [u for u in range(n) if len(adj[u]) % 2 != 0]
        
        # 2. Pair them up
        # If odd count is 0, we have an Eulerian graph
        # If odd count > 0, we add dummy edges to make it Eulerian
        dummy_edges = []
        if len(odd_vertices) > 0:
            # Pair arbitrarily
            for i in range(0, len(odd_vertices), 2):
                u = odd_vertices[i]
                v = odd_vertices[i+1]
                dummy_edges.append((u, v))
                # Add to adj for Eulerian circuit finding
                adj[u].add(v)
                adj[v].add(u)
        
        # 3. Find Eulerian Circuit
        # Hierholzer's algorithm
        # We need to be careful: adj is modified. We should use a copy or track used edges.
        # Let's use a standard Hierholzer implementation that removes edges as we go.
        
        # Re-initialize adj for Hierholzer to avoid messing up original logic if we need to retry?
        # We are inside a function, so adj is local.
        
        # Stack for Hierholzer
        stack = []
        circuit = []
        
        # Start from any node with degree > 0
        start_node = None
        for u in range(n):
            if adj[u]:
                start_node = u
                break
                
        if start_node is None:
            return [] # No edges? Shouldn't happen if edges not empty.

        stack.append(start_node)
        
        while stack:
            u = stack[-1]
            if adj[u]:
                # Pick a neighbor
                v = next(iter(adj[u]))
                # Remove edge (u, v)
                adj[u].remove(v)
                adj[v].remove(u)
                stack.append(v)
            else:
                circuit.append(stack.pop())
                
        # circuit is in reverse order
        circuit.reverse()
        
        # 4. Process Circuit to extract paths
        # The circuit is a list of vertices: v0, v1, ..., vk, v0 (conceptually, but Hierholzer gives v0...vk)
        # Actually, Hierholzer gives the sequence of vertices in the circuit.
        # The edges are (circuit[i], circuit[i+1]).
        
        # We need to identify where the dummy edges are in this circuit.
        # A dummy edge is an edge (u, v) in dummy_edges.
        
        # The circuit represents a closed loop if we consider (circuit[-1], circuit[0]) as an edge?
        # No, Hierholzer's output `circuit` is the sequence of vertices.
        # The edges traversed are (circuit[i], circuit[i+1]) for i in 0..len(circuit)-2.
        # And the final edge is (circuit[-1], circuit[0]) to close the loop?
        # Actually, the standard output is the path of vertices.
        # Let's verify:
        # Graph 1-2-3-1.
        # Start 1.
        # Stack: [1]. Adj[1]={2,3}. Pick 2.
        # Stack: [1, 2]. Adj[2]={1,3}. Remove 1-2. Adj[2]={3}. Pick 3.
        # Stack: [1, 2, 3]. Adj[3]={1,2}. Remove 2-3. Adj[3]={1}. Pick 1.
        # Stack: [1, 2, 3, 1]. Adj[1]={3}. Remove 3-1. Adj[1]={}.
        # Pop 1 -> Circuit: [1]
        # Top 3. Adj[3]={}. Pop 3 -> Circuit: [1, 3]
        # Top 2. Adj[2]={}. Pop 2 -> Circuit: [1, 3, 2]
        # Top 1. Adj[1]={}. Pop 1 -> Circuit: [1, 3, 2, 1]
        
        # So circuit is [1, 3, 2, 1].
        # Edges: (1,3), (3,2), (2,1).
        # This covers all edges.
        
        # Now, if we have dummy edges, we need to cut the circuit at those edges.
        # The circuit is a sequence of vertices.
        # If we have no dummy edges, the circuit is a single cycle.
        # We need to convert this cycle into paths.
        # As discussed, a single cycle can be decomposed into 2 paths (or 1 if cycles allowed, but they aren't).
        # Wait, if we have 0 dummy edges, we have 1 cycle.
        # We can just take the circuit, and if it's a cycle, break it.
        # However, if we have dummy edges, the "circuit" includes the dummy edges.
        # We need to remove the dummy edges from the sequence.
        
        # Let's identify the indices of dummy edges in the circuit sequence.
        # An edge (u, v) is in the circuit if u=circuit[i] and v=circuit[i+1] OR v=circuit[i] and u=circuit[i+1].
        # Note: The circuit is closed, so the last edge is (circuit[-1], circuit[0]).
        
        dummy_indices = set()
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i+1]
            if (u, v) in dummy_edges or (v, u) in dummy_edges:
                dummy_indices.add(i)
        
        # Check the closing edge
        u, v = circuit[-1], circuit[0]
        if (u, v) in dummy_edges or (v, u) in dummy_edges:
            dummy_indices.add(len(circuit) - 1) # Mark the wrap-around edge
        
        # Now, construct the paths.
        # The paths are the segments of the circuit between the dummy edges.
        # If there are no dummy edges:
        if not dummy_indices:
            # It's a single cycle.
            # We must break it into 2 paths?
            # Or just 1 path if we treat the cycle as a path?
            # "Paths must be simple". A cycle is not simple.
            # So we break it.
            # Path 1: circuit[0] to circuit[-1] (excluding the closing edge)
            # Path 2: The closing edge?
            # Actually, if we have a cycle $v_0 \dots v_k v_0$.
            # Path 1: $v_0 \dots v_k$.
            # Path 2: $v_k \dots v_0$? No, that's the same edges.
            # We need to partition the edges.
            # Edges: $(v_0, v_1), \dots, (v_{k-1}, v_k), (v_k, v_0)$.
            # Path 1: $v_0, v_1, \dots, v_k$. (Covers $k$ edges).
            # Path 2: $v_k, v_0$. (Covers 1 edge).
            # Total 2 paths.
            
            if len(circuit) <= 2:
                # Single edge?
                # Circuit [u, v, u]? No, circuit [u, v] if 1 edge?
                # If 1 edge, odd vertices 2, so dummy edge 1. So this case (0 dummy) implies >= 2 edges.
                return []
            
            path1 = circuit[:-1] # v0 ... vk
            path2 = [circuit[-1], circuit[0]] # vk, v0
            return [path1, path2]
            
        else:
            # We have cuts.
            # The cuts are at indices in dummy_indices.
            # The segments are between these cuts.
            
            # Get sorted list of cut indices
            cuts = sorted(list(dummy_indices))
            
            paths = []
            
            # Helper to get path from circuit between two cuts
            # If cut i is at index i, the edge (circuit[i], circuit[i+1]) is a dummy edge.
            # The path starts at circuit[i+1] and ends at circuit[j] (where j is next cut index, handling wrap).
            
            for k_idx, cut in enumerate(cuts):
                start_idx = (cut + 1) % len(circuit)
                
                # Find next cut
                if k_idx + 1 < len(cuts):
                    end_cut = cuts[k_idx + 1]
                    end_idx = end_cut
                else:
                    end_cut = cuts[0]
                    end_idx = end_cut
                    
                # The path vertices are circuit[start_idx], circuit[start_idx+1], ..., circuit[end_idx]
                # Note: If start_idx > end_idx, we wrap around.
                
                path = []
                current = start_idx
                while True:
                    path.append(circuit[current])
                    if current == end_idx:
                        break
                    current = (current + 1) % len(circuit)
                    
                if len(path) > 1:
                    paths.append(path)
                    
            return paths

    results = []
    for idx, spec in enumerate(specs):
        if time.time() > DEADLINE - 10:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        n, p, seed = spec
        edges = instance(n, p, seed)
        
        remaining_time = DEADLINE - time.time() - 5
        time_for_this = remaining_time / (len(specs) - idx)
        
        paths = decompose_graph(n, edges, time_limit=time_for_this)
        results.append(paths)

    return results
