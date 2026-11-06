import random
import time
import sys

N = 18

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute has_edge_in for all masks
    # has_edge_in[mask] is True if the induced subgraph on vertices in mask has at least one edge.
    # We use a list of booleans of size 2^N.
    size = 1 << N
    has_edge_in = [False] * size
    
    # graph_mask is a list of N integers
    # We will compute has_edge_in after the graph is defined?
    # No, the graph changes during local search.
    # However, we can recompute `has_edge_in` and `is_tf` quickly for each candidate graph.
    # 2^18 = 262144 operations. In Python, a loop of 262k takes ~10-20ms.
    # This is acceptable for local search steps if we limit the number of steps or optimize.
    
    def compute_scores(graph_mask):
        """
        Computes the maximum size of a triangle-free induced subgraph.
        Uses dynamic programming over all subsets.
        """
        # 1. Compute has_edge_in for the current graph
        # has_edge_in[mask] = has_edge_in[mask ^ lsb] or (graph_mask[v] & (mask ^ lsb)) != 0
        # where v is the index of the lsb.
        
        # We can optimize by just computing is_tf directly?
        # is_tf[mask] = is_tf[mask ^ lsb] and not (edge_in[common_neighbors[v] & (mask ^ lsb)])
        
        # Let's precompute common_neighbors for this graph
        common_neighbors = [0] * N
        for u in range(N):
            for v in range(u + 1, N):
                if (graph_mask[u] >> v) & 1:
                    common_neighbors[u] |= (1 << v)
                    common_neighbors[v] |= (1 << u)
        
        # has_edge_in array
        # We need to know if a set of vertices contains an edge.
        # Let's compute `has_edge_in` for all masks.
        # This is O(2^N).
        # Then compute `is_tf` for all masks.
        
        has_edge_in[0] = False
        is_tf = [False] * size
        is_tf[0] = True
        
        best_size = 0
        
        # Iterate through all masks
        for mask in range(1, size):
            # Find least significant bit
            lsb = mask & (-mask)
            v = lsb.bit_length() - 1
            rest = mask ^ lsb
            
            # Check if rest has an edge
            # has_edge_in[mask] = has_edge_in[rest] or (graph_mask[v] & rest) != 0
            if not has_edge_in[rest]:
                if (graph_mask[v] & rest) != 0:
                    has_edge_in[mask] = True
                else:
                    has_edge_in[mask] = False
            else:
                has_edge_in[mask] = True
            
            # Check if mask is triangle-free
            # mask is TF if rest is TF and v does not form a triangle with any edge in rest.
            # v forms a triangle with rest if there is an edge (u,w) in rest such that u-w is edge, and u-v and w-v are edges.
            # This is equivalent to: the set of neighbors of v in rest contains an edge.
            # neighbors_of_v_in_rest = common_neighbors[v] & rest
            # If has_edge_in[neighbors_of_v_in_rest] is True, then triangle exists.
            
            if is_tf[rest]:
                neigh_v_rest = common_neighbors[v] & rest
                if has_edge_in[neigh_v_rest]:
                    is_tf[mask] = False
                else:
                    is_tf[mask] = True
                    # Update best size
                    # popcount(mask)
                    # We can compute popcount on the fly or precompute.
                    # Precomputing popcount for 2^N is fast.
                    # Or just use bin(mask).count('1') which is slow.
                    # Let's maintain a popcount array or use int.bit_count() (Python 3.8+)
                    if mask.bit_count() > best_size:
                        best_size = mask.bit_count()
            else:
                is_tf[mask] = False
        
        return best_size

    def is_k4_free(graph_mask):
        # Check if graph is K4-free
        # For each edge (u,v), check if common neighbors contain an edge.
        for u in range(N):
            for v in range(u + 1, N):
                if (graph_mask[u] >> v) & 1:
                    common = graph_mask[u] & graph_mask[v]
                    # Check if common contains an edge
                    # Iterate bits of common
                    c = common
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if (graph_mask[w] & common) & (common ^ lsb):
                            return False
        return True

    def random_k4_free_graph():
        graph_mask = [0] * N
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if random.random() < 0.5:
                    # Check if adding (u,v) creates a K4
                    common = graph_mask[u] & graph_mask[v]
                    c = common
                    has_edge = False
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if (graph_mask[w] & common) & (common ^ lsb):
                            has_edge = True
                            break
                    if not has_edge:
                        graph_mask[u] |= (1 << v)
                        graph_mask[v] |= (1 << u)
                        edges.append([u, v])
        return edges, graph_mask

    def local_search(edges, graph_mask):
        current_score = compute_scores(graph_mask)
        
        # To speed up local search, we can use a heuristic score or just accept the cost of compute_scores.
        # compute_scores is ~20ms. 
        # If we do 1000 iterations, that's 20 seconds.
        # We have ~140 seconds.
        # We can do ~500-700 full local search passes if each pass is small?
        # Or one long pass.
        
        improved = True
        iterations = 0
        while improved and iterations < 100:
            improved = False
            iterations += 1
            if time.time() > deadline - 10:
                break
                
            # Try adding edges
            for u in range(N):
                for v in range(u + 1, N):
                    if (graph_mask[u] >> v) & 1:
                        continue
                    # Check K4-free
                    common = graph_mask[u] & graph_mask[v]
                    c = common
                    has_edge = False
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if (graph_mask[w] & common) & (common ^ lsb):
                            has_edge = True
                            break
                    if has_edge:
                        continue
                    
                    # Add edge
                    graph_mask[u] |= (1 << v)
                    graph_mask[v] |= (1 << u)
                    edges.append([u, v])
                    
                    new_score = compute_scores(graph_mask)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        graph_mask[u] &= ~(1 << v)
                        graph_mask[v] &= ~(1 << u)
                        edges.pop()
                        
            # Try removing edges
            for i in range(len(edges) - 1, -1, -1):
                u, v = edges[i]
                # Remove edge
                graph_mask[u] &= ~(1 << v)
                graph_mask[v] &= ~(1 << u)
                edges.pop(i)
                
                new_score = compute_scores(graph_mask)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    graph_mask[u] |= (1 << v)
                    graph_mask[v] |= (1 << u)
                    edges.insert(i, [u, v])
                    
        return edges, current_score

    best_edges = None
    best_score = N + 1
    
    # Start with complete tripartite
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    tripartite_edges = []
    tripartite_mask = [0] * N
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    tripartite_edges.append([a, b])
                    tripartite_mask[a] |= (1 << b)
                    tripartite_mask[b] |= (1 << a)
    
    tripartite_score = compute_scores(tripartite_mask)
    if tripartite_score < best_score:
        best_score = tripartite_score
        best_edges = tripartite_edges[:]
    
    # Local search from tripartite
    ls_edges, ls_score = local_search(tripartite_edges[:], tripartite_mask[:])
    if ls_score < best_score:
        best_score = ls_score
        best_edges = ls_edges[:]
    
    # Random restarts
    while time.time() < deadline - 20:
        edges, mask = random_k4_free_graph()
        score = compute_scores(mask)
        if score < best_score:
            best_score = score
            best_edges = edges[:]
        
        # Local search
        ls_edges, ls_score = local_search(edges[:], mask[:])
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges[:]
            
    return best_edges if best_edges else []

# Ensure the function is called correctly
if __name__ == "__main__":
    result = construct()
    print(f"Found graph with {len(result)} edges")
