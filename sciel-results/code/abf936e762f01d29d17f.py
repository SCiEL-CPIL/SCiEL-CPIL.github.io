import time
import random

def construct():
    N = 15
    deadline = time.time() + 50

    def is_valid(edges):
        """Check if no 3-edge path has endpoints of equal degree."""
        deg = [0] * N
        adj = [set() for _ in range(N)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)

        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def get_degree_seq(edges):
        deg = [0] * N
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg

    # 1. Construct the seed bipartite graph
    # A = 0..6, B = 7..14
    # Target degrees A: 1, 2, 3, 4, 5, 6, 7
    # Target degrees B: 0, 1, 2, 3, 4, 5, 6, 7
    
    # We will use a greedy construction to find the specific edges
    # Sort A by target degree descending
    a_indices = list(range(7))
    a_targets = [7, 6, 5, 4, 3, 2, 1] # Corresponding to indices 0..6
    
    # Sort B by target degree descending
    b_indices = list(range(7, 14))
    b_targets = [7, 6, 5, 4, 3, 2, 1, 0] # Corresponding to indices 7..14
    
    # Map actual indices to sorted positions
    # a_sorted[i] is the vertex in A with target a_targets[i]
    # We need to map vertex -> target
    # Let's just assign targets to vertices 0..6 and 7..14
    # Vertex 0 gets target 7, Vertex 1 gets 6, ..., Vertex 6 gets 1
    # Vertex 7 gets target 7, Vertex 8 gets 6, ..., Vertex 14 gets 0
    
    edges = []
    # Greedy connection
    # For each a in A (sorted by target desc), connect to the b's with highest remaining capacity
    b_remaining = [7, 6, 5, 4, 3, 2, 1, 0]
    b_vertices = [7, 8, 9, 10, 11, 12, 13, 14]
    
    a_vertices = [0, 1, 2, 3, 4, 5, 6]
    a_targets = [7, 6, 5, 4, 3, 2, 1]
    
    for i, a_v in enumerate(a_vertices):
        target = a_targets[i]
        # Select 'target' b's with highest remaining capacity
        # Create list of (remaining, index)
        candidates = [(b_remaining[j], j) for j in range(8)]
        candidates.sort(reverse=True)
        
        for k in range(target):
            rem, idx = candidates[k]
            if rem == 0:
                break # Should not happen if sequence is valid
            b_v = b_vertices[idx]
            edges.append([a_v, b_v])
            b_remaining[idx] -= 1
    
    # Verify seed is valid
    if not is_valid(edges):
        # Fallback to star if construction failed
        edges = [[0, v] for v in range(1, N)]
        return edges

    best_edges = edges[:]
    best_len = len(edges)

    # 2. Local Search: Try to add edges
    all_possible_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edge_set = set(tuple(sorted(e)) for e in edges)
    missing_edges = [e for e in all_possible_edges if e not in edge_set]
    
    # Shuffle missing edges for diversity
    random.shuffle(missing_edges)

    # Try to add edges greedily
    while missing_edges and time.time() < deadline:
        progress = False
        # Re-shuffle if we get stuck? Or just iterate
        for e in missing_edges:
            if time.time() > deadline:
                break
            if e in edge_set:
                continue
            new_edges = best_edges + [list(e)]
            if is_valid(new_edges):
                best_edges = new_edges
                edge_set.add(e)
                missing_edges.remove(e)
                best_len = len(best_edges)
                progress = True
                break # Restart loop with new state
        
        if not progress:
            # Try removing an edge and re-adding to escape local optimum?
            # Or just break if no progress in a full pass
            # Let's try a simple perturbation: remove a random edge, then try to add
            if best_len > 0 and random.random() < 0.5:
                # Remove a random edge
                rem_idx = random.randint(0, len(best_edges) - 1)
                rem_edge = best_edges[rem_idx]
                edge_set.remove(rem_edge)
                best_edges.remove(rem_edge)
                missing_edges.append(rem_edge)
                # Try to add it back or others
                # The loop above will handle adding
            else:
                break

    return best_edges
