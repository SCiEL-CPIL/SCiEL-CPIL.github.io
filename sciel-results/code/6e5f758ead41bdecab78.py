import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def solve_instance(n, p, seed):
        edges, adj = generate_graph(n, p, seed)
        if not edges:
            return []

        uncovered = set(edges)
        partition = []
        
        # Precompute degrees for ordering
        degrees = [len(adj[i]) for i in range(n)]

        while uncovered:
            if time.time() > deadline - 5:
                # Fallback: if time is up, we must still cover all edges.
                # Adding them as pairs is the only safe way to ensure validity,
                # even if it's not optimal.
                for e in uncovered:
                    partition.append(list(e))
                break

            # Pick an edge to seed the clique
            # Heuristic: pick edge with highest sum of degrees
            # To save time, just pick the first one or a random one?
            # Picking the one with highest degree sum is better.
            # But iterating all uncovered edges to find max is O(E).
            # Let's just pick an arbitrary edge from the set (fast) or the first.
            # Actually, picking high-degree edges first helps create larger cliques.
            # We can approximate by maintaining a list of active vertices.
            
            # For simplicity and speed, pick the first edge in the set.
            # Sets don't have order, but iteration is fast.
            seed_edge = next(iter(uncovered))
            u, v = seed_edge
            
            # Find maximal clique containing u and v
            clique = {u, v}
            
            # Candidates must be neighbors of both u and v
            # And must be connected to all current clique members (initially just u,v)
            candidates = adj[u] & adj[v]
            
            # Remove u and v from candidates
            candidates.discard(u)
            candidates.discard(v)
            
            # Greedily add vertices to clique
            # Sort candidates by degree (descending) to find larger cliques faster
            # degrees are static from initial graph, which is a good proxy
            sorted_candidates = sorted(candidates, key=lambda x: degrees[x], reverse=True)
            
            for w in sorted_candidates:
                # Check if w is connected to all in clique
                is_connected = True
                for c in clique:
                    if w not in adj[c]:
                        is_connected = False
                        break
                if is_connected:
                    clique.add(w)
            
            # Remove all edges within the clique from uncovered
            clique_list = list(clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    edge = (min(a, b), max(a, b))
                    if edge in uncovered:
                        uncovered.remove(edge)
            
            partition.append(sorted(clique_list))

        return partition

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)

    return [res1, res2]
