"""Clique partition of dense random graphs - minimize number of cliques."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    _start = time.time()
    _DEADLINE = 145.0  # safety margin before 150s kill

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_clique_partition_bitset(edges, n, start_time, deadline):
        if not edges:
            return []

        cliques = []
        
        # Initialize bitset adjacency for remaining graph
        rem_adj = [0] * n
        for u, v in edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)

        def get_active_vertices():
            return [i for i in range(n) if rem_adj[i] != 0]

        while True:
            if time.time() > deadline - 1.0:
                break

            active_verts = get_active_vertices()
            if not active_verts:
                break

            # Sort active vertices by degree (popcount) descending
            active_verts.sort(key=lambda v: rem_adj[v].bit_count(), reverse=True)
            
            best_clique = []
            best_size = 0
            
            # Try top K candidates as starting vertices
            K = min(10, len(active_verts))
            
            for start in active_verts[:K]:
                # Build clique starting from 'start'
                candidates = [start]
                avail = rem_adj[start]
                
                while avail:
                    # Find vertex u in avail that has the most neighbors in avail
                    best_u = -1
                    max_local_deg = -1
                    a = avail
                    while a:
                        lsb = a & (-a)
                        u = lsb.bit_length() - 1
                        local_deg = (rem_adj[u] & avail).bit_count()
                        if local_deg > max_local_deg:
                            max_local_deg = local_deg
                            best_u = u
                        a ^= lsb
                    
                    if best_u == -1:
                        break
                        
                    candidates.append(best_u)
                    avail &= rem_adj[best_u]
                
                if len(candidates) > best_size:
                    best_size = len(candidates)
                    best_clique = candidates

            if not best_clique:
                # Fallback: take any edge
                v = active_verts[0]
                u_bit = rem_adj[v] & (-rem_adj[v])
                u = u_bit.bit_length() - 1
                best_clique = [v, u]
                best_size = 2

            # Remove edges in the clique from rem_adj
            for i in range(len(best_clique)):
                u = best_clique[i]
                for j in range(i + 1, len(best_clique)):
                    v = best_clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
            
            cliques.append(best_clique)

        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = generate_edges(n, p, seed)
        
        # Allocate time budget
        # 70s per instance, 5s buffer
        instance_deadline = _start + 70.0
        
        cliques = greedy_clique_partition_bitset(edges, n, _start, instance_deadline)
        
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
