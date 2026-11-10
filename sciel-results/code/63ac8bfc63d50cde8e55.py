"""Rewritten program for clique partition of split graphs."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # N[idx] = bitmask of C-vertices adjacent to I-vertex (idx + k)
        N = [0] * (n - k)
        for u, v in edges:
            if u < k and v >= k:
                N[v - k] |= (1 << u)
            elif v < k and u >= k:
                N[u - k] |= (1 << v)

        cliques = []
        
        # covered[u] = bitmask of vertices v such that edge (u, v) in C is covered
        covered = [0] * k

        # Phase 1: Add clique {i} U N(i) for each i in I
        for idx in range(n - k):
            if time.time() > deadline:
                break
            mask = N[idx]
            if mask == 0:
                continue
            
            i = idx + k
            # The clique is {i} U {v | v in N(i)}
            # We need to add this clique to the list
            # And mark internal edges as covered
            
            # Extract vertices in N(i)
            verts = []
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            clique = [i] + verts
            cliques.append(clique)
            
            # Mark edges within verts as covered
            for a_idx in range(len(verts)):
                a = verts[a_idx]
                for b_idx in range(a_idx + 1, len(verts)):
                    b = verts[b_idx]
                    covered[a] |= (1 << b)
                    covered[b] |= (1 << a)

        # Phase 2: Cover remaining C-edges
        # rem[u] = bitmask of v such that edge (u, v) is NOT covered
        rem = [0] * k
        for u in range(k):
            # All possible neighbors in C
            all_c = ((1 << k) - 1) ^ (1 << u)
            # Remove covered
            rem[u] = all_c & ~covered[u]
        
        active = set()
        for u in range(k):
            if rem[u]:
                active.add(u)

        while active:
            if time.time() > deadline:
                break
            
            # Find Max Clique in the subgraph induced by active vertices, 
            # where edges are defined by rem
            # Greedy Max Clique
            S = []
            candidates = 0
            for v in active:
                candidates |= (1 << v)
            
            # Sort candidates by degree in the candidate subgraph to help greedy?
            # Or just iterate.
            # To ensure we get a good clique, we can use a simple greedy:
            # Pick vertex with max degree, add it, intersect candidates with its neighbors.
            
            while candidates:
                best_v = -1
                best_deg = -1
                cm = candidates
                while cm:
                    lsb = cm & (-cm)
                    v = lsb.bit_length() - 1
                    # Degree of v in the current candidate set
                    # rem[v] contains all uncovered neighbors.
                    # We only care about neighbors in 'candidates'
                    deg = bin(rem[v] & candidates).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    cm ^= lsb
                
                if best_v < 0:
                    break
                    
                S.append(best_v)
                # Update candidates: must be neighbors of best_v in rem
                candidates &= rem[best_v]
                candidates &= ~(1 << best_v)
                
                if time.time() > deadline:
                    break

            if not S:
                # Fallback: pick an edge
                # Find any edge in active
                edge_found = False
                for u in active:
                    m = rem[u]
                    while m:
                        lsb = m & (-m)
                        v = lsb.bit_length() - 1
                        if v in active:
                            S = [u, v]
                            edge_found = True
                            break
                        m ^= lsb
                    if edge_found:
                        break
                if not S:
                    break

            # Add clique S
            cliques.append(sorted(S))
            
            # Remove edges covered by S from rem
            for a_idx in range(len(S)):
                a = S[a_idx]
                for b_idx in range(a_idx + 1, len(S)):
                    b = S[b_idx]
                    rem[a] &= ~(1 << b)
                    rem[b] &= ~(1 << a)
            
            # Update active
            active = set(v for v in active if rem[v])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
