import random
import time
import itertools
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def pc(x):
        return x.bit_count()

    def check_regular_mask(adj, mask, n):
        """Check if induced subgraph on mask is regular."""
        degs = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = pc(adj[v] & mask)
            degs.append(deg)
            if len(degs) > 1 and degs[0] != deg:
                return False
        return len(set(degs)) == 1

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def full_check(adj, n):
        """Check all subsets of size >= 6."""
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if check_regular_mask(adj, mask, n):
                    return False
        return True

    def build_polarity_graph(n):
        """
        Build a polarity graph over GF(q) where q is chosen so that
        the number of points is close to n. We use the standard construction:
        Points are 1D vectors in GF(q)^2, i.e., (a,1) for a in GF(q) plus (0,1).
        Actually, let's use a simpler approach: use a known small prime power.
        
        For n=21, we can try GF(4) or GF(5) constructions.
        Let's use a different approach: take a random algebraic structure.
        
        Actually, let's try a circulant-based approach with specific connection sets
        that are known to avoid regular induced subgraphs.
        """
        # Try multiple deterministic constructions
        constructions = []
        
        # Construction 1: Based on quadratic residues
        for p in [19, 23, 29, 31]:
            if p >= n:
                adj = [0] * n
                # Use quadratic residues mod p
                for i in range(n):
                    for j in range(i+1, n):
                        d = (j - i) % p
                        # Check if d is a quadratic residue mod p
                        if d == 0:
                            continue
                        # Compute d^((p-1)/2) mod p
                        if pow(d, (p-1)//2, p) == 1:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                constructions.append(adj)
        
        # Construction 2: Use a fixed-seed PRNG with specific seed
        for seed in [42, 123, 456, 789, 101, 202, 303]:
            rng = random.Random(seed)
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            constructions.append(adj)
        
        # Construction 3: Use a specific known good pattern
        # Try a "balanced" construction where each vertex has degree ~n/2
        for density in [0.4, 0.45, 0.5, 0.55, 0.6]:
            rng = random.Random(999)
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < density:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            constructions.append(adj)
            
        return constructions

    def check_with_numpy(adj, n):
        """Use NumPy for faster subset checking."""
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            mask = adj[i]
            for j in range(n):
                if mask & (1 << j):
                    A[i, j] = 1
        
        # Check all subsets of size >= 6
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                # Get degrees in induced subgraph
                degs = []
                for v in combo:
                    deg = 0
                    for w in combo:
                        if v != w and A[v, w] == 1:
                            deg += 1
                    degs.append(deg)
                    if len(degs) > 1 and degs[0] != deg:
                        break
                if len(set(degs)) == 1:
                    return False
        return True

    # First, try deterministic constructions
    for n in [21, 20]:
        if time.time() > deadline - 20:
            break
        
        constructions = build_polarity_graph(n)
        
        for adj in constructions:
            if time.time() > deadline - 10:
                break
            if full_check(adj, n):
                return {'n': n, 'edges': get_edges(adj, n)}
        
        # If deterministic didn't work, try random search with incremental construction
        best_adj = None
        best_n_found = 0

        # Multiple random restarts with incremental construction
        for restart in range(20):
            if time.time() > deadline - 5:
                break

            adj = [0] * n
            current_n = 0

            # Build vertex by vertex
            for v in range(n):
                if time.time() > deadline - 5:
                    break

                # Try random edge patterns for this vertex
                found_valid = False
                for attempt in range(30):
                    # Randomly decide edges from v to existing vertices
                    candidate_adj = adj[:]
                    for u in range(v):
                        if random.random() < 0.5:
                            candidate_adj[v] |= (1 << u)
                            candidate_adj[u] |= (1 << v)

                    # Check if adding vertex v with these edges creates any regular subgraph of size >= 6
                    valid = True
                    existing = list(range(v))
                    for k in range(5, min(v, n)):  # k existing vertices, total k+1
                        if k + 1 < 6:
                            continue
                        for combo in itertools.combinations(existing, k):
                            mask = 0
                            for w in combo:
                                mask |= (1 << w)
                            mask |= (1 << v)
                            if check_regular_mask(candidate_adj, mask, n):
                                valid = False
                                break
                        if not valid:
                            break

                    if valid:
                        adj = candidate_adj
                        current_n = v + 1
                        found_valid = True
                        break

                if not found_valid:
                    break

            if current_n > best_n_found:
                best_n_found = current_n
                best_adj = adj[:]

            if current_n == n:
                if full_check(adj, n):
                    return {'n': n, 'edges': get_edges(adj, n)}

        if best_adj and best_n_found >= 6:
            if full_check(best_adj, best_n_found):
                return {'n': best_n_found, 'edges': get_edges(best_adj, best_n_found)}

    # Fallback: try smaller n with random + full check
    for n in [18, 15, 12, 10]:
        if time.time() > deadline - 5:
            break
        for restart in range(30):
            if time.time() > deadline - 5:
                break
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if full_check(adj, n):
                return {'n': n, 'edges': get_edges(adj, n)}

    return {'n': 5, 'edges': []}
