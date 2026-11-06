import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    n = 20
    # Bitmask representation: adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * n
    edge_set = set()

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        key = (u, v) if u < v else (v, u)
        edge_set.add(key)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        key = (u, v) if u < v else (v, u)
        edge_set.discard(key)

    def is_valid_add(u, v):
        """Check if adding edge (u,v) creates a K3,3 subgraph.
        
        A K3,3 exists iff there exist a1, a2 in N(u)\\{v} and b1, b2 in N(v)\\{u}
        such that all four edges (a1,b1),(a1,b2),(a2,b1),(a2,b2) exist.
        Equivalent: there exist a1 != a2 in N(u)\\{v} with |N(a1) ∩ N(a2) ∩ N(v)\\{u}| >= 2.
        """
        # N(u) excluding v
        nu_mask = adj[u] & ~(1 << v)
        # N(v) excluding u
        nv_mask = adj[v] & ~(1 << u)
        
        if nu_mask == 0 or nv_mask == 0:
            return True
        
        # Iterate over pairs in nu_mask
        # For efficiency, collect the bits
        nu_bits = []
        mask = nu_mask
        while mask:
            lsb = mask & (-mask)
            nu_bits.append(lsb.bit_length() - 1)
            mask ^= lsb
        
        len_nu = len(nu_bits)
        if len_nu < 2:
            return True
        
        # For each pair (a1, a2) in nu_bits, check if they share >= 2 neighbors in nv_mask
        for i in range(len_nu):
            a1 = nu_bits[i]
            common1 = adj[a1] & nv_mask
            if common1 == 0:
                continue
            # Check if common1 has at least 2 bits
            if (common1 & (common1 - 1)) == 0:
                continue  # only 1 bit
            # common1 has >= 2 bits, now check if any other a2 in nu also connects to at least 2 of these
            # Actually, we need: exists a2 != a1 in nu such that |N(a2) ∩ N(a1) ∩ nv_mask| >= 2
            # So for each a2 in nu_bits (a2 != a1), check if adj[a2] & common1 has >= 2 bits
            for j in range(len_nu):
                if j == i:
                    continue
                a2 = nu_bits[j]
                common2 = adj[a2] & common1
                if common2 != 0 and (common2 & (common2 - 1)) != 0:
                    # common2 has >= 2 bits
                    return False
        return True

    # Initialize with a random sparse graph
    random.seed(42)
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.15:
                add_edge(u, v)

    best_edges = len(edge_set)
    best_edge_set = set(edge_set)
    best_adj = list(adj)

    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995

    while time.time() < deadline and T > T_min:
        # Generate a random move: add or remove a random edge
        if random.random() < 0.6:
            # Try to add a random non-edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] >> v) & 1]
            if not non_edges:
                T *= cooling
                continue
            u, v = random.choice(non_edges)
            if is_valid_add(u, v):
                add_edge(u, v)
                delta = 1
                # Always accept valid additions
                if len(edge_set) > best_edges:
                    best_edges = len(edge_set)
                    best_edge_set = set(edge_set)
                    best_adj = list(adj)
            else:
                # Invalid addition, skip
                T *= cooling
                continue
        else:
            # Try to remove a random edge
            if not edge_set:
                T *= cooling
                continue
            u, v = random.choice(list(edge_set))
            remove_edge(u, v)
            delta = -1
            # Accept with probability exp(delta/T) = exp(-1/T)
            if random.random() < math.exp(delta / T):
                # Accept
                if len(edge_set) > best_edges:
                    best_edges = len(edge_set)
                    best_edge_set = set(edge_set)
                    best_adj = list(adj)
            else:
                # Reject, undo
                add_edge(u, v)

        T *= cooling

        # Periodically try to greedily add all possible edges
        if random.random() < 0.1:
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] >> v) & 1]
            random.shuffle(non_edges)
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                if is_valid_add(u, v):
                    add_edge(u, v)
            if len(edge_set) > best_edges:
                best_edges = len(edge_set)
                best_edge_set = set(edge_set)
                best_adj = list(adj)

    # Final greedy pass to add any remaining valid edges
    non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not (best_adj[u] >> v) & 1]
    random.shuffle(non_edges)
    for u, v in non_edges:
        if time.time() > deadline:
            break
        # Check validity against best_adj
        nu_mask = best_adj[u] & ~(1 << v)
        nv_mask = best_adj[v] & ~(1 << u)
        if nu_mask == 0 or nv_mask == 0:
            best_edge_set.add((u, v) if u < v else (v, u))
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
            continue
        nu_bits = []
        mask = nu_mask
        while mask:
            lsb = mask & (-mask)
            nu_bits.append(lsb.bit_length() - 1)
            mask ^= lsb
        valid = True
        for i in range(len(nu_bits)):
            a1 = nu_bits[i]
            common1 = best_adj[a1] & nv_mask
            if common1 == 0 or (common1 & (common1 - 1)) == 0:
                continue
            for j in range(len(nu_bits)):
                if j == i:
                    continue
                a2 = nu_bits[j]
                common2 = best_adj[a2] & common1
                if common2 != 0 and (common2 & (common2 - 1)) != 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            best_edge_set.add((u, v) if u < v else (v, u))
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)

    # Restore best state
    adj = best_adj
    edge_set = best_edge_set

    return [[u, v] for u, v in edge_set]
