"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    n = 16
    adj = [0] * n
    
    def has_edge(u, v):
        return (adj[u] >> v) & 1
        
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_neighbors(u, exclude=None):
        mask = adj[u]
        if exclude is not None:
            mask &= ~(1 << exclude)
        return [i for i in range(n) if (mask >> i) & 1]

    def find_one_q3(u, v):
        """
        Returns a list of 12 edges forming a Q3 containing (u,v), or None.
        Assumes (u,v) is a potential edge (either present or not, but we check the other 11).
        """
        neigh_u = get_neighbors(u, v)
        neigh_v = get_neighbors(v, u)
        
        for i in range(len(neigh_u)):
            a = neigh_u[i]
            for j in range(i + 1, len(neigh_u)):
                b = neigh_u[j]
                
                common_ab = adj[a] & adj[b]
                
                for k in range(len(neigh_v)):
                    c = neigh_v[k]
                    for l in range(k + 1, len(neigh_v)):
                        d = neigh_v[l]
                        
                        # Pattern 1
                        if has_edge(a, c) and has_edge(b, d):
                            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            cand_f = common_ab & ~forbidden
                            if cand_f:
                                mask = cand_f
                                while mask:
                                    lsb = mask & -mask
                                    f = lsb.bit_length() - 1
                                    mask -= lsb
                                    forbidden_e = forbidden | (1 << f)
                                    cand_e = adj[c] & adj[d] & adj[f] & ~forbidden_e
                                    if cand_e:
                                        # Found one
                                        lsb_e = cand_e & -cand_e
                                        e = lsb_e.bit_length() - 1
                                        return [[u,v], [u,a], [u,b], [v,c], [v,d], [a,c], [b,d], [a,f], [b,f], [c,f], [d,f], [c,e], [d,e], [f,e]]
                        # Pattern 2
                        if has_edge(a, d) and has_edge(b, c):
                            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            cand_f = common_ab & ~forbidden
                            if cand_f:
                                mask = cand_f
                                while mask:
                                    lsb = mask & -mask
                                    f = lsb.bit_length() - 1
                                    mask -= lsb
                                    forbidden_e = forbidden | (1 << f)
                                    cand_e = adj[c] & adj[d] & adj[f] & ~forbidden_e
                                    if cand_e:
                                        lsb_e = cand_e & -cand_e
                                        e = lsb_e.bit_length() - 1
                                        return [[u,v], [u,a], [u,b], [v,c], [v,d], [a,d], [b,c], [a,f], [b,f], [c,f], [d,f], [c,e], [d,e], [f,e]]
        return None

    def count_q3s_using_edge(u, v):
        neigh_u = get_neighbors(u, v)
        neigh_v = get_neighbors(v, u)
        count = 0
        for i in range(len(neigh_u)):
            a = neigh_u[i]
            for j in range(i + 1, len(neigh_u)):
                b = neigh_u[j]
                common_ab = adj[a] & adj[b]
                for k in range(len(neigh_v)):
                    c = neigh_v[k]
                    for l in range(k + 1, len(neigh_v)):
                        d = neigh_v[l]
                        if has_edge(a, c) and has_edge(b, d):
                            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            cand_f = common_ab & ~forbidden
                            if cand_f:
                                mask = cand_f
                                while mask:
                                    lsb = mask & -mask
                                    f = lsb.bit_length() - 1
                                    mask -= lsb
                                    forbidden_e = forbidden | (1 << f)
                                    cand_e = adj[c] & adj[d] & adj[f] & ~forbidden_e
                                    if cand_e:
                                        count += cand_e.bit_count()
                        if has_edge(a, d) and has_edge(b, c):
                            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            cand_f = common_ab & ~forbidden
                            if cand_f:
                                mask = cand_f
                                while mask:
                                    lsb = mask & -mask
                                    f = lsb.bit_length() - 1
                                    mask -= lsb
                                    forbidden_e = forbidden | (1 << f)
                                    cand_e = adj[c] & adj[d] & adj[f] & ~forbidden_e
                                    if cand_e:
                                        count += cand_e.bit_count()
        return count

    def total_violations():
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                if has_edge(u, v):
                    total += count_q3s_using_edge(u, v)
        return total // 12 # Each Q3 counted 12 times

    def edge_count():
        return sum(has_edge(u, v) for u in range(n) for v in range(u + 1, n))

    best_adj = [0] * n
    best_edges = 0
    
    # Multi-start SA
    for seed in [42, 123, 999, 555, 777]:
        random.seed(seed)
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.6: # Start denser
                    add_edge(u, v)
        
        # Repair phase
        max_repair_steps = 1000
        for _ in range(max_repair_steps):
            viol = total_violations()
            if viol == 0:
                break
            # Find a Q3 and remove an edge
            # Iterate edges to find one in a Q3
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if has_edge(u, v):
                        q3 = find_one_q3(u, v)
                        if q3:
                            # Remove a random edge from the Q3
                            e = random.choice(q3)
                            remove_edge(e[0], e[1])
                            found = True
                            break
                if found:
                    break
            if not found:
                # Should not happen if viol > 0
                break
        
        # SA phase
        T = 5.0
        T_min = 0.01
        alpha = 0.9995
        steps = 50000
        
        current_violations = total_violations()
        current_edges = edge_count()
        
        for step in range(steps):
            T *= alpha
            if T < T_min: T = T_min
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            if u > v: u, v = v, u
            
            if has_edge(u, v):
                delta = count_q3s_using_edge(u, v)
                # Removing reduces violations by delta
                # New viol = current_violations - delta
                # We want to minimize violations, then maximize edges.
                # If delta > 0, it's good.
                # If delta == 0, it's neutral for viol, bad for edges.
                accept = False
                if current_violations > 0:
                    if delta > 0:
                        accept = True
                    else:
                        # Maybe accept with small prob to escape?
                        # But removing an edge that doesn't fix a Q3 is bad.
                        accept = False
                else:
                    # current_violations == 0
                    # Removing an edge (delta must be 0) is bad.
                    accept = False
                
                if accept:
                    remove_edge(u, v)
                    current_violations -= delta
                    current_edges -= 1
            else:
                delta = count_q3s_using_edge(u, v)
                # Adding increases violations by delta
                if current_violations == 0:
                    if delta == 0:
                        accept = True
                    else:
                        # Allow with prob exp(-delta/T)
                        if random.random() < np.exp(-delta / T):
                            accept = True
                        else:
                            accept = False
                else:
                    # current_violations > 0
                    # Adding is bad.
                    # Only accept if it helps? It doesn't.
                    # Maybe accept to escape local min?
                    if random.random() < np.exp(-delta / T):
                        accept = True
                    else:
                        accept = False

                if accept:
                    add_edge(u, v)
                    current_violations += delta
                    current_edges += 1
            
            # Update best
            if current_violations == 0:
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            # Also track best with low violations?
            # No, we need 0 violations.

    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges

# EVOLVE-BLOCK-END
