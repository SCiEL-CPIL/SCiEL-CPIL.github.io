import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin for 150s limit

    def time_up():
        return time.time() > DEADLINE

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def max_clique_bk(rem, n, active_mask, seed_vertex):
        """
        Find max clique containing seed_vertex in the subgraph induced by active_mask,
        using 'rem' for edge existence.
        """
        if active_mask == 0:
            return []

        if not (active_mask & (1 << seed_vertex)):
            return []

        best_clique = [seed_vertex]
        best_size = 1

        def color_sort(P_mask):
            """Greedy coloring using bit operations for speed."""
            pm = P_mask
            verts_p = []
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                verts_p.append(idx)
                pm ^= lsb

            if not verts_p:
                return []

            # Use rem for degree calculation
            verts_p.sort(key=lambda v: (rem[v] & P_mask).bit_count(), reverse=True)

            v_color = {}
            result = []
            for v in verts_p:
                used = 0
                # Use rem for neighbor check
                neigh = rem[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    if u in v_color:
                        used |= (1 << v_color[u])
                    nm ^= lsb
                c = 1
                while (used >> c) & 1:
                    c += 1
                v_color[v] = c
                result.append((c, v))

            result.sort(key=lambda x: x[0])
            return result

        def bk(R, P, X):
            nonlocal best_clique, best_size

            if P == 0 and X == 0:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = list(R)
                return

            if time_up():
                return

            colored = color_sort(P)
            for color, v in reversed(colored):
                if len(R) + color <= best_size:
                    break

                R.append(v)
                # Use rem for intersection
                bk(R, P & rem[v], X & rem[v])
                R.pop()
                P ^= (1 << v)
                X |= (1 << v)

                if time_up():
                    return

        # P is neighbors of seed_vertex in the active subgraph, restricted by rem
        P = rem[seed_vertex] & active_mask
        X = 0
        R = [seed_vertex]
        
        bk(R, P, X)
        
        return best_clique

    def solve_instance(n, p, seed):
        adj_orig = generate_graph(n, p, seed)
        rem = adj_orig[:]
        cliques = []

        active_mask = 0
        for i in range(n):
            if rem[i]:
                active_mask |= (1 << i)

        while active_mask and not time_up():
            # Find the edge with highest clique potential
            best_edge = None
            best_potential = -1
            
            # Iterate through all remaining edges
            for u in range(n):
                if not (active_mask & (1 << u)):
                    continue
                mu = rem[u] & active_mask
                while mu:
                    lsb = mu & -mu
                    v = lsb.bit_length() - 1
                    if v > u:  # Only consider each edge once
                        # Compute potential: common neighbors of u and v in rem
                        # These are vertices w such that (u,w) and (v,w) are in rem
                        common = rem[u] & rem[v] & active_mask
                        # Remove u and v from common (they are not in common usually, but just in case)
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        potential = common.bit_count()
                        
                        if potential > best_potential:
                            best_potential = potential
                            best_edge = (u, v)
                    mu ^= lsb
            
            if best_edge is None:
                break
            
            u, v = best_edge
            
            # Find max clique containing both u and v
            # The clique must be a subset of {u, v} U (common neighbors in rem)
            
            common = rem[u] & rem[v] & active_mask
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            if common == 0:
                clique = [u, v]
            else:
                # Find vertex in common with highest degree within common (using rem)
                common_verts = []
                cm = common
                while cm:
                    lsb = cm & -cm
                    idx = lsb.bit_length() - 1
                    common_verts.append(idx)
                    cm ^= lsb
                
                best_seed = common_verts[0]
                max_deg = -1
                for w in common_verts:
                    deg = (rem[w] & common).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_seed = w
                
                # Find max clique containing best_seed in subgraph induced by common
                # Passing rem ensures we only use remaining edges
                sub_clique = max_clique_bk(rem, n, common, best_seed)
                
                if not sub_clique:
                    clique = [u, v]
                else:
                    clique = [u, v] + sub_clique
            
            # Ensure clique is valid (all pairs connected in rem)
            valid_clique = []
            for x in clique:
                is_valid = True
                for y in valid_clique:
                    if not (rem[x] & (1 << y)):
                        is_valid = False
                        break
                if is_valid:
                    valid_clique.append(x)
            
            if len(valid_clique) < 2:
                # Fallback to just the edge
                valid_clique = [u, v]
            
            cliques.append(valid_clique)
            
            # Remove edges of the clique from rem
            for i, x in enumerate(valid_clique):
                for j in range(i + 1, len(valid_clique)):
                    y = valid_clique[j]
                    rem[x] &= ~(1 << y)
                    rem[y] &= ~(1 << x)
            
            # Update active_mask
            active_mask = 0
            for i in range(n):
                if rem[i]:
                    active_mask |= (1 << i)

        # Fallback: cover remaining edges with pairs
        if active_mask:
            for u in range(n):
                nb = rem[u]
                while nb:
                    lsb = nb & -nb
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                        rem[u] &= ~(1 << v)
                        rem[v] &= ~(1 << u)
                    nb ^= lsb

        return cliques

    # Instance 1
    cliques1 = solve_instance(32, 0.6, 601)
    
    # Instance 2
    cliques2 = solve_instance(64, 0.55, 602)
    
    return [cliques1, cliques2]
