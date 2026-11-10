import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin before 150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count()

    def max_clique_in_induced(n, adj, mask, rng):
        """
        Find max clique in G[mask].
        adj is the global adjacency list (remaining edges).
        mask is a bitmask of vertices to consider.
        """
        best_clique = []

        def expand(R, P):
            nonlocal best_clique
            if not P:
                if len(R) > len(best_clique):
                    best_clique = R[:]
                return

            pc = popcount(P)
            if len(R) + pc <= len(best_clique):
                return

            # Pivot selection: find vertex in P with max degree in P
            max_deg = -1
            candidates = []
            P_copy = P
            while P_copy:
                v_bit = P_copy & (-P_copy)
                v = v_bit.bit_length() - 1
                P_copy ^= v_bit
                deg = popcount(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    candidates = [v]
                elif deg == max_deg:
                    candidates.append(v)

            if not candidates:
                return

            u = rng.choice(candidates)
            Q = P & ~adj[u]

            while Q:
                # Pick v from Q with max degree in P
                max_v_deg = -1
                v_candidates = []
                Q_copy = Q
                while Q_copy:
                    v_bit = Q_copy & (-Q_copy)
                    i = v_bit.bit_length() - 1
                    Q_copy ^= v_bit
                    deg = popcount(P & adj[i])
                    if deg > max_v_deg:
                        max_v_deg = deg
                        v_candidates = [i]
                    elif deg == max_v_deg:
                        v_candidates.append(i)

                if not v_candidates:
                    break

                v = rng.choice(v_candidates)

                new_R = R + [v]
                new_P = P & adj[v]
                expand(new_R, new_P)

                P &= ~(1 << v)
                Q = Q & P

        expand([], mask)
        return best_clique

    def find_maximal_cliques(n, adj, rng, deadline):
        """
        Find a set of maximal cliques in the graph.
        Returns list of cliques (each as a list of vertices).
        """
        cliques = []
        remaining_mask = (1 << n) - 1

        while remaining_mask and time.time() < deadline:
            # Pick a random vertex from remaining
            verts = []
            mask_copy = remaining_mask
            while mask_copy:
                v_bit = mask_copy & (-mask_copy)
                v = v_bit.bit_length() - 1
                mask_copy ^= v_bit
                verts.append(v)
            
            if not verts:
                break

            # Try multiple random starting vertices
            best_clique = None
            for start in verts[:min(5, len(verts))]:
                # Build candidate set: start + neighbors of start
                cand_mask = (1 << start) | (adj[start] & remaining_mask)
                clique = max_clique_in_induced(n, adj, cand_mask, rng)
                if best_clique is None or len(clique) > len(best_clique):
                    best_clique = clique
            
            if best_clique and len(best_clique) >= 2:
                # Check if this clique is maximal (not contained in a larger one)
                # For now, just add it and mark its edges as used
                cliques.append(best_clique)
                # Remove vertices that are "covered" by this clique from remaining
                # Actually, we should remove the edges, but for maximal clique extraction,
                # we'll track which edges are covered
                for v in best_clique:
                    remaining_mask &= ~(1 << v)
            
            if time.time() > deadline:
                break

        return cliques

    def edge_set_of_clique(clique):
        """Return set of edges (as frozensets) in a clique."""
        edges = set()
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                edges.add(frozenset([clique[i], clique[j]]))
        return edges

    def lagrangian_clique_partition(n, adj, edges, rng, deadline):
        """
        Lagrangian relaxation approach:
        1. Find a pool of candidate cliques
        2. Iteratively select non-overlapping cliques using weight adjustment
        3. Cover remaining edges with 2-cliques
        """
        # Step 1: Find candidate cliques
        candidate_cliques = []
        
        # Strategy A: Find cliques from random edges
        edge_list = list(edges)
        rng.shuffle(edge_list)
        
        for u, v in edge_list[:min(len(edge_list), 200)]:
            if time.time() > deadline:
                break
            common = adj[u] & adj[v]
            if common:
                clique_core = max_clique_in_induced(n, adj, common, rng)
                if clique_core:
                    clique = clique_core + [u, v]
                    # Remove duplicates
                    clique_set = frozenset(clique)
                    if clique_set not in [frozenset(c) for c in candidate_cliques]:
                        candidate_cliques.append(clique)
        
        # Strategy B: Find cliques from high-degree vertices
        degs = [popcount(adj[i]) for i in range(n)]
        sorted_verts = sorted(range(n), key=lambda x: -degs[x])
        
        for v in sorted_verts[:min(20, n)]:
            if time.time() > deadline:
                break
            neigh_mask = adj[v]
            if neigh_mask:
                clique = max_clique_in_induced(n, adj, neigh_mask | (1 << v), rng)
                if len(clique) >= 2:
                    clique_set = frozenset(clique)
                    if clique_set not in [frozenset(c) for c in candidate_cliques]:
                        candidate_cliques.append(clique)
        
        # Step 2: Lagrangian selection
        # Weight for each clique = number of edges it covers
        clique_weights = []
        for clique in candidate_cliques:
            w = len(clique) * (len(clique) - 1) // 2
            clique_weights.append(w)
        
        # Conflict graph: two cliques conflict if they share an edge
        num_cliques = len(candidate_cliques)
        conflicts = [set() for _ in range(num_cliques)]
        clique_edges = [edge_set_of_clique(c) for c in candidate_cliques]
        
        for i in range(num_cliques):
            for j in range(i + 1, num_cliques):
                if clique_edges[i] & clique_edges[j]:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        
        # Lagrangian iteration
        best_selection = []
        best_score = -1
        dual_vars = [0.0] * num_cliques  # penalty for each clique
        
        for iteration in range(50):
            if time.time() > deadline:
                break
            
            # Compute adjusted weights
            adjusted_weights = []
            for i in range(num_cliques):
                conflict_penalty = sum(dual_vars[j] for j in conflicts[i])
                adj_w = clique_weights[i] - conflict_penalty
                adjusted_weights.append(adj_w)
            
            # Greedy selection: pick clique with best adjusted weight, remove conflicts
            selected = []
            used = [False] * num_cliques
            
            # Sort by adjusted weight descending
            order = sorted(range(num_cliques), key=lambda i: -adjusted_weights[i])
            
            for i in order:
                if used[i]:
                    continue
                if adjusted_weights[i] <= 0:
                    continue
                selected.append(i)
                used[i] = True
                for j in conflicts[i]:
                    used[j] = True
            
            # Score: total edges covered
            covered_edges = set()
            for i in selected:
                covered_edges |= clique_edges[i]
            
            score = len(covered_edges)
            if score > best_score:
                best_score = score
                best_selection = selected[:]
            
            # Update dual variables
            for i in range(num_cliques):
                if i in selected:
                    dual_vars[i] *= 0.9  # decrease penalty for selected
                else:
                    dual_vars[i] += 0.1 * (1 + len(conflicts[i]))  # increase for unselected
        
        # Step 3: Build partition from best selection
        selected_cliques = [candidate_cliques[i] for i in best_selection]
        covered_edge_set = set()
        for clique in selected_cliques:
            for e in edge_set_of_clique(clique):
                covered_edge_set.add(e)
        
        # Cover remaining edges with 2-cliques
        remaining_edges = []
        for e in edges:
            if frozenset(e) not in covered_edge_set:
                remaining_edges.append(list(e))
        
        partition = selected_cliques + remaining_edges
        return partition

    def greedy_partition_max(n, adj, rng, deadline):
        """
        Greedy: repeatedly find edge with MAX common neighbors,
        extract max clique containing it, remove those edges.
        """
        rem_adj = adj[:]
        cliques = []

        while True:
            if time.time() > deadline:
                return None

            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            # Find edge (u,v) with MAX common neighbors in remaining graph
            max_common = -1
            best_u, best_v = -1, -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                # Iterate neighbors v > u
                v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    v_bit = v_mask & (-v_mask)
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit

                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)
                    if cnt > max_common:
                        max_common = cnt
                        best_u, best_v = u, v

            if best_u == -1:
                break

            # Find max clique containing best_u, best_v
            common_mask = rem_adj[best_u] & rem_adj[best_v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            # Remove edges of this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def greedy_partition_min(n, adj, rng, deadline):
        """
        Greedy: repeatedly find edge with MIN common neighbors,
        extract max clique containing it, remove those edges.
        """
        rem_adj = adj[:]
        cliques = []

        while True:
            if time.time() > deadline:
                return None

            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            # Find edge (u,v) with MIN common neighbors in remaining graph
            min_common = n + 1
            best_u, best_v = -1, -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                # Iterate neighbors v > u
                v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    v_bit = v_mask & (-v_mask)
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit

                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)
                    if cnt < min_common:
                        min_common = cnt
                        best_u, best_v = u, v
                        if min_common == 0:
                            break
                if min_common == 0:
                    break

            if best_u == -1:
                break

            # Find max clique containing best_u, best_v
            common_mask = rem_adj[best_u] & rem_adj[best_v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            # Remove edges of this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def greedy_partition_weighted(n, adj, rng, deadline, weight_alpha=0.5):
        """
        Greedy: find edge with weighted score combining common neighbors
        and edge density. weight_alpha controls tradeoff.
        """
        rem_adj = adj[:]
        cliques = []

        while True:
            if time.time() > deadline:
                return None

            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            # Find edge with best weighted score
            best_score = -1
            best_u, best_v = -1, -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                deg_u = popcount(rem_adj[u])
                v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    v_bit = v_mask & (-v_mask)
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit

                    deg_v = popcount(rem_adj[v])
                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)
                    
                    # Score: balance between common neighbors and degree
                    score = weight_alpha * cnt + (1 - weight_alpha) * (deg_u + deg_v) / 2
                    if score > best_score:
                        best_score = score
                        best_u, best_v = u, v

            if best_u == -1:
                break

            # Find max clique containing best_u, best_v
            common_mask = rem_adj[best_u] & rem_adj[best_v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            # Remove edges of this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def validate_partition(n, adj, partition):
        """
        Validate that partition is a valid edge clique partition.
        Returns True if valid, False otherwise.
        """
        # Check each clique is a valid clique with >= 2 vertices
        for clique in partition:
            if len(clique) < 2:
                return False
            # Check all pairs are adjacent
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if not (adj[u] & (1 << v)):
                        return False
        
        # Check every edge is covered exactly once
        edge_count = {}
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edge_count[frozenset([u, v])] = 0
        
        for clique in partition:
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = frozenset([clique[i], clique[j]])
                    if e in edge_count:
                        edge_count[e] += 1
                    else:
                        return False  # Edge not in original graph
        
        # Every original edge must be covered exactly once
        for e, cnt in edge_count.items():
            if cnt != 1:
                return False
        
        return True

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)

        # Baseline: each edge as its own 2-clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)

        start = time.time()
        attempt = 0
        while time.time() < start + time_limit:
            rng = random.Random(seed * 1000 + attempt)
            
            # Try different strategies
            if attempt % 5 == 0:
                result = greedy_partition_max(n, adj, rng, DEADLINE)
            elif attempt % 5 == 1:
                result = greedy_partition_min(n, adj, rng, DEADLINE)
            elif attempt % 5 == 2:
                result = greedy_partition_weighted(n, adj, rng, DEADLINE, weight_alpha=0.3)
            elif attempt % 5 == 3:
                result = greedy_partition_weighted(n, adj, rng, DEADLINE, weight_alpha=0.7)
            else:
                # Lagrangian relaxation approach
                result = lagrangian_clique_partition(n, adj, edges, rng, DEADLINE)
            
            if result is not None:
                # Validate the partition
                if validate_partition(n, adj, result):
                    count = len(result)
                    if count < best_count:
                        best_count = count
                        best_cliques = result
            attempt += 1

        return best_cliques

    # Time budget allocation
    t1 = 15.0
    t2 = 90.0

    current = time.time()
    remaining = DEADLINE - current
    if t1 + t2 > remaining:
        scale = remaining / (t1 + t2)
        t1 *= scale
        t2 *= scale

    result1 = solve_instance(32, 0.6, 601, t1)

    if time.time() > DEADLINE:
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, t2)

    return [result1, result2]
