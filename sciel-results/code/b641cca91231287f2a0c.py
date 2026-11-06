import time
import random
import itertools

def construct():
    start_time = time.time()
    timeout = 290  # Safety margin before 300s limit

    # Fallback: Paley 37
    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    n_seed, edges_seed = get_paley(37)

    # Try to build n=42
    n_target = 42
    all_5sets = list(itertools.combinations(range(n_target), 5))
    
    # Precompute: for each edge (u,v), which 5-sets contain both u and v
    # This allows fast violation updates
    edge_to_5sets = {}
    for idx, combo in enumerate(all_5sets):
        for i in range(5):
            for j in range(i + 1, 5):
                edge = (combo[i], combo[j])
                if edge not in edge_to_5sets:
                    edge_to_5sets[edge] = []
                edge_to_5sets[edge].append(idx)

    # Helper to check if a 5-set is a clique or independent given adj
    def is_bad_5set(combo, adj):
        # Check Clique
        ok_clique = True
        # Check Independent
        ok_indep = True
        for i in range(5):
            u = combo[i]
            for j in range(i + 1, 5):
                v = combo[j]
                if adj[u] & (1 << v):
                    ok_indep = False
                else:
                    ok_clique = False
                if not ok_clique and not ok_indep:
                    return False
        return ok_clique or ok_indep

    # Local Search
    best_result = (n_seed, edges_seed)

    for restart in range(50):  # More restarts with faster updates
        if time.time() - start_time > timeout:
            break

        # Initialize with moderate density to reduce initial violations
        random.seed(42 + restart)
        adj = [0] * n_target
        edges = set()
        p = 0.5  # Standard random graph
        for u in range(n_target):
            for v in range(u + 1, n_target):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.add((u, v))

        # Initial violation set
        violations = set()
        for idx, c in enumerate(all_5sets):
            if is_bad_5set(c, adj):
                violations.add(idx)

        # Local Search Loop
        max_steps = 20000
        for step in range(max_steps):
            if time.time() - start_time > timeout:
                break

            if not violations:
                # Success!
                edge_list = list(edges)
                return n_target, edge_list

            # Pick a random violation
            v_idx = random.choice(list(violations))
            v_set = all_5sets[v_idx]
            
            # Determine if it's a clique or independent set
            is_clique = True
            is_indep = True
            for i in range(5):
                u = v_set[i]
                for j in range(i + 1, 5):
                    v = v_set[j]
                    if adj[u] & (1 << v):
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break

            # Pick an edge to flip within this 5-set
            # For clique: remove an edge (any of the 10 edges)
            # For independent: add an edge (any of the 10 non-edges)
            pairs = list(itertools.combinations(v_set, 2))
            
            if is_clique:
                # Remove a random edge
                u, v = random.choice(pairs)
                if (u, v) not in edges:
                    continue  # Should not happen
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.remove((u, v))
            else:
                # Add a random non-edge
                # Find non-edges
                non_edges = [(u, v) for u, v in pairs if (u, v) not in edges]
                if not non_edges:
                    continue
                u, v = random.choice(non_edges)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))

            # Update violations using precomputed mapping
            # Only 5-sets containing (u,v) can change
            if (u, v) in edge_to_5sets:
                for affected_idx in edge_to_5sets[(u, v)]:
                    if is_bad_5set(all_5sets[affected_idx], adj):
                        violations.add(affected_idx)
                    else:
                        violations.discard(affected_idx)

    return best_result
