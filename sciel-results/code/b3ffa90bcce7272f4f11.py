import time
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Maximize the number of edges in a 4-cycle-free subgraph of the 7-dimensional hypercube.
    Uses a multi-start local search with simulated annealing to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before the 120s limit

    d = 7
    n_vertices = 1 << d
    n_edges_total = n_vertices * d // 2

    # Precompute all edges and their bit-dimension
    # Edge represented as (min(u,v), max(u,v))
    edge_list = []
    edge_dim = []
    edge_index = {}  # (u, v) -> index in edge_list
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edge_list)
                edge_list.append((u, v))
                edge_dim.append(i)
                edge_index[(u, v)] = idx

    # Precompute 4-cycles (2-dimensional faces)
    # A 4-cycle is formed by two vertices u, v that differ in exactly 2 bits.
    # The 4 edges of the face are: (u, u^(1<<i)), (u, u^(1<<j)), (v, v^(1<<i)), (v, v^(1<<j))
    # where i, j are the two differing bits.
    cycles = []  # list of tuples of 4 edge indices
    cycle_of_edge = [[] for _ in range(n_edges_total)]  # for each edge, list of cycle indices it belongs to

    for u in range(n_vertices):
        # u differs from v in exactly 2 bits
        for i in range(d):
            for j in range(i + 1, d):
                v = u ^ (1 << i) ^ (1 << j)
                if u < v:
                    # The 4 edges of the face:
                    # u -- u^(1<<i)
                    # u -- u^(1<<j)
                    # v -- v^(1<<i) = u^(1<<j)
                    # v -- v^(1<<j) = u^(1<<i)
                    e1 = edge_index[(min(u, u ^ (1 << i)), max(u, u ^ (1 << i)))]
                    e2 = edge_index[(min(u, u ^ (1 << j)), max(u, u ^ (1 << j)))]
                    e3 = edge_index[(min(v, v ^ (1 << i)), max(v, v ^ (1 << i)))]
                    e4 = edge_index[(min(v, v ^ (1 << j)), max(v, v ^ (1 << j)))]
                    cyc_idx = len(cycles)
                    cycles.append((e1, e2, e3, e4))
                    cycle_of_edge[e1].append(cyc_idx)
                    cycle_of_edge[e2].append(cyc_idx)
                    cycle_of_edge[e3].append(cyc_idx)
                    cycle_of_edge[e4].append(cyc_idx)

    n_cycles = len(cycles)

    # State: x[e] = 1 if edge e is selected, 0 otherwise
    # Objective: maximize sum(x[e]) subject to no cycle having all 4 edges selected.
    # Penalty approach: for each cycle, if all 4 edges are selected, penalty = 1.
    # Score = sum(x) - penalty * violation_count, but we want to maximize sum(x) with no violations.
    # Better: track number of edges and number of violations separately.

    def random_initial_solution(rng):
        """Generate a random valid solution (no 4-cycles) by random edge inclusion."""
        x = np.zeros(n_edges_total, dtype=np.int8)
        for e in range(n_edges_total):
            # Check if adding this edge would create a 4-cycle
            ok = True
            for cyc_idx in cycle_of_edge[e]:
                e1, e2, e3, e4 = cycles[cyc_idx]
                other = [cyc for cyc in (e1, e2, e3, e4) if cyc != e]
                if all(x[c] == 1 for c in other):
                    ok = False
                    break
            if ok:
                x[e] = 1
        return x

    def count_edges(x):
        return int(np.sum(x))

    def count_violations(x):
        """Count number of 4-cycles where all 4 edges are selected."""
        violations = 0
        for e1, e2, e3, e4 in cycles:
            if x[e1] and x[e2] and x[e3] and x[e4]:
                violations += 1
        return violations

    def valid_solution(x):
        """Check if solution is valid (no 4-cycles)."""
        for e1, e2, e3, e4 in cycles:
            if x[e1] and x[e2] and x[e3] and x[e4]:
                return False
        return True

    def greedy_construct(rng):
        """Greedy construction with random edge ordering."""
        x = np.zeros(n_edges_total, dtype=np.int8)
        order = list(range(n_edges_total))
        rng.shuffle(order)
        for e in order:
            ok = True
            for cyc_idx in cycle_of_edge[e]:
                e1, e2, e3, e4 = cycles[cyc_idx]
                other = [cyc for cyc in (e1, e2, e3, e4) if cyc != e]
                if all(x[c] == 1 for c in other):
                    ok = False
                    break
            if ok:
                x[e] = 1
        return x

    def local_search(x, rng, max_iter=50000, deadline=None):
        """Local search: try flipping edges to increase count while maintaining validity."""
        x = x.copy()
        current_count = count_edges(x)
        best_count = current_count
        best_x = x.copy()
        
        # Precompute for faster violation checking
        # For each edge, store which cycles it belongs to (already have cycle_of_edge)
        
        for it in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            # Try to add a random edge
            e = rng.randint(0, n_edges_total - 1)
            if x[e] == 0:
                # Check if adding e creates a 4-cycle
                ok = True
                for cyc_idx in cycle_of_edge[e]:
                    e1, e2, e3, e4 = cycles[cyc_idx]
                    other = [cyc for cyc in (e1, e2, e3, e4) if cyc != e]
                    if all(x[c] == 1 for c in other):
                        ok = False
                        break
                if ok:
                    x[e] = 1
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_x = x.copy()
            else:
                # Try to remove a random edge (less likely to help, but for diversity)
                # Actually, removing an edge never helps increase count, so skip
                pass
            
            # Also try: remove an edge that's part of a violation and add another
            # But we maintain validity, so no violations should exist
            
            # Alternative: try a random edge addition more aggressively
            e2 = rng.randint(0, n_edges_total - 1)
            if x[e2] == 0:
                ok = True
                for cyc_idx in cycle_of_edge[e2]:
                    e1, e2_, e3, e4 = cycles[cyc_idx]
                    other = [cyc for cyc in (e1, e2_, e3, e4) if cyc != e2]
                    if all(x[c] == 1 for c in other):
                        ok = False
                        break
                if ok:
                    x[e2] = 1
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_x = x.copy()
        
        return best_x, best_count

    def simulated_annealing(x, rng, max_iter=200000, t0=1.0, t1=0.01, deadline=None):
        """Simulated annealing to escape local optima."""
        x = x.copy()
        current_count = count_edges(x)
        best_count = current_count
        best_x = x.copy()
        
        t = t0
        cooling_rate = (t1 / t0) ** (1.0 / max_iter)
        
        for it in range(max_iter):
            if deadline and time.time() > deadline:
                break
            t *= cooling_rate
            
            # Random move: flip a random edge
            e = rng.randint(0, n_edges_total - 1)
            old_val = x[e]
            x[e] = 1 - old_val
            
            new_count = current_count + (1 if old_val == 0 else -1)
            
            # Check validity
            valid = True
            if old_val == 0:
                # We added edge e; check if any cycle is now fully selected
                for cyc_idx in cycle_of_edge[e]:
                    e1, e2, e3, e4 = cycles[cyc_idx]
                    if x[e1] and x[e2] and x[e3] and x[e4]:
                        valid = False
                        break
            
            if not valid:
                # Reject: revert
                x[e] = old_val
            else:
                delta = new_count - current_count
                if delta >= 0 or rng.random() < np.exp(delta / t):
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_x = x.copy()
                else:
                    x[e] = old_val
        
        return best_x, best_count

    # Multi-start strategy
    rng = random.Random(42)
    best_overall = 0
    best_x_overall = None

    # Start 1: Greedy
    x = greedy_construct(rng)
    count = count_edges(x)
    if count > best_overall:
        best_overall = count
        best_x_overall = x.copy()
    
    # Local search from greedy
    x_ls, count_ls = local_search(x, rng, max_iter=30000, deadline=deadline)
    if count_ls > best_overall:
        best_overall = count_ls
        best_x_overall = x_ls.copy()

    # Multiple random restarts with local search
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        x = random_initial_solution(rng)
        count = count_edges(x)
        
        # Local search
        x_ls, count_ls = local_search(x, rng, max_iter=20000, deadline=deadline)
        if count_ls > best_overall:
            best_overall = count_ls
            best_x_overall = x_ls.copy()
        
        # Simulated annealing from the local search result
        if time.time() < deadline - 10:
            x_sa, count_sa = simulated_annealing(x_ls, rng, max_iter=50000, t0=1.0, t1=0.05, deadline=deadline)
            if count_sa > best_overall:
                best_overall = count_sa
                best_x_overall = x_sa.copy()

    # Final: try to polish the best solution with more intensive local search
    if best_x_overall is not None and time.time() < deadline - 5:
        x_pol, count_pol = local_search(best_x_overall, rng, max_iter=100000, deadline=deadline)
        if count_pol > best_overall:
            best_overall = count_pol
            best_x_overall = x_pol.copy()

    # Convert best solution to edge list
    result = []
    for e in range(n_edges_total):
        if best_x_overall[e] == 1:
            u, v = edge_list[e]
            result.append([u, v])
    
    return result
