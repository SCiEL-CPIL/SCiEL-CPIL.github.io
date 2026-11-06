"""Combined approach: bitmask-based Q3 detection + Max-SAT style simulated annealing."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    total_edges = n * (n - 1) // 2  # 120

    # Q3 structure: 8 vertices, 12 edges
    q3_vertices = list(range(8))
    q3_edges = set()
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.add((i, j))
    q3_adj = {v: set() for v in range(8)}
    for a, b in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)

    # Precompute all 8-subsets
    all_8sets = list(combinations(range(16), 8))

    # For each pair of vertices, precompute which 8-sets contain both
    pair_to_8sets = {}
    for vs in all_8sets:
        for i in range(8):
            for j in range(i + 1, 8):
                pair = (vs[i], vs[j])
                if pair not in pair_to_8sets:
                    pair_to_8sets[pair] = []
                pair_to_8sets[pair].append(vs)

    def contains_q3_bitmask(vs, adj_bits):
        """Fast Q3 check using bitmasks. Returns True if vs induces Q3."""
        # Build local adjacency masks
        local_adj = [0] * 8
        for i in range(8):
            u = vs[i]
            mask = adj_bits[u]
            for j in range(i + 1, 8):
                if (mask >> vs[j]) & 1:
                    local_adj[i] |= (1 << j)
                    local_adj[j] |= (1 << i)

        # Quick degree check: each vertex needs degree >= 3 within the 8-set
        for i in range(8):
            if bin(local_adj[i]).count('1') < 3:
                return False

        # Backtracking with pruning
        assignment = [-1] * 8
        vs_used = [False] * 8

        def backtrack(q3_v):
            if q3_v == 8:
                return True
            for vs_i in range(8):
                if vs_used[vs_i]:
                    continue
                ok = True
                for nb in q3_adj[q3_v]:
                    if nb < q3_v:
                        if not ((local_adj[vs_i] >> assignment[nb]) & 1):
                            ok = False
                            break
                if ok:
                    assignment[q3_v] = vs_i
                    vs_used[vs_i] = True
                    if backtrack(q3_v + 1):
                        return True
                    vs_used[vs_i] = False
                    assignment[q3_v] = -1
            return False

        return backtrack(0)

    def count_q3_violations(adj_bits):
        """Count number of 8-sets that contain a Q3 subgraph."""
        count = 0
        for vs in all_8sets:
            if contains_q3_bitmask(vs, adj_bits):
                count += 1
        return count

    def is_q3_free(adj_bits):
        """Check if graph is Q3-free."""
        for vs in all_8sets:
            if contains_q3_bitmask(vs, adj_bits):
                return False
        return True

    def edges_from_adj(adj_bits):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if (adj_bits[i] >> j) & 1:
                    edges.append((i, j))
        return edges

    # Start with a good initial solution
    random.seed(42)

    # Generate multiple random starts and pick the best
    best_adj = None
    best_edges = None
    best_count = 0
    best_violations = float('inf')

    for trial in range(30):
        if time.time() > deadline - 60:
            break

        # Random graph with varying density
        p = random.uniform(0.55, 0.80)
        adj = [0] * 16
        edge_count = 0
        for i in range(16):
            for j in range(i + 1, 16):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edge_count += 1

        violations = count_q3_violations(adj)

        # Try to repair by removing edges from violating Q3s
        if violations > 0:
            for _ in range(min(violations * 2, 20)):
                if time.time() > deadline - 60:
                    break
                # Find a violating 8-set
                found = False
                for vs in all_8sets:
                    if contains_q3_bitmask(vs, adj):
                        found = True
                        # Remove a random edge from this Q3
                        vs_list = list(vs)
                        random.shuffle(vs_list)
                        for i in range(8):
                            for j in range(i + 1, 8):
                                u, v = vs_list[i], vs_list[j]
                                if (adj[u] >> v) & 1:
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                    edge_count -= 1
                                    break
                            else:
                                continue
                            break
                        break
                if not found:
                    break
            violations = count_q3_violations(adj)

        # Track best by: prefer Q3-free, then more edges
        if violations == 0 and edge_count > best_count:
            best_count = edge_count
            best_adj = adj[:]
            best_violations = 0
        elif violations < best_violations:
            best_violations = violations
            if best_adj is None or (violations == 0 and edge_count > best_count):
                best_count = edge_count
                best_adj = adj[:]

    if best_adj is None:
        # Fallback: empty graph
        best_adj = [0] * 16
        best_count = 0

    # Simulated annealing on the Max-SAT formulation
    # Objective: maximize (edge_count - penalty * violations)
    current_adj = best_adj[:]
    current_edges = edges_from_adj(current_adj)
    current_count = len(current_edges)
    current_violations = count_q3_violations(current_adj)
    current_score = current_count - 10 * current_violations

    best_sa_adj = best_adj[:]
    best_sa_count = current_count
    best_sa_score = current_score

    temperature = 5.0
    cooling = 0.9997
    min_temp = 0.1

    # Precompute all possible edges for random selection
    all_possible_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]

    while time.time() < deadline - 5:
        # Choose a random edge to flip
        u, v = random.choice(all_possible_edges)
        has_edge = (current_adj[u] >> v) & 1

        # Compute delta in violations and edge count
        if has_edge:
            # Removing edge: count decreases by 1
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            # Only check 8-sets containing both u and v
            relevant = pair_to_8sets.get((u, v), [])
            old_viol = 0
            new_viol = 0
            for vs in relevant:
                if contains_q3_bitmask(vs, current_adj):
                    old_viol += 1
                if contains_q3_bitmask(vs, new_adj):
                    new_viol += 1
            delta_viol = new_viol - old_viol
            delta_edges = -1
        else:
            # Adding edge: count increases by 1
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            relevant = pair_to_8sets.get((u, v), [])
            old_viol = 0
            new_viol = 0
            for vs in relevant:
                if contains_q3_bitmask(vs, current_adj):
                    old_viol += 1
                if contains_q3_bitmask(vs, new_adj):
                    new_viol += 1
            delta_viol = new_viol - old_viol
            delta_edges = 1

        delta_score = delta_edges - 10 * delta_viol

        # Acceptance criterion
        if delta_score >= 0 or random.random() < pow(2.718281828, delta_score / temperature):
            current_adj = new_adj
            current_count += delta_edges
            current_violations += delta_viol
            current_score += delta_score

            if current_violations == 0 and current_count > best_sa_count:
                best_sa_count = current_count
                best_sa_adj = current_adj[:]
                best_sa_score = current_score
            elif current_score > best_sa_score:
                best_sa_score = current_score
                if current_violations == 0:
                    best_sa_count = current_count
                    best_sa_adj = current_adj[:]

        temperature = max(min_temp, temperature * cooling)

    # Use the best Q3-free solution found
    if best_sa_violations == 0 if 'best_sa_violations' in dir() else True:
        # Check if best_sa_adj is Q3-free
        if is_q3_free(best_sa_adj):
            final_adj = best_sa_adj
            final_count = best_sa_count
        else:
            final_adj = best_adj
            final_count = best_count
    else:
        final_adj = best_adj
        final_count = best_count

    # Final greedy pass: try to add any missing edges
    final_adj = final_adj[:]
    for i in range(16):
        for j in range(i + 1, 16):
            if time.time() > deadline - 2:
                break
            if not ((final_adj[i] >> j) & 1):
                test_adj = final_adj[:]
                test_adj[i] |= (1 << j)
                test_adj[j] |= (1 << i)
                relevant = pair_to_8sets.get((i, j), [])
                creates_q3 = False
                for vs in relevant:
                    if contains_q3_bitmask(vs, test_adj):
                        creates_q3 = True
                        break
                if not creates_q3:
                    final_adj = test_adj
                    final_count += 1
        if time.time() > deadline - 2:
            break

    result = edges_from_adj(final_adj)
    return result
