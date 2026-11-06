"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }

    # Precompute all 8-subsets
    all_8sets = list(combinations(range(16), 8))
    # Precompute 8-sets containing each pair (for fast incremental checks)
    pair_to_8sets = {}
    for vs in all_8sets:
        for i in range(8):
            for j in range(i+1, 8):
                pair = (vs[i], vs[j])
                if pair not in pair_to_8sets:
                    pair_to_8sets[pair] = []
                pair_to_8sets[pair].append(vs)

    def contains_q3_fast(vs, adj_bits):
        """Check if 8 vertices vs induce a Q3. adj_bits[v] = bitmask of neighbors."""
        # Build local adjacency
        local_adj = [0] * 8
        for i in range(8):
            u = vs[i]
            mask = adj_bits[u]
            for j in range(8):
                if j != i and (mask >> vs[j]) & 1:
                    local_adj[i] |= (1 << j)
        
        # Degree check
        for i in range(8):
            if bin(local_adj[i]).count('1') < 3:
                return False
        
        # Backtrack
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

    def check_edge_creates_q3(u, v, adj_bits):
        """Check if adding edge (u,v) creates a Q3. Only check 8-sets containing u and v."""
        if u > v:
            u, v = v, u
        relevant_8sets = pair_to_8sets.get((u, v), [])
        for vs in relevant_8sets:
            if contains_q3_fast(vs, adj_bits):
                return True
        return False

    def edges_to_adj_bits(edges):
        adj = [0] * 16
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_q3_free(edges):
        adj = edges_to_adj_bits(edges)
        for vs in all_8sets:
            if contains_q3_fast(vs, adj):
                return False
        return True

    def greedy_add(edges_set, adj_bits):
        """Greedily add any possible edge."""
        changed = True
        while changed and time.time() < deadline - 30:
            changed = False
            for i in range(16):
                for j in range(i+1, 16):
                    if (i, j) in edges_set:
                        continue
                    new_adj = adj_bits[:]
                    new_adj[i] |= (1 << j)
                    new_adj[j] |= (1 << i)
                    if not check_edge_creates_q3(i, j, new_adj):
                        adj_bits = new_adj
                        edges_set.add((i, j))
                        changed = True
                        if time.time() > deadline - 30:
                            break
                if time.time() > deadline - 30:
                    break
        return edges_set, adj_bits

    # Initial dense random Q3-free graph
    random.seed(42)
    best_edges = set()
    best_adj = [0] * 16
    best_count = 0

    # Generate initial random graph
    all_possible = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    for attempt in range(50):
        if time.time() > deadline - 60:
            break
        # Random graph with density ~0.7
        candidate = set()
        for e in all_possible:
            if random.random() < 0.7:
                candidate.add(e)
        
        # Remove edges to make Q3-free
        adj = edges_to_adj_bits(candidate)
        # Find all Q3s and remove one edge from each
        q3_count = 0
        for vs in all_8sets:
            if contains_q3_fast(vs, adj):
                q3_count += 1
                # Remove a random edge from this Q3
                vs_list = list(vs)
                random.shuffle(vs_list)
                for i in range(8):
                    for j in range(i+1, 8):
                        u, v = vs_list[i], vs_list[j]
                        if (u, v) in candidate or (v, u) in candidate:
                            # Remove this edge
                            if (u, v) in candidate:
                                candidate.remove((u, v))
                            else:
                                candidate.remove((v, u))
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            break
                    else:
                        continue
                    break
                if q3_count > 10:  # Too many Q3s, restart
                    break
        
        if q3_count <= 10 and len(candidate) > best_count:
            best_edges = candidate
            best_adj = adj
            best_count = len(candidate)
            # Greedy add
            best_edges, best_adj = greedy_add(best_edges, best_adj)
            best_count = len(best_edges)

    # Iterated Local Search with swap moves
    current_edges = set(best_edges)
    current_adj = best_adj[:]
    current_count = len(current_edges)

    # Temperature for simulated annealing
    temperature = 1.0
    cooling = 0.995

    while time.time() < deadline - 10:
        # Generate a perturbation: remove 1-2 edges, add 1-2 edges
        perturbed_edges = set(current_edges)
        perturbed_adj = current_adj[:]
        
        # Remove 1-2 random edges
        remove_count = random.randint(1, 2)
        removed = []
        for _ in range(remove_count):
            if not perturbed_edges:
                break
            e = random.choice(list(perturbed_edges))
            u, v = e
            perturbed_edges.remove(e)
            perturbed_adj[u] &= ~(1 << v)
            perturbed_adj[v] &= ~(1 << u)
            removed.append(e)
        
        # Try to add back the removed edges or new edges
        added = 0
        # First try to add back removed edges
        for e in removed:
            if time.time() > deadline - 10:
                break
            u, v = e
            if (u, v) not in perturbed_edges:
                new_adj = perturbed_adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                if not check_edge_creates_q3(u, v, new_adj):
                    perturbed_adj = new_adj
                    perturbed_edges.add((u, v))
                    added += 1
        
        # Then try to add new random edges
        new_edge_attempts = random.randint(1, 3)
        for _ in range(new_edge_attempts):
            if time.time() > deadline - 10:
                break
            e = random.choice(all_possible)
            u, v = e
            if (u, v) not in perturbed_edges:
                new_adj = perturbed_adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                if not check_edge_creates_q3(u, v, new_adj):
                    perturbed_adj = new_adj
                    perturbed_edges.add((u, v))
                    added += 1
        
        perturbed_count = len(perturbed_edges)
        
        # Accept if better, or with probability based on simulated annealing
        if perturbed_count > current_count:
            current_edges = perturbed_edges
            current_adj = perturbed_adj
            current_count = perturbed_count
            if current_count > best_count:
                best_count = current_count
                best_edges = set(current_edges)
        else:
            # Simulated annealing acceptance
            delta = current_count - perturbed_count
            if delta > 0:
                accept_prob = 1.0 / (1.0 + delta * temperature)
                if random.random() < accept_prob:
                    current_edges = perturbed_edges
                    current_adj = perturbed_adj
                    current_count = perturbed_count
        
        # Cool down
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01

    # Final greedy pass
    best_edges, best_adj = greedy_add(best_edges, best_adj)
    best_count = len(best_edges)

    result = [[u, v] for (u, v) in best_edges]
    return result
# EVOLVE-BLOCK-END
