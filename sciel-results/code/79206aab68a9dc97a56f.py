import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    n = 20

    def popcount(x):
        return bin(x).count('1')

    def can_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if (adj[u] >> v) & 1:
            return False

        # 1. Pair (u,v) common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False

        # 2. For all w in N(v) \ {u}: pair (u,w) gains v as common neighbor
        #    Need popcount(adj[u] & adj[w]) <= 1 (before adding v to adj[u])
        mask_w = adj[v]
        while mask_w:
            lsb = mask_w & (-mask_w)
            w = lsb.bit_length() - 1
            mask_w ^= lsb
            if w == u:
                continue
            if popcount(adj[u] & adj[w]) > 1:
                return False

        # 3. For all w in N(u) \ {v}: pair (v,w) gains u as common neighbor
        #    Need popcount(adj[v] & adj[w]) <= 1 (before adding u to adj[v])
        mask_w = adj[u]
        while mask_w:
            lsb = mask_w & (-mask_w)
            w = lsb.bit_length() - 1
            mask_w ^= lsb
            if w == v:
                continue
            if popcount(adj[v] & adj[w]) > 1:
                return False

        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2

    def get_candidate_edges(adj):
        """Get all missing edges that are 'near' addable, sorted by constraint tightness."""
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    common = adj[u] & adj[v]
                    c = popcount(common)
                    if c <= 2:
                        # Check how many other constraints are violated
                        violations = 0
                        mask_w = adj[v]
                        while mask_w:
                            lsb = mask_w & (-mask_w)
                            w = lsb.bit_length() - 1
                            mask_w ^= lsb
                            if w == u:
                                continue
                            if popcount(adj[u] & adj[w]) > 1:
                                violations += 1
                        mask_w = adj[u]
                        while mask_w:
                            lsb = mask_w & (-mask_w)
                            w = lsb.bit_length() - 1
                            mask_w ^= lsb
                            if w == v:
                                continue
                            if popcount(adj[v] & adj[w]) > 1:
                                violations += 1
                        candidates.append((violations, c, u, v))
        candidates.sort()
        return candidates

    def greedy_add(adj):
        """Greedily add any valid edge until none remain, prioritizing tight constraints."""
        improved = True
        while improved:
            improved = False
            # Get candidates sorted by how close they are to being addable
            candidates = get_candidate_edges(adj)
            for _, _, u, v in candidates:
                if (adj[u] >> v) & 1:
                    continue
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
                    break  # Restart with new state
        return adj

    def tabu_search(adj_init, tabu_tenure=20, max_steps=100000):
        """Tabu search with aspiration criterion."""
        adj = list(adj_init)
        best_adj = list(adj)
        best_count = edge_count(adj)
        current_count = best_count

        tabu_list = {}  # (u, v) -> step when it becomes non-tabu
        current_step = 0
        no_improve = 0

        while current_step < max_steps:
            if time.time() > deadline:
                break

            # Get candidate edges
            candidates = get_candidate_edges(adj)

            # Also consider removing edges (less frequently)
            if random.random() < 0.15:
                # Try removing a random edge
                all_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            all_edges.append((u, v))
                if all_edges:
                    u, v = random.choice(all_edges)
                    remove_edge(adj, u, v)
                    new_count = current_count - 1
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = list(adj)
                        no_improve = 0
                    else:
                        no_improve += 1
                    current_count = new_count
                    tabu_list[(u, v)] = current_step + tabu_tenure
                    current_step += 1
                    continue

            # Try adding edges, preferring non-tabu
            best_move = None
            best_move_score = -1
            for violations, c, u, v in candidates:
                if (adj[u] >> v) & 1:
                    continue
                # Tabu check
                if (u, v) in tabu_list and tabu_list[(u, v)] > current_step:
                    continue
                # Score: prefer fewer violations, then fewer common neighbors
                score = -violations * 10 - c
                if can_add(adj, u, v):
                    score += 100  # Strong preference for valid additions
                if score > best_move_score:
                    best_move_score = score
                    best_move = (u, v)

            if best_move is None:
                # All moves are tabu or invalid, pick best tabu move (aspiration)
                for violations, c, u, v in candidates:
                    if (adj[u] >> v) & 1:
                        continue
                    if can_add(adj, u, v):
                        best_move = (u, v)
                        break
                if best_move is None:
                    break

            u, v = best_move
            if can_add(adj, u, v):
                add_edge(adj, u, v)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = list(adj)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Should not happen if we checked, but handle gracefully
                no_improve += 1

            tabu_list[(u, v)] = current_step + tabu_tenure
            current_step += 1

            # Clean up old tabu entries
            if current_step % 100 == 0:
                tabu_list = {k: v for k, v in tabu_list.items() if v > current_step}

            if no_improve > 5000:
                break

        return best_adj, best_count

    # Multi-start strategy
    best_adj = None
    best_count = 0

    # 1. Deterministic Greedy Start
    adj0 = [0] * n
    adj0 = greedy_add(adj0)
    c0 = edge_count(adj0)
    if c0 > best_count:
        best_count = c0
        best_adj = list(adj0)

    # 2. Multiple randomized starts with tabu search
    num_starts = 8
    for seed in range(num_starts):
        if time.time() > deadline - 15:
            break

        random.seed(seed * 7919 + 12345)

        # Initialize with random valid edges
        adj_greedy = [0] * n
        num_random = random.randint(15, 30)
        for _ in range(num_random):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not ((adj_greedy[u] >> v) & 1):
                if can_add(adj_greedy, u, v):
                    add_edge(adj_greedy, u, v)

        # Greedy completion
        adj_greedy = greedy_add(adj_greedy)
        c_greedy = edge_count(adj_greedy)

        if c_greedy > best_count:
            best_count = c_greedy
            best_adj = list(adj_greedy)

        # Tabu search refinement
        adj_tabu, c_tabu = tabu_search(adj_greedy, tabu_tenure=random.randint(15, 30), max_steps=50000)
        if c_tabu > best_count:
            best_count = c_tabu
            best_adj = list(adj_tabu)

        # Refine global best
        if time.time() < deadline - 20:
            adj_tabu2, c_tabu2 = tabu_search(best_adj, tabu_tenure=random.randint(15, 30), max_steps=30000)
            if c_tabu2 > best_count:
                best_count = c_tabu2
                best_adj = list(adj_tabu2)

    # Final polish on best solution
    if best_adj is not None and time.time() < deadline - 10:
        adj_final, c_final = tabu_search(best_adj, tabu_tenure=25, max_steps=80000)
        if c_final > best_count:
            best_count = c_final
            best_adj = list(adj_final)

    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])

    return edges
