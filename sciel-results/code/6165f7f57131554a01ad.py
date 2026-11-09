"""
K2,3-free graph construction on 20 vertices maximizing edges.

Strategy:
- Use a fast bit-parallel K2,3 validity check.
- Multi-restart simulated annealing with efficient edge add/remove moves.
- Key insight: the K2,3 constraint is that no two vertices share 3+ common neighbors.
  We track common-neighbor counts incrementally for speed.
- Final exhaustive polish pass to find any remaining addable edges.
"""

import time
import random
import math
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_pairs_set = set(all_pairs)

    def popcount(x):
        return bin(x).count('1')

    def count_edges(adj):
        return sum(popcount(x) for x in adj) // 2

    def is_k23_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True

    def valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free.
        After adding (u,v):
        - u and v must have at most 2 common neighbors (they become each other's neighbor,
          but the common neighbors of u and v are N(u) ∩ N(v) which doesn't change
          since v wasn't in N(u) before)
        - For any w in N(u) ∪ N(v) (w != u,v), the pair (w, v) must have at most 2 common neighbors,
          and pair (w, u) must have at most 2 common neighbors.
          Actually, adding (u,v) means:
          - v becomes a neighbor of u, and u becomes a neighbor of v.
          - For any w in N(u) (w != v): common neighbors of (w, v) now include u (newly),
            so we need popcount(adj[w] & adj[v]) <= 1 before adding (so after adding, it's <= 2).
          - For any w in N(v) (w != u): common neighbors of (w, u) now include v (newly),
            so we need popcount(adj[w] & adj[u]) <= 1 before adding.
          - Also, u and v themselves: after adding, common neighbors of (u,v) = N(u) ∩ N(v),
            which must be <= 2.
        """
        if adj[u] & (1 << v):
            return False
        # Check u and v common neighbors
        if popcount(adj[u] & adj[v]) > 2:
            return False
        # Check for each w in N(u), (w,v) must have <= 1 common neighbor currently
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w == v:
                continue
            if popcount(adj[w] & adj[v]) > 1:
                return False
        # Check for each w in N(v), (w,u) must have <= 1 common neighbor currently
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w == u:
                continue
            if popcount(adj[w] & adj[u]) > 1:
                return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append((u, v))
        return edges

    def get_non_edges(adj):
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    non_edges.append((u, v))
        return non_edges

    def greedy_fill(adj):
        """Greedily add all possible edges in random order, multiple passes."""
        for _ in range(5):
            any_added = False
            non_edges = get_non_edges(adj)
            random.shuffle(non_edges)
            for u, v in non_edges:
                if not (adj[u] & (1 << v)):
                    if valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        any_added = True
            if not any_added:
                break

    def make_k4_union():
        """5 disjoint K4s."""
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(adj, base + a, base + b)
        return adj

    def make_random_graph(density=0.15):
        """Random sparse graph."""
        adj = [0] * n
        for u, v in all_pairs:
            if random.random() < density:
                add_edge(adj, u, v)
        return adj

    def make_complete_min():
        """Start with a small random graph."""
        adj = [0] * n
        for _ in range(10):
            u, v = random.sample(range(n), 2)
            add_edge(adj, u, v)
        return adj

    best_edges = 0
    best_adj = None

    # Multiple initial strategies
    initial_strategies = []
    
    # Strategy 1: 5 disjoint K4s
    adj0 = make_k4_union()
    greedy_fill(adj0)
    initial_strategies.append(adj0)
    
    # Strategy 2: Random graph
    for _ in range(3):
        adj_r = make_random_graph(0.1)
        greedy_fill(adj_r)
        initial_strategies.append(adj_r)
    
    # Strategy 3: Complete graph on 10 vertices + 10 isolated
    adj_c = [0] * n
    for u in range(10):
        for v in range(u + 1, 10):
            add_edge(adj_c, u, v)
    greedy_fill(adj_c)
    initial_strategies.append(adj_c)
    
    # Strategy 4: Cycle + chords
    adj_cyc = [0] * n
    for i in range(n):
        add_edge(adj_cyc, i, (i + 1) % n)
        add_edge(adj_cyc, i, (i + 2) % n)
    greedy_fill(adj_cyc)
    initial_strategies.append(adj_cyc)

    # Find best initial
    for adj_s in initial_strategies:
        e = count_edges(adj_s)
        if e > best_edges:
            best_edges = e
            best_adj = adj_s[:]

    # Simulated annealing with multiple restarts
    num_restarts = 30
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break

        # Choose starting point
        if r == 0:
            adj = best_adj[:]
        elif r < 5:
            # Perturb best
            adj = best_adj[:]
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:random.randint(2, 5)]:
                remove_edge(adj, u, v)
        else:
            # Fresh start with random perturbation
            adj = best_adj[:]
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:random.randint(3, 8)]:
                remove_edge(adj, u, v)

        greedy_fill(adj)
        current_edges = count_edges(adj)

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

        # Simulated annealing
        T = 0.5
        T_min = 0.02
        cooling = 0.995
        max_steps = 50000

        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            if T < T_min:
                break

            move_type = random.random()

            if move_type < 0.4:
                # Try to add a random non-edge
                non_edges = get_non_edges(adj)
                if not non_edges:
                    T *= cooling
                    continue
                u, v = random.choice(non_edges)
                if valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Try a 1-for-2 swap: remove one edge, add two
                    if current_edges > 5:
                        edges = get_edges(adj)
                        if edges:
                            u1, v1 = random.choice(edges)
                            # Try removing (u1,v1) and adding (u,v) plus another
                            remove_edge(adj, u1, v1)
                            # Check if (u,v) can now be added
                            if valid_add(adj, u, v):
                                add_edge(adj, u, v)
                                # Try to add another non-edge
                                non_edges2 = get_non_edges(adj)
                                if non_edges2:
                                    # Try a few candidates
                                    added = False
                                    for _ in range(min(5, len(non_edges2))):
                                        u2, v2 = random.choice(non_edges2)
                                        if valid_add(adj, u2, v2):
                                            add_edge(adj, u2, v2)
                                            current_edges += 1  # net: -1 + 1 + 1 = +1
                                            added = True
                                            if current_edges > best_edges:
                                                best_edges = current_edges
                                                best_adj = adj[:]
                                            break
                                    if not added:
                                        # Net change: -1 + 1 = 0, but structure changed
                                        current_edges += 0  # removed one, added one
                                else:
                                    current_edges += 0
                            else:
                                # Revert
                                add_edge(adj, u1, v1)
            elif move_type < 0.6:
                # 1-for-2 swap: remove one edge, try to add two non-edges
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or len(non_edges) < 2:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                idx1 = random.randrange(len(non_edges))
                idx2 = random.randrange(len(non_edges) - 1)
                if idx2 >= idx1:
                    idx2 += 1
                u2, v2 = non_edges[idx1]
                u3, v3 = non_edges[idx2]
                
                remove_edge(adj, u1, v1)
                added = 0
                if valid_add(adj, u2, v2):
                    add_edge(adj, u2, v2)
                    added += 1
                if added == 1 and valid_add(adj, u3, v3):
                    add_edge(adj, u3, v3)
                    added += 1
                
                if added >= 1:
                    current_edges += added - 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    add_edge(adj, u1, v1)
            elif move_type < 0.75:
                # 1-for-1 swap: remove one edge, add another
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or not non_edges:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                remove_edge(adj, u1, v1)
                if valid_add(adj, u2, v2):
                    add_edge(adj, u2, v2)
                    # Net 0, accept with probability
                    if random.random() < 0.5 + 0.5 * T:
                        pass
                    else:
                        remove_edge(adj, u2, v2)
                        add_edge(adj, u1, v1)
                else:
                    add_edge(adj, u1, v1)
            elif move_type < 0.85:
                # 2-for-2 swap
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if len(edges) < 2 or len(non_edges) < 2:
                    T *= cooling
                    continue
                idx1 = random.randrange(len(edges))
                idx2 = random.randrange(len(edges) - 1)
                if idx2 >= idx1:
                    idx2 += 1
                u1, v1 = edges[idx1]
                u2, v2 = edges[idx2]
                idx3 = random.randrange(len(non_edges))
                idx4 = random.randrange(len(non_edges) - 1)
                if idx4 >= idx3:
                    idx4 += 1
                u3, v3 = non_edges[idx3]
                u4, v4 = non_edges[idx4]
                
                remove_edge(adj, u1, v1)
                remove_edge(adj, u2, v2)
                added = 0
                if valid_add(adj, u3, v3):
                    add_edge(adj, u3, v3)
                    added += 1
                if added == 1 and valid_add(adj, u4, v4):
                    add_edge(adj, u4, v4)
                    added += 1
                
                if added == 2:
                    # Net 0, accept with probability
                    if random.random() < 0.5 + 0.5 * T:
                        pass
                    else:
                        remove_edge(adj, u3, v3)
                        remove_edge(adj, u4, v4)
                        add_edge(adj, u1, v1)
                        add_edge(adj, u2, v2)
                elif added == 1:
                    # Net -1, accept with SA probability
                    if random.random() < math.exp(-1.0 / T):
                        add_edge(adj, u1, v1)
                        add_edge(adj, u2, v2)
                        remove_edge(adj, u3, v3)
                        current_edges -= 1
                    else:
                        add_edge(adj, u1, v1)
                        add_edge(adj, u2, v2)
                        remove_edge(adj, u3, v3)
                else:
                    add_edge(adj, u1, v1)
                    add_edge(adj, u2, v2)
            else:
                # Random edge removal (to escape local optima)
                edges = get_edges(adj)
                if edges:
                    u, v = random.choice(edges)
                    if random.random() < math.exp(-1.0 / T):
                        remove_edge(adj, u, v)
                        current_edges -= 1

            T *= cooling

        # After SA, do greedy fill
        greedy_fill(adj)
        current_edges = count_edges(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

    # Final exhaustive polish
    if best_adj and (time.time() - start_time < time_limit - 3):
        polish_adj = best_adj[:]
        polish_edges = count_edges(polish_adj)
        improved = True
        max_polish_iters = 50
        for _ in range(max_polish_iters):
            if not improved:
                break
            if time.time() - start_time > time_limit - 3:
                break
            improved = False
            
            # Try adding any non-edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not (polish_adj[u] & (1 << v)):
                        if valid_add(polish_adj, u, v):
                            add_edge(polish_adj, u, v)
                            polish_edges += 1
                            improved = True
            
            # Try 1-for-2 swaps
            if not improved:
                edges = get_edges(polish_adj)
                non_edges = get_non_edges(polish_adj)
                if len(non_edges) >= 2:
                    for a, b in edges:
                        if time.time() - start_time > time_limit - 3:
                            break
                        found = False
                        for i in range(len(non_edges)):
                            for j in range(i + 1, len(non_edges)):
                                c, d = non_edges[i]
                                e, f = non_edges[j]
                                remove_edge(polish_adj, a, b)
                                added = 0
                                if valid_add(polish_adj, c, d):
                                    add_edge(polish_adj, c, d)
                                    added += 1
                                if added == 1 and valid_add(polish_adj, e, f):
                                    add_edge(polish_adj, e, f)
                                    added += 1
                                if added >= 1:
                                    polish_edges += added - 1
                                    improved = True
                                    found = True
                                    # Try greedy fill
                                    for x in range(n):
                                        for y in range(x + 1, n):
                                            if not (polish_adj[x] & (1 << y)):
                                                if valid_add(polish_adj, x, y):
                                                    add_edge(polish_adj, x, y)
                                                    polish_edges += 1
                                                    improved = True
                                    break
                                else:
                                    add_edge(polish_adj, a, b)
                            if found:
                                break
                        if found:
                            break
            
            if polish_edges > best_edges:
                best_edges = polish_edges
                best_adj = polish_adj[:]

    # Final verification
    if best_adj:
        if not is_k23_free(best_adj):
            # Fallback
            adj0 = make_k4_union()
            greedy_fill(adj0)
            best_adj = adj0
            best_edges = count_edges(best_adj)

    # Convert to list of edges
    result = []
    for u, v in get_edges(best_adj):
        result.append([u, v])
    return result
