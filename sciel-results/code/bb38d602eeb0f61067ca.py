import random
import time
import math
import numpy as np

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Precompute edge indices for fast access
    edge_idx = {}
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx[(u, v)] = len(all_edges)
            all_edges.append((u, v))
    num_possible_edges = len(all_edges)  # 120

    def adj_to_matrix(adj):
        """Convert bitmask adjacency list to numpy adjacency matrix"""
        A = np.zeros((n, n), dtype=np.int8)
        for u in range(n):
            mask = adj[u]
            for v in range(n):
                if mask & (1 << v):
                    A[u, v] = 1
        return A

    def matrix_to_adj(A):
        """Convert numpy adjacency matrix to bitmask adjacency list"""
        adj = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if A[u, v]:
                    mask |= (1 << v)
            adj[u] = mask
        return adj

    def get_degs(A):
        """Get degree vector from adjacency matrix"""
        return A.sum(axis=1)

    def calc_score_fast(A):
        """Calculate max degree sum over all triangles using matrix ops"""
        degs = A.sum(axis=1)  # shape (n,)
        # Triangles: A @ A @ A diagonal gives 2 * (number of triangles through each vertex)
        # Actually, (A^3)[i,i] = 2 * (number of triangles containing vertex i)
        # But we need the actual triangles to compute degree sums
        # 
        # For each triangle (u,v,w), score = degs[u] + degs[v] + degs[w]
        # We can enumerate triangles more efficiently:
        # For each edge (u,v), common neighbors form triangles
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    # Common neighbors of u and v
                    common = np.where((A[u] & A[v]) > 0)[0]
                    for w in common:
                        if w > v:  # avoid duplicates
                            s = degs[u] + degs[v] + degs[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    def calc_score_adj(adj, degs):
        """Calculate score from bitmask adjacency and degree list"""
        max_sum = 0
        for u in range(n):
            if not adj[u]:
                continue
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        if w > v:
                            s = degs[u] + degs[v] + degs[w]
                            if s > max_sum:
                                max_sum = s
                        common ^= lsb
        return max_sum

    def generate_baseline():
        """K_{8,8} plus edge (0,1)"""
        A = np.zeros((n, n), dtype=np.int8)
        for u in range(8):
            for v in range(8, 16):
                A[u, v] = 1
                A[v, u] = 1
        A[0, 1] = 1
        A[1, 0] = 1
        return A

    def get_edges(A):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    edges.append([u, v])
        return edges

    def random_swap(A):
        """Pick random edge to remove and random non-edge to add"""
        # Find all edges and non-edges
        edge_list = []
        nonedge_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    edge_list.append((u, v))
                else:
                    nonedge_list.append((u, v))
        if not edge_list or not nonedge_list:
            return None, None
        e_rm = random.choice(edge_list)
        e_add = random.choice(nonedge_list)
        return e_rm, e_add

    def apply_swap(A, e_rm, e_add):
        """Apply swap to adjacency matrix"""
        u1, v1 = e_rm
        u2, v2 = e_add
        A[u1, v1] = 0
        A[v1, u1] = 0
        A[u2, v2] = 1
        A[v2, u2] = 1

    def revert_swap(A, e_rm, e_add):
        """Revert swap"""
        u1, v1 = e_rm
        u2, v2 = e_add
        A[u1, v1] = 1
        A[v1, u1] = 1
        A[u2, v2] = 0
        A[v2, u2] = 0

    # Start with baseline
    A_best = generate_baseline()
    score_best = calc_score_fast(A_best)

    A_current = A_best.copy()
    score_current = score_best

    # Simulated annealing with focus on high-degree-sum triangles
    initial_temp = 3.0
    cooling_rate = 0.9995
    min_temp = 0.05

    step = 0
    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline:
        step += 1

        if no_improve > max_no_improve:
            # Restart with perturbation
            A_current = A_best.copy()
            score_current = score_best
            no_improve = 0
            # Apply several random swaps
            n_kicks = random.randint(3, 10)
            for _ in range(n_kicks):
                if time.time() > deadline:
                    break
                e_rm, e_add = random_swap(A_current)
                if e_rm is None:
                    break
                apply_swap(A_current, e_rm, e_add)
            score_current = calc_score_fast(A_current)
            initial_temp = max(initial_temp * 0.7, 1.0)
            continue

        e_rm, e_add = random_swap(A_current)
        if e_rm is None:
            continue

        apply_swap(A_current, e_rm, e_add)
        new_score = calc_score_fast(A_current)

        delta = new_score - score_current
        temp = max(initial_temp * (cooling_rate ** step), min_temp)

        if delta <= 0:
            score_current = new_score
            no_improve = 0
            if new_score < score_best:
                score_best = new_score
                A_best = A_current.copy()
        else:
            if random.random() < math.exp(-delta / temp):
                score_current = new_score
                no_improve = 0
            else:
                revert_swap(A_current, e_rm, e_add)
                no_improve += 1

    return {'n': n, 'edges': get_edges(A_best)}
