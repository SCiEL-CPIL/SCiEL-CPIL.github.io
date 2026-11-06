import numpy as np
import time
import random
import math

def count_c4(adj):
    """
    Count the number of 4-cycles in the graph represented by adjacency matrix adj.
    Uses the formula: C4 = sum_{i<j} C(common_neighbors(i,j), 2)
    """
    n = adj.shape[0]
    A2 = adj @ adj
    # For each pair (i,j), the number of common neighbors is A2[i,j]
    # Each C4 corresponds to choosing 2 common neighbors for 2 opposite vertices
    # C4 count = (1/4) * sum_{i != j} C(A2[i,j], 2) but we need to be careful
    # Actually: each C4 is counted 4 times (once for each pair of opposite vertices)
    # So C4 = (1/4) * sum_{i<j} C(A2[i,j], 2) * 2 ... let me think again
    # A C4 has 4 vertices a,b,c,d in cycle. Opposite pairs are (a,c) and (b,d).
    # For opposite pair (a,c), common neighbors are b and d, so C(common,2) = 1.
    # Each C4 contributes 1 to C(A2[a,c],2) and 1 to C(A2[b,d],2).
    # So sum_{i<j} C(A2[i,j],2) counts each C4 exactly 2 times.
    # Therefore C4 = sum_{i<j} C(A2[i,j],2) / 2
    vals = np.triu(A2, k=1).astype(np.int64)
    return int(np.sum(vals * (vals - 1)) // 2)

def count_c4s_containing_edge(adj, u, v):
    """
    Count the number of C4s that contain the edge (u, v).
    A C4 containing (u,v) has form u-a-b-v where a in N(u)\{v}, b in N(v)\{u}, (a,b) in E, a != b.
    """
    nu = np.where(adj[u] == 1)[0]
    nv = np.where(adj[v] == 1)[0]
    count = 0
    if len(nu) < len(nv):
        for a in nu:
            if a == v:
                continue
            count += np.sum(adj[a, nv])
    else:
        for b in nv:
            if b == u:
                continue
            count += np.sum(adj[b, nu])
    return int(count)

def count_new_c4s_for_edge(adj, u, v):
    """
    Count the number of new C4s created by adding edge (u, v) to the graph.
    This equals the number of length-3 paths from u to v that don't use edge (u,v).
    """
    nu = np.where(adj[u] == 1)[0]
    nv = np.where(adj[v] == 1)[0]
    count = 0
    if len(nu) < len(nv):
        for a in nu:
            if a == v:
                continue
            count += np.sum(adj[a, nv])
    else:
        for b in nv:
            if b == u:
                continue
            count += np.sum(adj[b, nu])
    return int(count)

def compute_edge_c4_counts(adj, edge_set):
    """
    Compute the number of C4s containing each edge in the graph.
    Returns a dict mapping edge -> count.
    """
    n = adj.shape[0]
    edge_c4s = {}
    for (u, v) in edge_set:
        edge_c4s[(u, v)] = count_c4s_containing_edge(adj, u, v)
    return edge_c4s

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 140  # 10s margin from 150s limit

    # 1. Construct the base graph: Orthogonal Polarity Graph of PG(2,5)
    # This gives a graph with 93 edges and high symmetry
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))

    adj = np.zeros((n, n), dtype=np.int8)
    edge_set = set()

    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edge_set.add((i, j))

    current_edges = len(edge_set)
    edges_to_remove = current_edges - m

    # Greedily remove edges that are in the most C4s
    if edges_to_remove > 0:
        for _ in range(edges_to_remove):
            if time.time() > deadline:
                break
            # Compute C4 count for each edge
            edge_c4_counts = {}
            for (u, v) in edge_set:
                edge_c4_counts[(u, v)] = count_c4s_containing_edge(adj, u, v)
            
            # Find the edge with the most C4s
            max_c4s = -1
            worst_edge = None
            for e, c in edge_c4_counts.items():
                if c > max_c4s:
                    max_c4s = c
                    worst_edge = e
            
            if worst_edge is not None:
                u, v = worst_edge
                adj[u, v] = 0
                adj[v, u] = 0
                edge_set.remove(worst_edge)

    # If we still need to add edges (shouldn't happen with polarity graph having 93 edges)
    current_edges = len(edge_set)
    edges_to_add = m - current_edges

    if edges_to_add > 0:
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        added = 0
        while added < edges_to_add and time.time() < deadline:
            min_c4s = float('inf')
            best_edge = None
            for u, v in all_possible_edges:
                if (u, v) in edge_set:
                    continue
                c = count_new_c4s_for_edge(adj, u, v)
                if c < min_c4s:
                    min_c4s = c
                    best_edge = (u, v)
            if best_edge:
                u, v = best_edge
                adj[u, v] = 1
                adj[v, u] = 1
                edge_set.add(best_edge)
                added += 1
            else:
                break

    # Phase 1: Exhaustive 1-edge-swap local search with proper delta computation
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]

        # Precompute C4 counts for all edges
        edge_c4s = {}
        for e in current_edges_list:
            edge_c4s[e] = count_c4s_containing_edge(adj, e[0], e[1])

        for e_rm in current_edges_list:
            if time.time() > deadline:
                break
            u1, v1 = e_rm
            c4s_rm = edge_c4s[e_rm]
            
            for e_add in non_edges:
                u2, v2 = e_add
                # Compute the true delta: C4s added by e_add minus C4s removed by e_rm
                # But we need to account for interactions
                # Quick estimate: use the individual counts
                c4s_add = count_new_c4s_for_edge(adj, u2, v2)
                delta = c4s_add - c4s_rm
                
                if delta < 0:
                    # Perform the swap
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1

                    new_c4 = count_c4(adj)

                    if new_c4 < best_c4:
                        best_c4 = new_c4
                        best_adj = adj.copy()
                        best_edge_set = set(edge_set)
                        edge_set.remove(e_rm)
                        edge_set.add(e_add)
                        current_c4 = new_c4
                        improved = True
                        adj = best_adj.copy()
                        break
                    else:
                        # Revert
                        adj[u1, v1] = 1
                        adj[v1, u1] = 1
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0

    # Phase 2: Simulated annealing with better move generation
    T = 15.0
    T_min = 0.1
    cooling_rate = 0.9995

    current_edges_list = list(edge_set)
    current_c4 = best_c4
    adj = best_adj.copy()
    edge_set = set(best_edge_set)

    # Precompute non-edges for faster sampling
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while T > T_min and time.time() < deadline:
        # Random edge removal
        idx_rm = random.randint(0, len(current_edges_list) - 1)
        e_rm = current_edges_list[idx_rm]
        u1, v1 = e_rm

        # Random edge addition from non-edges
        while True:
            u2, v2 = random.choice(all_pairs)
            if (u2, v2) not in edge_set:
                break

        # Perform swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1

        new_c4 = count_c4(adj)
        delta = new_c4 - current_c4

        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current_c4 = new_c4
            edge_set.remove(e_rm)
            edge_set.add((u2, v2))
            current_edges_list[idx_rm] = (u2, v2)

            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
        else:
            # Revert
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0

        T *= cooling_rate

    # Phase 3: Final intensive local search
    adj = best_adj.copy()
    edge_set = set(best_edge_set)
    current_c4 = best_c4
    
    # Multiple rounds of local search
    for round_num in range(10):
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_edges_list = list(edge_set)
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]

            # Precompute C4 counts for all edges
            edge_c4s = {}
            for e in current_edges_list:
                edge_c4s[e] = count_c4s_containing_edge(adj, e[0], e[1])

            for e_rm in current_edges_list:
                if time.time() > deadline:
                    break
                u1, v1 = e_rm
                c4s_rm = edge_c4s[e_rm]
                
                for e_add in non_edges:
                    u2, v2 = e_add
                    c4s_add = count_new_c4s_for_edge(adj, u2, v2)
                    delta = c4s_add - c4s_rm
                    
                    if delta < 0:
                        adj[u1, v1] = 0
                        adj[v1, u1] = 0
                        adj[u2, v2] = 1
                        adj[v2, u2] = 1

                        new_c4 = count_c4(adj)

                        if new_c4 < best_c4:
                            best_c4 = new_c4
                            best_adj = adj.copy()
                            best_edge_set = set(edge_set)
                            edge_set.remove(e_rm)
                            edge_set.add(e_add)
                            current_c4 = new_c4
                            improved = True
                            adj = best_adj.copy()
                            break
                        else:
                            adj[u1, v1] = 1
                            adj[v1, u1] = 1
                            adj[u2, v2] = 0
                            adj[v2, u2] = 0

    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
