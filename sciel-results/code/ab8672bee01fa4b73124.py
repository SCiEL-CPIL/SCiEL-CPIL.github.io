import time
import random
import itertools
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Fast SA for n=17 with structured initialization
    n17 = 17
    r17 = _sa_search_structured(17, deadline - 50.0, 0)
    
    if r17 is None:
        r17 = _sa_search(17, deadline - 45.0, 1)
    
    if r17 is None:
        n_small = 9
        colors_small = [min(j - i, n_small - j + i) - 1 for i in range(n_small) for j in range(i + 1, n_small)]
        return {'n': n_small, 'colors': colors_small}
    
    best = r17
    
    # Try to extend to n=18
    if time.time() < deadline - 10.0:
        r18 = _backtrack_extend_bit(17, r17['colors'], deadline - 5.0)
        if r18 is not None:
            return r18
    
    return best


def _sa_search_structured(n, deadline, seed=0):
    """SA with structured initialization for faster convergence."""
    random.seed(seed)
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1
    num_edges = n * (n - 1) // 2
    
    cycles = []
    edge_to_cycles = {e: [] for e in range(num_edges)}
    for subset in itertools.combinations(range(n), 6):
        seen = set()
        for perm in itertools.permutations(subset):
            mn = min(perm)
            mi = perm.index(mn)
            rot = perm[mi:] + perm[:mi]
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            if rot in seen:
                continue
            seen.add(rot)
            edges = []
            for k in range(6):
                u, v = rot[k], rot[(k + 1) % 6]
                a, b = min(u, v), max(u, v)
                edges.append(edge_map[(a, b)])
            cid = len(cycles)
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(cid)
    
    if not cycles:
        return {'n': n, 'colors': [0] * num_edges}
    
    # Structured initialization using Z_n pattern
    color_of_diff = {}
    if n == 17:
        powers = [1, 3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6]
        for k, d in enumerate(powers):
            c = k // 4
            color_of_diff[d] = c
            color_of_diff[17 - d] = c
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d in color_of_diff:
                colors.append(color_of_diff[d])
            else:
                colors.append(random.randint(0, 3))
    
    cycle_counts = [[0, 0, 0, 0] for _ in range(len(cycles))]
    total_viol = 0
    for c_id in range(len(cycles)):
        for e in cycles[c_id]:
            cycle_counts[c_id][colors[e]] += 1
        for c in range(4):
            if cycle_counts[c_id][c] == 6:
                total_viol += 1
                break
    
    if total_viol == 0:
        return {'n': n, 'colors': colors}
    
    T = 1.5
    cooling = 0.9997
    min_T = 0.03
    
    while time.time() < deadline:
        if total_viol == 0:
            return {'n': n, 'colors': colors}
        e = random.randint(0, num_edges - 1)
        oc = colors[e]
        nc = random.randint(0, 3)
        if nc == oc:
            continue
        delta = 0
        for c_id in edge_to_cycles[e]:
            cc = cycle_counts[c_id]
            old_v = 1 if cc[oc] == 6 else 0
            cc[oc] -= 1
            cc[nc] += 1
            new_v = 1 if cc[nc] == 6 else 0
            delta += new_v - old_v
        if delta < 0 or random.random() < 2.718281828 ** (-delta / T):
            colors[e] = nc
            total_viol += delta
        else:
            for c_id in edge_to_cycles[e]:
                cc = cycle_counts[c_id]
                cc[oc] += 1
                cc[nc] -= 1
        T = max(min_T, T * cooling)
    
    if total_viol == 0:
        return {'n': n, 'colors': colors}
    return None


def _sa_search(n, deadline, seed=0):
    """Simulated annealing to find a C6-free 4-coloring of K_n."""
    random.seed(seed)
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1
    num_edges = n * (n - 1) // 2
    
    cycles = []
    edge_to_cycles = {e: [] for e in range(num_edges)}
    for subset in itertools.combinations(range(n), 6):
        seen = set()
        for perm in itertools.permutations(subset):
            mn = min(perm)
            mi = perm.index(mn)
            rot = perm[mi:] + perm[:mi]
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            if rot in seen:
                continue
            seen.add(rot)
            edges = []
            for k in range(6):
                u, v = rot[k], rot[(k + 1) % 6]
                a, b = min(u, v), max(u, v)
                edges.append(edge_map[(a, b)])
            cid = len(cycles)
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(cid)
    
    if not cycles:
        return {'n': n, 'colors': [0] * num_edges}
    
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    cycle_counts = [[0, 0, 0, 0] for _ in range(len(cycles))]
    total_viol = 0
    for c_id in range(len(cycles)):
        for e in cycles[c_id]:
            cycle_counts[c_id][colors[e]] += 1
        for c in range(4):
            if cycle_counts[c_id][c] == 6:
                total_viol += 1
                break
    
    T = 2.0
    cooling = 0.9995
    min_T = 0.05
    
    while time.time() < deadline:
        if total_viol == 0:
            return {'n': n, 'colors': colors}
        e = random.randint(0, num_edges - 1)
        oc = colors[e]
        nc = random.randint(0, 3)
        if nc == oc:
            continue
        delta = 0
        for c_id in edge_to_cycles[e]:
            cc = cycle_counts[c_id]
            old_v = 1 if cc[oc] == 6 else 0
            cc[oc] -= 1
            cc[nc] += 1
            new_v = 1 if cc[nc] == 6 else 0
            delta += new_v - old_v
        if delta < 0 or random.random() < 2.718281828 ** (-delta / T):
            colors[e] = nc
            total_viol += delta
        else:
            for c_id in edge_to_cycles[e]:
                cc = cycle_counts[c_id]
                cc[oc] += 1
                cc[nc] -= 1
        T = max(min_T, T * cooling)
    
    if total_viol == 0:
        return {'n': n, 'colors': colors}
    return None


def _backtrack_extend_bit(n_old, old_colors, deadline):
    """Bit-parallel backtracking to extend coloring to n_old+1."""
    n_new = n_old + 1
    n_old_edges = n_old * (n_old - 1) // 2
    n_new_edges = n_new * (n_new - 1) // 2
    
    edge_map = {}
    idx = 0
    for i in range(n_new):
        for j in range(i + 1, n_new):
            edge_map[(i, j)] = idx
            idx += 1
    
    new_edge_indices = [edge_map[(j, n_new - 1)] for j in range(n_old)]
    new_edge_set = set(new_edge_indices)
    
    cycles = []
    edge_to_cycles = {e: [] for e in range(n_new_edges)}
    for subset in itertools.combinations(range(n_new), 6):
        seen = set()
        for perm in itertools.permutations(subset):
            mn = min(perm)
            mi = perm.index(mn)
            rot = perm[mi:] + perm[:mi]
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            if rot in seen:
                continue
            seen.add(rot)
            edges = []
            for k in range(6):
                u, v = rot[k], rot[(k + 1) % 6]
                a, b = min(u, v), max(u, v)
                edges.append(edge_map[(a, b)])
            cid = len(cycles)
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(cid)
    
    colors = list(old_colors)
    for e in new_edge_indices:
        colors[e] = -1
    
    # Order by constraint count
    edge_cc = {}
    for e in new_edge_indices:
        cnt = 0
        for c_id in edge_to_cycles[e]:
            ne = sum(1 for ee in cycles[c_id] if ee in new_edge_set)
            if ne >= 2:
                cnt += 1
        edge_cc[e] = cnt
    ordered = sorted(new_edge_indices, key=lambda e: -edge_cc[e])
    
    def check_conflict(e, c):
        for c_id in edge_to_cycles[e]:
            cyc = cycles[c_id]
            for edge in cyc:
                if edge == e:
                    continue
                if colors[edge] != c:
                    break
            else:
                return True
        return False
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == len(ordered):
            return True
        e = ordered[idx]
        for c in range(4):
            if not check_conflict(e, c):
                colors[e] = c
                if backtrack(idx + 1):
                    return True
                colors[e] = -1
        return False
    
    if backtrack(0):
        return {'n': n_new, 'colors': colors}
    return None


# EVOLVE-BLOCK-END
