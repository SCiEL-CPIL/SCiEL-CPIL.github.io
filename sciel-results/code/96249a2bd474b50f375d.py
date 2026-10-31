import time
import random
import sys
import os
import numpy as np
from collections import deque, defaultdict

def construct(k, g):
    deadline = time.time() + 110

    if k != 3 or g != 16:
        return _fallback_construct(k, g, deadline)

    # For k=3, g=16: Use voltage lift of the Petersen graph (10 vertices)
    # with modulus M. Total vertices = 10*M.
    # Need 10*M >= 512 (Moore bound), so M >= 52.

    # Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5.
    # Outer pentagon: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9

    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    n_base = 10
    m_edges = len(base_edges)
    edge_index = {e: i for i, e in enumerate(base_edges)}

    # Enumerate all simple cycles
    def enumerate_simple_cycles(edges, n_vertices):
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        cycles = set()

        def dfs(start, current, visited, path):
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 3:
                    cycle = tuple(path)
                    min_idx = min(range(len(cycle)), key=lambda i: cycle[i])
                    normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
                    cycles.add(normalized)
                elif neighbor not in visited and neighbor > start:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)

        for v in range(n_vertices):
            dfs(v, v, {v}, [v])

        return [list(c) for c in cycles]

    all_cycles = enumerate_simple_cycles(base_edges, n_base)

    def cycle_voltage_sum(cycle, voltages):
        total = 0
        L = len(cycle)
        for i in range(L):
            u = cycle[i]
            v = cycle[(i + 1) % L]
            if (u, v) in edge_index:
                total += voltages[edge_index[(u, v)]]
            elif (v, u) in edge_index:
                total -= voltages[edge_index[(v, u)]]
            else:
                raise ValueError(f"Edge ({u},{v}) not found")
        return total

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    best_result = None

    # Try prime moduli
    min_M = (512 + n_base - 1) // n_base  # ceil(512/10) = 52
    if min_M % 2 == 0:
        min_M += 1

    for M in range(max(min_M, 5), 300):
        if time.time() > deadline - 20:
            break

        if not is_prime(M):
            continue

        # For prime M: need all cycle voltages != 0 mod M
        max_tries = 1000
        found = None
        for _ in range(max_tries):
            voltages = [random.randint(1, M - 1) for _ in range(m_edges)]
            valid = True
            for cycle in all_cycles:
                S = cycle_voltage_sum(cycle, voltages) % M
                if S == 0:
                    valid = False
                    break
            if valid:
                found = voltages
                break

        if found is None:
            continue

        # Build the lifted graph
        n_lifted = n_base * M
        edges = []
        for (u, v), eidx in edge_index.items():
            c = found[eidx]
            for a in range(M):
                u_node = u * M + a
                v_node = v * M + ((a + c) % M)
                edges.append((u_node, v_node))

        if check_girth_fast(n_lifted, edges, g):
            best_result = (n_lifted, edges)
            break

    if best_result is not None:
        return best_result

    # Try composite moduli
    for M in range(max(min_M, 5), 300):
        if time.time() > deadline - 10:
            break
        if is_prime(M):
            continue

        max_tries = 500
        found = None
        for _ in range(max_tries):
            voltages = [random.randint(0, M - 1) for _ in range(m_edges)]
            valid = True
            for cycle in all_cycles:
                S = cycle_voltage_sum(cycle, voltages) % M
                if S == 0:
                    valid = False
                    break
                L = len(cycle)
                # Cycle length in lift: L * M / gcd(S, M)
                g_s = _gcd(S, M)
                lift_len = L * M // g_s
                if lift_len < g:
                    valid = False
                    break
            if valid:
                found = voltages
                break

        if found is None:
            continue

        n_lifted = n_base * M
        edges = []
        for (u, v), eidx in edge_index.items():
            c = found[eidx]
            for a in range(M):
                u_node = u * M + a
                v_node = v * M + ((a + c) % M)
                edges.append((u_node, v_node))

        if check_girth_fast(n_lifted, edges, g):
            best_result = (n_lifted, edges)
            break

    if best_result is not None:
        return best_result

    return 0, []

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def check_girth_fast(n, edges, g):
    if n == 0:
        return True

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    max_depth = g - 1

    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _fallback_construct(k, g, deadline):
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    return 0, []
