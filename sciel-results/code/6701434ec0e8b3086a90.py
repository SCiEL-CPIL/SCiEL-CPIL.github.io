"""Rewritten program: structured local-search repair for C4-free graphs."""
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # For n vertices, we need min_degree >= n - 39
    # So for n=47: min_deg >= 8, for n=46: min_deg >= 7, etc.

    best_result = {'n': 39, 'edges': []}
    best_n = 39

    for n in range(47, 38, -1):
        if time.time() > deadline:
            break

        min_deg_needed = n - 39
        if min_deg_needed <= 0:
            # Trivial: empty graph works
            best_result = {'n': n, 'edges': []}
            best_n = n
            continue

        success = False
        max_attempts = 200

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            random.seed(attempt * 7919 + n * 31)

            # Build adjacency as list of sets for fast common-neighbor checks
            adj = [set() for _ in range(n)]
            edges = []

            # Strategy: Start with a random regular-ish graph, then repair
            # 1. First, try to build a base graph with sufficient degree
            # 2. Remove edges that create C4
            # 3. Greedily add edges to restore degree while maintaining C4-free

            # Step 1: Build initial graph - try to get close to min_deg_needed
            # Use a greedy approach with random ordering
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)

            # Add edges greedily, checking C4 constraint
            for (u, v) in all_edges:
                if time.time() > deadline:
                    break
                if v in adj[u]:
                    continue
                # Check if adding (u,v) creates a C4
                # This happens if u and v have a common neighbor
                common = adj[u] & adj[v]
                if len(common) == 0:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])

            # Check current min degree
            min_deg = min(len(adj[i]) for i in range(n))

            # If we already meet the requirement, we're done
            if min_deg >= min_deg_needed:
                success = True
                best_result = {'n': n, 'edges': edges}
                best_n = n
                break

            # Otherwise, try local search: add more edges to deficient vertices
            # Identify vertices with degree < min_deg_needed
            deficient = [i for i in range(n) if len(adj[i]) < min_deg_needed]

            # Try to add edges to deficient vertices
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Sort deficient vertices by current degree (most deficient first)
                deficient.sort(key=lambda x: len(adj[x]))

                for u in deficient:
                    if time.time() > deadline:
                        break
                    if len(adj[u]) >= min_deg_needed:
                        continue

                    # Find potential neighbors for u
                    candidates = [v for v in range(n) if v not in adj[u] and v != u]
                    random.shuffle(candidates)

                    for v in candidates:
                        if len(adj[u]) >= min_deg_needed:
                            break
                        # Check if adding (u,v) creates a C4
                        common = adj[u] & adj[v]
                        if len(common) == 0:
                            adj[u].add(v)
                            adj[v].add(u)
                            edges.append([u, v])
                            improved = True
                            break

                # Update deficient list
                deficient = [i for i in range(n) if len(adj[i]) < min_deg_needed]

            # Final check
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_needed:
                success = True
                best_result = {'n': n, 'edges': edges}
                best_n = n
                break

        if success:
            break

    # Verify the result
    if best_n > 39:
        n = best_result['n']
        edges = best_result['edges']
        # Quick verification: check no C4 and min degree
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Check min degree
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg < n - 39:
            best_result = {'n': 39, 'edges': []}
            return best_result

        # Check C4: C4 exists iff there exist two vertices with >= 2 common neighbors
        has_c4 = False
        for u in range(n):
            for v in range(u+1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    has_c4 = True
                    break
            if has_c4:
                break

        if has_c4:
            best_result = {'n': 39, 'edges': []}

    return best_result
