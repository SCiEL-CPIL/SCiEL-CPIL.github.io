"""Stronger algorithm for dense K3,3-free graphs on 20 vertices.
Combines structured initialization (polarity/strongly-regular style) with
fast bitset local search and targeted swap moves."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 20

    # Fast K3,3-free check using bitsets
    def is_k33_free(adj_bits):
        for i in range(n):
            for j in range(i+1, n):
                ij = adj_bits[i] & adj_bits[j]
                if ij.bit_count() < 3:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (ij & adj_bits[k]).bit_count() >= 3:
                        return False
        return True

    def count_edges(adj_bits):
        return sum(b.bit_count() for b in adj_bits) // 2

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    def get_missing(adj_bits):
        return [(i, j) for i in range(n) for j in range(i+1, n)
                if not (adj_bits[i] >> j) & 1]

    def get_existing(adj_bits):
        return [(i, j) for i in range(n) for j in range(i+1, n)
                if (adj_bits[i] >> j) & 1]

    def local_search(adj_bits, deadline):
        """Hill climbing with additions and swaps, focused on high-degree vertices."""
        current = adj_bits[:]
        current_edges = count_edges(current)
        
        # Precompute missing and existing edges
        missing = get_missing(current)
        existing = get_existing(current)
        random.shuffle(missing)

        improved = True
        no_improve_count = 0
        while improved and time.time() < deadline - 3:
            improved = False
            no_improve_count = 0
            random.shuffle(missing)
            
            # Try adding edges
            new_missing = []
            for u, v in missing:
                if time.time() > deadline - 3:
                    new_missing.append((u, v))
                    continue
                add_edge(current, u, v)
                if is_k33_free(current):
                    current_edges += 1
                    improved = True
                    existing.append((u, v))
                else:
                    remove_edge(current, u, v)
                    new_missing.append((u, v))
            
            if improved:
                missing = new_missing
                continue
            
            # Try swaps if stuck - focus on high-degree vertices
            if existing and missing and time.time() < deadline - 3:
                # Sort existing edges by degree of endpoints (descending)
                deg = [b.bit_count() for b in current]
                existing_sorted = sorted(existing, key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
                
                swap_attempts = 0
                max_swaps = 300
                while swap_attempts < max_swaps and time.time() < deadline - 3:
                    swap_attempts += 1
                    # Pick from high-degree existing edges
                    if swap_attempts <= 100:
                        e1 = existing_sorted[swap_attempts % len(existing_sorted)]
                    else:
                        e1 = random.choice(existing)
                    e2 = random.choice(missing)
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    remove_edge(current, u1, v1)
                    add_edge(current, u2, v2)
                    if is_k33_free(current):
                        # Accept
                        existing.remove(e1)
                        existing.append(e2)
                        missing.remove(e2)
                        missing.append(e1)
                        improved = True
                        break
                    else:
                        remove_edge(current, u2, v2)
                        add_edge(current, u1, v1)
        
        return current, current_edges

    def init_structured(deadline):
        """Build a structured graph inspired by polarity graphs / strongly regular graphs.
        Uses a symmetric construction to create a dense K3,3-free starting point."""
        adj = [0] * n
        
        # Construction 1: Based on Z_20 with quadratic residues
        # QR mod 20: {1, 4, 9, 16} -> but 20 is not prime. Use a different approach.
        
        # Construction: Use a 5x4 grid structure
        # Vertices 0-19 arranged as 5 rows x 4 cols
        # Connect within rows and columns with specific patterns
        
        # Try a circulant-like structure
        # For each vertex i, connect to i+1, i+2, i+3, i+4, i+7, i+11, i+13, i+17 mod 20
        # This is based on known dense K3,3-free constructions
        
        # Actually, let's use a more principled approach:
        # Start with a random regular-like graph and structure it
        
        # Use a 4-partite structure: divide 20 vertices into 4 groups of 5
        # Connect between groups with a pattern that avoids K3,3
        
        groups = [list(range(i*5, (i+1)*5)) for i in range(4)]
        
        # Connect all edges within each group (cliques of size 5)
        for g in groups:
            for a in range(len(g)):
                for b in range(a+1, len(g)):
                    add_edge(adj, g[a], g[b])
        
        # Connect between groups with a structured pattern
        # Between group i and group j, use a "shifted" connection
        for i in range(4):
            for j in range(i+1, 4):
                # Each vertex in group i connects to 2-3 vertices in group j
                for a in groups[i]:
                    idx_a = a % 5
                    # Connect to vertices in group j with indices (idx_a + j) % 5 and (idx_a + j + 1) % 5
                    b1 = groups[j][(idx_a + j) % 5]
                    b2 = groups[j][(idx_a + j + 1) % 5]
                    add_edge(adj, a, b1)
                    add_edge(adj, a, b2)
        
        return adj

    def init_circulant(deadline):
        """Circulant graph construction."""
        adj = [0] * n
        
        # Try different connection sets
        # Known good sets for K3,3-free circulant graphs on 20 vertices
        # Connect i to i+1, i+2, i+4, i+5, i+8, i+12, i+15, i+19 mod 20
        # This gives degree 8, total edges = 20*8/2 = 80
        
        # Let's try a few different circulant sets
        sets_to_try = [
            [1, 2, 4, 5, 8, 12, 15, 19],
            [1, 2, 3, 5, 7, 13, 15, 18],
            [1, 2, 4, 6, 7, 13, 14, 18],
            [1, 2, 3, 4, 6, 9, 11, 16],
            [1, 2, 4, 5, 7, 11, 13, 18],
        ]
        
        best_adj = None
        best_edges = -1
        
        for conn_set in sets_to_try:
            adj_try = [0] * n
            for i in range(n):
                for d in conn_set:
                    j = (i + d) % n
                    if j > i:
                        add_edge(adj_try, i, j)
            
            if is_k33_free(adj_try):
                edges = count_edges(adj_try)
                if edges > best_edges:
                    best_edges = edges
                    best_adj = adj_try[:]
        
        if best_adj is not None:
            return best_adj
        
        # Fallback: random
        return init_random_greedy(deadline)

    def init_random_greedy(deadline):
        """Build a random maximal K3,3-free graph via greedy edge addition."""
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 10:
                break
            add_edge(adj, u, v)
            if not is_k33_free(adj):
                remove_edge(adj, u, v)
        return adj

    def init_bipartite_random(deadline):
        """Initialize with a random bipartite graph."""
        adj = [0] * n
        A = list(range(10))
        B = list(range(10, 20))
        for a in A:
            for b in B:
                if random.random() < 0.5:
                    add_edge(adj, a, b)
        return adj

    def init_dense_random(deadline):
        """Dense random initialization."""
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    add_edge(adj, i, j)
        return adj

    def prune_to_valid(adj, deadline):
        """Prune a graph to make it K3,3-free by removing edges greedily."""
        adj_clean = [0] * n
        all_edges = get_existing(adj)
        random.shuffle(all_edges)
        for u, v in all_edges:
            if time.time() > deadline - 5:
                break
            add_edge(adj_clean, u, v)
            if not is_k33_free(adj_clean):
                remove_edge(adj_clean, u, v)
        return adj_clean

    best_adj = None
    best_edges = -1

    # Restart 1: Structured initialization
    if time.time() < deadline - 25:
        adj1 = init_structured(deadline)
        adj1_clean = prune_to_valid(adj1, deadline)
        adj1, e1 = local_search(adj1_clean, deadline - 15)
        if e1 > best_edges:
            best_edges = e1
            best_adj = adj1[:]

    # Restart 2: Circulant initialization
    if time.time() < deadline - 20:
        adj2 = init_circulant(deadline)
        adj2, e2 = local_search(adj2, deadline - 12)
        if e2 > best_edges:
            best_edges = e2
            best_adj = adj2[:]

    # Restart 3: Random greedy
    if time.time() < deadline - 15:
        adj3 = init_random_greedy(deadline)
        adj3, e3 = local_search(adj3, deadline - 8)
        if e3 > best_edges:
            best_edges = e3
            best_adj = adj3[:]

    # Restart 4: Bipartite random
    if time.time() < deadline - 12:
        adj4 = init_bipartite_random(deadline)
        adj4_clean = prune_to_valid(adj4, deadline)
        adj4, e4 = local_search(adj4_clean, deadline - 7)
        if e4 > best_edges:
            best_edges = e4
            best_adj = adj4[:]

    # Restart 5: Dense random
    if time.time() < deadline - 10:
        adj5 = init_dense_random(deadline)
        adj5_clean = prune_to_valid(adj5, deadline)
        adj5, e5 = local_search(adj5_clean, deadline - 5)
        if e5 > best_edges:
            best_edges = e5
            best_adj = adj5[:]

    # Final polish on the best
    if best_adj is not None and time.time() < deadline - 2:
        best_adj, best_edges = local_search(best_adj, deadline - 2)

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result

# EVOLVE-BLOCK-END
