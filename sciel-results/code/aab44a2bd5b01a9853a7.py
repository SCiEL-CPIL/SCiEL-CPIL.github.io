import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    n = 20
    
    # Bitset utilities
    def count_edges(adj):
        return sum(b.bit_count() for b in adj) // 2

    def is_k33_free(adj):
        """
        Check if the graph is K3,3-free.
        A graph is K3,3-free iff no 3 vertices have 3 or more common neighbors.
        """
        # Precompute common neighbors for all pairs to speed up triple checks?
        # Actually, just iterating triples is O(n^3) = 8000.
        # For each triple (u, v, w), check if |N(u) & N(v) & N(w)| >= 3.
        for u in range(n):
            for v in range(u + 1, n):
                uv = adj[u] & adj[v]
                # Optimization: if uv has < 3 bits, no triple (u, v, w) can have 3 common neighbors
                if uv.bit_count() < 3:
                    continue
                for w in range(v + 1, n):
                    if (uv & adj[w]).bit_count() >= 3:
                        return False
        return True

    def find_k33(adj):
        """
        Find a K3,3 subgraph (return the 6 vertices) or None.
        Returns (u, v, w, x, y, z) such that {u,v,w} and {x,y,z} form K3,3.
        """
        for u in range(n):
            for v in range(u + 1, n):
                uv = adj[u] & adj[v]
                if uv.bit_count() < 3:
                    continue
                for w in range(v + 1, n):
                    common = uv & adj[w]
                    if common.bit_count() >= 3:
                        # Extract 3 neighbors from common
                        neighbors = []
                        mask = common
                        while len(neighbors) < 3 and mask:
                            lsb = mask & -mask
                            idx = lsb.bit_length() - 1
                            neighbors.append(idx)
                            mask ^= lsb
                        return (u, v, w, neighbors[0], neighbors[1], neighbors[2])
        return None

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def init_greedy_prune(deadline):
        """
        Start with a dense random graph and prune K3,3s.
        """
        adj = [0] * n
        # Start with p=0.8
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.8:
                    add_edge(adj, i, j)
        
        # Prune
        while time.time() < deadline - 10:
            k33 = find_k33(adj)
            if k33 is None:
                break
            u, v, w, x, y, z = k33
            # Remove a random edge from the K3,3
            # Edges are all pairs between {u,v,w} and {x,y,z}
            edges_to_remove = []
            for a in (u, v, w):
                for b in (x, y, z):
                    if (adj[a] >> b) & 1:
                        edges_to_remove.append((a, b))
            if edges_to_remove:
                rem_u, rem_v = random.choice(edges_to_remove)
                remove_edge(adj, rem_u, rem_v)
            else:
                break # Should not happen
        
        return adj

    def local_search(adj, deadline):
        """
        Hill climbing with 1-opt and 2-opt moves.
        """
        current = adj[:]
        current_edges = count_edges(current)
        
        # Precompute lists of existing and missing edges
        existing = []
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if (current[i] >> j) & 1:
                    existing.append((i, j))
                else:
                    missing.append((i, j))
        
        improved = True
        rounds = 0
        while improved and time.time() < deadline - 5:
            improved = False
            rounds += 1
            
            # 1-Opt: Try adding missing edges
            # Shuffle to avoid bias
            random.shuffle(missing)
            new_missing = []
            for u, v in missing:
                if time.time() > deadline - 5:
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
            
            # 2-Opt: Swap one existing edge for one missing edge
            if not existing or not missing:
                break
                
            # Try a limited number of swaps
            max_swaps = 200
            swap_attempts = 0
            while swap_attempts < max_swaps and time.time() < deadline - 5:
                swap_attempts += 1
                # Pick random existing and missing
                e1 = random.choice(existing)
                e2 = random.choice(missing)
                
                u1, v1 = e1
                u2, v2 = e2
                
                # Perform swap
                remove_edge(current, u1, v1)
                add_edge(current, u2, v2)
                
                if is_k33_free(current):
                    # Update lists
                    existing.remove(e1)
                    existing.append(e2)
                    missing.remove(e2)
                    missing.append(e1)
                    improved = True
                else:
                    # Revert
                    remove_edge(current, u2, v2)
                    add_edge(current, u1, v1)
            
            # 3-Opt (optional, slower): Remove 2 existing, add 2 missing
            if len(existing) >= 2 and len(missing) >= 2 and time.time() < deadline - 5:
                max_3opt = 50
                three_opt_attempts = 0
                while three_opt_attempts < max_3opt and time.time() < deadline - 5:
                    three_opt_attempts += 1
                    e1 = random.choice(existing)
                    # Pick a second distinct existing edge
                    e2 = random.choice([e for e in existing if e != e1])
                    m1 = random.choice(missing)
                    m2 = random.choice([m for m in missing if m != m1])
                    
                    u1, v1 = e1
                    u2, v2 = e2
                    w1, x1 = m1
                    w2, x2 = m2
                    
                    remove_edge(current, u1, v1)
                    remove_edge(current, u2, v2)
                    add_edge(current, w1, x1)
                    add_edge(current, w2, x2)
                    
                    if is_k33_free(current):
                        existing.remove(e1)
                        existing.remove(e2)
                        existing.extend([m1, m2])
                        missing.remove(m1)
                        missing.remove(m2)
                        missing.extend([e1, e2])
                        improved = True
                    else:
                        remove_edge(current, w1, x1)
                        remove_edge(current, w2, x2)
                        add_edge(current, u1, v1)
                        add_edge(current, u2, v2)

        return current, current_edges

    best_adj = None
    best_edges = -1

    # Multiple restarts
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts

    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize
        adj = init_greedy_prune(deadline - 5)
        
        # Local Search
        adj, edges = local_search(adj, deadline - 2)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]

    # Final Polish on the best solution
    if best_adj is not None and time.time() < deadline - 2:
        best_adj, best_edges = local_search(best_adj, deadline - 1)

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result

# EVOLVE-BLOCK-END
