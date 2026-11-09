"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    n = 20
    
    # Represent graph as list of integers where bit i is set if edge (v, i) exists
    # This allows fast bitwise AND for common neighbors
    
    def is_k33_free(adj_bits):
        """
        Check if graph is K3,3-free.
        K3,3 exists iff there are 3 vertices with >= 3 common neighbors.
        Using bitsets: for each triple (i, j, k), check popcount(adj[i] & adj[j] & adj[k]) >= 3
        """
        # Optimization: Pre-compute pairwise intersections? 
        # For n=20, triples are 1140. 
        # popcount is fast in Python 3.10+ (int.bit_count())
        
        for i in range(n):
            for j in range(i+1, n):
                ij = adj_bits[i] & adj_bits[j]
                # If ij has < 2 bits, no triple involving i,j can have 3 common neighbors
                # Actually we need common neighbors of i, j, k.
                # Let C_ij be common neighbors of i and j.
                # We need to know if there exists k such that |C_ij & N(k)| >= 3.
                # This is equivalent to: does there exist k such that k has at least 3 neighbors in C_ij?
                
                # Let's stick to the simple triple loop first, it's small enough.
                # But we can prune: if popcount(ij) < 3, skip inner loop for k? No, k is part of the triple.
                # Wait, the condition is: exists i, j, k such that |N(i) ∩ N(j) ∩ N(k)| >= 3.
                # Let S = N(i) ∩ N(j). We need to check if there is a k such that |S ∩ N(k)| >= 3.
                # Note k cannot be i or j.
                
                # Efficient check for fixed i, j:
                # S = N(i) & N(j)
                # If popcount(S) < 3, then for any k, |S ∩ N(k)| <= popcount(S) < 3. So skip.
                if ij.bit_count() < 3:
                    continue
                
                # Now check all k != i, j
                # We can iterate through bits of S? No, k is any vertex.
                # But if k is not in S, it doesn't matter? No, k is one of the 3 vertices in the partition.
                # The 3 common neighbors are in S.
                # So we just need to check: is there any vertex k (k!=i, k!=j) such that k is connected to at least 3 vertices in S?
                
                # Let's precompute for this pair (i,j)
                # We can just loop k from 0 to n-1
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # Count common neighbors of i, j, k
                    # Common neighbors are in S ∩ N(k)
                    common = (ij & adj_bits[k]).bit_count()
                    if common >= 3:
                        return False
        return True

    def count_edges(adj_bits):
        total = 0
        for i in range(n):
            total += adj_bits[i].bit_count()
        return total // 2

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    # Strategy:
    # 1. Start with a random dense graph or a known structure.
    #    Erdős-Rényi G(20, 0.5) is likely to have K3,3.
    #    We want a MAXIMAL K3,3-free graph.
    #    Let's start with an empty graph and add edges greedily in a random order.
    #    Then do local search (add, remove, swap) to improve.
    
    # Initial construction: Random Greedy
    adj_bits = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline - 5:
            break
        add_edge(adj_bits, u, v)
        if is_k33_free(adj_bits):
            pass # Keep
        else:
            remove_edge(adj_bits, u, v)
            
    best_adj = adj_bits[:]
    best_edges = count_edges(adj_bits)
    
    # Local Search Phase
    # We will try to add edges that are currently missing.
    # And try to swap edges.
    
    current_adj = best_adj[:]
    current_edges = count_edges(current_adj)
    
    # List of missing edges
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] >> j) & 1]
    random.shuffle(missing)
    
    # List of existing edges
    existing = [(i, j) for i in range(n) for j in range(i+1, n) if (current_adj[i] >> j) & 1]
    
    # We will iterate through missing edges and try to add them.
    # If adding fails, we might try to remove an existing edge to make room?
    # Simpler: Just try to add missing edges in random order repeatedly.
    # If stuck, try swapping: remove one existing, add one missing.
    
    # To be fast, we'll use a while loop with time checks.
    
    # Optimization for is_k33_free: 
    # The current is_k33_free is O(n^3 * bit_ops). 
    # n=20, triples=1140. 
    # 1140 * (few ops) is very fast. ~10us per check? 
    # So we can do ~100,000 checks in 1s.
    
    # Let's do a simple hill climbing:
    # 1. Try adding each missing edge. If valid, accept.
    # 2. If no missing edge can be added, try swaps.
    
    progress = True
    while progress and time.time() < deadline - 10:
        progress = False
        # Re-shuffle missing to avoid bias
        random.shuffle(missing)
        
        # Try adding edges
        new_missing = []
        for u, v in missing:
            if time.time() > deadline - 10:
                new_missing.append((u, v))
                continue
            add_edge(current_adj, u, v)
            if is_k33_free(current_adj):
                current_edges += 1
                progress = True
                # Update existing/missing lists
                existing.append((u, v))
                # Remove from missing? We are iterating, so just don't add to new_missing
            else:
                remove_edge(current_adj, u, v)
                new_missing.append((u, v))
        
        if progress:
            # Rebuild missing list
            missing = new_missing
            continue
        
        # If no simple addition worked, try swaps
        # Pick a random existing edge to remove, and a random missing edge to add
        # This is O(n^2) swaps per iteration potentially.
        # Let's limit swaps to a reasonable number per "stuck" phase.
        
        if existing and missing:
            # Try a few random swaps
            swaps_attempted = 0
            while swaps_attempted < 100 and time.time() < deadline - 10:
                e1 = random.choice(existing)
                e2 = random.choice(missing)
                
                # Try swap: remove e1, add e2
                u1, v1 = e1
                u2, v2 = e2
                
                remove_edge(current_adj, u1, v1)
                add_edge(current_adj, u2, v2)
                
                if is_k33_free(current_adj):
                    # Accept swap
                    existing.remove(e1)
                    existing.append(e2)
                    missing.remove(e2)
                    missing.append(e1)
                    progress = True
                    # Continue searching for more swaps
                else:
                    # Revert
                    remove_edge(current_adj, u2, v2)
                    add_edge(current_adj, u1, v1)
                
                swaps_attempted += 1
    
    # Final check: try adding edges again with a fresh shuffle, just in case
    if time.time() < deadline - 5:
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] >> j) & 1]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            add_edge(current_adj, u, v)
            if is_k33_free(current_adj):
                pass
            else:
                remove_edge(current_adj, u, v)

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (current_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result

# EVOLVE-BLOCK-END
