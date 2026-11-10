"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _build_adj_bitset(adj_matrix, n):
    """Build bitset representation from numpy adjacency matrix."""
    adj_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in range(i + 1, n):
            if adj_matrix[i, j]:
                bits |= (1 << j)
        adj_bits[i] = bits
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j]:
                adj_bits[j] |= (1 << i)
    return adj_bits

def _count_six_violations_fast(adj_bits, n, six_masks):
    """Count 6-vertex subsets inducing regular graphs using precomputed masks."""
    count = 0
    for mask in six_masks:
        # Check if induced subgraph on this mask is regular
        first_deg = -1
        regular = True
        # Get vertices in mask
        verts = []
        for v in range(n):
            if mask & (1 << v):
                verts.append(v)
        for u in verts:
            nb = adj_bits[u] & mask
            d = nb.bit_count()
            if first_deg == -1:
                first_deg = d
            elif d != first_deg:
                regular = False
                break
        if regular:
            count += 1
    return count

def _has_regular_induced(adj_bits, n):
    """Check any induced subgraph of size >= 6 that is regular."""
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            first_deg = -1
            regular = True
            for u in subset:
                nb = adj_bits[u] & mask
                d = nb.bit_count()
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    regular = False
                    break
            if regular:
                return True
    return False

def _adj_to_edges(adj_bits, n):
    """Convert bitset adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_bits[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute all 6-subset masks for n=22
    six_masks_22 = [0] * 74613  # C(22,6) = 74613
    idx = 0
    for subset in combinations(range(22), 6):
        m = 0
        for v in subset:
            m |= (1 << v)
        six_masks_22[idx] = m
        idx += 1
    
    # Try n from 22 down to 6
    for n in range(22, 5, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Precompute 6-masks for this n
        six_masks = []
        for subset in combinations(range(n), 6):
            m = 0
            for v in subset:
                m |= (1 << v)
            six_masks.append(m)
        
        best_count = float('inf')
        best_adj = None
        
        num_restarts = 60 if n <= 20 else 30
        max_steps = 400
        
        for restart in range(num_restarts):
            if time.time() > deadline - 3.0:
                break
            
            # Random initial graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Quick check
            if not _has_regular_induced(adj, n):
                return {'n': n, 'edges': _adj_to_edges(adj, n)}
            
            current_count = _count_six_violations_fast(adj, n, six_masks)
            
            # Local search: flip edges to reduce violations
            for step in range(max_steps):
                if time.time() > deadline - 2.0:
                    break
                
                if current_count == 0:
                    if not _has_regular_induced(adj, n):
                        return {'n': n, 'edges': _adj_to_edges(adj, n)}
                    break
                
                # Pick a random edge to flip
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                old = (adj[i] >> j) & 1
                if old:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                
                new_count = _count_six_violations_fast(adj, n, six_masks)
                
                if new_count <= current_count:
                    current_count = new_count
                    if new_count < best_count:
                        best_count = new_count
                        best_adj = adj[:]
                else:
                    # Revert
                    if old:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    else:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
            
            if best_adj is not None and best_count < 5 and time.time() < deadline - 5.0:
                # Continue from best
                adj = best_adj[:]
                current_count = best_count
                for step in range(200):
                    if time.time() > deadline - 2.0:
                        break
                    if current_count == 0:
                        if not _has_regular_induced(adj, n):
                            return {'n': n, 'edges': _adj_to_edges(adj, n)}
                        break
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j:
                        continue
                    old = (adj[i] >> j) & 1
                    if old:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    new_count = _count_six_violations_fast(adj, n, six_masks)
                    if new_count <= current_count:
                        current_count = new_count
                    else:
                        if old:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                        else:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
    
    # Fallback: return a small valid graph
    # Empty graph on 5 vertices is valid (no subset of size >= 6)
    return {'n': 5, 'edges': []}
# EVOLVE-BLOCK-END
