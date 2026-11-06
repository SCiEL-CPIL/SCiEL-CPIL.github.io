"""
Strategy: Simulated Annealing / Random Restart Hill Climbing with Swap Moves.
The current greedy approach gets stuck in local maxima. This approach explores
the space of Q3-free graphs by:
1. Starting from a known good seed (K3,13 or similar).
2. Performing random swaps (remove edge e1, add edge e2) if the result is Q3-free.
3. Accepting moves that increase edge count immediately.
4. Accepting moves that keep edge count same or decrease slightly with a probability
   based on a cooling schedule (Simulated Annealing) to escape local traps.
5. Restarting with random perturbations if stuck.
"""

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    n = 16
    # Q3 adjacency structure
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    
    # Precompute all 8-vertex subsets and map pairs to subsets for fast checking
    all_8sets = list(combinations(range(16), 8))
    pair_to_8sets = {}
    for vs in all_8sets:
        for i in range(8):
            for j in range(i+1, 8):
                pair = (vs[i], vs[j])
                if pair not in pair_to_8sets:
                    pair_to_8sets[pair] = []
                pair_to_8sets[pair].append(vs)

    def contains_q3_fast(vs, adj_bits):
        """Check if 8 vertices vs induce a Q3. adj_bits[v] = bitmask of neighbors."""
        local_adj = [0] * 8
        for i in range(8):
            u = vs[i]
            mask = adj_bits[u]
            for j in range(8):
                if j != i and (mask >> vs[j]) & 1:
                    local_adj[i] |= (1 << j)
        
        # Degree check
        for i in range(8):
            if bin(local_adj[i]).count('1') < 3:
                return False
        
        # Backtrack
        assignment = [-1] * 8
        vs_used = [False] * 8
        
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            for vs_i in range(8):
                if vs_used[vs_i]:
                    continue
                ok = True
                for nb in q3_adj[q3_v]:
                    if nb < q3_v:
                        if not ((local_adj[vs_i] >> assignment[nb]) & 1):
                            ok = False
                            break
                if ok:
                    assignment[q3_v] = vs_i
                    vs_used[vs_i] = True
                    if backtrack(q3_v + 1):
                        return True
                    vs_used[vs_i] = False
                    assignment[q3_v] = -1
            return False
        
        return backtrack(0)

    def check_edge_creates_q3(u, v, adj_bits):
        """Check if adding edge (u,v) creates a Q3. Only check 8-sets containing u and v."""
        if u > v:
            u, v = v, u
        relevant_8sets = pair_to_8sets.get((u, v), [])
        for vs in relevant_8sets:
            if contains_q3_fast(vs, adj_bits):
                return True
        return False

    def edges_to_adj_bits(edges):
        adj = [0] * 16
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    # Initialize with K3,13 (39 edges)
    # Vertices 0,1,2 connected to 3..15
    initial_edges = set((a, b) for a in range(3) for b in range(3, 16))
    best_edges = set(initial_edges)
    best_count = len(initial_edges)
    
    # Also try a few other random starts to diversify
    random.seed(42)
    
    for restart in range(5): # Multiple restarts
        if time.time() > deadline - 10:
            break
            
        # Start from best known or random
        if restart == 0:
            current_edges = set(best_edges)
        else:
            # Random dense graph, filter Q3s? Or just random start
            current_edges = set()
            # Try to build a random graph
            all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
            random.shuffle(all_pairs)
            adj = [0]*16
            for u, v in all_pairs:
                if time.time() > deadline - 5: break
                new_adj = adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                if not check_edge_creates_q3(u, v, new_adj):
                    adj = new_adj
                    current_edges.add((u, v))
        
        current_count = len(current_edges)
        if current_count > best_count:
            best_count = current_count
            best_edges = set(current_edges)
            
        adj = edges_to_adj_bits(current_edges)
        
        # Simulated Annealing Parameters
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        no_improve_steps = 0
        max_no_improve = 200
        
        while time.time() < deadline - 5 and T > T_min:
            if no_improve_steps > max_no_improve:
                # Restart or perturb
                break
                
            # Choose a move: Add, Remove, or Swap
            move_type = random.random()
            
            new_edges = set(current_edges)
            new_adj = adj[:]
            accepted = False
            delta = 0
            
            if move_type < 0.4:
                # Try to add an edge
                missing = [(i, j) for i in range(16) for j in range(i+1, 16) if (i,j) not in new_edges]
                if missing:
                    u, v = random.choice(missing)
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                    if not check_edge_creates_q3(u, v, new_adj):
                        new_edges.add((u, v))
                        delta = 1
                        accepted = True
                    else:
                        # Reject
                        new_adj = adj[:]
                        
            elif move_type < 0.7:
                # Try to remove an edge (to allow swaps or escape)
                if current_edges:
                    u, v = random.choice(list(current_edges))
                    new_edges.remove((u, v))
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                    delta = -1
                    accepted = True # Always accept removals in SA context to explore
                    
            else:
                # Swap: Remove one, Add one
                if current_edges:
                    u1, v1 = random.choice(list(current_edges))
                    missing = [(i, j) for i in range(16) for j in range(i+1, 16) if (i,j) not in current_edges]
                    if missing:
                        u2, v2 = random.choice(missing)
                        # Remove u1-v1
                        new_edges.remove((u1, v1))
                        new_adj[u1] &= ~(1 << v1)
                        new_adj[v1] &= ~(1 << u1)
                        # Add u2-v2
                        new_adj[u2] |= (1 << v2)
                        new_adj[v2] |= (1 << u2)
                        if not check_edge_creates_q3(u2, v2, new_adj):
                            new_edges.add((u2, v2))
                            delta = 0 # Net change 0
                            accepted = True
                        else:
                            # Reject swap
                            new_edges = set(current_edges)
                            new_adj = adj[:]

            if not accepted:
                continue
                
            # Calculate acceptance
            if delta > 0:
                # Always accept improvement
                current_edges = new_edges
                current_count += delta
                adj = new_adj
                no_improve_steps = 0
                if current_count > best_count:
                    best_count = current_count
                    best_edges = set(current_edges)
            else:
                # Accept with probability exp(-delta/T)
                # delta is -1 or 0
                if delta == 0:
                    # Neutral move, accept with high probability to explore
                    if random.random() < 0.5:
                        current_edges = new_edges
                        adj = new_adj
                        no_improve_steps += 1
                else: # delta == -1
                    p = 2.71828 ** (1.0 / T)
                    if random.random() < p:
                        current_edges = new_edges
                        current_count += delta
                        adj = new_adj
                        no_improve_steps += 1
                    else:
                        no_improve_steps += 1
            
            T *= cooling
            
        # End of SA run, update best if found something better during run (already done)

    result = [[u, v] for (u, v) in best_edges]
    return result

# EVOLVE-BLOCK-END
