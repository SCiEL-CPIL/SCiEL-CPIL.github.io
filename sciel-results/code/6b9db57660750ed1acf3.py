import time
import random
import math
from collections import Counter

DEADLINE = 110  # Safety margin for 150s limit

def _compute_score(n, adj_masks):
    """
    Computes the maximum degree sum over all triangles in the graph.
    Returns (max_sum, list_of_worst_triangles)
    """
    deg = [bin(mask).count('1') for mask in adj_masks]
    max_sum = 0
    max_triangles = []

    for i in range(n):
        mi = adj_masks[i]
        # Only consider j > i to avoid duplicates
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)

        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            mask_j_greater ^= lsb

            # Triangles containing i and j must have k in common neighbors
            common = mi & adj_masks[j]
            # Restrict k > j to avoid duplicates
            mask_k = common & ~((1 << (j + 1)) - 1)
            while mask_k:
                lsb_k = mask_k & -mask_k
                k = lsb_k.bit_length() - 1
                mask_k ^= lsb_k
                s = deg[i] + deg[j] + deg[k]
                if s > max_sum:
                    max_sum = s
                    max_triangles = [(i, j, k)]
                elif s == max_sum:
                    max_triangles.append((i, j, k))

    return max_sum, max_triangles

def _construct_edges_from_masks(adj_masks, n):
    edges = []
    for i in range(n):
        mi = adj_masks[i]
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            edges.append([i, j])
            mask_j_greater ^= lsb
    return edges

def _count_edges(adj_masks, n):
    count = 0
    for i in range(n):
        mi = adj_masks[i]
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        count += bin(mask_j_greater).count('1')
    return count

def _generate_initial_graphs(n, target_edges):
    """Generate several diverse initial graphs with exactly target_edges edges."""
    graphs = []
    
    # Graph 1: K_{8,8} + one internal edge
    adj1 = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj1[u] |= (1 << v)
            adj1[v] |= (1 << u)
    adj1[0] |= (1 << 1)
    adj1[1] |= (1 << 0)
    graphs.append(adj1)
    
    # Graph 2: K_{8,8} + different internal edge
    adj2 = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj2[u] |= (1 << v)
            adj2[v] |= (1 << u)
    adj2[0] |= (1 << 2)
    adj2[2] |= (1 << 0)
    graphs.append(adj2)
    
    # Graph 3: Random graph with exactly target_edges edges
    rng = random.Random(123)
    adj3 = [0] * n
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_possible)
    for u, v in all_possible[:target_edges]:
        adj3[u] |= (1 << v)
        adj3[v] |= (1 << u)
    graphs.append(adj3)
    
    # Graph 4: Another random graph
    rng2 = random.Random(456)
    adj4 = [0] * n
    all_possible2 = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng2.shuffle(all_possible2)
    for u, v in all_possible2[:target_edges]:
        adj4[u] |= (1 << v)
        adj4[v] |= (1 << u)
    graphs.append(adj4)
    
    # Graph 5: Near-regular random graph
    # Try to make degrees as equal as possible
    rng3 = random.Random(789)
    adj5 = [0] * n
    all_possible3 = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng3.shuffle(all_possible3)
    
    # Greedy: add edges trying to keep degrees balanced
    degs = [0] * n
    for u, v in all_possible3:
        if sum(1 for e in range(target_edges)) <= target_edges:
            pass
        # Simple: just take first target_edges
        break
    
    # Fallback: just use another random
    adj5 = [0] * n
    for u, v in all_possible3[:target_edges]:
        adj5[u] |= (1 << v)
        adj5[v] |= (1 << u)
    graphs.append(adj5)
    
    return graphs

def _sa_search(adj_masks, n, target_edges, time_budget, seed):
    """Simulated annealing search starting from given initial state."""
    random.seed(seed)
    
    start_time = time.time()
    
    current_score, current_max_tris = _compute_score(n, adj_masks)
    best_score = current_score
    best_adj_masks = list(adj_masks)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Maintain sets for current edges and non-edges
    current_edge_set = set()
    non_edge_set = set()
    for u in range(n):
        for v in range(u+1, n):
            if (adj_masks[u] >> v) & 1:
                current_edge_set.add((u, v))
            else:
                non_edge_set.add((u, v))
                
    # Use lists for fast random choice
    non_edge_list = list(non_edge_set)
    edge_list = list(current_edge_set)
    
    # Simulated Annealing parameters
    T = 8.0
    cooling_rate = 0.99995
    min_T = 0.05
    
    deg = [bin(mask).count('1') for mask in adj_masks]
    
    iterations = 0
    stuck_count = 0
    
    while time.time() - start_time < time_budget:
        iterations += 1
        
        # 1. Generate a candidate move: Remove one edge, Add one non-edge
        # Strategy: If we have worst triangles, try to break one
        if current_max_tris and random.random() < 0.7:
            # Pick a random worst triangle
            i, j, k = random.choice(current_max_tris)
            tri_edges = [(min(i,j), max(i,j)), (min(j,k), max(j,k)), (min(i,k), max(i,k))]
            rem_u, rem_v = random.choice(tri_edges)
        else:
            # Random edge
            if not edge_list:
                break
            rem_u, rem_v = random.choice(edge_list)
        
        rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
        
        # Pick a random non-edge to add
        if not non_edge_list:
            break
        add_u, add_v = random.choice(non_edge_list)
        add_edge = (min(add_u, add_v), max(add_u, add_v))
        
        # Quick check: avoid creating triangles with very high degree sum
        # This is a heuristic filter
        new_deg_u = deg[rem_u] - 1 + (1 if (add_u == rem_u or add_v == rem_u) else 0)
        new_deg_v = deg[rem_v] - 1 + (1 if (add_u == rem_v or add_v == rem_v) else 0)
        
        # Perform Swap
        # Remove rem_edge
        adj_masks[rem_u] ^= (1 << rem_v)
        adj_masks[rem_v] ^= (1 << rem_u)
        deg[rem_u] -= 1
        deg[rem_v] -= 1
        
        current_edge_set.discard(rem_edge)
        non_edge_set.add(rem_edge)
        non_edge_list.append(rem_edge)
        try:
            edge_list.remove(rem_edge)
        except ValueError:
            pass
        
        # Add add_edge
        adj_masks[add_u] ^= (1 << add_v)
        adj_masks[add_v] ^= (1 << add_u)
        deg[add_u] += 1
        deg[add_v] += 1
        
        non_edge_set.discard(add_edge)
        current_edge_set.add(add_edge)
        try:
            non_edge_list.remove(add_edge)
        except ValueError:
            pass
        edge_list.append(add_edge)

        # Calculate New Score
        new_score, new_max_tris = _compute_score(n, adj_masks)
        delta = new_score - current_score

        # Acceptance Criterion
        if delta < 0 or (T > min_T and random.random() < math.exp(-delta / T)):
            # Accept
            current_score = new_score
            current_max_tris = new_max_tris
            
            if current_score < best_score:
                best_score = current_score
                best_adj_masks = list(adj_masks)
                stuck_count = 0
            else:
                stuck_count += 1
        else:
            # Revert
            # Remove add_edge
            adj_masks[add_u] ^= (1 << add_v)
            adj_masks[add_v] ^= (1 << add_u)
            deg[add_u] -= 1
            deg[add_v] -= 1
            
            non_edge_set.add(add_edge)
            current_edge_set.discard(add_edge)
            non_edge_list.append(add_edge)
            try:
                edge_list.remove(add_edge)
            except ValueError:
                pass

            # Add rem_edge
            adj_masks[rem_u] ^= (1 << rem_v)
            adj_masks[rem_v] ^= (1 << rem_u)
            deg[rem_u] += 1
            deg[rem_v] += 1
            
            current_edge_set.add(rem_edge)
            non_edge_set.discard(rem_edge)
            edge_list.append(rem_edge)
            try:
                non_edge_list.remove(rem_edge)
            except ValueError:
                pass

        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    return best_score, best_adj_masks

def _polish(adj_masks, n, time_budget=10):
    """Deterministic local search: try all single swaps that improve score."""
    start_time = time.time()
    current_score, current_max_tris = _compute_score(n, adj_masks)
    best_score = current_score
    best_adj_masks = list(adj_masks)
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    current_edge_set = set()
    non_edge_set = set()
    for u in range(n):
        for v in range(u+1, n):
            if (adj_masks[u] >> v) & 1:
                current_edge_set.add((u, v))
            else:
                non_edge_set.add((u, v))
    
    non_edge_list = list(non_edge_set)
    
    improved = True
    while improved and (time.time() - start_time < time_budget):
        improved = False
        current_score, current_max_tris = _compute_score(n, best_adj_masks)
        
        # Rebuild sets from best_adj_masks
        current_edge_set = set()
        non_edge_set = set()
        for u in range(n):
            for v in range(u+1, n):
                if (best_adj_masks[u] >> v) & 1:
                    current_edge_set.add((u, v))
                else:
                    non_edge_set.add((u, v))
        
        non_edge_list = list(non_edge_set)
        
        # Try all swaps
        for rem_edge in list(current_edge_set):
            if time.time() - start_time >= time_budget:
                break
            rem_u, rem_v = rem_edge
            
            for add_edge in non_edge_list:
                if time.time() - start_time >= time_budget:
                    break
                add_u, add_v = add_edge
                
                # Apply swap
                test_masks = list(best_adj_masks)
                test_masks[rem_u] ^= (1 << rem_v)
                test_masks[rem_v] ^= (1 << rem_u)
                test_masks[add_u] ^= (1 << add_v)
                test_masks[add_v] ^= (1 << add_u)
                
                new_score, _ = _compute_score(n, test_masks)
                
                if new_score < best_score:
                    best_score = new_score
                    best_adj_masks = test_masks
                    improved = True
                    break
            
            if improved:
                break
    
    return best_score, best_adj_masks

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    
    # Generate diverse initial graphs
    initial_graphs = _generate_initial_graphs(n, target_edges)
    
    best_overall_score = float('inf')
    best_overall_adj = None
    
    # Evaluate each initial graph
    for idx, adj in enumerate(initial_graphs):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        score, _ = _compute_score(n, adj)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_adj = list(adj)
    
    # Run SA from the best initial graph and from a couple others
    time_per_sa = (DEADLINE - 20) / 3  # Reserve 20s for polish
    
    for idx in range(min(3, len(initial_graphs))):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        adj_copy = list(initial_graphs[idx])
        seed = 42 + idx * 1000
        
        sa_score, sa_adj = _sa_search(adj_copy, n, target_edges, time_per_sa, seed)
        
        if sa_score < best_overall_score:
            best_overall_score = sa_score
            best_overall_adj = sa_adj
    
    # Final polish
    if best_overall_adj is not None and time.time() - start_time < DEADLINE - 5:
        polish_time = min(15, DEADLINE - 5 - (time.time() - start_time))
        if polish_time > 2:
            final_score, final_adj = _polish(best_overall_adj, n, polish_time)
            if final_score < best_overall_score:
                best_overall_score = final_score
                best_overall_adj = final_adj
    
    # Fallback if something went wrong
    if best_overall_adj is None:
        best_overall_adj = initial_graphs[0]
    
    best_edges = _construct_edges_from_masks(best_overall_adj, n)
    
    # Verify edge count
    assert len(best_edges) == target_edges, f"Expected {target_edges} edges, got {len(best_edges)}"
    
    return {'n': n, 'edges': best_edges}
