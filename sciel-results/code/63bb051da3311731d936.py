import random
import time
import math
import numpy as np

N = 16
TARGET_EDGES = 65
DEADLINE = 135.0  # Safety margin before 150s kill

def _build_adj_matrix(edges):
    """Convert list of edges to a numpy adjacency matrix."""
    adj = np.zeros((N, N), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    return adj

def _calculate_score(adj):
    """
    Calculate the maximum degree sum over all triangles.
    Uses vectorized operations for speed.
    """
    deg = np.sum(adj, axis=1)
    # Find all triangles. 
    # A triangle (i, j, k) with i<j<k exists if adj[i,j] and adj[i,k] and adj[j,k].
    # We can iterate i, then find common neighbors of i, and check edges among them.
    max_sum = 0
    for i in range(N):
        # Neighbors of i with index > i
        neigh_i = np.where(adj[i, i+1:])[0] + i + 1
        if len(neigh_i) < 2:
            continue
        # For each pair of neighbors (j, k) of i, check if edge (j, k) exists
        # This is O(N^3) worst case but N=16 is small.
        # Vectorized check for edges between neighbors:
        # Create submatrix of adj for neighbors
        # Actually, simple loop is fast enough for N=16 in Python if optimized, 
        # but let's try to keep it simple and correct first.
        
        # Efficient triangle enumeration:
        # For each i, for each j > i where adj[i,j] is True:
        #   common = adj[i] & adj[j]
        #   for k in common where k > j:
        #      triangle (i, j, k)
        
        for j in range(i+1, N):
            if not adj[i, j]:
                continue
            # Common neighbors of i and j with index > j
            # adj[i] & adj[j] gives all common neighbors.
            # We need k > j.
            common = adj[i] & adj[j]
            # Mask for k > j
            mask = np.zeros(N, dtype=bool)
            mask[j+1:] = True
            common = common & mask
            
            if not np.any(common):
                continue
            
            # Get indices of common neighbors
            k_indices = np.where(common)[0]
            
            # Calculate degree sums for these triangles
            # deg[i] + deg[j] + deg[k] for k in k_indices
            sums = deg[i] + deg[j] + deg[k_indices]
            current_max = np.max(sums)
            if current_max > max_sum:
                max_sum = current_max
                
    return max_sum

def construct():
    t0 = time.time()
    rng = random.Random(42)

    # --- Phase 1: Initial Construction ---
    # Start with K_{8,8} + 1 internal edge. This is a known good structure for this problem.
    # Partitions: A = {0..7}, B = {8..15}
    edges = []
    for i in range(8):
        for j in range(8, 16):
            edges.append((i, j))
    # Add one internal edge to make 65
    edges.append((0, 1))
    
    # Verify count
    assert len(edges) == TARGET_EDGES, f"Initial edge count {len(edges)} != {TARGET_EDGES}"
    
    adj = _build_adj_matrix(edges)
    current_score = _calculate_score(adj)
    
    best_score = current_score
    best_adj = adj.copy()
    best_edges = set(edges)
    
    # --- Phase 2: Simulated Annealing with Targeted Moves ---
    # We maintain a set of edges and non-edges for fast sampling.
    edge_set = set(edges)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    non_edge_set = set(all_pairs) - edge_set
    
    start_temp = 5.0
    end_temp = 0.01
    max_iters = 200000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
    temp = start_temp
    
    # Precompute neighbors for potential moves? 
    # The "Targeted Move" idea: 
    # Identify the triangle with the max sum. 
    # Try to remove one of its edges.
    # If we remove an edge (u,v), we must add a non-edge (x,y).
    # To minimize score increase, we should pick (x,y) that doesn't create new high-sum triangles.
    
    def get_worst_triangle(adj):
        """Returns the triangle (i,j,k) with the maximum degree sum."""
        deg = np.sum(adj, axis=1)
        max_sum = -1
        worst_tri = None
        for i in range(N):
            for j in range(i+1, N):
                if not adj[i, j]:
                    continue
                common = adj[i] & adj[j]
                mask = np.zeros(N, dtype=bool)
                mask[j+1:] = True
                common = common & mask
                if not np.any(common):
                    continue
                k_indices = np.where(common)[0]
                sums = deg[i] + deg[j] + deg[k_indices]
                idx = np.argmax(sums)
                s = sums[idx]
                if s > max_sum:
                    max_sum = s
                    worst_tri = (i, j, k_indices[idx])
        return worst_tri, max_sum

    stall_count = 0
    stall_threshold = 5000
    iters = 0
    
    while iters < max_iters:
        if time.time() - t0 > DEADLINE:
            break
            
        # Check time every 1000 iters
        if iters % 1000 == 0:
            if time.time() - t0 > DEADLINE:
                break
            
        # Determine move
        # With probability 0.7, try to fix the worst triangle.
        # With probability 0.3, try a random swap.
        
        if rng.random() < 0.7 and current_score > 0:
            worst_tri, _ = get_worst_triangle(adj)
            if worst_tri:
                i, j, k = worst_tri
                # Pick one of the 3 edges of the worst triangle to remove
                tri_edges = [(i, j), (i, k), (j, k)]
                e_rm = rng.choice(tri_edges)
            else:
                e_rm = rng.choice(list(edge_set))
        else:
            e_rm = rng.choice(list(edge_set))
            
        # Pick a non-edge to add. 
        # Heuristic: Pick a non-edge that connects vertices with low degrees 
        # or is "far" from the removed edge to minimize triangle creation.
        # For simplicity and speed, random selection from non-edges is often sufficient 
        # if the removal step is targeted.
        if not non_edge_set:
            break
        e_add = rng.choice(list(non_edge_set))
        
        if e_rm == e_add:
            iters += 1
            temp *= cooling_rate
            continue
            
        # Apply Swap
        u, v = e_rm
        x, y = e_add
        
        # Update Adjacency
        adj[u, v] = False
        adj[v, u] = False
        adj[x, y] = True
        adj[y, x] = True
        
        # Update Sets
        edge_set.remove(e_rm)
        edge_set.add(e_add)
        non_edge_set.remove(e_add)
        non_edge_set.add(e_rm)
        
        # Calculate New Score
        new_score = _calculate_score(adj)
        
        # Acceptance Criterion
        delta = new_score - current_score
        if delta < 0:
            accept = True
        else:
            if temp < 1e-9:
                accept = False
            else:
                # Metropolis criterion
                prob = math.exp(-delta / temp)
                accept = rng.random() < prob
        
        if accept:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
                best_edges = set(edge_set)
                stall_count = 0
            else:
                stall_count += 1
        else:
            # Undo Swap
            adj[u, v] = True
            adj[v, u] = True
            adj[x, y] = False
            adj[y, x] = False
            edge_set.remove(e_add)
            edge_set.add(e_rm)
            non_edge_set.remove(e_rm)
            non_edge_set.add(e_add)
            
        iters += 1
        temp *= cooling_rate
        
        # Restart if stalled
        if stall_count > stall_threshold:
            temp = start_temp * 0.5
            stall_count = 0
            # Optionally randomize current state slightly? 
            # For now, just cool down and restart temperature.

    # --- Phase 3: Polish Phase (Deterministic Local Search) ---
    # Use the best solution found so far.
    adj = best_adj.copy()
    edge_set = set(best_edges)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    non_edge_set = set(all_pairs) - edge_set
    
    current_score = _calculate_score(adj)
    best_score = current_score
    best_adj = adj.copy()
    best_edges = set(edge_set)
    
    polish_deadline = DEADLINE - 5.0 # Leave 5s for return
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        # Find worst triangle
        worst_tri, cur_max = get_worst_triangle(adj)
        if not worst_tri:
            break
            
        i, j, k = worst_tri
        tri_edges = [(i, j), (i, k), (j, k)]
        
        # Try removing each edge of the worst triangle
        # and adding the "best" non-edge
        found_improvement = False
        for e_rm in tri_edges:
            u, v = e_rm
            # Find non-edge that minimizes new score
            best_add = None
            best_new_score = current_score
            
            # Sample a subset of non-edges for speed, or all if fast enough
            # N=16, max non-edges = 120-65=55. Checking all 55 is fast.
            for e_add in non_edge_set:
                x, y = e_add
                if e_rm == e_add:
                    continue
                # Check if adding (x,y) creates a triangle with high degree sum
                # Heuristic: If x and y have many common neighbors with high degree, skip
                # Or just calculate score.
                
                # Apply
                adj[u, v] = False
                adj[v, u] = False
                adj[x, y] = True
                adj[y, x] = True
                
                new_score = _calculate_score(adj)
                
                # Undo
                adj[u, v] = True
                adj[v, u] = True
                adj[x, y] = False
                adj[y, x] = False
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_add = e_add
            
            if best_add and best_new_score < current_score:
                # Apply the best move
                x, y = best_add
                u, v = e_rm
                adj[u, v] = False
                adj[v, u] = False
                adj[x, y] = True
                adj[y, x] = True
                
                edge_set.remove(e_rm)
                edge_set.add(best_add)
                non_edge_set.remove(best_add)
                non_edge_set.add(e_rm)
                
                current_score = best_new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
                    best_edges = set(edge_set)
                improved = True
                found_improvement = True
                break # Restart search from new state
        
        if not found_improvement:
            # Try random swaps to escape local minimum in polish phase?
            # No, polish phase should be strict. If stuck, we are at a local minimum.
            # We can try a few random swaps with strict acceptance to see if we can jump.
            for _ in range(10):
                e_rm = rng.choice(list(edge_set))
                e_add = rng.choice(list(non_edge_set))
                if e_rm == e_add: continue
                u, v = e_rm
                x, y = e_add
                adj[u, v] = False
                adj[v, u] = False
                adj[x, y] = True
                adj[y, x] = True
                new_score = _calculate_score(adj)
                if new_score < current_score:
                    edge_set.remove(e_rm)
                    edge_set.add(e_add)
                    non_edge_set.remove(e_add)
                    non_edge_set.add(e_rm)
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj.copy()
                        best_edges = set(edge_set)
                    improved = True
                    break
                else:
                    # Undo
                    adj[u, v] = True
                    adj[v, u] = True
                    adj[x, y] = False
                    adj[y, x] = False

    # Construct final edge list
    final_edges = [list(e) for e in best_edges]
    
    # Final validation
    assert len(final_edges) == TARGET_EDGES
    # Ensure no duplicates
    assert len(set(tuple(sorted(e)) for e in final_edges)) == TARGET_EDGES

    return {'n': N, 'edges': final_edges}

if __name__ == "__main__":
    res = construct()
    print(res)
