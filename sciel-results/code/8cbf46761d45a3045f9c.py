import itertools
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 290  # Safety margin for 300s limit

    # Helper to generate Paley graph
    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Fallback: Paley 37
    n_seed, edges_seed = get_paley(37)

    # Verify a graph has no K5 and no independent 5-set
    def is_valid(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for combo in itertools.combinations(range(n), 5):
            is_k5 = True
            is_i5 = True
            for i in range(5):
                u = combo[i]
                for j in range(i + 1, 5):
                    v = combo[j]
                    if adj[u] & (1 << v):
                        is_i5 = False
                    else:
                        is_k5 = False
                    if not is_k5 and not is_i5:
                        break
                if not is_k5 and not is_i5:
                    break
            if is_k5 or is_i5:
                return False
        return True

    # Try to find an n-vertex (5,5)-free graph using local search
    def try_find(n, max_attempts=12, max_steps=5000, seed_base=0):
        # Precompute 5-sets and edge-to-5set mapping
        all_5sets = list(itertools.combinations(range(n), 5))
        num_5sets = len(all_5sets)
        
        # For each 5-set, store the 10 edges (as index pairs within the 5-set)
        # and map each global edge to list of 5-set indices
        edge_to_5sets = {}
        # For each 5-set, store positions of each vertex in the combo
        combo_positions = []
        for idx, combo in enumerate(all_5sets):
            pos_map = {v: i for i, v in enumerate(combo)}
            combo_positions.append(pos_map)
            for i in range(5):
                for j in range(i + 1, 5):
                    edge = (combo[i], combo[j])
                    if edge not in edge_to_5sets:
                        edge_to_5sets[edge] = []
                    edge_to_5sets[edge].append(idx)
        
        # For efficient violation tracking:
        # For each 5-set, track number of edges present (0-10)
        # A 5-set is a K5 if all 10 edges present (count=10)
        # A 5-set is an independent 5-set if no edges present (count=0)
        # Otherwise it's fine
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout - 10:
                break
            
            random.seed(seed_base + attempt)
            
            # Try different initialization strategies
            if attempt % 3 == 0:
                # Random graph with p=0.5
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            elif attempt % 3 == 1:
                # Start from a sparse random graph
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.4:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            else:
                # Start from a dense random graph
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.6:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            # Compute edge counts for each 5-set
            edge_counts = [0] * num_5sets
            for idx in range(num_5sets):
                combo = all_5sets[idx]
                count = 0
                for i in range(5):
                    u = combo[i]
                    for j in range(i + 1, 5):
                        v = combo[j]
                        if adj[u] & (1 << v):
                            count += 1
                edge_counts[idx] = count
            
            # Count violations: edge_counts[i] == 0 (independent) or edge_counts[i] == 10 (clique)
            current_viol = sum(1 for c in edge_counts if c == 0 or c == 10)
            
            # Local search with simulated annealing
            temp = 2.0
            cooling = 0.9995
            best_viol = current_viol
            
            for step in range(max_steps):
                if time.time() - start_time > timeout - 10:
                    break
                
                if current_viol == 0:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append((i, j))
                    return n, edges
                
                # Pick random edge to flip
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                # Get 5-sets containing this edge
                affected = edge_to_5sets.get((i, j), [])
                
                # Determine if edge is currently present
                edge_present = bool(adj[i] & (1 << j))
                
                # For each affected 5-set, compute delta in edge count
                # If edge is present, flipping removes it (count decreases by 1)
                # If edge is absent, flipping adds it (count increases by 1)
                delta_per_5set = -1 if edge_present else 1
                
                # Compute violation change
                viol_delta = 0
                for idx in affected:
                    old_count = edge_counts[idx]
                    new_count = old_count + delta_per_5set
                    
                    # Check if old state was a violation
                    old_viol = 1 if (old_count == 0 or old_count == 10) else 0
                    # Check if new state is a violation
                    new_viol = 1 if (new_count == 0 or new_count == 10) else 0
                    
                    viol_delta += new_viol - old_viol
                
                # Accept if better or with probability based on temperature
                if viol_delta <= 0:
                    # Accept: update adj and edge_counts
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                    for idx in affected:
                        edge_counts[idx] += delta_per_5set
                    current_viol += viol_delta
                    if current_viol < best_viol:
                        best_viol = current_viol
                else:
                    # Accept with probability exp(-delta/temp)
                    prob = np.exp(-viol_delta / temp)
                    if random.random() < prob:
                        adj[i] ^= (1 << j)
                        adj[j] ^= (1 << i)
                        for idx in affected:
                            edge_counts[idx] += delta_per_5set
                        current_viol += viol_delta
                        if current_viol < best_viol:
                            best_viol = current_viol
                    # else: revert (no change needed)
                
                # Cool down
                temp *= cooling
            
            if current_viol == 0:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append((i, j))
                return n, edges
        
        return None, None

    # Try to find 42-vertex graph, then 41, 40, etc.
    for target_n in [42, 41, 40, 39, 38]:
        if time.time() - start_time > timeout - 30:
            break
        
        n_found, edges_found = try_find(target_n, max_attempts=12, max_steps=5000, seed_base=target_n * 100)
        
        if n_found == target_n and edges_found is not None:
            if is_valid(target_n, edges_found):
                return target_n, edges_found

    # Fallback: return the Paley 37
    return n_seed, edges_seed

# Test runner
if __name__ == "__main__":
    n, edges = construct()
    print(f"Vertices: {n}, Edges: {len(edges)}")
