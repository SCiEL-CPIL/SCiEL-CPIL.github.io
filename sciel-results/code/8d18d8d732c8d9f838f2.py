import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = 82.0  # safety margin before 90s kill
    
    n = 15
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_possible_set = set(all_possible)
    
    # Precompute bitmasks for fast adjacency
    def make_adj_bitmasks(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def compute_degrees(adj_bitmasks):
        return [bin(a).count('1') for a in adj_bitmasks]
    
    def is_valid_fast(edges_set):
        """Fast validity check using bitmasks."""
        adj = make_adj_bitmasks(edges_set)
        degrees = compute_degrees(adj)
        
        # Group vertices by degree
        deg_groups = {}
        for i in range(n):
            deg_groups.setdefault(degrees[i], set()).add(i)
        
        # For each pair (s, t) with same degree, check if 3-path exists
        for deg, group in deg_groups.items():
            group_list = list(group)
            for idx_s in range(len(group_list)):
                s = group_list[idx_s]
                for idx_t in range(idx_s + 1, len(group_list)):
                    t = group_list[idx_t]
                    # Check if path s-u-v-t exists with all distinct
                    # u in N(s), v in N(u) \ {s}, t in N(v) \ {u, s}
                    found = False
                    for u in range(n):
                        if u == s or u == t:
                            continue
                        if not (adj[s] & (1 << u)):
                            continue
                        # u is neighbor of s
                        for v in range(n):
                            if v == s or v == t or v == u:
                                continue
                            if not (adj[u] & (1 << v)):
                                continue
                            # v is neighbor of u, v != s, t, u
                            if adj[v] & (1 << t):
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
        return True
    
    def local_search(initial_edges, max_time=75.0):
        edges = set(tuple(sorted(e)) for e in initial_edges)
        
        # Ensure validity first
        if not is_valid_fast(edges):
            attempts = 0
            while not is_valid_fast(edges) and attempts < 2000 and time.time() - start_time < deadline:
                e_to_remove = random.choice(list(edges))
                edges.remove(e_to_remove)
                attempts += 1
            if not is_valid_fast(edges):
                return None, 0
        
        current_score = len(edges)
        
        # Phase 1: Greedy addition with degree-aware ordering
        improved = True
        pass_count = 0
        while improved and pass_count < 30 and time.time() - start_time < deadline:
            improved = False
            pass_count += 1
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)
            
            # Sort by a heuristic: prefer edges that connect low-degree vertices
            # to avoid creating too many same-degree pairs with paths
            adj = make_adj_bitmasks(edges)
            degrees = compute_degrees(adj)
            
            def edge_heuristic(e):
                u, v = e
                return degrees[u] + degrees[v]
            
            missing.sort(key=edge_heuristic)
            
            for e in missing:
                if time.time() - start_time > deadline:
                    return list(edges), current_score
                new_edges = edges | {e}
                if is_valid_fast(new_edges):
                    edges = new_edges
                    current_score = len(edges)
                    improved = True
                    break
        
        # Phase 2: Swap moves (remove 1, add 1)
        swap_improved = True
        max_swap_iterations = 40
        swap_iter = 0
        while swap_improved and swap_iter < max_swap_iterations and time.time() - start_time < deadline:
            swap_improved = False
            swap_iter += 1
            
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)
            existing = list(edges)
            random.shuffle(existing)
            
            adj = make_adj_bitmasks(edges)
            degrees = compute_degrees(adj)
            
            for e_remove in existing:
                if time.time() - start_time > deadline:
                    return list(edges), current_score
                edges_without = edges - {e_remove}
                
                # Filter candidates: prefer adding edges that don't increase degree disparity too much
                candidates = []
                for e_add in missing:
                    if e_add == e_remove:
                        continue
                    u, v = e_add
                    # Heuristic: prefer edges between vertices with similar degrees
                    candidates.append((abs(degrees[u] - degrees[v]), e_add))
                candidates.sort()
                
                for _, e_add in candidates:
                    new_edges = edges_without | {e_add}
                    if is_valid_fast(new_edges):
                        edges = new_edges
                        current_score = len(edges)
                        swap_improved = True
                        break
                if swap_improved:
                    break
            
            # After a swap, try to add more edges greedily
            if swap_improved:
                add_improved = True
                while add_improved and time.time() - start_time < deadline:
                    add_improved = False
                    missing = [e for e in all_possible if e not in edges]
                    random.shuffle(missing)
                    for e in missing:
                        new_edges = edges | {e}
                        if is_valid_fast(new_edges):
                            edges = new_edges
                            current_score = len(edges)
                            add_improved = True
                            break
        
        # Phase 3: 2-for-1 swaps (remove 2, add 1)
        two_for_one_improved = True
        max_21_iter = 20
        two_21_iter = 0
        while two_for_one_improved and two_21_iter < max_21_iter and time.time() - start_time < deadline:
            two_for_one_improved = False
            two_21_iter += 1
            
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)
            existing = list(edges)
            random.shuffle(existing)
            
            for i in range(min(len(existing), 40)):
                if time.time() - start_time > deadline:
                    return list(edges), current_score
                e_remove1 = existing[i]
                for j in range(i + 1, min(len(existing), 40)):
                    e_remove2 = existing[j]
                    edges_without = edges - {e_remove1, e_remove2}
                    
                    for e_add in missing[:30]:
                        if e_add == e_remove1 or e_add == e_remove2:
                            continue
                        new_edges = edges_without | {e_add}
                        if is_valid_fast(new_edges):
                            edges = new_edges
                            current_score = len(edges)
                            two_for_one_improved = True
                            break
                    if two_for_one_improved:
                        break
                if two_for_one_improved:
                    break
        
        # Phase 4: Final polish - try all missing edges in degree-ordered fashion
        if time.time() - start_time < deadline:
            adj = make_adj_bitmasks(edges)
            degrees = compute_degrees(adj)
            missing = [e for e in all_possible if e not in edges]
            
            def final_heuristic(e):
                u, v = e
                # Prefer edges that connect vertices with different degrees
                # to reduce the number of same-degree pairs
                return -abs(degrees[u] - degrees[v])
            
            missing.sort(key=final_heuristic)
            
            for e in missing:
                if time.time() - start_time > deadline:
                    break
                new_edges = edges | {e}
                if is_valid_fast(new_edges):
                    edges = new_edges
                    current_score = len(edges)
        
        return list(edges), current_score
    
    def perturbed_local_search(best_edges, perturbation_size=3):
        edges = set(tuple(sorted(e)) for e in best_edges)
        if len(edges) < perturbation_size:
            perturbation_size = max(1, len(edges) // 3)
        to_remove = random.sample(list(edges), perturbation_size)
        for e in to_remove:
            edges.remove(e)
        return local_search(list(edges))
    
    best_edges = None
    best_score = -1
    
    # Strategy 1: Random dense starts
    num_random_starts = 30
    for i in range(num_random_starts):
        if time.time() - start_time > deadline:
            break
        
        if i < num_random_starts // 3:
            density = 0.55 + 0.3 * random.random()
        elif i < 2 * num_random_starts // 3:
            density = 0.4 + 0.4 * random.random()
        else:
            density = 0.3 + 0.5 * random.random()
        
        num_initial = int(len(all_possible) * density)
        initial_edges = random.sample(all_possible, num_initial)
        
        res_edges, score = local_search(initial_edges)
        
        if res_edges is not None and score > best_score:
            best_score = score
            best_edges = res_edges
        
        if best_score > 90:
            break
    
    # Strategy 2: Structured starts - near-regular graphs
    for trial in range(15):
        if time.time() - start_time > deadline:
            break
        
        if trial % 3 == 0:
            # Hamiltonian cycle + random chords
            perm = list(range(n))
            random.shuffle(perm)
            initial_edges = set()
            for i in range(n):
                u = perm[i]
                v = perm[(i + 1) % n]
                initial_edges.add((min(u, v), max(u, v)))
            extra_count = random.randint(15, 30)
            extra = random.sample(all_possible, extra_count)
            for e in extra:
                if e not in initial_edges:
                    initial_edges.add(e)
            initial_edges = list(initial_edges)
        elif trial % 3 == 1:
            # Random near-regular: each vertex gets ~6-8 edges
            initial_edges = set()
            for i in range(n):
                target_deg = random.randint(6, 8)
                candidates = [j for j in range(n) if j != i]
                random.shuffle(candidates)
                for j in candidates:
                    e = (min(i, j), max(i, j))
                    if e in initial_edges:
                        continue
                    current_deg = sum(1 for u, v in initial_edges if u == i or v == i)
                    if current_deg >= target_deg:
                        break
                    initial_edges.add(e)
            initial_edges = list(initial_edges)
        else:
            # Start with a random bipartite-like structure
            left = list(range(7))
            right = list(range(7, 15))
            initial_edges = set()
            for i in left:
                for j in right:
                    if random.random() < 0.5:
                        initial_edges.add((i, j))
            # Add some intra-part edges
            for i in range(len(left)):
                for j in range(i + 1, len(left)):
                    if random.random() < 0.2:
                        initial_edges.add((left[i], left[j]))
            for i in range(len(right)):
                for j in range(i + 1, len(right)):
                    if random.random() < 0.2:
                        initial_edges.add((right[i], right[j]))
            initial_edges = list(initial_edges)
        
        res_edges, score = local_search(initial_edges)
        
        if res_edges is not None and score > best_score:
            best_score = score
            best_edges = res_edges
    
    # Strategy 3: Perturbation from best found
    if best_edges is not None:
        num_perturbations = 25
        for i in range(num_perturbations):
            if time.time() - start_time > deadline:
                break
            
            pert_size = random.randint(2, 6)
            res_edges, score = perturbed_local_search(best_edges, pert_size)
            
            if res_edges is not None and score > best_score:
                best_score = score
                best_edges = res_edges
    
    # Final: try to squeeze more from best
    if best_edges is not None and time.time() - start_time < deadline - 5:
        res_edges, score = local_search(best_edges)
        if res_edges is not None and score > best_score:
            best_score = score
            best_edges = res_edges
    
    if best_edges is None:
        return []
        
    result = [[int(u), int(v)] for u, v in best_edges]
    return result

if __name__ == "__main__":
    pass
