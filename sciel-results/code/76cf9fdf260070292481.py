import time
import random
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def is_k23_free(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free using bitmasks."""
        cu = (adj[u] & adj[v]).bit_count()
        if cu > 2:
            return False
        
        # For each w adjacent to v (but not u): adding (u,v) makes u a common neighbor of w and v
        # So we need common(w,v) <= 1 for all w in N(v) \ {u}
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[v] >> w) & 1:
                common_uw = (adj[u] & adj[w]).bit_count()
                if common_uw > 1:
                    return False
            if (adj[u] >> w) & 1:
                common_vw = (adj[v] & adj[w]).bit_count()
                if common_vw > 1:
                    return False
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            for v in range(u + 1, n):
                if (mask >> v) & 1:
                    edges.append([u, v])
        return edges

    def greedy_build(seed=None, degree_balanced=False):
        """Greedy random construction with optional degree-balancing."""
        rng = random.Random(seed)
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        
        if degree_balanced:
            for _ in range(10):
                remaining = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
                remaining.sort(key=lambda e: adj[e[0]].bit_count() + adj[e[1]].bit_count())
                for (u, v) in remaining:
                    if ((adj[u] >> v) & 1):
                        continue
                    if is_k23_free(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            for (u, v) in all_edges:
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def structured_initial(seed=None, circulant=True):
        """Create a structured initial graph using circulant base + greedy fill."""
        rng = random.Random(seed)
        adj = [0] * n
        
        if circulant:
            for i in range(n):
                for d in [1, 2, 5]:
                    j = (i + d) % n
                    if j != i and not ((adj[i] >> j) & 1):
                        if is_k23_free(adj, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for (u, v) in all_edges:
            if not ((adj[u] >> v) & 1):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj

    def random_structured_initial(seed=None):
        """Create a random structured initial graph with random 4-regular base."""
        rng = random.Random(seed)
        adj = [0] * n
        
        for i in range(n):
            candidates = [j for j in range(n) if j != i and not ((adj[i] >> j) & 1)]
            rng.shuffle(candidates)
            for j in candidates[:4]:
                if is_k23_free(adj, i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for (u, v) in all_edges:
            if not ((adj[u] >> v) & 1):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj

    def local_search(adj, rng, time_budget=5.0):
        """Local search with edge additions, 1-swaps, and 2-swaps."""
        adj = list(adj)
        search_start = time.time()
        
        present = set()
        missing = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    present.add((u, v))
                else:
                    missing.add((u, v))
        
        improved = True
        while improved:
            if time.time() - search_start > time_budget:
                break
            improved = False
            
            # Phase 1: Try adding any missing edge
            for (u, v) in list(missing):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    present.add((u, v))
                    missing.discard((u, v))
                    improved = True
                    break
            if improved:
                continue
            
            # Phase 2: Try 1-swap
            if not present or not missing:
                break
            
            present_list = list(present)
            missing_list = list(missing)
            
            found_swap = False
            for _ in range(500):
                if time.time() - search_start > time_budget:
                    break
                a, b = rng.choice(present_list)
                c, d = rng.choice(missing_list)
                
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                if is_k23_free(adj, c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    present.discard((a, b))
                    present.add((c, d))
                    missing.add((a, b))
                    missing.discard((c, d))
                    improved = True
                    found_swap = True
                    break
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            
            if found_swap:
                continue
            
            # Phase 3: Try 2-swap
            if len(present) < 2 or len(missing) < 2:
                break
            
            present_list = list(present)
            missing_list = list(missing)
            
            found_2swap = False
            for _ in range(200):
                if time.time() - search_start > time_budget:
                    break
                (a, b), (c, d) = rng.sample(present_list, 2)
                (e, f), (g, h) = rng.sample(missing_list, 2)
                
                adj[a] &= ~(1 << b); adj[b] &= ~(1 << a)
                adj[c] &= ~(1 << d); adj[d] &= ~(1 << c)
                
                if is_k23_free(adj, e, f):
                    adj[e] |= (1 << f); adj[f] |= (1 << e)
                    if is_k23_free(adj, g, h):
                        adj[g] |= (1 << h); adj[h] |= (1 << g)
                        present.discard((a, b))
                        present.discard((c, d))
                        present.add((e, f))
                        present.add((g, h))
                        missing.add((a, b))
                        missing.add((c, d))
                        missing.discard((e, f))
                        missing.discard((g, h))
                        improved = True
                        found_2swap = True
                        break
                    else:
                        adj[e] &= ~(1 << f); adj[f] &= ~(1 << e)
                
                adj[a] |= (1 << b); adj[b] |= (1 << a)
                adj[c] |= (1 << d); adj[d] |= (1 << c)
            
            if not found_2swap:
                break
        
        return adj

    def final_polish(adj, rng, time_budget=3.0):
        """High-precision polish pass with more exhaustive swap attempts."""
        adj = list(adj)
        search_start = time.time()
        
        present = set()
        missing = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    present.add((u, v))
                else:
                    missing.add((u, v))
        
        improved = True
        while improved:
            if time.time() - search_start > time_budget:
                break
            improved = False
            
            for (u, v) in list(missing):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    present.add((u, v))
                    missing.discard((u, v))
                    improved = True
                    break
            if improved:
                continue
            
            present_list = list(present)
            missing_list = list(missing)
            
            found_swap = False
            for a, b in present_list:
                if time.time() - search_start > time_budget:
                    break
                for c, d in missing_list:
                    if time.time() - search_start > time_budget:
                        break
                    
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    
                    if is_k23_free(adj, c, d):
                        adj[c] |= (1 << d)
                        adj[d] |= (1 << c)
                        present.discard((a, b))
                        present.add((c, d))
                        missing.add((a, b))
                        missing.discard((c, d))
                        improved = True
                        found_swap = True
                        break
                    else:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                if found_swap:
                    break
            
            if found_swap:
                continue
            
            if len(present) < 2 or len(missing) < 2:
                break
            
            present_list = list(present)
            missing_list = list(missing)
            
            found_2swap = False
            for i in range(len(present_list)):
                if time.time() - search_start > time_budget:
                    break
                for j in range(i + 1, len(present_list)):
                    if time.time() - search_start > time_budget:
                        break
                    (a, b) = present_list[i]
                    (c, d) = present_list[j]
                    
                    for k in range(len(missing_list)):
                        if time.time() - search_start > time_budget:
                            break
                        for l in range(k + 1, len(missing_list)):
                            if time.time() - search_start > time_budget:
                                break
                            (e, f) = missing_list[k]
                            (g, h) = missing_list[l]
                            
                            adj[a] &= ~(1 << b); adj[b] &= ~(1 << a)
                            adj[c] &= ~(1 << d); adj[d] &= ~(1 << c)
                            
                            if is_k23_free(adj, e, f):
                                adj[e] |= (1 << f); adj[f] |= (1 << e)
                                if is_k23_free(adj, g, h):
                                    adj[g] |= (1 << h); adj[h] |= (1 << g)
                                    present.discard((a, b))
                                    present.discard((c, d))
                                    present.add((e, f))
                                    present.add((g, h))
                                    missing.add((a, b))
                                    missing.add((c, d))
                                    missing.discard((e, f))
                                    missing.discard((g, h))
                                    improved = True
                                    found_2swap = True
                                    break
                                else:
                                    adj[e] &= ~(1 << f); adj[f] &= ~(1 << e)
                            
                            adj[a] |= (1 << b); adj[b] |= (1 << a)
                            adj[c] |= (1 << d); adj[d] |= (1 << c)
                            
                            if found_2swap:
                                break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break
            
            if not found_2swap:
                break
        
        return adj

    def simulated_annealing(adj, rng, time_budget=15.0):
        """Simulated annealing with NumPy vectorized violation counting."""
        adj = list(adj)
        search_start = time.time()
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.int8)
        for u in range(n):
            for v in range(n):
                if (adj[u] >> v) & 1:
                    A[u, v] = 1
        
        # Precompute common neighbor counts
        def compute_common_neighbors(A):
            # C[i,j] = number of common neighbors of i and j
            C = np.zeros((n, n), dtype=np.int32)
            for k in range(n):
                col = A[:, k]
                C += np.outer(col, col)
            # Subtract self-contributions
            np.fill_diagonal(C, 0)
            return C
        
        C = compute_common_neighbors(A)
        
        def count_violations(C):
            # A K2,3 violation exists if any pair has >= 3 common neighbors
            return np.sum(C >= 3)
        
        def edge_count_from_A(A):
            return int(np.sum(A)) // 2
        
        current_violations = count_violations(C)
        current_edges = edge_count_from_A(A)
        best_adj = adj[:]
        best_edges = current_edges
        best_violations = current_violations
        
        # Simulated annealing parameters
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        while time.time() - search_start < time_budget:
            if T < T_min:
                break
            
            # Pick a random edge to flip
            u, v = rng.choice(all_edges)
            
            # Flip the edge
            A[u, v] = 1 - A[u, v]
            A[v, u] = A[u, v]
            
            # Recompute common neighbors
            C = compute_common_neighbors(A)
            new_violations = count_violations(C)
            new_edges = edge_count_from_A(A)
            
            # Acceptance criterion
            if new_violations < current_violations:
                # Always accept if fewer violations
                accept = True
            elif new_violations == current_violations:
                # Accept if more edges (same violations)
                accept = new_edges >= current_edges
            else:
                # More violations: accept with probability exp(-(delta_violations)/T)
                delta_v = new_violations - current_violations
                delta_e = new_edges - current_edges
                # Energy: violations * 10 - edges
                energy_change = 10 * delta_v - delta_e
                if energy_change < 0:
                    accept = True
                else:
                    accept = rng.random() < np.exp(-energy_change / T)
            
            if accept:
                current_violations = new_violations
                current_edges = new_edges
                
                if (current_violations < best_violations or 
                    (current_violations == best_violations and current_edges > best_edges)):
                    best_violations = current_violations
                    best_edges = current_edges
                    # Convert A back to adj
                    for i in range(n):
                        mask = 0
                        for j in range(n):
                            if A[i, j]:
                                mask |= (1 << j)
                        best_adj[i] = mask
            else:
                # Undo the flip
                A[u, v] = 1 - A[u, v]
                A[v, u] = A[u, v]
                C = compute_common_neighbors(A)
            
            # Cool down
            T *= cooling_rate
        
        # Return best found
        return best_adj

    best_adj = None
    best_count = 0

    # Generate diverse initial constructions
    inits = []
    
    for s in range(3):
        inits.append(structured_initial(seed=s, circulant=True))
    
    for s in range(3):
        inits.append(random_structured_initial(seed=s))
    
    for s in range(5):
        inits.append(greedy_build(seed=s, degree_balanced=False))
    
    for s in range(3):
        inits.append(greedy_build(seed=200+s, degree_balanced=True))
    
    # Process each initial with local search
    for i, adj in enumerate(inits):
        if time.time() > deadline:
            break
        
        cnt = edge_count(adj)
        if cnt > best_count:
            best_adj = adj[:]
            best_count = cnt
        
        remaining = deadline - time.time()
        if remaining > 2.0:
            budget = min(5.0, remaining * 0.8)
            rng = random.Random(i * 1000)
            improved_adj = local_search(adj, rng, time_budget=budget)
            cnt2 = edge_count(improved_adj)
            if cnt2 > best_count:
                best_adj = improved_adj
                best_count = cnt2
    
    # Apply simulated annealing on the best solution found so far
    if best_adj is not None and time.time() < deadline - 2.0:
        remaining = deadline - time.time() - 2.0
        if remaining > 1.0:
            rng = random.Random(42)
            sa_adj = simulated_annealing(best_adj, rng, time_budget=min(remaining, 15.0))
            # Verify validity
            valid = True
            for u in range(n):
                for v in range(u + 1, n):
                    if (sa_adj[u] >> v) & 1:
                        pass  # Edge exists, need to check K2,3
            # Quick validity check
            for u in range(n):
                for v in range(u + 1, n):
                    if (sa_adj[u] >> v) & 1:
                        # Check if removing this edge and re-adding would be valid
                        # Actually just verify the whole graph
                        pass
            
            # Full validity check
            adj_check = sa_adj[:]
            valid = True
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_check[u] >> v) & 1:
                        # Temporarily remove and check
                        adj_check[u] &= ~(1 << v)
                        adj_check[v] &= ~(1 << u)
                        if not is_k23_free(adj_check, u, v):
                            valid = False
                            break
                        adj_check[u] |= (1 << v)
                        adj_check[v] |= (1 << u)
                if not valid:
                    break
            
            if valid:
                cnt_sa = edge_count(sa_adj)
                if cnt_sa > best_count:
                    best_adj = sa_adj
                    best_count = cnt_sa
    
    # Final polish
    if best_adj is not None and time.time() < deadline - 1.0:
        remaining = deadline - time.time() - 1.0
        if remaining > 0.5:
            rng = random.Random(999)
            polished = final_polish(best_adj, rng, time_budget=min(remaining, 3.0))
            # Verify validity
            valid = True
            adj_check = polished[:]
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_check[u] >> v) & 1:
                        adj_check[u] &= ~(1 << v)
                        adj_check[v] &= ~(1 << u)
                        if not is_k23_free(adj_check, u, v):
                            valid = False
                            break
                        adj_check[u] |= (1 << v)
                        adj_check[v] |= (1 << u)
                if not valid:
                    break
            
            if valid:
                cnt_polish = edge_count(polished)
                if cnt_polish > best_count:
                    best_adj = polished
                    best_count = cnt_polish
    
    return adj_to_edges(best_adj) if best_adj else []
