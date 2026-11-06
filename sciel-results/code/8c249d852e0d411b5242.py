"""Program for the graph construction task."""
import time
import random
import numpy as np

DEADLINE = 140  # safety margin before 150s hard kill

def _compute_score(adj, deg):
    """Compute the largest degree sum over triangles."""
    n = adj.shape[0]
    max_sum = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u, v]:
                continue
            for w in range(v + 1, n):
                if adj[u, w] and adj[v, w]:
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _initial_constructions():
    """Generate multiple diverse initial constructions."""
    n = 16
    target = 65
    constructions = []
    
    # 1. K_{8,8} + one internal edge
    adj = np.zeros((n, n), dtype=bool)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = True
            adj[v, u] = True
    adj[0, 1] = True
    adj[1, 0] = True
    constructions.append(adj.copy())
    
    # 2. K_{7,9} + internal edges
    adj = np.zeros((n, n), dtype=bool)
    for u in range(7):
        for v in range(7, 16):
            adj[u, v] = True
            adj[v, u] = True
    # 63 edges, need 2 more
    adj[0, 1] = True
    adj[1, 0] = True
    adj[2, 3] = True
    adj[3, 2] = True
    constructions.append(adj.copy())
    
    # 3. K_{6,10} + internal edges
    adj = np.zeros((n, n), dtype=bool)
    for u in range(6):
        for v in range(6, 16):
            adj[u, v] = True
            adj[v, u] = True
    # 60 edges, need 5 more
    internal_pairs = [(0,1), (0,2), (1,2), (6,7), (6,8)]
    for u, v in internal_pairs:
        adj[u, v] = True
        adj[v, u] = True
    constructions.append(adj.copy())
    
    # 4. Random near-bipartite
    for seed in range(10):
        random.seed(seed)
        perm = list(range(n))
        random.shuffle(perm)
        split = random.randint(7, 9)
        A = set(perm[:split])
        B = set(perm[split:])
        
        adj = np.zeros((n, n), dtype=bool)
        for u in A:
            for v in B:
                adj[u, v] = True
                adj[v, u] = True
        
        edge_count = len(A) * len(B)
        remaining = target - edge_count
        
        if remaining > 0:
            internal = []
            for part in [A, B]:
                plist = list(part)
                for i in range(len(plist)):
                    for j in range(i+1, len(plist)):
                        internal.append((plist[i], plist[j]))
            random.shuffle(internal)
            for u, v in internal[:remaining]:
                adj[u, v] = True
                adj[v, u] = True
        elif remaining < 0:
            cross = [(u, v) for u in A for v in B]
            random.shuffle(cross)
            for u, v in cross[:abs(remaining)]:
                adj[u, v] = False
                adj[v, u] = False
        
        constructions.append(adj.copy())
    
    return constructions

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_score = float('inf')
    best_adj = None
    
    def _elapsed():
        return time.time() - start_time
    
    def _time_left():
        return DEADLINE - _elapsed()
    
    def _find_worst_triangle(adj, deg):
        """Find the triangle with the maximum degree sum."""
        max_sum = -1
        worst_tri = None
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                for w in range(v + 1, n):
                    if adj[u, w] and adj[v, w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                            worst_tri = (u, v, w)
        return max_sum, worst_tri

    def _try_move(adj, deg, remove_edge, add_edge):
        """Try a specific move, return (new_adj, new_deg, new_score) or None if invalid."""
        if remove_edge == add_edge:
            return None
        u1, v1 = remove_edge
        u2, v2 = add_edge
        
        if not adj[u1, v1]:
            return None
        if adj[u2, v2]:
            return None
            
        new_adj = adj.copy()
        new_deg = deg.copy()
        
        new_adj[u1, v1] = False
        new_adj[v1, u1] = False
        new_deg[u1] -= 1
        new_deg[v1] -= 1
        
        new_adj[u2, v2] = True
        new_adj[v2, u2] = True
        new_deg[u2] += 1
        new_deg[v2] += 1
        
        new_score = _compute_score(new_adj, new_deg)
        return new_adj, new_deg, new_score

    # Try each initial construction
    for init_adj in _initial_constructions():
        if _time_left() < 10:
            break
        
        deg = init_adj.sum(axis=1).astype(int)
        score = _compute_score(init_adj, deg)
        
        if score < best_score:
            best_score = score
            best_adj = init_adj.copy()
        
        # Targeted Local Search
        current_adj = init_adj.copy()
        current_deg = deg.copy()
        current_score = score
        
        T = 1.0
        cooling = 0.999
        no_improve = 0
        max_no_improve = 5000
        
        for it in range(100000):
            if _time_left() < 5:
                break
            if no_improve > max_no_improve:
                break
            
            max_sum, worst_tri = _find_worst_triangle(current_adj, current_deg)
            
            # Generate candidate moves
            candidates = []
            
            if worst_tri:
                u, v, w = worst_tri
                # Try removing edges from the worst triangle
                for rem in [(u, v), (u, w), (v, w)]:
                    # Pick a random add edge, biased towards low degree vertices
                    # to keep degrees balanced
                    add_candidates = []
                    for a, b in all_pairs:
                        if current_adj[a, b]:
                            continue
                        # Heuristic: prefer edges that don't connect to high degree nodes
                        # or create triangles with low sums
                        # Simple heuristic: random selection is okay, but let's bias
                        # towards edges within the "same side" if we detect bipartiteness
                        # For now, just random but exclude edges that are obviously bad?
                        # Actually, just random add edge is fine if we have many iterations.
                        add_candidates.append((a, b))
                    
                    if add_candidates:
                        # Pick a few random add edges
                        for _ in range(5):
                            add = random.choice(add_candidates)
                            candidates.append((rem, add))
            
            # Also try random moves if stuck or to escape local minima
            if random.random() < 0.2 or not worst_tri:
                for _ in range(10):
                    rem = random.choice(all_pairs)
                    if not current_adj[rem[0], rem[1]]:
                        continue
                    add = random.choice(all_pairs)
                    if current_adj[add[0], add[1]]:
                        continue
                    candidates.append((rem, add))
            
            best_candidate = None
            best_candidate_score = float('inf')
            
            for rem, add in candidates:
                if _time_left() < 5:
                    break
                result = _try_move(current_adj, current_deg, rem, add)
                if result is None:
                    continue
                new_adj, new_deg, new_score = result
                
                if new_score < best_candidate_score:
                    best_candidate_score = new_score
                    best_candidate = (new_adj, new_deg, new_score)
            
            if best_candidate is None:
                no_improve += 1
                T *= cooling
                continue
                
            new_adj, new_deg, new_score = best_candidate
            delta = new_score - current_score
            
            # Accept if better, or with probability if worse (Simulated Annealing)
            if delta < 0 or (T > 0.01 and random.random() < np.exp(-delta / T)):
                current_adj = new_adj
                current_deg = new_deg
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            T *= cooling
        
        # Final hill-climbing pass on best found so far
        if _time_left() < 5:
            break
    
    # Final exhaustive local search on best_adj
    if best_adj is not None and _time_left() > 10:
        current_adj = best_adj.copy()
        current_deg = current_adj.sum(axis=1).astype(int)
        current_score = _compute_score(current_adj, current_deg)
        
        improved = True
        while improved and _time_left() > 5:
            improved = False
            max_sum, worst_tri = _find_worst_triangle(current_adj, current_deg)
            if not worst_tri:
                break
            
            u, v, w = worst_tri
            # Try removing edges from the worst triangle
            for rem in [(u, v), (u, w), (v, w)]:
                if _time_left() < 5:
                    break
                if not current_adj[rem[0], rem[1]]:
                    continue
                ru, rv = rem
                
                # Try adding any non-edge
                for add in all_pairs:
                    if _time_left() < 5:
                        break
                    if add == rem or current_adj[add[0], add[1]]:
                        continue
                    
                    test_adj = current_adj.copy()
                    test_deg = current_deg.copy()
                    
                    test_adj[ru, rv] = False
                    test_adj[rv, ru] = False
                    test_deg[ru] -= 1
                    test_deg[rv] -= 1
                    
                    test_adj[add[0], add[1]] = True
                    test_adj[add[1], add[0]] = True
                    test_deg[add[0]] += 1
                    test_deg[add[1]] += 1
                    
                    s = _compute_score(test_adj, test_deg)
                    if s < current_score:
                        current_score = s
                        current_adj = test_adj
                        current_deg = test_deg
                        improved = True
                        if s < best_score:
                            best_score = s
                            best_adj = current_adj.copy()
                        break
                if improved:
                    break
            if improved:
                # Restart the while loop to re-evaluate
                continue

    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
