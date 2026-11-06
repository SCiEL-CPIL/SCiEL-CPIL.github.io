"""Rewritten program for R(4,6) graph construction."""
import time
import random
import math
from itertools import combinations

def construct():
    deadline = time.time() + 115
    random.seed(42)
    
    def extract_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def has_k4(adj, n):
        """Check if graph contains a K4 using bitmask approach."""
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(j+1, n):
                    if not adj[i][k] or not adj[j][k]:
                        continue
                    for l in range(k+1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            return True
        return False

    def has_indep6(adj, n):
        """Check if graph has an independent set of size 6 using backtracking."""
        # For each vertex, its non-neighbors form the candidates for independent set
        def backtrack(candidates, size):
            if size == 6:
                return True
            if len(candidates) < 6 - size:
                return False
            for idx, v in enumerate(candidates):
                # Next candidates must be non-neighbors of v
                new_cands = [u for u in candidates[idx+1:] if not adj[v][u]]
                if backtrack(new_cands, size + 1):
                    return True
            return False
        return backtrack(list(range(n)), 0)

    def is_valid(adj, n):
        return not has_k4(adj, n) and not has_indep6(adj, n)

    def energy(adj, n):
        """Count K4s and independent 6-sets. 0 means valid."""
        k4_count = 0
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(j+1, n):
                    if not adj[i][k] or not adj[j][k]:
                        continue
                    for l in range(k+1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            k4_count += 1
        if k4_count > 0:
            return k4_count * 10  # Weight K4s more heavily
        
        # Count independent 6-sets
        indep_count = 0
        def count_indep(candidates, size):
            nonlocal indep_count
            if size == 6:
                indep_count += 1
                return
            if len(candidates) < 6 - size:
                return
            for idx, v in enumerate(candidates):
                new_cands = [u for u in candidates[idx+1:] if not adj[v][u]]
                count_indep(new_cands, size + 1)
        count_indep(list(range(n)), 0)
        return indep_count

    def find_k4(adj, n):
        """Find one K4 if it exists."""
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(j+1, n):
                    if not adj[i][k] or not adj[j][k]:
                        continue
                    for l in range(k+1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            return [i, j, k, l]
        return None

    def find_indep6(adj, n):
        """Find one independent 6-set if it exists."""
        def backtrack(candidates, chosen):
            if len(chosen) == 6:
                return chosen
            if len(candidates) < 6 - len(chosen):
                return None
            for idx, v in enumerate(candidates):
                new_cands = [u for u in candidates[idx+1:] if not adj[v][u]]
                res = backtrack(new_cands, chosen + [v])
                if res is not None:
                    return res
            return None
        return backtrack(list(range(n)), [])

    def flip(adj, n, u, v):
        adj[u][v] = adj[v][u] = not adj[u][v]

    def greedy_repair(adj, n, max_iter=2000):
        """Greedy repair: fix violations one at a time."""
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Check for K4 first
            k4 = find_k4(adj, n)
            if k4:
                # Remove an edge from the K4 that appears in fewest other K4s
                # Simple heuristic: try each edge, pick the one that reduces energy most
                best_edge = None
                best_energy = energy(adj, n)
                for u, v in combinations(k4, 2):
                    if adj[u][v]:
                        flip(adj, n, u, v)
                        e = energy(adj, n)
                        if e < best_energy:
                            best_energy = e
                            best_edge = (u, v)
                        flip(adj, n, u, v)  # undo
                if best_edge:
                    flip(adj, n, best_edge[0], best_edge[1])
                else:
                    # Just remove first edge
                    u, v = k4[0], k4[1]
                    flip(adj, n, u, v)
                continue
            
            # Check for independent 6-set
            indep = find_indep6(adj, n)
            if indep:
                # Add an edge within the independent set
                # Try each pair, pick the one that reduces energy most
                best_edge = None
                best_energy = energy(adj, n)
                for u, v in combinations(indep, 2):
                    if not adj[u][v]:
                        flip(adj, n, u, v)
                        e = energy(adj, n)
                        if e < best_energy:
                            best_energy = e
                            best_edge = (u, v)
                        flip(adj, n, u, v)  # undo
                if best_edge:
                    flip(adj, n, best_edge[0], best_edge[1])
                else:
                    u, v = indep[0], indep[1]
                    flip(adj, n, u, v)
                continue
            
            # No violations
            break
        
        return adj

    def simulated_annealing(adj, n, start_temp, end_temp, steps):
        """Simulated annealing to minimize energy."""
        current_energy = energy(adj, n)
        best_adj = [row[:] for row in adj]
        best_energy = current_energy
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            flip(adj, n, u, v)
            new_energy = energy(adj, n)
            
            delta = new_energy - current_energy
            temp = start_temp * (end_temp / start_temp) ** (step / max(steps, 1))
            
            if delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_adj = [row[:] for row in adj]
            else:
                flip(adj, n, u, v)
        
        return best_adj, best_energy

    # Start with the known 35-vertex product graph
    n35 = 35
    D5 = {1, 4}
    D7 = {1, 2, 4}
    adj35 = [[False]*n35 for _ in range(n35)]
    for i in range(n35):
        a1, b1 = divmod(i, 7)
        for j in range(i+1, n35):
            a2, b2 = divmod(j, 7)
            da = (a2 - a1) % 5
            db = (b2 - b1) % 7
            if (da in D5 and db in D7) or (da not in D5 and db not in D7):
                adj35[i][j] = adj35[j][i] = True
    
    best_result = None
    if is_valid(adj35, n35):
        best_result = {'n': n35, 'edges': extract_edges(adj35, n35)}
    
    # Try to find valid graphs for larger n using SA + greedy repair
    for target_n in [36, 37, 38, 39, 40, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for restart in range(8):
            if time.time() > deadline - 10:
                break
            if found:
                break
            
            # Different initialization strategies
            if restart == 0 and target_n == 35:
                adj = [row[:] for row in adj35]
            elif restart == 1:
                # Start from 35-vertex graph and extend
                if target_n > 35:
                    adj = [row[:] for row in adj35]
                    # Extend to target_n by adding vertices
                    adj = [[False]*target_n for _ in range(target_n)]
                    for i in range(35):
                        for j in range(i+1, 35):
                            adj[i][j] = adj[j][i] = adj35[i][j]
                    # Connect new vertices to old ones randomly
                    for i in range(35, target_n):
                        for j in range(35):
                            if random.random() < 0.5:
                                adj[i][j] = adj[j][i] = True
                    # Connect new vertices among themselves
                    for i in range(35, target_n):
                        for j in range(i+1, target_n):
                            if random.random() < 0.45:
                                adj[i][j] = adj[j][i] = True
                else:
                    adj = [[False]*target_n for _ in range(target_n)]
                    for i in range(target_n):
                        for j in range(i+1, target_n):
                            if random.random() < 0.45:
                                adj[i][j] = adj[j][i] = True
            elif restart == 2:
                adj = [[False]*target_n for _ in range(target_n)]
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if random.random() < 0.40:
                            adj[i][j] = adj[j][i] = True
            elif restart == 3:
                adj = [[False]*target_n for _ in range(target_n)]
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if random.random() < 0.50:
                            adj[i][j] = adj[j][i] = True
            elif restart == 4:
                # Start from complement of a random graph
                adj = [[False]*target_n for _ in range(target_n)]
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if random.random() < 0.55:
                            adj[i][j] = adj[j][i] = True
            else:
                adj = [[False]*target_n for _ in range(target_n)]
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if random.random() < random.uniform(0.35, 0.55):
                            adj[i][j] = adj[j][i] = True
            
            # Greedy repair first
            adj = greedy_repair(adj, target_n, 1000)
            e = energy(adj, target_n)
            
            if e == 0:
                found = True
                if best_result is None or target_n > best_result['n']:
                    best_result = {'n': target_n, 'edges': extract_edges(adj, target_n)}
                continue
            
            # Simulated annealing
            start_temp = max(1.0, e * 0.3)
            end_temp = 0.01
            sa_steps = min(5000, max(1000, (deadline - 10 - time.time()) * 100))
            adj, best_e = simulated_annealing(adj, target_n, start_temp, end_temp, int(sa_steps))
            
            if best_e == 0:
                found = True
                if best_result is None or target_n > best_result['n']:
                    best_result = {'n': target_n, 'edges': extract_edges(adj, target_n)}
        
        if found and best_result and best_result['n'] >= target_n:
            break
    
    # Final safety: if no valid graph found, return the 35-vertex one
    if best_result is None:
        best_result = {'n': n35, 'edges': extract_edges(adj35, n35)}
    
    return best_result
