import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Fallback: Paley graph on 17 vertices
    def get_paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}

    best_result = get_paley_17()
    best_n = 17

    # Target n values
    targets = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]

    for n in targets:
        if time.time() > deadline:
            break
            
        # Try to find a valid graph for this n
        result = try_find_graph_sa(n, deadline)
        if result:
            if is_valid(result):
                if n > best_n:
                    best_result = result
                    best_n = n
                # If we found 35, it's the known max, return immediately
                if n >= 35:
                    return best_result

    return best_result

def is_valid(graph):
    n = graph['n']
    edges = graph['edges']
    A = [[0]*n for _ in range(n)]
    for u, v in edges:
        A[u][v] = 1
        A[v][u] = 1

    # Check K4
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                for k in range(j+1, n):
                    if A[i][k] and A[j][k]:
                        for l in range(k+1, n):
                            if A[i][l] and A[j][l] and A[k][l]:
                                return False

    # Check Independent 6-set
    for i in range(n):
        for j in range(i+1, n):
            if not A[i][j]:
                for k in range(j+1, n):
                    if not A[i][k] and not A[j][k]:
                        for l in range(k+1, n):
                            if not A[i][l] and not A[j][l] and not A[k][l]:
                                for m in range(l+1, n):
                                    if not A[i][m] and not A[j][m] and not A[k][m] and not A[l][m]:
                                        for p in range(m+1, n):
                                            if not A[i][p] and not A[j][p] and not A[k][p] and not A[l][p] and not A[m][p]:
                                                return False
    return True

def try_find_graph_sa(n, deadline):
    # Simulated Annealing / Local Search with Bitmasks
    max_restarts = 50
    max_steps_per_restart = 5000
    
    # Different densities for initialization
    densities = [0.5, 0.45, 0.55, 0.4, 0.6]
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None
            
        p = densities[restart % len(densities)]
        
        # Initialize adjacency as list of bitmasks
        neighbors = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    neighbors[i] |= (1 << j)
                    neighbors[j] |= (1 << i)
        
        current_energy = calculate_energy(neighbors, n)
        
        if current_energy == 0:
            return build_graph(neighbors, n)

        best_energy = current_energy
        best_neighbors = neighbors[:]
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                return None
                
            # Pick a random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            # Flip edge
            if neighbors[u] & (1 << v):
                neighbors[u] &= ~(1 << v)
                neighbors[v] &= ~(1 << u)
            else:
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                
            new_energy = calculate_energy(neighbors, n)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_neighbors = neighbors[:]
                
                if new_energy == 0:
                    return build_graph(neighbors, n)
            else:
                # Revert if it's not an improvement? 
                # In simple hill climbing, yes. In SA, maybe accept worse.
                # Let's use simple hill climbing with restarts for stability.
                if neighbors[u] & (1 << v):
                    neighbors[u] &= ~(1 << v)
                    neighbors[v] &= ~(1 << u)
                else:
                    neighbors[u] |= (1 << v)
                    neighbors[v] |= (1 << u)
                    
        # End of steps for this restart
        # Use best found in this restart for next? Or just restart fresh?
        # Restart fresh is simpler.

    return None

def calculate_energy(neighbors, n):
    # Count K4s
    k4_count = 0
    for i in range(n):
        ni = neighbors[i]
        for j in range(i+1, n):
            if ni & (1 << j):
                common = ni & neighbors[j]
                # Count edges in common
                # If common has any edge, it forms K4s
                # To be fast, just check if common has an edge
                # But we need count?
                # Let's just count if there is ANY K4.
                # If we want gradient, we need count.
                # For now, let's just return 1 if any violation, 0 if none.
                # This is "Min-Conflict" style.
                
                # Check if any edge exists in common
                temp = common
                while temp:
                    lsb = temp & -temp
                    k = lsb.bit_length() - 1
                    if neighbors[k] & common:
                        # Found a K4
                        return 1
                    temp ^= lsb
                    
    # Check I6
    # Use backtracking to find one I6
    if find_indep_6(neighbors, n):
        return 1
        
    return 0

def find_indep_6(neighbors, n):
    # Find an independent set of size 6
    # This is finding a clique of size 6 in the complement
    comp_neighbors = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_neighbors[i] = (~neighbors[i] & all_mask) & ~(1 << i)
        
    def backtrack(clique, candidates):
        if len(clique) == 6:
            return True
        if not candidates:
            return False
        if len(clique) + len(candidates) < 6:
            return False
            
        for idx, v in enumerate(candidates):
            # Check if v connects to all in clique in comp
            valid = True
            for u in clique:
                if not (comp_neighbors[u] & (1 << v)):
                    valid = False
                    break
            if valid:
                new_candidates = []
                for w in candidates[idx+1:]:
                    if comp_neighbors[v] & (1 << w):
                        new_candidates.append(w)
                if backtrack(clique + [v], new_candidates):
                    return True
        return False
        
    # Order by degree in comp
    degrees = [bin(comp_neighbors[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    return backtrack([], order)

def build_graph(neighbors, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if neighbors[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
