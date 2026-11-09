import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Seed: Paley graph on 37
    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges
    
    best_n, best_edges = get_paley(37)
    
    # Verification function
    def is_valid(n, edges):
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        for comb in itertools.combinations(range(n), 5):
            is_clique = True
            is_indep = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if adj[comb[i]][comb[j]]:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                return False
        return True

    # Local Search
    def local_search(n, time_limit):
        # Try multiple random starts
        max_restarts = 50
        restart_time = time_limit / max_restarts
        
        for _ in range(max_restarts):
            if time.time() > time_limit:
                break
            
            # Random initialization
            adj = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i][j] = True
                        adj[j][i] = True
            
            def get_edges(adj):
                return [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
            
            def find_violation(adj):
                for comb in itertools.combinations(range(n), 5):
                    is_clique = True
                    is_indep = True
                    for i in range(5):
                        for j in range(i + 1, 5):
                            if adj[comb[i]][comb[j]]:
                                is_indep = False
                            else:
                                is_clique = False
                            if not is_clique and not is_indep:
                                break
                        if not is_clique and not is_indep:
                            break
                    if is_clique:
                        return comb
                    if is_indep:
                        return comb
                return None
            
            # Local search loop
            while time.time() < time_limit:
                viol = find_violation(adj)
                if viol is None:
                    return get_edges(adj)
                
                # Flip a random edge in the violation
                i, j = random.choice([(viol[a], viol[b]) for a in range(5) for b in range(a + 1, 5)])
                adj[i][j] = not adj[i][j]
                adj[j][i] = adj[i][j]
                
                # Optional: Random perturbation if stuck?
                # For now, rely on restarts.
        
        return None

    # Try larger N first
    for n in [42, 41, 40, 39, 38]:
        if time.time() > deadline - 10:  # Leave 10s for fallback
            break
        
        # Allocate time based on N
        # Give more time to 42
        if n == 42:
            t_limit = deadline - 50
        elif n == 41:
            t_limit = deadline - 20
        else:
            t_limit = deadline - 5
            
        if time.time() >= t_limit:
            continue
            
        edges = local_search(n, t_limit)
        if edges is not None:
            # Double check validity before returning
            if is_valid(n, edges):
                return n, edges
            else:
                # Should not happen if local_search is correct, but safe
                pass
                
    # Fallback to seed
    return best_n, best_edges
