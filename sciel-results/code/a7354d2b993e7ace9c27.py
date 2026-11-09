import time
import random
import sys
import itertools

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 145

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def iter_bits(x):
        while x:
            lsb = x & -x
            yield lsb.bit_length() - 1
            x ^= lsb

    def bron_kerbosch_maximal_cliques(adj, n, deadline):
        """Find all maximal cliques using Bron-Kerbosch with pivoting."""
        cliques = []
        
        def bk(R, P, X):
            if time.time() > deadline:
                return
            if P == 0 and X == 0:
                if len(R) >= 2:
                    cliques.append(R[:])
                return
            
            # Choose pivot u from P ∪ X to maximize |P ∩ N(u)|
            PX = P | X
            best_u = -1
            best_count = -1
            tmp = PX
            while tmp:
                lsb = tmp & -tmp
                u = lsb.bit_length() - 1
                cnt = (P & adj[u]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
                tmp ^= lsb
            
            if best_u == -1:
                return
            
            # Recurse on P \ N(u)
            candidates = P & ~adj[best_u]
            tmp = candidates
            while tmp:
                lsb = tmp & -tmp
                v = lsb.bit_length() - 1
                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P &= ~lsb
                X |= lsb
                tmp ^= lsb
        
        bk([], (1 << n) - 1, 0)
        return cliques

    def greedy_edge_partition(adj, n, deadline):
        """Greedy set cover with Lagrangian-style scoring: prioritize cliques that cover
        edges that are hard to cover (fewer candidate cliques contain them)."""
        # Find all maximal cliques
        max_cliques = bron_kerbosch_maximal_cliques(adj, n, deadline)
        
        if not max_cliques:
            return []
        
        # Track uncovered edges
        uncovered = [0] * n
        for u in range(n):
            uncovered[u] = adj[u]
        
        total_edges = sum(adj[u].bit_count() for u in range(n)) // 2
        if total_edges == 0:
            return []
        
        partition = []
        remaining = total_edges
        
        # Precompute edge-to-clique incidence for "hardness" estimation
        # For each edge, count how many maximal cliques contain it
        edge_clique_count = {}
        for C in max_cliques:
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    key = (min(u, v), max(u, v))
                    if key in adj[u] or adj[v] & (1 << u):  # edge exists
                        edge_clique_count[key] = edge_clique_count.get(key, 0) + 1
        
        # Greedy set cover with weighted scoring
        available = list(range(len(max_cliques)))
        
        while remaining > 0 and time.time() < deadline:
            best_idx = -1
            best_score = -1
            
            for idx in available:
                C = max_cliques[idx]
                if len(C) < 2:
                    continue
                # Score: sum of inverse clique counts for uncovered edges in this clique
                # (edges in fewer cliques are "harder" to cover, so weight them more)
                score = 0.0
                covered_count = 0
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if uncovered[u] & (1 << v):
                            key = (min(u, v), max(u, v))
                            cnt = edge_clique_count.get(key, 1)
                            score += 1.0 / max(cnt, 1)
                            covered_count += 1
                if covered_count == 0:
                    continue
                # Lagrangian-style: prefer higher score, break ties by more edges covered
                if score > best_score or (abs(score - best_score) < 1e-9 and covered_count > best_score):
                    best_score = score
                    best_idx = idx
            
            if best_idx == -1 or best_score <= 0:
                # Fallback: find any uncovered edge and make a 2-clique
                found = False
                for u in range(n):
                    tmp = uncovered[u]
                    while tmp:
                        lsb = tmp & -tmp
                        v = lsb.bit_length() - 1
                        if v > u:
                            partition.append([u, v])
                            uncovered[u] &= ~(1 << v)
                            uncovered[v] &= ~(1 << u)
                            remaining -= 1
                            found = True
                            break
                        tmp ^= lsb
                    if found:
                        break
                if not found:
                    break
                continue
            
            C = max_cliques[best_idx]
            partition.append(C[:])
            available.remove(best_idx)
            
            # Remove covered edges
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    if uncovered[u] & (1 << v):
                        uncovered[u] &= ~(1 << v)
                        uncovered[v] &= ~(1 << u)
                        remaining -= 1
        
        return partition

    def merge_cliques(partition, adj, n, deadline):
        """Merge cliques that together form a clique. Uses bitmask for speed."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            i = 0
            while i < len(partition) and time.time() < deadline:
                merged_any = False
                C1_mask = 0
                for u in partition[i]:
                    C1_mask |= (1 << u)
                
                for j in range(i + 1, len(partition)):
                    C2 = partition[j]
                    # Check if C1 ∪ C2 is a clique using bitmasks
                    is_clique = True
                    C2_mask = 0
                    for u in C2:
                        C2_mask |= (1 << u)
                    # Every vertex in C1 must be adjacent to every vertex in C2
                    for u in partition[i]:
                        if not (adj[u] & C2_mask) == C2_mask:
                            is_clique = False
                            break
                    if is_clique:
                        partition[i] = partition[i] + C2
                        del partition[j]
                        changed = True
                        merged_any = True
                        break
                if merged_any:
                    i = 0
                else:
                    i += 1
        return partition

    def validate(cliques, adj, n):
        for C in cliques:
            if len(C) < 2:
                return False
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    if not (adj[u] & (1 << v)):
                        return False
        edge_set = set()
        for C in cliques:
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        return False
                    edge_set.add(key)
        for u in range(n):
            for v in iter_bits(adj[u]):
                if v > u:
                    if (u, v) not in edge_set:
                        return False
        return True

    def solve_instance(n, p, seed, time_budget):
        adj = get_graph(n, p, seed)
        
        if not any(adj):
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        deadline = start_time + time_budget
        
        # Try greedy maximal clique approach with Lagrangian scoring
        try:
            partition = greedy_edge_partition(adj, n, deadline)
            if partition:
                partition = merge_cliques(partition, adj, n, deadline)
                if validate(partition, adj, n):
                    best_cliques = partition
                    best_count = len(partition)
        except:
            pass
        
        # Fallback: individual edges
        if best_cliques is None:
            best_cliques = []
            for u in range(n):
                for v in iter_bits(adj[u]):
                    if v > u:
                        best_cliques.append([u, v])
        
        return best_cliques

    results = []
    t1 = min(50, TOTAL_BUDGET * 0.4)
    results.append(solve_instance(32, 0.6, 601, t1))
    
    elapsed = time.time() - start_time
    t2 = max(15, TOTAL_BUDGET - elapsed - 2)
    results.append(solve_instance(64, 0.55, 602, t2))
    
    return results
