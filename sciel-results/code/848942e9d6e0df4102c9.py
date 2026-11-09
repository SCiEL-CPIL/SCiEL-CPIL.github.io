import time
import random
import sys

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
        """Greedy: repeatedly pick maximal clique covering most uncovered edges."""
        # Find all maximal cliques
        max_cliques = bron_kerbosch_maximal_cliques(adj, n, deadline)
        
        if not max_cliques:
            return []
        
        # Sort by size descending (larger cliques cover more edges)
        max_cliques.sort(key=len, reverse=True)
        
        # Track uncovered edges
        uncovered = [0] * n
        for u in range(n):
            uncovered[u] = adj[u]
        
        total_edges = sum(adj[u].bit_count() for u in range(n)) // 2
        if total_edges == 0:
            return []
        
        partition = []
        remaining = total_edges
        
        # Greedy set cover: pick clique covering most uncovered edges
        available = list(range(len(max_cliques)))
        
        while remaining > 0 and time.time() < deadline:
            best_idx = -1
            best_covered = 0
            
            for idx in available:
                C = max_cliques[idx]
                if len(C) < 2:
                    continue
                # Count how many edges of this clique are still uncovered
                covered = 0
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if uncovered[u] & (1 << v):
                            covered += 1
                if covered > best_covered:
                    best_covered = covered
                    best_idx = idx
            
            if best_idx == -1 or best_covered == 0:
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
        """Merge cliques that together form a clique."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            i = 0
            while i < len(partition) and time.time() < deadline:
                merged_any = False
                for j in range(i + 1, len(partition)):
                    C1, C2 = partition[i], partition[j]
                    # Check if C1 ∪ C2 is a clique
                    is_clique = True
                    for u in C1:
                        for v in C2:
                            if not (adj[u] & (1 << v)):
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        partition[i] = C1 + C2
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
        
        # Try greedy maximal clique approach
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
