import time
import itertools
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin
    
    n = 15
    # Partition: A = 0..6 (size 7), B = 7..14 (size 8)
    A = list(range(7))
    B = list(range(7, 15))
    
    # Start with complete bipartite K_{7,8}
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((min(a,b), max(a,b)))
    
    # Candidate intra-partition edges
    candidates = []
    for i, j in itertools.combinations(A, 2):
        candidates.append((i, j))
    for i, j in itertools.combinations(B, 2):
        candidates.append((i, j))
    
    def get_degrees(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edge_set):
        deg = get_degrees(edge_set)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        # Check every 3-edge path s-u-v-t
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def count_conflicts(edge_set, candidate_e):
        """Count how many existing edges conflict with adding candidate_e"""
        e = (min(candidate_e[0], candidate_e[1]), max(candidate_e[0], candidate_e[1]))
        if e in edge_set:
            return 0
        test_set = set(edge_set)
        test_set.add(e)
        # Check if adding this edge creates any violation
        if is_valid(test_set):
            return 0
        # Count which existing edges are involved in violations
        deg = get_degrees(test_set)
        adj = [set() for _ in range(n)]
        for u, v in test_set:
            adj[u].add(v)
            adj[v].add(u)
        conflict_count = 0
        for u, v in edge_set:
            # Check if removing this edge would fix the issue (simplified: just count total)
            pass
        return 1  # Just indicate it's invalid
    
    best_edges = set(base_edges)
    best_count = len(base_edges)
    
    # Try multiple random orderings
    for seed in range(50):
        if time.time() > DEADLINE:
            break
        
        rng = random.Random(seed)
        cand_list = list(candidates)
        rng.shuffle(cand_list)
        
        current = set(base_edges)
        
        # Priority-based insertion: sort by "safety" (fewer conflicts)
        # We'll do greedy with backtracking-lite
        for i, j in cand_list:
            if time.time() > DEADLINE:
                break
            e = (min(i, j), max(i, j))
            if e in current:
                continue
            current.add(e)
            if is_valid(current):
                pass
            else:
                current.discard(e)
        
        if len(current) > best_count:
            best_edges = set(current)
            best_count = len(current)
        
        # Local search: try to improve by removing one edge and adding two others
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            current = set(best_edges)
            # Try removing each edge and see if we can add more
            for e in list(current):
                if time.time() > DEADLINE:
                    break
                current.discard(e)
                # Try adding candidates
                for i, j in candidates:
                    if time.time() > DEADLINE:
                        break
                    ce = (min(i, j), max(i, j))
                    if ce in current or ce == e:
                        continue
                    current.add(ce)
                    if is_valid(current):
                        # Keep this improvement
                        best_edges = set(current)
                        best_count = len(current)
                        improved = True
                    else:
                        current.discard(ce)
    
    # Convert to list format
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
