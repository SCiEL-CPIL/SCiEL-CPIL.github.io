"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
EDGE_SET = set()
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    EDGE_SET.add((u, v))
    EDGE_SET.add((v, u))
    ADJ[u].add(v)
    ADJ[v].add(u)

def find_max_clique_size():
    """Find the size of the maximum clique using Bron-Kerbosch."""
    max_size = 0
    
    def bk(R, P_set, X):
        nonlocal max_size
        if not P_set and not X:
            if len(R) > max_size:
                max_size = len(R)
            return
        candidates = P_set | X
        if candidates:
            pivot = max(candidates, key=lambda v: len(P_set & ADJ[v]))
            candidates = P_set - ADJ[pivot]
        else:
            candidates = set()
        for v in list(candidates):
            bk(R | {v}, P_set & ADJ[v], X & ADJ[v])
            P_set.discard(v)
            X.add(v)
            
    bk(set(), set(range(N)), set())
    return max_size

def get_shortest_paths(u, v, branches_set, max_paths=10):
    """
    Get up to max_paths shortest simple paths from u to v.
    Branches (other than u, v) cannot be in the interior.
    """
    paths = []
    # BFS layer by layer to ensure shortest first
    visited = set()
    visited.add(u)
    current_paths = [(u,)]
    
    for depth in range(N):
        next_paths = []
        for path in current_paths:
            cur = path[-1]
            if cur == v:
                paths.append(path)
                if len(paths) >= max_paths:
                    return paths
                continue # Don't extend paths that reached destination
            
            for nb in sorted(ADJ[cur]):
                if nb in visited:
                    continue
                if nb in branches_set and nb != u and nb != v:
                    continue
                
                visited.add(nb)
                next_paths.append(path + (nb,))
        
        current_paths = next_paths
        if not current_paths:
            break
            
    return paths

def solve_path_packing(branches, deadline, rng):
    """
    Try to find disjoint paths for all pairs in branches.
    Returns dict mapping (u,v) -> path, or None.
    """
    bset = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    
    if not pairs:
        return {}
        
    # Precompute candidate paths for each pair
    pair_candidates = {}
    for u, v in pairs:
        cands = get_shortest_paths(u, v, bset, max_paths=5)
        if not cands:
            return None
        # Store as list of (path, interior_set)
        pair_candidates[(u,v)] = [(p, set(p[1:-1])) for p in cands]
        
    # Order pairs by number of candidates (most constrained first)
    ordered_pairs = sorted(pairs, key=lambda p: len(pair_candidates[p]))
    
    used_interiors = set()
    assigned = {}
    
    # Backtracking
    def backtrack(idx):
        if idx == len(ordered_pairs):
            return True
        
        if time.time() > deadline:
            return False
            
        u, v = ordered_pairs[idx]
        cands = pair_candidates[(u,v)]
        
        # Shuffle candidates to explore different options
        rng.shuffle(cands)
        
        for path, interiors in cands:
            # Check for conflict
            if interiors & used_interiors:
                continue
            
            # Assign
            used_interiors.update(interiors)
            assigned[(u,v)] = path
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            used_interiors.difference_update(interiors)
            del assigned[(u,v)]
            
        return False
        
    if backtrack(0):
        return assigned
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 75  # Safety margin for 90s limit
    
    # 1. Find max clique size as baseline
    omega = find_max_clique_size()
    
    rng = random.Random(42)
    
    best_result = None
    best_k = 0
    
    # 2. Try to find larger clique subdivisions
    # We iterate k from high to low? No, we want the max.
    # We can try k = 10, 9, ... down to omega+1.
    # If we find one for k, we can return it immediately? 
    # Yes, because we are checking higher k first.
    
    max_k_to_try = 12  # Upper bound heuristic
    
    for k in range(max_k_to_try, omega, -1):
        if time.time() > deadline:
            break
            
        # Generate candidate branch sets of size k
        candidates = []
        
        # Strategy A: Maximal Cliques of size close to k? 
        # We don't have them stored. Let's just use random high-degree sets.
        
        # Get top 20 vertices by degree
        degrees = [len(ADJ[i]) for i in range(N)]
        sorted_vertices = sorted(range(N), key=lambda x: degrees[x], reverse=True)
        pool = sorted_vertices[:20]
        
        # Add random sampling from pool
        # Also, ensure we try sets that are "dense"
        
        num_tries = 200
        
        for _ in range(num_tries):
            if time.time() > deadline:
                break
            # Pick k vertices from pool
            # Bias towards high degree
            chosen = rng.sample(pool, k)
            if chosen in candidates:
                continue
            candidates.append(chosen)
            
            # Also try specific structures: 
            # Take a clique of size k-1? We don't have that list.
            # Let's just stick to random for now, with more tries.
        
        # Additionally, if omega is close to k, we can modify max cliques
        # But we didn't store them. Let's re-run BK to get max cliques if k is small?
        # Or just rely on random.
        
        # For the first few k values, let's also try the actual max cliques if k <= omega
        # But we are going down from 12 to omega+1.
        
        # Let's add a few "greedy" candidates:
        # Start with top k vertices
        greedy_set = sorted_vertices[:k]
        if greedy_set not in candidates:
            candidates.append(greedy_set)
            
        # Try each candidate
        for bset in candidates:
            if time.time() > deadline:
                break
            
            # Quick check: Are all pairs connected?
            # get_shortest_paths will return empty if not.
            
            res = solve_path_packing(sorted(bset), deadline, rng)
            
            if res is not None:
                # Found a valid subdivision for k
                branches_out = [int(x) for x in sorted(bset)]
                paths_out = []
                for u, v in combinations(sorted(bset), 2):
                    # The key in res is (min, max) because of sorted(bset) in solve
                    # But solve uses combinations(sorted(branches), 2)
                    # So keys are (u,v) with u < v
                    p = res.get((u,v)) or res.get((v,u))
                    if p is None:
                        # Should not happen
                        p = [u, v]
                    paths_out.append([int(x) for x in p])
                
                best_result = {
                    'branches': branches_out,
                    'paths': paths_out
                }
                best_k = k
                # Since we are iterating k downwards, this is the maximum k found so far.
                # We can return immediately.
                return best_result
                
    # Fallback to omega (max clique)
    if best_k < omega:
        # Re-find a max clique to use as branches
        # We can just run BK again and keep the first max one
        cliques = []
        def bk2(R, P_set, X):
            if not P_set and not X:
                cliques.append(set(R))
                return
            candidates = P_set | X
            if candidates:
                pivot = max(candidates, key=lambda v: len(P_set & ADJ[v]))
                candidates = P_set - ADJ[pivot]
            else:
                candidates = set()
            for v in list(candidates):
                bk2(R | {v}, P_set & ADJ[v], X & ADJ[v])
                P_set.discard(v)
                X.add(v)
        bk2(set(), set(range(N)), set())
        
        max_clique = max(cliques, key=len)
        branches_out = [int(x) for x in sorted(max_clique)]
        paths_out = []
        for u, v in combinations(sorted(max_clique), 2):
            # In a clique, the path is just the edge
            paths_out.append([int(u), int(v)])
            
        best_result = {
            'branches': branches_out,
            'paths': paths_out
        }
        best_k = len(branches_out)

    # If even that failed (unlikely), fallback to 2
    if best_result is None:
        u, v = P['edges'][0]
        best_result = {
            'branches': [int(u), int(v)],
            'paths': [[int(u), int(v)]]
        }

    return best_result
# EVOLVE-BLOCK-END
