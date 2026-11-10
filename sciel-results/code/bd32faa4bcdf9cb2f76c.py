"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                lac = lcm_table[a][c]
                if lac != lab:
                    continue
                lbc = lcm_table[b][c]
                if lbc == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        return list(range(1, n+1))
    
    # For each x, store list of (a,b) pairs that with x form a bad triple
    x_conflict_pairs = {x: [] for x in range(1, n+1)}
    for (a, b, c) in bad_triples:
        x_conflict_pairs[a].append((b, c))
        x_conflict_pairs[b].append((a, c))
        x_conflict_pairs[c].append((a, b))
    
    # Count how many bad triples each element is in
    elem_triple_count = [0]*(n+1)
    for (a, b, c) in bad_triples:
        elem_triple_count[a] += 1
        elem_triple_count[b] += 1
        elem_triple_count[c] += 1
    
    def can_add(S_set, x):
        """Check if adding x to S_set creates any bad triple."""
        for (a, b) in x_conflict_pairs[x]:
            if a in S_set and b in S_set:
                return False
        return True
    
    def greedy_add(S_list, S_set, candidates):
        """Add elements from candidates in order, skipping those that conflict."""
        for x in candidates:
            if time.time() > deadline:
                break
            if x not in S_set and can_add(S_set, x):
                S_list.append(x)
                S_set.add(x)
    
    def local_search_improve(best_set, best_size, max_iterations=50):
        """Perform 1-swap and kick moves to improve the set."""
        S_set = set(best_set)
        improved = True
        iterations = 0
        
        while improved and time.time() < deadline and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try 1-swap: remove x from S, add y not in S
            for x in list(best_set):
                if time.time() > deadline:
                    break
                new_S_set = S_set - {x}
                # Find best y to add
                best_y = None
                for y in range(1, n+1):
                    if y in S_set:
                        continue
                    if can_add(new_S_set, y):
                        best_y = y
                        break
                if best_y is not None:
                    best_set = [z for z in best_set if z != x] + [best_y]
                    S_set = set(best_set)
                    improved = True
                    break
            
            # Try kick: remove one element, greedily add as many as possible
            if not improved and time.time() < deadline:
                for x in list(best_set):
                    if time.time() > deadline:
                        break
                    new_S_set = S_set - {x}
                    # Try to add multiple elements
                    added = []
                    for y in range(1, n+1):
                        if time.time() > deadline:
                            break
                        if y in S_set or y in added:
                            continue
                        if can_add(new_S_set | set(added), y):
                            added.append(y)
                    
                    if len(added) > 1:
                        new_set = [z for z in best_set if z != x] + added
                        if len(new_set) > best_size:
                            best_set = new_set
                            best_size = len(new_set)
                            S_set = set(new_set)
                            improved = True
                            break
        
        # Final greedy add
        if time.time() < deadline:
            S_set = set(best_set)
            for x in range(1, n+1):
                if time.time() > deadline:
                    break
                if x not in S_set and can_add(S_set, x):
                    best_set.append(x)
                    S_set.add(x)
                    best_size = len(best_set)
        
        return best_set, best_size
    
    random.seed(42)
    
    best_set = None
    best_size = 0
    
    # Multiple random restarts with different seeds
    for restart in range(30):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Deterministic seed: large range
            candidates = list(range(65, 129))
        elif restart == 1:
            # Large range + some smaller
            candidates = list(range(65, 129)) + list(range(33, 65))
        elif restart == 2:
            # Random order
            candidates = list(range(1, n+1))
            random.shuffle(candidates)
        elif restart == 3:
            # Sorted by ascending triple count (least constrained first)
            candidates = list(range(1, n+1))
            candidates.sort(key=lambda x: elem_triple_count[x])
        elif restart == 4:
            # Sorted by descending triple count
            candidates = list(range(1, n+1))
            candidates.sort(key=lambda x: -elem_triple_count[x])
        elif restart < 15:
            # Random subsets of large range
            large_range = list(range(65, 129))
            random.shuffle(large_range)
            candidates = large_range + [x for x in range(1, 65) if random.random() < 0.4]
        elif restart < 25:
            # Start with mid-range
            mid_range = list(range(40, 100))
            random.shuffle(mid_range)
            candidates = mid_range + [x for x in range(1, 40) if random.random() < 0.5] + [x for x in range(100, 129) if random.random() < 0.5]
        else:
            candidates = list(range(1, n+1))
            random.shuffle(candidates)
        
        # Greedy construction
        S_list = []
        S_set = set()
        greedy_add(S_list, S_set, candidates)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = list(S_list)
        
        # Local search to improve
        if time.time() < deadline:
            best_set, best_size = local_search_improve(best_set, best_size)
    
    # Final: ensure we have at least one element
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
