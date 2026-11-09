import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples_and_map(max_n):
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    bad_triples = []
    conflict_map = {x: [] for x in range(1, n + 1)}
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    conflict_map[a].append((b, c))
                    conflict_map[b].append((a, c))
                    conflict_map[c].append((a, b))
                    
    return bad_triples, conflict_map

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    max_n = 128
    numbers = list(range(1, max_n + 1))
    
    bad_triples, conflict_map = _precompute_bad_triples_and_map(max_n)
    
    best_set = set()
    best_size = 0
    random.seed(42)
    
    while time.time() < deadline:
        S = set(numbers)
        
        # Phase 1: Remove elements to resolve all bad triples
        active_triples = [t for t in bad_triples if all(x in S for x in t)]
        
        while active_triples:
            if time.time() > deadline:
                break
            
            t_idx = random.randint(0, len(active_triples) - 1)
            a, b, c = active_triples[t_idx]
            
            counts = {a: 0, b: 0, c: 0}
            for t in active_triples:
                if a in t: counts[a] += 1
                if b in t: counts[b] += 1
                if c in t: counts[c] += 1
            
            max_count = max(counts.values())
            candidates = [x for x in counts if counts[x] == max_count]
            to_remove = random.choice(candidates)
            
            S.remove(to_remove)
            
            active_triples = [t for t in active_triples if to_remove not in t]
            
        # Phase 2: Try to add back elements
        candidates_to_add = [x for x in numbers if x not in S]
        random.shuffle(candidates_to_add)
        
        for x in candidates_to_add:
            if time.time() > deadline:
                break
            
            can_add = True
            for y, z in conflict_map[x]:
                if y in S and z in S:
                    can_add = False
                    break
            
            if can_add:
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    return list(best_set)
