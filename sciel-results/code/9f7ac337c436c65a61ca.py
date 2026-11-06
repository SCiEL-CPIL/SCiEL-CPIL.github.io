import random
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for cleanup/return
    
    n = 16
    # Generate all triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Generate all K(2,2,2)s
    k222s = []
    
    def get_pairings(items):
        if not items:
            yield []
            return
        first = items[0]
        rest = items[1:]
        for i, second in enumerate(rest):
            pair = (first, second)
            remaining = rest[:i] + rest[i+1:]
            for sub_pairings in get_pairings(remaining):
                yield [pair] + sub_pairings

    for verts in combinations(range(n), 6):
        seen_pairings = set()
        for pairing in get_pairings(verts):
            sorted_pairs = tuple(sorted(tuple(sorted(p)) for p in pairing))
            if sorted_pairs in seen_pairings:
                continue
            seen_pairings.add(sorted_pairs)
            
            p1, p2, p3 = sorted_pairs
            edges = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        t = (a, b, c)
                        edges.append(triple_to_idx[t])
            k222s.append(tuple(edges))
            
    num_k222s = len(k222s)
    
    # Map edge index to list of k222 indices
    edge_to_k222s = [[] for _ in range(num_triples)]
    for k_idx, edges in enumerate(k222s):
        for e_idx in edges:
            edge_to_k222s[e_idx].append(k_idx)
    
    # State
    present = [False] * num_triples
    counts = [0] * num_k222s
    
    def add_edge(e_idx):
        for k_idx in edge_to_k222s[e_idx]:
            if counts[k_idx] >= 7:
                return False
        present[e_idx] = True
        for k_idx in edge_to_k222s[e_idx]:
            counts[k_idx] += 1
        return True
        
    def remove_edge(e_idx):
        if not present[e_idx]:
            return
        present[e_idx] = False
        for k_idx in edge_to_k222s[e_idx]:
            counts[k_idx] -= 1
            
    def get_num_edges():
        return sum(present)
        
    def greedy_add():
        candidates = [i for i in range(num_triples) if not present[i]]
        random.shuffle(candidates)
        for e_idx in candidates:
            add_edge(e_idx)
            
    def repair():
        for k_idx in range(num_k222s):
            if counts[k_idx] >= 8:
                # Remove a random edge from this k222
                e_idx = random.choice(k222s[k_idx])
                remove_edge(e_idx)
        
    # Initialize
    for i in range(num_triples):
        if random.random() < 0.5:
            present[i] = True
            for k_idx in edge_to_k222s[i]:
                counts[k_idx] += 1
                
    repair()
    greedy_add()
    
    best_edges = get_num_edges()
    best_state = present[:]
    
    # Iterated Local Search
    while time.time() < deadline:
        # Perturbation
        present_indices = [i for i in range(num_triples) if present[i]]
        if len(present_indices) > 10:
            num_remove = random.randint(5, 15)
            to_remove = random.sample(present_indices, num_remove)
            for e_idx in to_remove:
                remove_edge(e_idx)
                
        greedy_add()
        curr_edges = get_num_edges()
        
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_state = present[:]
            
    # Convert best_state to list of triples
    result = []
    for i, is_present in enumerate(best_state):
        if is_present:
            result.append(list(triples[i]))
            
    return result
