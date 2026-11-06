"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # 1. Precompute all bad triples
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We store conflicts for each vertex v: list of pairs (a, b) such that {v, a, b} is bad
    # To make it efficient, we store for each v, a set of "conflict pairs"
    # But checking "both a and b in S" is O(1) if we use a boolean array for S.
    
    N = 128
    vertices = list(range(1, N + 1))
    
    # conflicts[v] will be a list of tuples (a, b)
    conflicts = {v: [] for v in vertices}
    
    # Precompute LCMs for all pairs to speed up triple checking?
    # There are ~8000 pairs. Storing lcm is fine.
    lcm_pair = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_pair[(i, j)] = lcm(i, j)
    
    # Find all bad triples
    # Iterate over combinations of 3
    # To optimize, we can group by LCM?
    # But N=128 is small. C(128,3) = 340,000.
    # Checking 340,000 triples with precomputed pair LCMs is very fast in Python.
    
    for a, b, c in combinations(vertices, 3):
        l_ab = lcm_pair[(a, b)]
        l_ac = lcm_pair[(a, c)]
        if l_ab == l_ac:
            l_bc = lcm_pair[(b, c)]
            if l_ab == l_bc:
                # This triple is bad
                # Add (b, c) to conflicts[a]
                # Add (a, c) to conflicts[b]
                # Add (a, b) to conflicts[c]
                conflicts[a].append((b, c))
                conflicts[b].append((a, c))
                conflicts[c].append((a, b))
    
    # 2. Greedy Initialization
    def greedy_init():
        selected = set()
        # Randomize order for diversity
        vs = vertices[:]
        random.shuffle(vs)
        for v in vs:
            # Check if v can be added
            can_add = True
            for u, w in conflicts[v]:
                if u in selected and w in selected:
                    can_add = False
                    break
            if can_add:
                selected.add(v)
        return selected
    
    # 3. Local Search
    def local_search(init_set):
        S = set(init_set)
        best_size = len(S)
        best_set = set(S)
        
        # Helper to check if a vertex v can be added to a set S
        def can_add(v, S):
            for u, w in conflicts[v]:
                if u in S and w in S:
                    return False
            return True
            
        # Helper to find minimal hitting set for conflicts of v with S
        # Returns a list of vertices in S that must be removed to add v
        # We want a small set R such that for every (u, w) in conflicts[v] with u,w in S, R intersects {u, w}.
        # This is a hitting set problem on the graph of conflicts restricted to S.
        # For simplicity and speed, we use a greedy approximation or exact for small sizes.
        # Given the structure, the number of conflicting pairs is likely small.
        
        def find_removals(v, S):
            # Collect all pairs (u, w) from conflicts[v] where u, w in S
            pairs_to_hit = []
            for u, w in conflicts[v]:
                if u in S and w in S:
                    pairs_to_hit.append((u, w))
            
            if not pairs_to_hit:
                return []
            
            # Greedy Hitting Set:
            # Pick a vertex that hits the most pairs.
            # This is an approximation.
            # For better results, if len(pairs_to_hit) is small, we can try all subsets?
            # No, let's stick to a simple greedy or just check if size 1 is possible.
            
            # Check if there is a single vertex that hits all pairs
            # This is the intersection of all pairs? No, it's a vertex common to all pairs?
            # If all pairs share a common vertex, that vertex is a hitting set of size 1.
            # Otherwise, we might need size 2.
            
            # Let's try to find a hitting set of size 1
            candidates = set()
            for u, w in pairs_to_hit:
                candidates.add(u)
                candidates.add(w)
            
            for c in candidates:
                hits_all = True
                for u, w in pairs_to_hit:
                    if c != u and c != w:
                        hits_all = False
                        break
                if hits_all:
                    return [c]
            
            # If no size 1, try size 2
            # If we can't find a small hitting set, we return a large set (indicating high cost)
            # Greedy approximation:
            remaining_pairs = set(pairs_to_hit)
            hitting_set = []
            while remaining_pairs:
                # Find vertex with max degree in the hypergraph of pairs
                deg = {}
                for u, w in remaining_pairs:
                    deg[u] = deg.get(u, 0) + 1
                    deg[w] = deg.get(w, 0) + 1
                
                if not deg:
                    break
                    
                best_v = max(deg, key=deg.get)
                hitting_set.append(best_v)
                
                # Remove all pairs hit by best_v
                new_remaining = set()
                for u, w in remaining_pairs:
                    if u != best_v and w != best_v:
                        new_remaining.add((u, w))
                remaining_pairs = new_remaining
                
                # Safety break
                if len(hitting_set) > 5:
                    break
                    
            return hitting_set

        # Main Loop
        no_improvement_count = 0
        while time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to add a vertex with 0 cost (no removals)
            added = False
            # Randomize order of candidates
            candidates = [v for v in vertices if v not in S]
            random.shuffle(candidates)
            for v in candidates:
                if can_add(v, S):
                    S.add(v)
                    improved = True
                    added = True
                    break
            
            if not added:
                # 2. Try 1-for-1 swaps (remove 1, add 1)
                # We look for v not in S, and a hitting set R of size 1
                swapped = False
                candidates = [v for v in vertices if v not in S]
                random.shuffle(candidates)
                for v in candidates:
                    rem = find_removals(v, S)
                    if len(rem) == 1:
                        u = rem[0]
                        # Perform swap
                        S.remove(u)
                        S.add(v)
                        improved = True
                        swapped = True
                        break
                
                if not swapped:
                    # 3. Try random 1-for-2 swaps (remove 2, add 1) - usually bad, but might escape
                    # Or try to remove a vertex to allow 2 additions?
                    # Let's try a random perturbation: remove a random vertex, then try to add greedily
                    if random.random() < 0.1: # 10% chance to perturb
                        if S:
                            victim = random.choice(list(S))
                            S.remove(victim)
                            # Try to add back greedily
                            added_any = False
                            candidates = [x for x in vertices if x not in S]
                            random.shuffle(candidates)
                            for c in candidates:
                                if can_add(c, S):
                                    S.add(c)
                                    added_any = True
                                    # Don't break, try to add more
                            if not added_any:
                                # Re-add victim if possible?
                                if can_add(victim, S):
                                    S.add(victim)
                            improved = True # Consider perturbation as a move to allow further search
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # If no improvement for a while, restart with a new greedy init
            if no_improvement_count > 50:
                S = greedy_init()
                no_improvement_count = 0
                # Ensure we don't lose the best
                if len(S) < best_size:
                    # We keep best_set separate, so this is fine. 
                    # But we want to search from a good starting point.
                    # Maybe we should merge? For now, just restart.
                    pass

        return best_set

    # Run Local Search multiple times with different seeds
    best_overall = set()
    best_size_overall = 0
    
    for i in range(10): # 10 restarts
        if time.time() > deadline - 10.0:
            break
            
        init = greedy_init()
        result = local_search(init)
        
        if len(result) > best_size_overall:
            best_size_overall = len(result)
            best_overall = result
            
        # Early exit if we found a very large set (e.g., > 100)
        if best_size_overall > 100:
            break

    return list(best_overall)

# EVOLVE-BLOCK-END
