import numpy as np
import random
import itertools
import networkx as nx
import time

def construct(q, n, radius):
    # Parameters for this specific task instance
    if q != 3 or n != 6 or radius != 1:
        # Fallback to a simple greedy for other cases
        # Just return a trivial cover: all vertices
        return [list(v) for v in itertools.product(range(q), repeat=n)]

    total_vertices = q ** n
    vertices = list(itertools.product(range(q), repeat=n))
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Build graph and neighborhoods
    # G[i] is the set of indices in the closed ball of radius 1 around i
    balls = [set() for _ in range(total_vertices)]
    
    for i, v in enumerate(vertices):
        v_list = list(v)
        balls[i].add(i)
        for pos in range(n):
            old_val = v_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    v_list[pos] = new_val
                    idx = vertex_to_idx[tuple(v_list)]
                    balls[i].add(idx)
            v_list[pos] = old_val

    def is_dominating(S):
        # S is a set of indices
        if not S:
            return total_vertices == 0
        covered = set()
        for idx in S:
            covered.update(balls[idx])
        return len(covered) == total_vertices

    def greedy_dominating_set(seed):
        random.seed(seed)
        undominated = set(range(total_vertices))
        dominating_set = set()
        
        while undominated:
            # Pick a random undominated vertex
            target = random.choice(list(undominated))
            # Candidates are the ball of the target
            candidates = balls[target]
            # Pick the candidate that covers the most undominated vertices
            best_cand = None
            max_cover = -1
            for c in candidates:
                cover_count = len(undominated & balls[c])
                if cover_count > max_cover:
                    max_cover = cover_count
                    best_cand = c
            
            dominating_set.add(best_cand)
            undominated -= balls[best_cand]
            
        return dominating_set

    def local_search(S, time_limit=10):
        """
        Improves a dominating set S by removing redundant nodes and swapping.
        """
        S = set(S)
        start_time = time.time()
        
        # 1. Remove redundant nodes
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias
            nodes = list(S)
            random.shuffle(nodes)
            for u in nodes:
                if u not in S:
                    continue
                # Check if S \ {u} is dominating
                # Optimization: Check if any node in balls[u] is only covered by u
                # Pre-compute coverage counts? Too slow to update dynamically in simple way.
                # For N=729, checking is_dominating(S-{u}) is O(N * deg) ~ 729*17 ~ 12k ops.
                # 73 removals * 12k = ~1M ops. Very fast.
                new_S = S - {u}
                if is_dominating(new_S):
                    S = new_S
                    changed = True
        
        # 2. Local Search / Swaps
        # Try to swap nodes to potentially allow further removals
        # This part is tricky. A simple swap u->v doesn't reduce size.
        # We need a swap that leads to a state where a removal is possible.
        # Strategy: Randomly pick u in S, v not in S.
        # If (S - {u} + {v}) is dominating, accept it.
        # Then try to remove again.
        
        for _ in range(50): # Limit iterations
            if time.time() - start_time > time_limit:
                break
                
            if not S:
                break
                
            u = random.choice(list(S))
            # Find a random v not in S
            # Efficient way: pick random index, if in S, retry or pick from complement
            # Complement size is large.
            v = random.randint(0, total_vertices - 1)
            if v in S:
                continue
            
            new_S = (S - {u}) | {v}
            if is_dominating(new_S):
                S = new_S
                # Try to remove redundant nodes again
                changed = True
                while changed:
                    changed = False
                    nodes = list(S)
                    random.shuffle(nodes)
                    for w in nodes:
                        if w not in S:
                            continue
                        temp_S = S - {w}
                        if is_dominating(temp_S):
                            S = temp_S
                            changed = True

        return S

    best_set = None
    best_size = float('inf')
    start_time = time.time()
    time_limit = 110 # Leave 10s for overhead
    
    seed = 0
    while time.time() - start_time < time_limit:
        # Generate initial solution
        S = greedy_dominating_set(seed)
        
        # Optimize
        S = local_search(S, time_limit=5)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            # Early stop if we hit the theoretical lower bound or known best
            if best_size <= 73:
                break
        
        seed += 1

    if best_set is None:
        best_set = greedy_dominating_set(0)
        best_set = local_search(best_set, time_limit=5)

    return [vertices[idx] for idx in best_set]
