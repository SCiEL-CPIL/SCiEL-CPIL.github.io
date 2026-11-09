import random
from itertools import combinations
from math import gcd
import time

N = 128

def compute_conflict_graph():
    """
    Build the conflict graph where vertices are [1..N] and an edge (u,v) exists
    if there is some w such that {u,v,w} forms a bad triple (all pairwise GCDs equal).
    We want a maximum independent set in this graph.
    """
    # adjacency as bitsets for speed
    adj = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    # Bad triple (a, b, c): all three pairs conflict
                    adj[a] |= (1 << b) | (1 << c)
                    adj[b] |= (1 << a) | (1 << c)
                    adj[c] |= (1 << a) | (1 << b)
    
    return adj

def max_independent_set(adj, deadline):
    """
    Branch-and-bound maximum independent set solver using bitsets.
    """
    best_set = [0]  # best independent set as bitset
    best_size = [0]
    
    def popcount(x):
        return bin(x).count('1')
    
    def bb(remaining, current, depth):
        if time.time() > deadline:
            return
        
        # Upper bound: |current| + |remaining|
        if len(current) + popcount(remaining) <= best_size[0]:
            return
        
        if remaining == 0:
            if len(current) > best_size[0]:
                best_size[0] = len(current)
                best_set[0] = 0
                for v in current:
                    best_set[0] |= (1 << v)
            return
        
        # Pick a vertex with max degree in remaining subgraph for branching
        # Use a simple heuristic: pick the first vertex
        # Better: pick vertex that maximizes the bound
        
        # Find a vertex to branch on
        # Use a greedy coloring bound for tighter pruning
        # For now, simple branching on highest-degree vertex
        
        # Get list of vertices in remaining
        verts = []
        temp = remaining
        while temp:
            lowbit = temp & (-temp)
            v = lowbit.bit_length() - 1
            verts.append(v)
            temp ^= lowbit
        
        if not verts:
            return
        
        # Choose vertex with maximum degree in the remaining graph
        max_deg = -1
        chosen = verts[0]
        for v in verts:
            deg = popcount(adj[v] & remaining)
            if deg > max_deg:
                max_deg = deg
                chosen = v
        
        # Branch 1: include chosen
        new_current = current + [chosen]
        new_remaining = remaining & ~(1 << chosen)
        # Remove all neighbors of chosen from remaining
        new_remaining &= ~adj[chosen]
        bb(new_remaining, new_current, depth + 1)
        
        # Branch 2: exclude chosen
        new_remaining = remaining & ~(1 << chosen)
        bb(new_remaining, current, depth + 1)
    
    # Start with all vertices
    all_verts = (1 << (N + 1)) - 2  # bits 1..N set
    bb(all_verts, [], 0)
    
    # Convert best_set bitset to list
    result = []
    bs = best_set[0]
    for v in range(1, N + 1):
        if bs & (1 << v):
            result.append(v)
    return result

def greedy_mis(adj, deadline):
    """
    Greedy maximum independent set: repeatedly pick vertex with fewest conflicts.
    """
    remaining = set(range(1, N + 1))
    selected = set()
    
    while remaining and time.time() < deadline:
        # Pick vertex with minimum degree in remaining subgraph
        min_deg = float('inf')
        chosen = None
        for v in remaining:
            deg = sum(1 for u in remaining if u != v and (adj[v] >> u) & 1)
            if deg < min_deg:
                min_deg = deg
                chosen = v
        
        if chosen is None:
            break
        
        selected.add(chosen)
        remaining.discard(chosen)
        # Remove all neighbors of chosen from remaining
        neighbors = set()
        temp = adj[chosen]
        while temp:
            lowbit = temp & (-temp)
            u = lowbit.bit_length() - 1
            if 1 <= u <= N:
                neighbors.add(u)
            temp ^= lowbit
        remaining -= neighbors
    
    return list(selected)

def local_improve(s, adj, deadline):
    """
    Try to improve set s by adding elements that don't conflict.
    """
    s_set = set(s)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x in s_set:
                continue
            # Check if x can be added: no two elements in s_set should both be neighbors of x
            # Actually, x conflicts with y if there exists z such that {x,y,z} is bad
            # In the conflict graph, x and y are adjacent if they form a bad triple with some z
            # So x can be added iff x is not adjacent to any two elements in s_set that are also adjacent to each other?
            # Wait, the conflict graph edge (u,v) means u and v cannot both be in the independent set.
            # So x can be added iff x is not adjacent to any element in s_set.
            can_add = True
            for y in s_set:
                if (adj[x] >> y) & 1:
                    can_add = False
                    break
            if can_add:
                s_set.add(x)
                improved = True
                break
    
    return list(s_set)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit (evaluator kills at 90s)
    
    # Build conflict graph
    adj = compute_conflict_graph()
    
    # Get initial solution via greedy
    best_set = greedy_mis(adj, deadline)
    
    # Try to improve
    best_set = local_improve(best_set, adj, deadline)
    best_size = len(best_set)
    
    # Try branch-and-bound with time limit
    if time.time() < deadline:
        bb_result = max_independent_set(adj, deadline)
        if len(bb_result) > best_size:
            best_set = bb_result
            best_size = len(bb_result)
    
    # Try multiple greedy restarts with different orderings
    while time.time() < deadline - 5:
        # Random perturbation: shuffle which vertices we consider first
        random.shuffle(list(range(1, N + 1)))
        
        # Greedy with random tie-breaking
        remaining = set(range(1, N + 1))
        selected = set()
        
        while remaining and time.time() < deadline:
            # Pick vertex with minimum degree, break ties randomly
            min_deg = float('inf')
            candidates = []
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and (adj[v] >> u) & 1)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            
            if not candidates:
                break
            
            chosen = random.choice(candidates)
            selected.add(chosen)
            remaining.discard(chosen)
            neighbors = set()
            temp = adj[chosen]
            while temp:
                lowbit = temp & (-temp)
                u = lowbit.bit_length() - 1
                if 1 <= u <= N:
                    neighbors.add(u)
                temp ^= lowbit
            remaining -= neighbors
        
        improved = local_improve(list(selected), adj, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    return best_set
