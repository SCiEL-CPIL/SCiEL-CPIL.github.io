import time
import sys
import random
import numpy as np
from collections import deque
from itertools import product

def _bfs_girth(adj):
    """Compute the girth of an undirected graph via BFS from each vertex."""
    n = len(adj)
    best = float('inf')
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle = dist[u] + dist[v] + 1
                    if cycle < best:
                        best = cycle
        if best <= 2:
            break
    return best if best < float('inf') else 0

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (from seed)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _petersen_edges():
    """Return the 15 edges of the Petersen graph on vertices 0..9."""
    edges = []
    # Outer cycle
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner star
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i + 1) % 5]))
    # Spokes
    for i in range(5):
        edges.append((i, i + 5))
    return edges

def _check_girth_nb_fast(adj, min_girth):
    """Check if graph has girth >= min_girth using non-backtracking walks."""
    n = len(adj)
    for s in range(n):
        for v in adj[s]:
            dist = {}
            dist[(s, v)] = 1
            q = deque([(s, v)])
            while q:
                u, w = q.popleft()
                d = dist[(u, w)]
                if d >= min_girth:
                    continue
                for x in adj[w]:
                    if x == u:
                        continue
                    if (w, x) in dist:
                        if d + 1 < min_girth:
                            return False
                    else:
                        dist[(w, x)] = d + 1
                        q.append((w, x))
    return True

def _precompute_petersen_walks():
    """
    Precompute non-backtracking closed walks of length 5-9 in the Petersen graph.
    Returns a list of tuples (length, list_of_edge_indices).
    Edge indices correspond to the order in _petersen_edges().
    """
    edges = _petersen_edges()
    # Map edge to index
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    
    adj = [[] for _ in range(10)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    walks = []
    # We need walks of length 5, 6, 7, 8, 9
    for L in range(5, 10):
        for start in range(10):
            # DFS
            # State: (current_node, prev_node, path_edge_indices)
            stack = [(start, -1, [])]
            while stack:
                curr, prev, path = stack.pop()
                if len(path) == L:
                    if curr == start:
                        # Check non-backtracking at the end
                        # The walk is v0...vL=v0.
                        # Edges are e1...eL.
                        # Non-backtracking means v_i != v_{i-2}.
                        # This is implicitly handled by the DFS if we don't go back to prev.
                        # But we must ensure the last step doesn't "undo" the second-to-last?
                        # Actually, non-backtracking walk definition: v_{i-1} != v_{i+1}.
                        # In our DFS, we always move to a node != prev.
                        # For the closing edge, we go from curr to start.
                        # We need curr != prev (already true) and start != prev (second to last node).
                        # The second to last node is the one we came from to get to curr?
                        # No, the path is nodes v0, v1, ..., vL.
                        # The stack stores (curr, prev, path).
                        # If we pop (curr, prev, path), the next node must be != prev.
                        # If we close the loop, next is start.
                        # We need start != prev.
                        # Also, the walk must be simple in terms of edges? No, non-backtracking allows repeating edges as long as not immediately reversed.
                        # However, for girth, we care about cycles.
                        # A non-backtracking closed walk of length L corresponds to a cycle of length <= L.
                        # If L < 10, we must ensure the sum is non-zero.
                        
                        # Check if start == prev?
                        # If L=1, start==prev is impossible.
                        # If L=2, path=[e1, e2]. v0->v1->v0. e1=(v0,v1), e2=(v1,v0).
                        # This is backtracking. Our DFS prevents moving to prev.
                        # So if we are at v1 and prev is v0, we cannot go to v0.
                        # So L=2 walks are not generated.
                        
                        # For L=5, we might have a 5-cycle.
                        # We need to ensure the sequence of edges doesn't backtrack.
                        # The DFS constraint `nxt != prev` ensures no immediate backtracking.
                        # Does it ensure the whole walk is non-backtracking?
                        # Yes, if we never go back to the node we just came from.
                        
                        # One edge case: The closing edge.
                        # We are at `curr`. We want to go to `start`.
                        # We must ensure `start != prev`.
                        if start != prev:
                            walks.append((L, tuple(path)))
                    continue
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    e_idx = edge_to_idx[(curr, nxt)]
                    stack.append((nxt, curr, path + [e_idx]))
    return walks

def _evaluate_chromosome(chrom, walks, group_order):
    """
    Evaluate a voltage assignment.
    Returns the number of violating walks (sum == 0 mod group_order).
    """
    violations = 0
    for L, edge_seq in walks:
        # Calculate sum of voltages
        # chrom is a list of voltages for each edge index
        s = 0
        for e_idx in edge_seq:
            s += chrom[e_idx]
        if s % group_order == 0:
            violations += 1
    return violations

def _construct_4_10_cage_ga():
    """
    Use Genetic Algorithm to find a valid voltage assignment for the Petersen graph
    lifted by Z_32, resulting in a 320-vertex 4-regular graph of girth >= 10.
    """
    start_time = time.time()
    deadline = start_time + 110  # 40s margin for construction and verification
    
    group_order = 32
    edges = _petersen_edges()
    
    # Precompute walks
    walks = _precompute_petersen_walks()
    # Optional: Remove duplicate walks?
    # Different walks might have the same edge sequence?
    # In a simple graph, a non-backtracking walk is uniquely determined by its edge sequence?
    # Not necessarily, but for small graphs, it's a good approximation.
    # Actually, if the edge sequence is the same, the voltage sum is the same.
    # So we can unique the edge sequences to speed up evaluation.
    unique_walks = list(set(walks))
    
    # Initialize Population
    pop_size = 200
    population = []
    for _ in range(pop_size):
        chrom = [random.randint(1, group_order - 1) for _ in range(15)]
        population.append(chrom)
    
    best_chrom = None
    best_score = float('inf')
    
    # GA Loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Evaluate
        scored_pop = []
        for chrom in population:
            score = _evaluate_chromosome(chrom, unique_walks, group_order)
            scored_pop.append((score, chrom))
            if score < best_score:
                best_score = score
                best_chrom = chrom[:]
                
        if best_score == 0:
            # Found a valid assignment!
            break
            
        # Selection: Tournament
        new_population = []
        for _ in range(pop_size):
            # Pick 3 random candidates
            candidates = random.sample(scored_pop, 3)
            winner = min(candidates, key=lambda x: x[0])
            new_population.append(winner[1][:])
            
        # Crossover and Mutation
        mutated_pop = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                p1 = new_population[i]
                p2 = new_population[i+1]
                
                # Uniform Crossover
                c1 = []
                c2 = []
                for j in range(15):
                    if random.random() < 0.5:
                        c1.append(p1[j])
                        c2.append(p2[j])
                    else:
                        c1.append(p2[j])
                        c2.append(p1[j])
                        
                # Mutation
                for j in range(15):
                    if random.random() < 0.1:
                        c1[j] = random.randint(1, group_order - 1)
                    if random.random() < 0.1:
                        c2[j] = random.randint(1, group_order - 1)
                        
                mutated_pop.append(c1)
                mutated_pop.append(c2)
            else:
                mutated_pop.append(new_population[i])
                
        population = mutated_pop

    if best_chrom is not None and best_score == 0:
        # Construct the graph
        adj = [[] for _ in range(320)]
        edge_set = set()
        
        for u, v in edges:
            alpha = best_chrom[edges.index((u,v))] if (u,v) in [(e[0],e[1]) for e in edges] else best_chrom[edges.index((v,u))]
            # Note: edges list contains (u,v). We need to map edge to voltage.
            # best_chrom index corresponds to the index in _petersen_edges()
            
        # Re-do mapping carefully
        for i, (u, v) in enumerate(edges):
            alpha = best_chrom[i]
            for g in range(group_order):
                u_lift = u * group_order + g
                v_lift = v * group_order + (g + alpha) % group_order
                if u_lift < v_lift:
                    edge_key = (u_lift, v_lift)
                else:
                    edge_key = (v_lift, u_lift)
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    adj[u_lift].append(v_lift)
                    adj[v_lift].append(u_lift)
                    
        # Verify
        for i in range(320):
            if len(adj[i]) != 4:
                return None
                
        if not _check_girth_nb_fast(adj, 10):
            return None
            
        return 320, list(edge_set)
        
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    if k == 4 and g == 10:
        # Try GA first
        try:
            res = _construct_4_10_cage_ga()
            if res is not None:
                n, edges = res
                return n, edges
        except Exception:
            pass
            
        # Fallback to LUW
        n, edges = _luw_construct(k, g)
        return n, edges
    else:
        n, edges = _luw_construct(k, g)
        return n, edges
