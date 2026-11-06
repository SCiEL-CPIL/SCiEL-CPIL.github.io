import time
import random
import math

def construct():
    """
    Constructs a simple graph on 20 vertices with edge colours 0..19.
    Maximizes the number of edges while ensuring every 4-cycle is rainbow.
    
    Strategy:
    1. Use Simulated Annealing to find a dense C4-free graph.
    2. State: Set of edges.
    3. Moves: Add an edge, Remove an edge, Swap an edge.
    4. Energy: Penalty for C4s + Penalty for missing edges (to maximize density).
       Specifically, we want to minimize: C4_count * BIG_PENALTY - edge_count.
       Or simpler: Maximize edge_count - C4_count * BIG_PENALTY.
       Since we need strict C4-freeness, we treat any C4 as a hard constraint violation
       with a very high penalty, but SA allows temporary violations to escape local optima.
       
    5. We start with a random dense graph and try to fix it.
    """
    n = 20
    deadline = time.time() + 115  # Safety margin
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m_total = len(all_edges)
    
    # Helper to count C4s efficiently using bitmasks or sets
    # For n=20, we can represent adjacency as a bitmask or list of sets.
    # Let's use a list of sets for speed in intersection, or numpy arrays if needed.
    # With n=20, sets are fast enough.
    
    def count_c4s(adj):
        """
        Counts the number of 4-cycles in the graph defined by adj.
        A 4-cycle is u-v-x-y-u with u<v<x<y? No, standard cycle counting.
        We count each set of 4 vertices that form a C4 exactly once.
        For a set of 4 vertices {a,b,c,d}, there are 3 ways to form a cycle:
        (a-b-c-d-a), (a-b-d-c-a), (a-c-b-d-a).
        Actually, it's easier to count pairs of vertices with 2 common neighbors.
        Number of C4s = sum_{u < v} (C(common_neighbors(u,v), 2))
        """
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u].intersection(adj[v])
                k = len(common)
                if k >= 2:
                    count += k * (k - 1) // 2
        return count

    def get_score(adj, edges_set):
        """
        Score to maximize: Number of edges - Penalty * Number of C4s.
        We want C4s to be 0.
        """
        c4s = count_c4s(adj)
        # We want to maximize edges, but C4s are bad.
        # If C4s > 0, the graph is invalid.
        # Let's use a large penalty to ensure C4-free graphs are preferred.
        # Max possible edges = 190. Max C4s is much smaller.
        # Let's say penalty = 1000.
        return len(edges_set) - 1000 * c4s

    def random_initial_graph(density=0.6):
        """
        Creates a random graph with given edge probability.
        """
        adj = [set() for _ in range(n)]
        edges_set = set()
        for u, v in all_edges:
            if random.random() < density:
                adj[u].add(v)
                adj[v].add(u)
                edges_set.add((u, v))
        return adj, edges_set

    # Simulated Annealing
    best_adj = None
    best_edges_set = set()
    best_score = -10**9
    
    # Initial temperature
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Start with a few random restarts
    num_restarts = 5
    edges_per_restart = 20000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        adj, edges_set = random_initial_graph(density=0.5 + 0.1 * r)
        current_score = get_score(adj, edges_set)
        
        if current_score > best_score:
            best_score = current_score
            best_adj = [s.copy() for s in adj]
            best_edges_set = set(edges_set)
            
        T = 10.0
        
        for step in range(edges_per_restart):
            if time.time() > deadline:
                break
                
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.5:
                # Add an edge
                u, v = random.choice(all_edges)
                if (u, v) in edges_set:
                    continue
                # Tentative add
                adj[u].add(v)
                adj[v].add(u)
                edges_set.add((u, v))
                
                new_score = get_score(adj, edges_set)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_adj = [s.copy() for s in adj]
                        best_edges_set = set(edges_set)
                else:
                    # Reject
                    adj[u].remove(v)
                    adj[v].remove(u)
                    edges_set.remove((u, v))
                    
            elif move_type < 0.8:
                # Remove an edge
                if not edges_set:
                    continue
                u, v = random.choice(list(edges_set))
                # Tentative remove
                adj[u].remove(v)
                adj[v].remove(u)
                edges_set.remove((u, v))
                
                new_score = get_score(adj, edges_set)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_adj = [s.copy() for s in adj]
                        best_edges_set = set(edges_set)
                else:
                    # Reject
                    adj[u].add(v)
                    adj[v].add(u)
                    edges_set.add((u, v))
                    
            else:
                # Swap: Remove one, Add another
                if not edges_set:
                    continue
                u1, v1 = random.choice(list(edges_set))
                u2, v2 = random.choice(all_edges)
                if (u2, v2) in edges_set:
                    continue
                if (u1, v1) == (u2, v2):
                    continue
                
                # Tentative swap
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                edges_set.remove((u1, v1))
                
                adj[u2].add(v2)
                adj[v2].add(u2)
                edges_set.add((u2, v2))
                
                new_score = get_score(adj, edges_set)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_adj = [s.copy() for s in adj]
                        best_edges_set = set(edges_set)
                else:
                    # Reject
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    edges_set.add((u1, v1))
                    
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    edges_set.remove((u2, v2))
            
            T *= cooling_rate
            if T < min_T:
                T = min_T

    # Final check: If best graph has C4s, try to fix it greedily or return empty?
    # The problem requires valid colouring. If we return a graph with C4s, it's invalid (score 0).
    # We must ensure the final graph is C4-free.
    # If best_adj has C4s, we can try to remove edges to break them until C4-free.
    
    final_adj = [s.copy() for s in best_adj]
    final_edges = set(best_edges_set)
    
    # Repair phase: Remove edges involved in C4s until none remain.
    # This might reduce edge count significantly, but ensures validity.
    # Actually, if SA found a high score, it likely has 0 C4s because of the penalty.
    # Let's verify.
    c4_count = count_c4s(final_adj)
    if c4_count > 0:
        # Greedy repair: while C4 exists, remove a random edge from a C4
        # Identify C4s
        while count_c4s(final_adj) > 0:
            # Find a C4
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    common = final_adj[u].intersection(final_adj[v])
                    if len(common) >= 2:
                        # Pick two common neighbors x, y
                        x, y = list(common)[:2]
                        # Remove edge (u, x) or (v, x) etc to break the C4 u-x-v-y-u
                        # Removing (u, x) breaks it.
                        e = (u, x) if u < x else (x, u)
                        final_adj[u].remove(x)
                        final_adj[x].remove(u)
                        final_edges.remove(e)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
            if time.time() > deadline:
                break

    # Convert to required format
    # Colours: Since graph is C4-free, any colouring works. Use 0.
    coloured_edges = []
    for u, v in final_edges:
        coloured_edges.append([u, v, 0])
    
    return coloured_edges
