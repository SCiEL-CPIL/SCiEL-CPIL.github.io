import numpy as np
import time
import random
import math

def count_c4(adj):
    """
    Count the number of 4-cycles in the graph represented by adjacency matrix adj.
    Uses the formula: C4 = 1/2 * sum_{u<v} C(common_neighbors(u,v), 2)
    """
    n = adj.shape[0]
    A2 = adj @ adj
    # For each pair u < v, count common neighbors, then C(common, 2)
    # A2[u,v] gives number of length-2 paths from u to v = common neighbors
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            cn = A2[u, v]
            if cn >= 2:
                total += cn * (cn - 1)
    return total // 2

def construct():
    n = 31
    m = 91
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We stop at 135s to be safe.
    deadline = start_time + 135

    # 1. Construct the base graph: Orthogonal Polarity Graph of PG(2,5)
    # This graph has 31 vertices.
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    adj = np.zeros((n, n), dtype=np.int8)
    edge_set = set()
    
    # Define polarity: P and Q are adjacent if dot product is 0 in GF(5)
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edge_set.add((i, j))
    
    current_edges = len(edge_set)
    
    # 2. Greedy addition of remaining edges
    # We need to add (m - current_edges) edges.
    # For each potential edge, calculate the number of new C4s it creates.
    
    def count_new_c4s(u, v, adj):
        # New C4s involving (u, v) = number of edges (w, x) where w in N(u) \ {v} and x in N(v) \ {u}
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        count = 0
        # Optimize: iterate over smaller set
        if len(nu) < len(nv):
            for w in nu:
                # Check neighbors of w in nv
                count += np.sum(adj[w, nv])
        else:
            for x in nv:
                count += np.sum(adj[x, nu])
        return count

    # Candidate edges: all non-edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
    
    edges_to_add = m - current_edges
    for _ in range(edges_to_add):
        if time.time() > deadline:
            break
        best_edge = None
        min_c4s = float('inf')
        
        # To speed up, we can sample a subset if the list is large, but 384 is small.
        # We shuffle to break ties.
        random.shuffle(all_possible_edges)
        
        for u, v in all_possible_edges:
            c4s = count_new_c4s(u, v, adj)
            if c4s < min_c4s:
                min_c4s = c4s
                best_edge = (u, v)
        
        if best_edge is not None:
            u, v = best_edge
            adj[u, v] = 1
            adj[v, u] = 1
            edge_set.add(best_edge)
            all_possible_edges.remove(best_edge)
    
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_edge_set = set(edge_set)
    
    # 3. Local Search (Simulated Annealing) with Smart Moves
    # Instead of purely random swaps, we bias towards removing edges in many C4s
    # and adding edges in few C4s.
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    current_edges_list = list(edge_set)
    
    # Precompute C4 participation for each edge to guide the search
    # An edge (u,v) is in a C4 if there are common neighbors w,x with (w,x) edge.
    # Actually, it's simpler to just use the full count for evaluation, 
    # but we can use a heuristic for move selection.
    
    steps = 0
    while T > T_min and time.time() < deadline:
        steps += 1
        
        # Strategy: Pick a random edge to remove, but with some bias?
        # For now, random removal is standard, but let's try to be smart.
        # Let's stick to random removal for diversity, but smart addition.
        
        idx_rm = random.randint(0, len(current_edges_list) - 1)
        e_rm = current_edges_list[idx_rm]
        u1, v1 = e_rm
        
        # Pick a random non-edge to add
        # To ensure we don't pick an edge that creates too many C4s, 
        # we could sample a few candidates and pick the best.
        
        # Sample 5 candidate non-edges
        candidates = []
        for _ in range(5):
            while True:
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2: continue
                if u2 > v2: u2, v2 = v2, u2
                e_add = (u2, v2)
                if e_add not in edge_set:
                    candidates.append(e_add)
                    break
        
        best_add = None
        best_delta = float('inf')
        
        for e_add in candidates:
            u2, v2 = e_add
            
            # Apply swap
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            
            new_c4 = count_c4(adj)
            delta = new_c4 - current_c4
            
            # Revert for next candidate
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            
            if delta < best_delta:
                best_delta = delta
                best_add = e_add
        
        if best_add is None:
            continue
            
        u2, v2 = best_add
        
        # Apply the best swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        new_c4 = count_c4(adj)
        delta = new_c4 - current_c4
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_c4 = new_c4
            edge_set.remove(e_rm)
            edge_set.add(best_add)
            current_edges_list[idx_rm] = best_add
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edge_set = set(edge_set)
        else:
            # Reject
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            
        T *= cooling_rate
        
    # 4. Final Hill Climbing Pass
    # Try to improve further by only accepting moves that strictly reduce C4 count.
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(current_edges_list)):
                if time.time() > deadline - 5:
                    break
                e_rm = current_edges_list[i]
                u1, v1 = e_rm
                
                # Try a few random non-edges
                for _ in range(10):
                    u2 = random.randint(0, n-1)
                    v2 = random.randint(0, n-1)
                    if u2 == v2: continue
                    if u2 > v2: u2, v2 = v2, u2
                    e_add = (u2, v2)
                    if e_add in edge_set: continue
                    
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    
                    new_c4 = count_c4(adj)
                    
                    if new_c4 < current_c4:
                        current_c4 = new_c4
                        edge_set.remove(e_rm)
                        edge_set.add(e_add)
                        current_edges_list[i] = e_add
                        improved = True
                        if current_c4 < best_c4:
                            best_c4 = current_c4
                            best_edge_set = set(edge_set)
                    else:
                        adj[u1, v1] = 1
                        adj[v1, u1] = 1
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0

    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]

# To make the local search faster, we can optimize count_c4 for the specific swap?
# But for n=31, count_c4 is ~microseconds. The loop overhead is dominant.
# We can try to run this.
