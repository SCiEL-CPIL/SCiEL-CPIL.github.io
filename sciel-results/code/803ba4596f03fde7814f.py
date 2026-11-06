import numpy as np
import time
import random
import math

def count_c4(adj):
    """
    Count the number of 4-cycles in the graph represented by adjacency matrix adj.
    """
    n = adj.shape[0]
    A2 = adj @ adj
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = A2[mask]
    return int(np.sum(vals * (vals - 1)) // 2)

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Construct the base graph: Orthogonal Polarity Graph of PG(2,5)
    # This graph has 31 vertices.
    # Points in PG(2,5) represented as (x, y, z) in GF(5)^3, normalized.
    # We use the standard representative:
    # (1, y, z) for y, z in 0..4 (25 points)
    # (0, 1, z) for z in 0..4 (5 points)
    # (0, 0, 1) (1 point)
    # Total 31 points.
    
    points = []
    # Index 0-24: (1, y, z)
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    # Index 25-29: (0, 1, z)
    for z in range(5):
        points.append((0, 1, z))
    # Index 30: (0, 0, 1)
    points.append((0, 0, 1))
    
    adj = np.zeros((n, n), dtype=np.int8)
    edge_set = set()
    
    # Define polarity: P and Q are adjacent if dot product is 0 in GF(5)
    # dot(P, Q) = x1*x2 + y1*y2 + z1*z2 mod 5 == 0
    # Note: This is a symmetric bilinear form.
    # If dot(P, P) != 0, P is not absolute (no loop).
    # If dot(P, P) == 0, P is absolute. In simple graph, we don't add loop, but P is on its polar line.
    # The edge set is {(P, Q) : P != Q, dot(P, Q) == 0}.
    
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
    # For GF(5) orthogonal polarity, we expect ~81 edges.
    # We need 91 edges. So we need to add more edges.
    
    # Calculate initial C4 count
    current_c4 = count_c4(adj)
    
    # 2. Greedy addition of remaining edges
    # We need to add (m - current_edges) edges.
    # For each potential edge not in edge_set, calculate the number of new C4s it creates.
    # New C4s involving (u, v) = number of edges (w, x) such that w in N(u), x in N(v).
    # This is because adding (u,v) creates a C4 u-w-x-v-u for every edge (w,x) connecting N(u) and N(v).
    
    def count_new_c4s(u, v, adj):
        # u and v are not currently connected
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        count = 0
        # Optimize: iterate over smaller set
        if len(nu) < len(nv):
            for w in nu:
                count += np.sum(adj[w, nv])
        else:
            for x in nv:
                count += np.sum(adj[x, nu])
        
        return count

    # Candidate edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
    
    edges_to_add = m - current_edges
    for _ in range(edges_to_add):
        if time.time() > deadline - 60:
            break
        best_edge = None
        min_c4s = float('inf')
        
        # Shuffle to break ties randomly
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
            # Remove from candidate list
            all_possible_edges.remove(best_edge)
    
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)
    
    # 3. Local Search (Simulated Annealing) to refine
    # We will use a bitmask representation for faster delta calculations if possible, 
    # but for n=31, numpy is fast enough. 
    # Key improvement: Efficient delta calculation.
    
    def get_delta(adj, u, v, x, y):
        # Remove (u,v), Add (x,y)
        # 1. Calculate C4s lost by removing (u,v) in current graph
        # C4s containing edge (u,v) = number of edges (w,z) with w in N(u)\{v}, z in N(v)\{u}
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        # Exclude v from nu and u from nv
        # Since adj is int8, we can just mask or filter
        nu_masked = [w for w in nu if w != v]
        nv_masked = [z for z in nv if z != u]
        
        lost = 0
        if len(nu_masked) < len(nv_masked):
            for w in nu_masked:
                lost += np.sum(adj[w, nv_masked])
        else:
            for z in nv_masked:
                lost += np.sum(adj[z, nu_masked])
        
        # 2. Temporarily remove (u,v)
        adj[u, v] = 0
        adj[v, u] = 0
        
        # 3. Calculate C4s gained by adding (x,y) in the modified graph
        # C4s containing edge (x,y) in new graph = number of edges (w,z) with w in N'(x), z in N'(y)
        # N'(x) = N(x) (since (u,v) removal doesn't affect x unless x==u or x==v, which it isn't as (x,y) is non-edge and distinct)
        # Wait, if x==u or x==v, then N changes. 
        # But (x,y) is a non-edge in the ORIGINAL graph. 
        # If we remove (u,v), does (x,y) become an edge? No.
        # Does N(x) change? Only if x==u or x==v.
        # If x==u, then (u,y) was a non-edge. Removing (u,v) doesn't change N(u) except v is removed.
        # So N'(u) = N(u) \ {v}.
        # Let's handle this generally.
        
        nx = np.where(adj[x] == 1)[0]
        ny = np.where(adj[y] == 1)[0]
        
        # If x == u, we must exclude v from nx (since we removed u-v). 
        # But wait, v is not in nx because (u,v) is removed? No, we set adj[u,v]=0. 
        # So np.where(adj[x]==1) will naturally exclude v if x==u.
        # Similarly for y==v.
        
        gained = 0
        if len(nx) < len(ny):
            for w in nx:
                gained += np.sum(adj[w, ny])
        else:
            for z in ny:
                gained += np.sum(adj[z, nx])
        
        # 4. Restore (u,v) for evaluation (we don't apply yet)
        adj[u, v] = 1
        adj[v, u] = 1
        
        return gained - lost

    T = 5.0
    T_min = 0.1
    cooling_rate = 0.995
    
    current_edges_list = list(edge_set)
    
    steps = 0
    while T > T_min and time.time() < deadline - 5:
        steps += 1
        
        # Pick random edge to remove
        idx_rm = random.randint(0, len(current_edges_list) - 1)
        e_rm = current_edges_list[idx_rm]
        u, v = e_rm
        
        # Pick random non-edge to add
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) not in edge_set:
                break
        
        delta = get_delta(adj, u, v, x, y)
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            # Apply removal
            adj[u, v] = 0
            adj[v, u] = 0
            edge_set.remove(e_rm)
            current_edges_list[idx_rm] = (x, y)
            
            # Apply addition
            adj[x, y] = 1
            adj[y, x] = 1
            edge_set.add((x, y))
            
            current_c4 += delta
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
                
        T *= cooling_rate
    
    # Final polish: try all single-edge swaps from best solution
    if time.time() < deadline - 10:
        adj = best_adj.copy()
        edge_set = set(best_edge_set)
        
        current_c4 = best_c4
        
        # Try removing each edge and adding each non-edge
        for u, v in list(edge_set):
            if time.time() > deadline - 10:
                break
            
            # Calculate lost C4s
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            nu_masked = [w for w in nu if w != v]
            nv_masked = [z for z in nv if z != u]
            
            lost = 0
            if len(nu_masked) < len(nv_masked):
                for w in nu_masked:
                    lost += np.sum(adj[w, nv_masked])
            else:
                for z in nv_masked:
                    lost += np.sum(adj[z, nu_masked])
            
            adj[u, v] = 0
            adj[v, u] = 0
            
            best_gain = -1
            best_add = None
            
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) in edge_set:
                        continue
                    # Calculate gained C4s
                    ni = np.where(adj[i] == 1)[0]
                    nj = np.where(adj[j] == 1)[0]
                    gained = 0
                    if len(ni) < len(nj):
                        for w in ni:
                            gained += np.sum(adj[w, nj])
                    else:
                        for z in nj:
                            gained += np.sum(adj[z, ni])
                    
                    net = gained - lost
                    if net < best_gain:
                        best_gain = net
                        best_add = (i, j)
            
            adj[u, v] = 1
            adj[v, u] = 1
            
            if best_add is not None and best_gain < 0:
                i, j = best_add
                adj[u, v] = 0
                adj[v, u] = 0
                edge_set.remove((u, v))
                
                adj[i, j] = 1
                adj[j, i] = 1
                edge_set.add((i, j))
                
                current_c4 += best_gain
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = adj.copy()
                    best_edge_set = set(edge_set)
    
    # Return best found
    if best_edge_set and len(best_edge_set) == m:
        return [[u, v] for u, v in best_edge_set]
    elif best_edge_set:
        # If we don't have exactly m edges, try to fix
        edges = list(best_edge_set)
        if len(edges) < m:
            # Add random non-edges
            all_non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in best_edge_set]
            random.shuffle(all_non_edges)
            for e in all_non_edges[:m - len(edges)]:
                best_edge_set.add(e)
        elif len(edges) > m:
            # Remove random edges
            random.shuffle(edges)
            for e in edges[:len(edges) - m]:
                best_edge_set.remove(e)
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
