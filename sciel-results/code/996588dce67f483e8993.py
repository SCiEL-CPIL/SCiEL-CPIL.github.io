import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin before 150s kill
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute Q3 structure
    # Q3 vertices: 0..7 (binary 000..111)
    # Edge if Hamming distance = 1
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.add((i, j))
    
    # Q3 adjacency
    q3_adj = [set() for _ in range(8)]
    for (a, b) in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)
    
    def has_q3_in_subgraph(adj, vertices):
        """Check if the induced subgraph on 'vertices' (8 vertices) is isomorphic to Q3.
        adj is a list of sets or a 2D boolean array."""
        if len(vertices) != 8:
            return False
        
        # Count edges
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                u, v = vertices[i], vertices[j]
                if isinstance(adj, list) and isinstance(adj[0], set):
                    if v in adj[u]:
                        edge_count += 1
                else:
                    if adj[u][v]:
                        edge_count += 1
        
        if edge_count != 12:
            return False
        
        # Check 3-regular
        degrees = [0]*8
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                u, v = vertices[i], vertices[j]
                if isinstance(adj, list) and isinstance(adj[0], set):
                    if v in adj[u]:
                        degrees[i] += 1
                else:
                    if adj[u][v]:
                        degrees[i] += 1
        
        if any(d != 3 for d in degrees):
            return False
        
        # Check bipartite
        color = [-1]*8
        color[0] = 0
        queue = [0]
        is_bipartite = True
        while queue:
            u_idx = queue.pop(0)
            u = vertices[u_idx]
            for v_idx in range(8):
                if v_idx == u_idx:
                    continue
                v = vertices[v_idx]
                if isinstance(adj, list) and isinstance(adj[0], set):
                    if v in adj[u]:
                        if color[v_idx] == -1:
                            color[v_idx] = 1 - color[u_idx]
                            queue.append(v_idx)
                        elif color[v_idx] == color[u_idx]:
                            is_bipartite = False
                            break
            if not is_bipartite:
                break
        
        if not is_bipartite:
            return False
        
        # If 3-regular, bipartite, 8 vertices, 12 edges -> Q3
        return True
    
    def count_q3s(adj_sets):
        """Count the number of Q3 subgraphs in the graph represented by adj_sets (list of sets)."""
        count = 0
        for subset in combinations(range(n), 8):
            if has_q3_in_subgraph(adj_sets, list(subset)):
                count += 1
        return count
    
    def is_q3_free(adj_sets):
        """Check if graph is Q3-free."""
        for subset in combinations(range(n), 8):
            if has_q3_in_subgraph(adj_sets, list(subset)):
                return False
        return True
    
    def add_edge(adj_sets, u, v):
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    
    def remove_edge(adj_sets, u, v):
        adj_sets[u].discard(v)
        adj_sets[v].discard(u)
    
    def edge_count(adj_sets):
        return sum(len(s) for s in adj_sets) // 2
    
    # Strategy: 
    # 1. Start with a random dense graph (e.g., p=0.8) and remove edges to eliminate Q3s.
    # 2. Use a randomized greedy approach: start empty, add edges if they don't create Q3.
    # 3. Use local search to improve.
    
    best_adj = [set() for _ in range(n)]
    best_count = 0
    
    # Seed: K_{3,13} has 39 edges. Let's try to beat that.
    # Actually, let's try a better seed: a random 7-regular bipartite graph or similar.
    # Or just start with K_16 and remove edges.
    
    # Let's try multiple strategies:
    
    # Strategy A: Greedy add with random order
    for seed in range(200):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed)
        
        # Try different initial densities
        if seed < 100:
            # Start empty, add edges greedily
            adj_sets = [set() for _ in range(n)]
            edges = all_edges[:]
            random.shuffle(edges)
            
            for (u, v) in edges:
                if time.time() > deadline - 5:
                    break
                
                # Try adding edge
                add_edge(adj_sets, u, v)
                
                # Check if this creates a Q3
                # Fast check: only check Q3s that contain (u,v)
                # We need to find if there exists a Q3 containing edge (u,v)
                
                # In Q3, given edge (u,v), the structure is fixed.
                # u has 2 other neighbors, v has 2 other neighbors.
                # Let's enumerate possibilities.
                
                found_q3 = False
                
                # Neighbors of u (besides v)
                neigh_u = [w for w in adj_sets[u] if w != v]
                # Neighbors of v (besides u)
                neigh_v = [w for w in adj_sets[v] if w != u]
                
                # We need to choose 2 from neigh_u and 2 from neigh_v
                # such that they form a Q3 with u, v.
                
                for (a, b) in combinations(neigh_u, 2):
                    for (c, d) in combinations(neigh_v, 2):
                        if len({a, b, c, d}) != 4:
                            continue
                        
                        # Now we need e, f such that:
                        # e is adjacent to a and b
                        # f is adjacent to c and d
                        # e is adjacent to f
                        # Also, a-c, b-d must be edges? 
                        # Let's re-verify the Q3 structure.
                        #
                        # Q3: 000, 001, 010, 011, 100, 101, 110, 111
                        # Let u=000, v=001.
                        # Neighbors of u: 001, 010, 100. So a,b are 010, 100 (in some order).
                        # Neighbors of v: 000, 011, 101. So c,d are 011, 101 (in some order).
                        #
                        # Edges in Q3:
                        # 000-001 (u-v)
                        # 000-010 (u-a)
                        # 000-100 (u-b)
                        # 001-011 (v-c)
                        # 001-101 (v-d)
                        # 010-011 (a-c)
                        # 010-110 (a-e) -> e=110
                        # 100-101 (b-d)
                        # 100-110 (b-e) -> e=110
                        # 011-111 (c-f) -> f=111
                        # 101-111 (d-f) -> f=111
                        # 110-111 (e-f)
                        #
                        # So the required edges are:
                        # u-v, u-a, u-b, v-c, v-d, a-c, a-e, b-d, b-e, c-f, d-f, e-f
                        #
                        # We already have u-v, u-a, u-b, v-c, v-d.
                        # We need to check: a-c, b-d, and existence of e, f.
                        # e must be adjacent to a and b.
                        # f must be adjacent to c and d.
                        # e must be adjacent to f.
                        # e, f must be distinct from u, v, a, b, c, d.
                        
                        # Check a-c and b-d
                        if not (c in adj_sets[a] and d in adj_sets[b]):
                            continue
                        
                        # Find candidates for e
                        cand_e = [w for w in range(n) 
                                  if w not in {u, v, a, b, c, d} 
                                  and a in adj_sets[w] and b in adj_sets[w]]
                        
                        # Find candidates for f
                        cand_f = [w for w in range(n) 
                                  if w not in {u, v, a, b, c, d} 
                                  and c in adj_sets[w] and d in adj_sets[w]]
                        
                        for e in cand_e:
                            for f in cand_f:
                                if e == f:
                                    continue
                                if f in adj_sets[e]:
                                    found_q3 = True
                                    break
                            if found_q3:
                                break
                        if found_q3:
                            break
                    if found_q3:
                        break
                
                if found_q3:
                    remove_edge(adj_sets, u, v)
                    continue
                # Else, keep the edge
        
            if edge_count(adj_sets) > best_count:
                best_count = edge_count(adj_sets)
                best_adj = [s.copy() for s in adj_sets]
        
        else:
            # Strategy B: Start dense, remove edges
            p = 0.9
            adj_sets = [set() for _ in range(n)]
            for (u, v) in all_edges:
                if random.random() < p:
                    add_edge(adj_sets, u, v)
            
            # Remove edges to eliminate Q3s
            # Find all Q3s, remove one edge from each
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                # Find a Q3
                for subset in combinations(range(n), 8):
                    if time.time() > deadline - 5:
                        break
                    if has_q3_in_subgraph(adj_sets, list(subset)):
                        # Remove a random edge from this Q3
                        verts = list(subset)
                        q3_edges_here = []
                        for i in range(8):
                            for j in range(i+1, 8):
                                u, v = verts[i], verts[j]
                                if v in adj_sets[u]:
                                    q3_edges_here.append((u, v))
                        
                        if q3_edges_here:
                            u, v = random.choice(q3_edges_here)
                            remove_edge(adj_sets, u, v)
                            changed = True
                            break
                
                if time.time() > deadline - 5:
                    break
            
            if edge_count(adj_sets) > best_count:
                best_count = edge_count(adj_sets)
                best_adj = [s.copy() for s in adj_sets]
    
    # Local search to improve best_adj
    if time.time() < deadline - 10:
        adj_sets = [s.copy() for s in best_adj]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding a random missing edge
            missing = [(u, v) for (u, v) in all_edges if v not in adj_sets[u]]
            random.shuffle(missing)
            
            for (u, v) in missing:
                if time.time() > deadline - 5:
                    break
                
                add_edge(adj_sets, u, v)
                
                # Check if Q3-free
                if is_q3_free(adj_sets):
                    improved = True
                    best_count = edge_count(adj_sets)
                    best_adj = [s.copy() for s in adj_sets]
                else:
                    remove_edge(adj_sets, u, v)
    
    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in adj_sets[u]:
            if u < v:
                result.append([u, v])
    
    return result

# EVOLVE-BLOCK-END
