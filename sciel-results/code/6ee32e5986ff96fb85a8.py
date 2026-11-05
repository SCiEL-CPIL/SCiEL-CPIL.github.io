import time
import numpy as np
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin for 150s limit
    
    n = 20
    num_colours = 20
    
    # Precompute all 4-cycles as sets of 4 edges
    # A 4-cycle is a set of 4 edges forming a cycle of length 4
    # For vertices a,b,c,d: cycles are (a,b,c,d), (a,b,d,c), (a,c,b,d)
    # But we need to be careful about which edges form actual 4-cycles in the graph
    
    # For each pair of vertices, precompute which other vertices can form a 4-cycle with them
    # Actually, let's just use the adjacency-based check
    
    def can_add_edge(adj, u, v, colour):
        """Check if adding edge (u,v) with given colour creates a bad 4-cycle."""
        # A 4-cycle containing edge (u,v) needs two other edges (u,w) and (v,w') 
        # where w and w' are connected, OR (u,w) and (v,w) for same w... 
        # Actually: 4-cycle a-b-c-d-a means edges (a,b),(b,c),(c,d),(d,a)
        # If we add edge (u,v), it could be any of the 4 positions.
        # Let's think: edge (u,v) is in 4-cycle if there exist w,x such that
        # (u,w), (w,x), (x,v) exist and colours are all distinct from each other and from c(u,v)
        # OR (u,w), (w,x), (x,v) where the cycle is u-w-x-v-u
        
        # More systematically: for edge (u,v) to be in a 4-cycle, we need
        # vertices w, x such that edges (u,w), (w,x), (x,v) all exist
        # and the 4 edges (u,v), (u,w), (w,x), (x,v) have 4 distinct colours.
        
        # For each w adjacent to u (w != v):
        for w in range(n):
            if w == v or w == u:
                continue
            if adj[u, w] == -1:
                continue
            cw = adj[u, w]
            if cw == colour:
                continue  # would create 2-cycle with same colour? No, but let's check
            # For each x adjacent to w (x != u):
            for x in range(n):
                if x == u or x == w:
                    continue
                if adj[w, x] == -1:
                    continue
                cx = adj[w, x]
                if cx == colour or cx == cw:
                    continue
                # Check if (x,v) exists
                if x == v:
                    continue  # that's a triangle, not 4-cycle
                if adj[x, v] == -1:
                    continue
                cv = adj[x, v]
                if cv == colour or cv == cw or cv == cx:
                    continue
                # We have a potential 4-cycle: u-v-x-w-u with edges
                # (u,v)=colour, (v,x)=cv, (x,w)=cx, (w,u)=cw
                # Wait, I need to be more careful about the cycle structure.
                # Edge (u,v) plus path u-w-x-v forms 4-cycle u-v-x-w-u
                # Edges: (u,v), (v,x), (x,w), (w,u)
                # Colours: colour, cv, cx, cw - all distinct, so this is VALID
                # We only care about BAD cycles, so this is fine.
                pass
        # Actually, the check above is wrong. Let me redo.
        # When we ADD edge (u,v) with colour C, we need to check all 4-cycles 
        # that include this new edge. A 4-cycle containing (u,v) has form:
        # u - v - a - b - u where edges (u,v), (v,a), (a,b), (b,u) exist
        # So for each a adjacent to v (a != u), for each b adjacent to a (b != v):
        #   if (b,u) exists and b != v and b != a and b != u:
        #     check colours of (u,v), (v,a), (a,b), (b,u) are all distinct
        
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[v, a] == -1:
                continue
            ca = adj[v, a]
            if ca == colour:
                continue
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if adj[a, b] == -1:
                    continue
                cb = adj[a, b]
                if cb == colour or cb == ca:
                    continue
                if adj[b, u] == -1:
                    continue
                cu = adj[b, u]
                if cu == colour or cu == ca or cu == cb:
                    continue
                # Found a bad 4-cycle
                return False
        return True
    
    def can_add_edge_fast(adj, neighbours, u, v, colour):
        """Faster check using neighbor lists."""
        # For each a in neighbours[v] (a != u):
        for a in neighbours[v]:
            if a == u:
                continue
            ca = adj[v, a]
            if ca == colour:
                continue
            # For each b in neighbours[a] (b != v, b != a):
            for b in neighbours[a]:
                if b == v or b == a:
                    continue
                if b == u:
                    # This gives cycle u-v-a-u which is a triangle, skip
                    continue
                cb = adj[a, b]
                if cb == colour or cb == ca:
                    continue
                # Check (b, u) exists
                if adj[b, u] == -1:
                    continue
                cu = adj[b, u]
                if cu == colour or cu == ca or cu == cb:
                    continue
                return False
        return True
    
    def greedy_construct(seed_offset=0):
        """Greedy construction: try to add edges one by one."""
        random.seed(seed_offset)
        
        adj = np.full((n, n), -1, dtype=int)
        neighbours = [set() for _ in range(n)]
        
        # Generate all possible (u, v, c) triples
        all_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                for c in range(num_colours):
                    all_edges.append((u, v, c))
        
        # Shuffle with seed
        random.shuffle(all_edges)
        
        added_count = 0
        skipped = []
        
        for (u, v, c) in all_edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            if can_add_edge_fast(adj, neighbours, u, v, c):
                adj[u, v] = c
                adj[v, u] = c
                neighbours[u].add(v)
                neighbours[v].add(u)
                added_count += 1
            else:
                skipped.append((u, v, c))
        
        # Second pass: retry skipped edges
        for (u, v, c) in skipped:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            if can_add_edge_fast(adj, neighbours, u, v, c):
                adj[u, v] = c
                adj[v, u] = c
                neighbours[u].add(v)
                neighbours[v].add(u)
                added_count += 1
        
        # Build result
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] != -1:
                    result.append([u, v, adj[u, v]])
        
        return result, added_count
    
    best_result = []
    best_count = 0
    
    # Try multiple seeds
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        result, count = greedy_construct(seed_offset=i)
        if count > best_count:
            best_count = count
            best_result = result
    
    # Verify the best result
    if best_result:
        adj = np.full((n, n), -1, dtype=int)
        for [u, v, c] in best_result:
            adj[u, v] = c
            adj[v, u] = c
        
        # Check all 4-cycles
        valid = True
        for a in range(n):
            if not valid:
                break
            for b in range(a + 1, n):
                if not valid:
                    break
                for c in range(b + 1, n):
                    if not valid:
                        break
                    for d in range(c + 1, n):
                        # Check cycle a-b-c-d-a
                        if adj[a,b] != -1 and adj[b,c] != -1 and adj[c,d] != -1 and adj[d,a] != -1:
                            if len({adj[a,b], adj[b,c], adj[c,d], adj[d,a]}) < 4:
                                valid = False
                                break
                        # Check cycle a-b-d-c-a
                        if adj[a,b] != -1 and adj[b,d] != -1 and adj[d,c] != -1 and adj[c,a] != -1:
                            if len({adj[a,b], adj[b,d], adj[d,c], adj[c,a]}) < 4:
                                valid = False
                                break
                        # Check cycle a-c-b-d-a
                        if adj[a,c] != -1 and adj[c,b] != -1 and adj[b,d] != -1 and adj[d,a] != -1:
                            if len({adj[a,c], adj[c,b], adj[b,d], adj[d,a]}) < 4:
                                valid = False
                                break
        
        if not valid:
            # Fallback
            res = []
            for v in range(1, n):
                res.append([0, v, 0])
            return res
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
