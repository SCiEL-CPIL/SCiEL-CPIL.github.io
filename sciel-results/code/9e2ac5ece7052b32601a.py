import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
                        [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14],
                        [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20],
                        [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
BASE_EDGES = set(tuple(sorted(e)) for e in P['edges'])
D_TARGET = 4

# Precompute all candidate edges (not in base, not distance-2 on path)
ALL_CANDIDATES = []
for u in range(N):
    for v in range(u + 1, N):
        if (u, v) not in BASE_EDGES and abs(u - v) >= 3:
            ALL_CANDIDATES.append((u, v))

CAND_IDX = {e: i for i, e in enumerate(ALL_CANDIDATES)}
NUM_CAND = len(ALL_CANDIDATES)

# Precompute triangle incompatibility: two candidate edges (a,b) and (c,d) are incompatible
# if adding both creates a triangle. A triangle exists if there's a vertex w adjacent to both
# endpoints of one edge in the other edge's sense... actually let's think:
# Adding edge (u,v) and (x,y) creates a triangle if there exists a vertex z such that
# z is adjacent to both u and v (forming triangle u-v-z) OR z adjacent to both x and y,
# OR the edges share a vertex and the third edge exists.
# More precisely: triangle u-v-w exists if edges (u,v), (v,w), (w,u) all present.
# With base path edges, (u,v) is added. Triangle u-v-w needs (v,w) and (w,u) in graph.
# (v,w) is in base if |v-w|=1, or added. (w,u) similarly.
# For two added edges e1=(u1,v1) and e2=(u2,v2):
# They form a triangle if they share a vertex, say u1=u2, and (v1,u2) i.e. (v1,v2) is in graph
# (base or added). Or if they don't share a vertex, we need a third edge connecting them.
# Actually the simplest: just check all triples.

# Better: precompute for each candidate edge, which other candidate edges are incompatible
# (would form a triangle when both added, considering only base edges + these two)
incompat = [set() for _ in range(NUM_CAND)]
for i in range(NUM_CAND):
    u1, v1 = ALL_CANDIDATES[i]
    for j in range(i + 1, NUM_CAND):
        u2, v2 = ALL_CANDIDATES[j]
        # Check if adding both creates a triangle
        # Triangle 1: u1-v1-w where w is common neighbor of u1 and v1 in base+{e2}
        # In base, common neighbors of u1 and v1: vertices at distance 1 from both
        # On a path, |u1-v1|>=3, so no common neighbors in base alone.
        # With e2 added: check if e2 provides a common neighbor
        # e2=(u2,v2). Common neighbor of u1 and v1 via e2:
        #   u2 adjacent to both u1 and v1? i.e., (u1,u2) in base or e2, and (v1,u2) in base or e2
        #   This is getting complex. Let me just do a direct check.
        
        # Build adjacency with base + e1 + e2, check for triangle
        # But that's O(N^3) per pair. NUM_CAND ~ 276, so ~38k pairs * 13k checks = too slow.
        
        # Simpler: a triangle with edges from {e1, e2} and base edges.
        # Case A: e1 and e2 share a vertex. WLOG u1=u2.
        #   Then triangle u1-v1-v2 exists if (v1,v2) is in base (|v1-v2|=1) 
        #   OR if (v1,v2) is e1 or e2 (but e1 is (u1,v1)=(u1,u2), e2 is (u2,v2), so (v1,v2) 
        #   would need to be a base edge or... it's not e1 or e2 unless v1=u2 which contradicts).
        #   So: if u1==u2 or u1==v2 or v1==u2 or v1==v2, check if the other pair is base-adjacent.
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            # Find the shared vertex and the two other endpoints
            verts = {u1, v1, u2, v2}
            if len(verts) == 2:
                # Same edge, shouldn't happen since i!=j
                continue
            # The three vertices of potential triangle
            # If u1==u2: triangle is u1, v1, v2. Edge (v1,v2) needed.
            #   (v1,v2) in base? |v1-v2|==1
            # If u1==v2: triangle is u1, v1, u2. Edge (v1,u2) needed. |v1-u2|==1
            # etc.
            if u1 == u2:
                if abs(v1 - v2) == 1:
                    incompat[i].add(j)
                    incompat[j].add(i)
            elif u1 == v2:
                if abs(v1 - u2) == 1:
                    incompat[i].add(j)
                    incompat[j].add(i)
            elif v1 == u2:
                if abs(u1 - v2) == 1:
                    incompat[i].add(j)
                    incompat[j].add(i)
            elif v1 == v2:
                if abs(u1 - u2) == 1:
                    incompat[i].add(j)
                    incompat[j].add(i)
        # Case B: e1 and e2 don't share a vertex. Then a triangle would need a third edge
        # connecting them, which would have to be a base edge. 
        # Triangle u1-v1-w: w must be adjacent to both u1 and v1. In base, w is at distance 1 
        # from both, so |u1-w|=1 and |v1-w|=1, meaning |u1-v1|=2, but we excluded those.
        # So no triangles from non-sharing candidate edges with only base edges.
        # But wait: what if e2 itself provides the connection? 
        # Triangle u1-v1-u2: need (v1,u2) in graph. (v1,u2) is base if |v1-u2|=1.
        # Triangle u1-v1-v2: need (u1,v2) in graph. (u1,v2) is base if |u1-v2|=1.
        # Triangle u2-v2-u1: need (v2,u1) in graph. Same as above.
        # Triangle u2-v2-v1: need (u2,v1) in graph. Same as above.
        # So: if any of these cross-edges are base edges, we have a triangle.
        else:
            if abs(v1 - u2) == 1 or abs(v1 - v2) == 1 or abs(u1 - u2) == 1 or abs(u1 - v2) == 1:
                # Check more carefully:
                # Triangle u1,v1,u2: edges (u1,v1)=e1, (v1,u2)=base?, (u2,u1)=? 
                #   (u2,u1) is not base (|u2-u1|>=3 since both candidates), not e1 or e2.
                #   So need (v1,u2) in base AND (u1,u2) in graph. (u1,u2) is not in base 
                #   (|u1-u2| could be anything), not e1 or e2. So no.
                # Let me reconsider. A triangle is 3 vertices with all 3 edges present.
                # Vertices involved: u1,v1,u2,v2 (all distinct in this case).
                # Possible triangles using at least one of e1,e2:
                #   {u1,v1,u2}: edges (u1,v1)=e1, (v1,u2)=?, (u2,u1)=?
                #     (v1,u2) in base? |v1-u2|==1. (u2,u1) in base? |u2-u1|==1.
                #     Need BOTH to be base edges (or added, but only e1,e2 added).
                #     (u2,u1) is not e1 or e2 (e1 is u1-v1, e2 is u2-v2). So (u2,u1) must be base.
                #     So: |v1-u2|==1 AND |u2-u1|==1
                #   {u1,v1,v2}: edges (u1,v1)=e1, (v1,v2)=?, (v2,u1)=?
                #     (v1,v2) not e1 or e2. Must be base: |v1-v2|==1.
                #     (v2,u1) not e1 or e2. Must be base: |v2-u1|==1.
                #     So: |v1-v2|==1 AND |v2-u1|==1
                #   {u2,v2,u1}: edges (u2,v2)=e2, (v2,u1)=?, (u1,u2)=?
                #     (v2,u1) base? |v2-u1|==1. (u1,u2) base? |u1-u2|==1.
                #     So: |v2-u1|==1 AND |u1-u2|==1
                #   {u2,v2,v1}: edges (u2,v2)=e2, (v2,v1)=?, (v1,u2)=?
                #     (v2,v1) base? |v2-v1|==1. (v1,u2) base? |v1-u2|==1.
                #     So: |v2-v1|==1 AND |v1-u2|==1
                if (abs(v1 - u2) == 1 and abs(u1 - u2) == 1) or \
                   (abs(v1 - v2) == 1 and abs(u1 - v2) == 1) or \
                   (abs(v2 - u1) == 1 and abs(u1 - u2) == 1) or \
                   (abs(v1 - v2) == 1 and abs(v1 - u2) == 1):
                    incompat[i].add(j)
                    incompat[j].add(i)

# Also: a single candidate edge is always triangle-free with base (since |u-v|>=3 means no common neighbor in path)
# And we need to check: does adding e1 create a triangle with base? No, as established.

def compute_diameter(added_idx):
    """Compute diameter given set of candidate edge indices added."""
    adj = [set() for _ in range(N)]
    for u, v in BASE_EDGES:
        adj[u].add(v)
        adj[v].add(u)
    for idx in added_idx:
        u, v = ALL_CANDIDATES[idx]
        adj[u].add(v)
        adj[v].add(u)
    
    max_d = 0
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                    if dist[v] > max_d:
                        max_d = dist[v]
        if -1 in dist:
            return float('inf')
    return max_d

def is_compatible(added_set, new_idx):
    """Check if adding new_idx to added_set maintains triangle-freeness."""
    for idx in added_set:
        if idx in incompat[new_idx]:
            return False
    return True

def branch_and_bound(deadline):
    """Search for minimum number of edges to achieve diameter <= 4."""
    best_sol = None
    best_size = NUM_CAND + 1
    
    # Try sizes from 1 upward
    for size in range(1, 15):  # Up to 14 edges
        if time.time() > deadline:
            break
        
        # Generate combinations of size 'size' from candidates
        # Use pruning: only consider compatible sets
        found = False
        
        def dfs(combo, start, remaining):
            nonlocal found, best_sol, best_size
            if time.time() > deadline:
                return True  # Signal stop
            if len(combo) == size:
                # Check diameter
                d = compute_diameter(combo)
                if d <= D_TARGET:
                    best_sol = list(combo)
                    best_size = size
                    found = True
                    return True
                return False
            
            # Pruning: if we can't fill 'remaining' more slots with compatible edges
            avail = []
            for i in range(start, NUM_CAND):
                if i not in set(combo) and is_compatible(set(combo), i):
                    avail.append(i)
            
            if len(avail) < remaining:
                return False
            
            for i in avail:
                combo.append(i)
                if dfs(combo, i + 1, remaining - 1):
                    combo.pop()
                    return True
                combo.pop()
            return False
        
        # Initial call
        if dfs([], 0, size):
            if found:
                break
        else:
            # No solution of this size found (or timed out)
            if time.time() > deadline:
                break
    
    if best_sol is not None:
        return [[ALL_CANDIDATES[i][0], ALL_CANDIDATES[i][1]] for i in best_sol]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    sol = branch_and_bound(deadline)
    
    if sol is None:
        # Fallback: return empty (shouldn't happen, but safe)
        return []
    
    # Ensure types
    out = [[int(u), int(v)] for u, v in sol]
    return out
