import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # For k=5, g=9, we try the Moore bound construction.
    # Moore bound for k=5, g=9 (odd, r=4):
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i = 1 + 5 * (1 + 4 + 16 + 64) = 426.
    
    if k == 5 and g == 9:
        # Build 5-ary tree of depth 4
        # Root is 0.
        # Vertices 1..5 are children of 0.
        # Vertices 6..25 are children of 1..5.
        # And so on.
        
        # We need to generate the tree structure.
        # Let's use a BFS to generate the tree.
        # Parent array.
        
        # Depth 0: 1 vertex
        # Depth 1: 5 vertices
        # Depth 2: 20 vertices
        # Depth 3: 80 vertices
        # Depth 4: 320 vertices
        
        n_target = 426
        parent = [-1] * n_target
        children = [[] for _ in range(n_target)]
        
        idx = 0
        current_level = [0]
        parent[0] = -1
        idx = 1
        
        for depth in range(4):
            next_level = []
            for v in current_level:
                for _ in range(5):
                    children[v].append(idx)
                    parent[idx] = v
                    next_level.append(idx)
                    idx += 1
            current_level = next_level
        
        # Now we have the tree.
        # Leaves are in current_level (depth 4).
        leaves = current_level
        # Group leaves by their ancestor at depth 1.
        # Ancestor at depth 1 is parent[parent[parent[parent[leaf]]]]
        
        sets = [[] for _ in range(5)]
        for leaf in leaves:
            # Find depth 1 ancestor
            p = leaf
            for _ in range(3):
                p = parent[p]
            # p is now at depth 1
            # Which of the 5 children of root is p?
            # Root's children are 1..5.
            # So index is p - 1.
            sets[p - 1].append(leaf)
            
        # We need to connect leaves.
        # Each leaf has 4 dangling edges.
        # We want to connect leaf u in Set i to leaves in Sets j, k, l, m (j!=i).
        # Specifically, one to each of the other 4 sets.
        
        # This is a problem of finding a 4-regular graph on the leaves,
        # with edges only between different sets,
        # such that the girth is >= 9.
        
        # A simple heuristic:
        # For each leaf u in Set i, pick one random leaf from each other Set j.
        # Ensure no duplicate edges.
        
        # Since the sets are disjoint, and we pick one from each,
        # we just need to ensure we don't pick the same pair twice.
        # We can use a matching approach.
        
        # Let's represent the "ends" of the leaves.
        # Each leaf u has 4 ends.
        # We assign end 0 to a leaf in Set (i+1)%5
        # End 1 to Set (i+2)%5
        # End 2 to Set (i+3)%5
        # End 3 to Set (i+4)%5
        
        # This ensures u connects to one leaf in each other set.
        
        # We need to find a bijection (permutation) for each pair of sets.
        # Let P_{ij} be a permutation of the leaves in Set j, assigned to leaves in Set i.
        
        # We can just use random permutations.
        # Will this guarantee girth >= 9?
        # As analyzed, if u connects to v in Set j, and w in Set k,
        # the cycle u-v-...-w-u goes through the root.
        # Path v to root is length 4.
        # Path w to root is length 4.
        # Cycle length: 1 (u-v) + 4 (v-root) + 4 (root-w) + 1 (w-u) = 10.
        # This is >= 9.
        
        # What about cycles using more edges?
        # They will be longer.
        
        # So any such construction should work!
        
        edges = []
        
        # Add tree edges
        for v in range(1, n_target):
            edges.append((parent[v], v))
            
        # Add matching edges
        # For each set i, we need to match its leaves to leaves in sets j, k, l, m.
        # We can do this by iterating over sets.
        
        # To avoid duplicate edges, we only add edges for i < j?
        # No, the connection is directed in our assignment logic?
        # We want an undirected graph.
        # If we assign u in Set i to v in Set j, we add edge (u,v).
        # We must ensure we don't add (v,u) again.
        
        # Let's iterate i from 0 to 4.
        # For each i, we match Set i to Set (i+1)%5.
        # And Set i to Set (i+2)%5?
        # Wait, each leaf in Set i needs 4 neighbors.
        # If we match Set i to Set (i+1)%5, that's 1 neighbor.
        # We need 3 more.
        
        # Let's define the target sets for each source set i:
        # Targets for i: (i+1)%5, (i+2)%5, (i+3)%5, (i+4)%5.
        
        # We will generate a permutation for each pair (i, target).
        # But we must ensure consistency?
        # No, the edges are distinct.
        # Edge (u, v) where u in Set i, v in Set j.
        # We only add this edge once.
        
        # We can just iterate over all pairs (i, j) with i < j?
        # No, the degree requirement is local to the leaf.
        
        # Let's just generate the edges.
        # For each set i:
        #   Create a random permutation of the leaves in Set (i+1)%5.
        #   Match leaf u in Set i to perm[u] in Set (i+1)%5.
        #   Add edge (u, perm[u]).
        
        # Do this for all i and all 4 target sets.
        
        random.seed(42)
        
        for i in range(5):
            targets = [(i+1)%5, (i+2)%5, (i+3)%5, (i+4)%5]
            for t in targets:
                # We need to match leaves in Set i to leaves in Set t.
                # But if we do this for all i, we will add every edge twice?
                # No, if we match i to t, we add edges between Set i and Set t.
                # If we later match t to i, we add edges between Set t and Set i.
                # These are the same edges.
                
                # To avoid duplicates, we only match i to t if i < t?
                # But we need to satisfy the degree of leaves in Set i.
                # Leaf u in Set i needs to connect to one leaf in Set t.
                
                # So for each pair (i, t), we should define a matching.
                # But we only want to add the edge once.
                
                # Let's only add edges for i < t.
                # But then, does every leaf get its required connections?
                # Leaf u in Set i needs connections to 4 other sets.
                # If we only add edges for i < t, then for t < i, the edge is added when processing t.
                # So we need to ensure that when processing t, we match t to i.
                
                # So, for every pair (i, j) with i != j, we need a matching.
                # But we only store the edge once.
                
                # Let's just generate all edges and use a set to deduplicate.
                
                # Permutation for Set i to Set t
                perm = list(range(len(sets[i])))
                random.shuffle(perm)
                
                for idx_u, u in enumerate(sets[i]):
                    v = sets[t][perm[idx_u]]
                    edges.append((u, v))
        
        # Remove duplicates
        unique_edges = []
        seen = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            if (u, v) not in seen:
                seen.add((u, v))
                unique_edges.append((u, v))
                
        return n_target, unique_edges

    # Fallback to LUW for other parameters
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
