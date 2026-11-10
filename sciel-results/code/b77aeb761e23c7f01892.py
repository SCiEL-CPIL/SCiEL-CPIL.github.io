import time
import random
import numpy as np

def construct():
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    # Adjacency as bitmasks for fast operations
    adj = [0] * N
    adj_list = [[] for _ in range(N)]

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        adj_list[u].append(v)
        adj_list[v].append(u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj_list[u].remove(v)
        adj_list[v].remove(u)

    def edge_count():
        return sum(bin(adj[i]).count('1') for i in range(N)) // 2

    def can_add(u, v):
        """
        Check if adding edge (u, v) creates a C6.
        C6: u-a-b-c-d-v (then v-u closes it).
        So we need a path of length 4 from u to v in the existing graph (without using the new edge).
        Equivalently, is there a path u-a-b-c-d-v where all intermediate nodes are distinct and not u,v?
        We check: exists a in N(u), d in N(v), such that there is a path a-b-c-d of length 2 (i.e., b in N(a) cap N(d), b != u, v, a, d)
        Wait, path length 4 from u to v: u-a, a-b, b-c, c-d, d-v.
        So we need a path of length 3 from a to d where a in N(u) and d in N(v).
        Actually, simpler: check if there is a path of length 5 from u to v? No, the cycle is u...v-u.
        The path in the existing graph must be u-a-b-c-d-v. That is a path of length 5.
        So we check if there is a path of length 5 from u to v in the existing graph.
        Let's just do a bounded BFS/DFS from u up to depth 5, avoiding v until the end.
        """
        # Check if there is a path u -> a -> b -> c -> d -> v in existing graph.
        # a in N(u), d in N(v), and path a->b->c->d of length 3.
        # Let's iterate over a in N(u) and d in N(v) and check if a and d are connected by a path of length 3 avoiding u, v, and themselves.
        # Path a-b-c-d: b in N(a) \ {u,v,a}, c in N(b) \ {u,v,a,b,d}, and c adjacent to d? No, c-d is the last edge.
        # So: exists a in N(u), d in N(v), b, c such that:
        # a-b, b-c, c-d are edges, all distinct, none are u or v.
        
        # Let's do a BFS from all neighbors of u up to depth 3, tracking the node.
        # Actually, just check: is there a path of length 3 from any a in N(u) to any d in N(v)?
        # Let's use bitmasks.
        # N(u) bitmask: adj[u] & ~(1<<v)
        # We want to know if there exists a path a-b-c-d where a in N(u), d in N(v).
        # Let's iterate over a in N(u):
        for a_bit in range(N):
            if adj[u] & (1 << a_bit):
                a = a_bit
                if a == v:
                    continue
                # Now from a, we need to reach some d in N(v) in exactly 2 more steps (a-b-c-d).
                # So we need a path of length 2 from a to some d in N(v).
                # That means exists b in N(a), c in N(b) such that c is adjacent to d? No, c-d is the edge.
                # Path: a-b-c-d. So c must be in N(b) and c must be adjacent to some d in N(v).
                # So: exists b in N(a) \ {u,v,a}, such that N(b) \ {u,v,a,b} intersects N(v)?
                # Wait, the last node is d, and c-d is an edge. So c must be a neighbor of some d in N(v).
                # So N(c) must intersect N(v).
                # Let's just iterate b in N(a):
                for b_bit in range(N):
                    if (adj[a] & (1 << b_bit)) and b_bit not in (u, v, a):
                        b = b_bit
                        # Now from b, we need to reach d in N(v) in 1 step via c.
                        # So exists c in N(b) \ {u,v,a,b} such that c is adjacent to some d in N(v).
                        # c-d is an edge, so c in N(d) for some d in N(v).
                        # So N(b) \ {u,v,a,b} must intersect (union of N(d) for d in N(v)) \ {u,v}?
                        # Actually, we just need c such that c is adjacent to at least one d in N(v), and c != u, v, a, b.
                        # Let's precompute N(v) neighbors: neighbors of nodes in N(v).
                        # Actually, simpler: for each c in N(b), check if c is adjacent to any d in N(v) (c != u,v,a,b).
                        # To speed up: let's compute the set of all nodes that are adjacent to some node in N(v).
                        # That is the union of adj[d] for d in N(v).
                        # Let's call this set S.
                        # Then we need c in N(b) \ {u,v,a,b} such that c is in S? No, c must be adjacent to a specific d in N(v).
                        # Actually, if c is in N(b) and c is adjacent to some d in N(v), then the path a-b-c-d exists.
                        # So we need: (N(b) & S) is non-empty, where S = union of adj[d] for d in N(v), and we exclude u,v,a,b from N(b).
                        # But c must also be distinct from a and b.
                        # Let's just do it directly but with bitmasks.
                        pass
        
        # This is getting complicated. Let's just do a BFS from u to depth 4, and check if we can reach v in one more step.
        # Actually, the original can_add in the seed was doing BFS from u to depth 4 and checking if v is a neighbor of the node at depth 4.
        # That is correct and probably fast enough with bitmasks? Let's stick with BFS but optimize.
        # Actually, let's just use the original approach but with bitmasks for neighbor iteration.
        
        # Let's try a different approach: check if there is a path of length 5 from u to v.
        # We can compute the set of nodes reachable from u in exactly k steps.
        # R[0] = {u}
        # R[1] = N(u) \ {v}  (we can't go to v yet, we want to reach v at step 5)
        # R[2] = (N(R[1]) \ {u, v}) 
        # R[3] = (N(R[2]) \ {u, v})
        # R[4] = (N(R[3]) \ {u, v})
        # Then check if R[4] intersects N(v). If yes, there is a path of length 5.
        
        R = [0] * 5
        R[0] = (1 << u)
        R[1] = adj[u] & ~(1 << u) & ~(1 << v)
        # Actually, we must ensure simple paths, so we need to avoid revisiting nodes.
        # BFS with visited set is tricky with bitmasks if we want to track the path.
        # Given N=40, maybe the original DFS is okay if we limit the branching.
        # Let's just use the original can_add but optimize the inner loop.
        
        # Actually, let's just go with the original DFS but with a small optimization: prune if depth remaining > 0 but no unvisited neighbors.
        # For now, let's just use the original logic.
        
        # Original can_add logic:
        stack = [(u, 0, 1 << u)]
        while stack:
            node, depth, mask = stack.pop()
            if depth == 4:
                if adj[node] & (1 << v):
                    return False
                continue
            # Get neighbors not in mask and not v
            nbs = adj[node] & ~(mask | (1 << v))
            while nbs:
                lsb = nbs & (-nbs)
                nb = lsb.bit_length() - 1
                stack.append((nb, depth + 1, mask | lsb))
                nbs ^= lsb
        return True

    # Initialize with 8 disjoint K5s
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                add_edge(base + a, base + b)
    
    current_edges = edge_count()
    best_edges = current_edges
    best_adj = adj[:]
    best_adj_list = [lst[:] for lst in adj_list]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    missing_edges = [p for p in all_pairs if not (adj[p[0]] & (1 << p[1]))]
    
    while time.time() < deadline:
        random.shuffle(missing_edges)
        added_in_pass = 0
        new_missing = []
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if can_add(u, v):
                add_edge(u, v)
                added_in_pass += 1
                current_edges += 1
            else:
                new_missing.append((u, v))
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            best_adj_list = [lst[:] for lst in adj_list]
        
        missing_edges = new_missing
        
        if added_in_pass == 0:
            if current_edges > 0:
                kick_size = random.randint(2, 5)
                existing_edges = [(u, v) for u in range(N) for v in adj_list[u] if u < v]
                random.shuffle(existing_edges)
                for _ in range(kick_size):
                    if not existing_edges:
                        break
                    u, v = existing_edges.pop()
                    remove_edge(u, v)
                    current_edges -= 1
                    missing_edges.append((u, v))
    
    final_edges = []
    for u in range(N):
        for v in best_adj_list[u]:
            if u < v:
                final_edges.append([u, v])
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
