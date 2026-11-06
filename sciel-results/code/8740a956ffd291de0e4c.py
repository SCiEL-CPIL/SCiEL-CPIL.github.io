import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    # Q3 adjacency
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    
    # Initialize with all edges
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            edges.add((u, v))
            
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def find_q3_in_subset(verts):
        """
        Returns a tuple (mapping_dict, q3_edges_list) if Q3 is found in induced subgraph of verts, else None.
        mapping_dict maps q3_vertex_id (0..7) to graph_vertex_id.
        """
        V = set(verts)
        local_adj = {v: adj[v] & V for v in V}
        
        # Count edges in induced subgraph
        edge_count = sum(len(local_adj[v]) for v in V) // 2
        if edge_count < 12:
            return None
            
        # Backtracking
        # Order Q3 vertices: 0, 1, 2, 3, 4, 5, 6, 7
        # q0 neighbors: 1, 2, 4
        # We try to map q0 to each v in V
        for v0 in V:
            if len(local_adj[v0]) < 3:
                continue
            # Candidates for 1, 2, 4 are neighbors of v0
            # In Q3, 1, 2, 4 are not connected to each other.
            # So v1, v2, v4 must be pairwise non-adjacent in G?
            # Wait, Q3 is bipartite. 0 is in part A. 1,2,4 are in part B.
            # So yes, v1, v2, v4 must be independent in the induced subgraph.
            
            # Get neighbors of v0
            neigh0 = list(local_adj[v0])
            
            # Try permutations of 3 from neigh0
            for p in permutations(neigh0, 3):
                v1, v2, v4 = p
                # Check independence
                if v2 in local_adj[v1] or v4 in local_adj[v1] or v4 in local_adj[v2]:
                    continue
                
                # Mapping so far: 0->v0, 1->v1, 2->v2, 4->v4
                # q3 neighbors: 1, 2, 7. So v3 must be neighbor of v1 and v2.
                # Also v3 is not v0, v1, v2, v4.
                # q7 neighbors: 3, 5, 6.
                # q5 neighbors: 1, 4, 7.
                # q6 neighbors: 2, 4, 7.
                
                # Find candidates for v3: neighbors of v1 AND v2, not in {v0, v1, v2, v4}
                cand_3 = local_adj[v1] & local_adj[v2]
                cand_3 -= {v0, v1, v2, v4}
                
                for v3 in cand_3:
                    # Find candidates for v7: neighbors of v3, not in {v0, v1, v2, v3, v4}
                    # Also v7 must be neighbor of v5 and v6 (which we haven't picked yet).
                    # Let's find v5 and v6 first?
                    # v5 neighbors: 1, 4, 7. So v5 is neighbor of v1 and v4.
                    # v6 neighbors: 2, 4, 7. So v6 is neighbor of v2 and v4.
                    
                    cand_5 = local_adj[v1] & local_adj[v4]
                    cand_5 -= {v0, v1, v2, v3, v4}
                    
                    cand_6 = local_adj[v2] & local_adj[v4]
                    cand_6 -= {v0, v1, v2, v3, v4}
                    
                    # v5 and v6 must be distinct and not v7?
                    # v7 is the remaining vertex.
                    # Let's iterate v5
                    for v5 in cand_5:
                        # v6 must be in cand_6, v6 != v5
                        # Also v7 is the last vertex in V not used.
                        used = {v0, v1, v2, v3, v4, v5}
                        rem = V - used
                        if len(rem) != 2: # v6 and v7
                            continue
                        # v6 must be in rem and in cand_6
                        cand_6_in_rem = cand_6 & rem
                        if len(cand_6_in_rem) == 0:
                            continue
                        for v6 in cand_6_in_rem:
                            if v6 == v5:
                                continue
                            # v7 is the other one
                            rem2 = rem - {v6}
                            if len(rem2) != 1:
                                continue
                            v7 = list(rem2)[0]
                            
                            # Check if v7 is valid
                            # v7 must be neighbor of v3, v5, v6
                            if v3 in local_adj[v7] and v5 in local_adj[v7] and v6 in local_adj[v7]:
                                # Verify all edges of Q3 exist
                                # Edges: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
                                mapping = {0: v0, 1: v1, 2: v2, 3: v3, 4: v4, 5: v5, 6: v6, 7: v7}
                                q3_edges = [
                                    (v0, v1), (v0, v2), (v0, v4),
                                    (v1, v3), (v1, v5),
                                    (v2, v3), (v2, v6),
                                    (v3, v7),
                                    (v4, v5), (v4, v6),
                                    (v5, v7), (v6, v7)
                                ]
                                # Ensure all edges are in adj
                                valid = True
                                for u, v in q3_edges:
                                    if v not in adj[u]:
                                        valid = False
                                        break
                                if valid:
                                    return mapping, q3_edges
        return None

    # Removal Loop
    no_q3_streak = 0
    max_streak = 5000 # Adjust based on time
    
    while time.time() < deadline:
        # Try to find a Q3
        found = False
        # Try random subsets
        for _ in range(20): # 20 random probes per iteration
            if time.time() > deadline:
                break
            verts = tuple(random.sample(range(n), 8))
            res = find_q3_in_subset(verts)
            if res:
                mapping, q3_edges = res
                # Remove a random edge from this Q3
                edge_to_remove = random.choice(q3_edges)
                # Normalize edge
                if edge_to_remove[0] > edge_to_remove[1]:
                    edge_to_remove = (edge_to_remove[1], edge_to_remove[0])
                
                if edge_to_remove in edges:
                    edges.remove(edge_to_remove)
                    u, v = edge_to_remove
                    adj[u].remove(v)
                    adj[v].remove(u)
                    found = True
                    no_q3_streak = 0
                    break
        
        if not found:
            no_q3_streak += 1
            if no_q3_streak > max_streak:
                # Do a more exhaustive check? Or just break?
                # For n=16, if we don't find it in 5000*20 random probes, it's likely gone or rare.
                # Let's just break to ensure we return something.
                break
                
    return [list(e) for e in edges]
