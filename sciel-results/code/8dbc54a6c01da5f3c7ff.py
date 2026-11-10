import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def greedy_max_clique_partition(adj, n, deadline):
        """
        Greedy partition by repeatedly extracting the maximum clique.
        1. Start with the full graph (all edges present).
        2. Find a maximum clique in the current graph.
        3. Add this clique to the partition.
        4. Remove all edges within this clique from the graph.
        5. Repeat until no edges remain.
        
        This is a heuristic for the minimum edge clique cover problem.
        """
        # graph[u] is a bitmask of neighbors v such that edge (u,v) exists
        graph = adj[:]
        cliques = []
        
        # Helper to find max clique using Bron-Kerbosch with pivot
        # We use a recursive backtracking approach but optimized with bitmasks.
        # Since n is small (<=64), we can use integers as bitsets.
        
        def max_clique(R, P, X):
            """
            R: current clique (bitmask)
            P: candidates (bitmask)
            X: excluded (bitmask)
            Returns the largest clique found in this branch.
            """
            best = R
            
            # If P and X are empty, we have a maximal clique
            if P == 0 and X == 0:
                return R
            
            # Choose a pivot u from P | X to minimize branching
            # The neighbors of u in P are the only ones that need to be considered
            # if we branch on non-neighbors of u? No, standard BK:
            # Branch on v in P \ N(u).
            # If P \ N(u) is empty, we have a maximal clique (since all P are neighbors of u, 
            # and u is in X or P? If u in X, all P are neighbors of u, so R U P is a clique? 
            # No, R U P is a clique if P is a clique. But P is just candidates.
            # Actually, if P \ N(u) is empty, then every v in P is a neighbor of u.
            # Since u is in X, u is not added to R. But this implies that any clique extending R 
            # within P must be a subset of N(u) intersect P.
            # Standard BK theorem: If P \ N(u) is empty, then R is a maximal clique? 
            # No, it means we don't need to branch further? 
            # Actually, if P \ N(u) is empty, then for any v in P, v is a neighbor of u.
            # Since u is in X, u was excluded. This doesn't necessarily mean R is maximal.
            # Wait, the standard BK algorithm says:
            # If P == 0 and X == 0: report R.
            # Else: choose u in P | X. For each v in P \ N(u): recurse.
            # If P \ N(u) is empty, the loop doesn't execute, so we return best (which is R).
            # Is R a maximal clique? Yes, because if there were a vertex w in P that could be added,
            # w would be a neighbor of all in R. Also w is in P.
            # If u is a pivot, and P \ N(u) is empty, it means all vertices in P are neighbors of u.
            # This doesn't prevent R from being extendable by a vertex in P that is NOT a neighbor of u?
            # No, the set to branch on is P \ N(u). If this set is empty, we don't branch.
            # This is correct because any maximal clique containing R and some subset of P 
            # must contain a vertex from P \ N(u)? 
            # Actually, the logic is: any maximal clique in the subgraph induced by R U P 
            # that contains R must contain at least one vertex from P \ N(u) OR be equal to R 
            # if R U (P \ N(u)) is not a clique?
            # Let's trust the standard BK implementation.
            
            pivot_mask = P | X
            if pivot_mask == 0:
                # No pivot, branch on all P
                candidates = P
            else:
                # Find a pivot u with max degree in P
                # For efficiency, we can just pick the first bit, or try to find a good one.
                # A common heuristic is to pick the vertex in P|X with the most neighbors in P.
                # Since n is small, we can just iterate.
                best_pivot = -1
                max_deg = -1
                pm = pivot_mask
                while pm:
                    lsb = pm & -pm
                    u = lsb.bit_length() - 1
                    deg = (graph[u] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_pivot = u
                    pm ^= lsb
                
                if best_pivot != -1:
                    # Branch on P \ N(u)
                    candidates = P & ~graph[best_pivot]
                else:
                    candidates = P
            
            # Iterate over candidates
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                
                # Branch: add v to R
                new_R = R | lsb
                # New P: neighbors of v in current P
                new_P = P & graph[v]
                # New X: neighbors of v in current X
                new_X = X & graph[v]
                
                # Recurse
                res = max_clique(new_R, new_P, new_X)
                if res.bit_count() > best.bit_count():
                    best = res
                
                # Update P and X for next iteration
                P ^= lsb
                X |= lsb
                c ^= lsb
                
                # Time check
                if time.time() > deadline:
                    return best
            
            return best

        # Main loop
        # We need to ensure we find a clique of size >= 2.
        # If the max clique is size 1, we just pick any edge.
        
        while True:
            # Check if any edges remain
            has_edges = False
            for u in range(n):
                if graph[u]:
                    has_edges = True
                    break
            
            if not has_edges:
                break
            
            # Find max clique
            # Start with R=0, P=all vertices, X=0
            # But we want a clique of vertices.
            # We can restrict P to vertices that have at least one neighbor?
            # Or just run on all vertices.
            # To avoid finding a size 1 clique when edges exist, 
            # we can check the result.
            
            best_clique_mask = max_clique(0, (1 << n) - 1, 0)
            
            # If best_clique_mask has size 1, it means there are no edges? 
            # Or the algorithm failed to find an edge?
            # If there are edges, there must be a clique of size 2.
            # If max_clique returns size 1, we should pick any edge.
            
            if best_clique_mask.bit_count() < 2:
                # Find any edge
                for u in range(n):
                    if graph[u]:
                        lsb = graph[u] & -graph[u]
                        v = lsb.bit_length() - 1
                        best_clique_mask = (1 << u) | (1 << v)
                        break
            
            # Extract vertices
            verts = []
            mask = best_clique_mask
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                verts.append(v)
                mask ^= lsb
            
            if len(verts) >= 2:
                cliques.append(verts)
                
                # Remove edges within this clique
                # For each pair (u, v) in clique, remove edge from graph
                for i in range(len(verts)):
                    u = verts[i]
                    for j in range(i + 1, len(verts)):
                        v = verts[j]
                        graph[u] &= ~(1 << v)
                        graph[v] &= ~(1 << u)
            
            # Time check
            if time.time() > deadline:
                break
        
        # Safety: cover any remaining edges as size-2 cliques
        # This ensures validity if the main loop timed out or missed edges
        for u in range(n):
            mask = graph[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    cliques.append([u, v])
                    graph[u] &= ~(1 << v)
                    graph[v] &= ~(1 << u)
                mask ^= lsb
        
        return cliques

    start_time = time.time()
    # Allow 145s for computation, leaving 5s buffer for return/overhead
    deadline = start_time + 145
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    cliques_1 = greedy_max_clique_partition(adj_1, 32, deadline)
    cliques_2 = greedy_max_clique_partition(adj_2, 64, deadline)
    
    return [cliques_1, cliques_2]

if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total: {len(res[0]) + len(res[1])}")
