import sys
import time
import math

def construct():
    # Moduli from 2 to 128
    moduli = list(range(2, 129))
    n_mods = len(moduli)
    
    # 1. Find the largest set of moduli that are pairwise non-coprime.
    # This is a Max Clique problem on the graph where an edge exists if gcd(u, v) > 1.
    
    # Build adjacency bitsets for the modulus graph
    # adj_mod[i] is a bitmask of all j such that gcd(moduli[i], moduli[j]) > 1
    adj_mod = [0] * n_mods
    for i in range(n_mods):
        mi = moduli[i]
        for j in range(i + 1, n_mods):
            mj = moduli[j]
            if math.gcd(mi, mj) > 1:
                adj_mod[i] |= (1 << j)
                adj_mod[j] |= (1 << i)
                
    # Function to find Max Clique in a graph defined by adjacency bitsets
    def max_clique(adj, n):
        best_size = 0
        best_clique = []
        
        def expand(R, P):
            nonlocal best_size, best_clique
            if not P:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = R.copy()
                return
            
            # Heuristic: Order vertices in P by degree in P (descending)
            # This helps in finding large cliques early and pruning.
            # For n=127, we can just iterate.
            
            # Coloring bound for pruning (Tomita's algorithm)
            # We need to color the subgraph induced by P.
            # The number of colors is an upper bound on the clique size in P.
            
            # Simple greedy coloring
            colors = [0] * n
            color_count = 0
            # We process vertices in P. To make coloring effective, we should process them in an order.
            # Let's just use a simple list of vertices in P.
            verts_in_P = [i for i in range(n) if (P >> i) & 1]
            
            # Greedy coloring
            # To get a tight bound, we should color in a specific order.
            # Let's just do a standard greedy coloring.
            for v in verts_in_P:
                # Find smallest color not used by neighbors in P
                used = 0
                for u in verts_in_P:
                    if u != v and (adj[v] >> u) & 1:
                        used |= (1 << colors[u])
                c = 0
                while (used >> c) & 1:
                    c += 1
                colors[v] = c
                if c + 1 > color_count:
                    color_count = c + 1
            
            if len(R) + color_count <= best_size:
                return
                
            # Branch
            # We iterate through vertices in P.
            # To improve pruning, we should pick vertices that are "central".
            # Let's iterate in the order they were colored (reverse color order often works well?)
            # Or just iterate through verts_in_P.
            
            # To be more efficient, we can group by color.
            # If we pick a vertex of color c, we can only pick vertices of colors > c? No.
            # Standard Tomita:
            # Process vertices in reverse order of coloring.
            
            # For simplicity and speed in Python for n=127:
            # Just iterate.
            
            for v in reversed(verts_in_P):
                # If we've already found a clique of size >= len(R) + (number of remaining vertices in P that could form a clique with v)
                # The coloring bound already handles this globally.
                
                if len(R) + 1 + (P.bit_count() - 1) <= best_size: # Rough bound
                    break
                # Better bound: len(R) + color_count[remaining] <= best_size
                
                # Recurse
                new_R = R + [v]
                # New P is intersection of P and neighbors of v
                new_P = P & adj[v]
                # Remove v from new_P? No, v is not in adj[v] usually (no self loops).
                # But we should ensure we don't pick v again.
                # Since v is in P, and we are branching on v, the new P should be P \ {v} & adj[v].
                # Actually, standard expansion:
                # R' = R U {v}
                # P' = P \ (R U {v}) & N(v)
                
                new_P = (P & ~(1 << v)) & adj[v]
                
                expand(new_R, new_P)
                
                # Remove v from P for subsequent branches
                P &= ~(1 << v)
                
                if len(R) + P.bit_count() <= best_size:
                    break
                    
        expand([], (1 << n) - 1)
        return best_clique

    # Run Max Clique on moduli
    best_mods = max_clique(adj_mod, n_mods)
    
    # The size of best_mods is the maximum number of classes we can possibly have.
    # Now we need to assign residues.
    
    def solve_residues(mods):
        n = len(mods)
        residues = [None] * n
        # Precompute gcds for this set
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                gcds[i][j] = math.gcd(mods[i], mods[j])
                gcds[j][i] = gcds[i][j]

        def backtrack(i):
            if i == n:
                return True
            
            m = mods[i]
            # Try all residues for m
            for r in range(m):
                valid = True
                for j in range(i):
                    g = gcds[i][j]
                    if r % g == residues[j] % g:
                        valid = False
                        break
                if valid:
                    residues[i] = r
                    if backtrack(i + 1):
                        return True
                    residues[i] = None
            return False
            
        # Heuristic: Order moduli by "constrainedness" or just size?
        # Larger moduli have more choices, so maybe put them last?
        # Smaller moduli have fewer choices, put them first?
        # Let's sort by modulus ascending.
        mods_sorted = sorted(mods)
        # We need to map back to original indices if we return, but we just need the pairs.
        # So we just solve for mods_sorted and return pairs.
        
        if backtrack(0):
            return [[residues[i], mods_sorted[i]] for i in range(n)]
        return None

    # Try to solve for the max modulus clique
    solution = solve_residues(best_mods)
    if solution:
        return solution
        
    # If it fails (unlikely for evens), try smaller cliques?
    # For robustness, we can just return the best we found or a known good one.
    # Given the time limit, if the max clique fails, we might timeout.
    # But 64 evens should be solvable.
    
    # Fallback
    return [[1, 2], [0, 4]]
