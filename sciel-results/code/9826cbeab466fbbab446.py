"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    N = 40
    M = 401
    deadline = time.time() + 110.0
    
    # Helper to compute max book size given adj matrix
    def get_max_book(adj):
        # tri[u,v] = sum_w adj[u,w]*adj[v,w]
        # For edges (u,v), book size is tri[u,v]
        # We can compute this via matrix multiplication
        # adj @ adj gives (u,v) = number of common neighbors
        # But we need to be careful: adj @ adj includes self-loops?
        # adj[u,u] = 0, so (adj@adj)[u,v] = sum_w adj[u,w]*adj[v,w]
        # This is exactly the number of common neighbors, i.e., triangles containing edge (u,v) IF (u,v) is an edge.
        # Wait, the definition of book size for edge (u,v) is the number of triangles containing (u,v).
        # A triangle is (u,v,w). This exists if (u,v), (v,w), (w,u) are all edges.
        # So for a fixed edge (u,v), the number of such triangles is the number of w such that (u,w) and (v,w) are edges.
        # This is exactly the number of common neighbors of u and v.
        # So book_size(u,v) = common_neighbors(u,v).
        # We only care about this if (u,v) is an edge.
        
        tri_matrix = adj @ adj
        # Mask out non-edges
        # adj is 0/1. We want max of tri_matrix where adj is 1.
        # Note: tri_matrix is symmetric.
        # We can just take max of tri_matrix * adj (element-wise)
        # Or np.where(adj==1, tri_matrix, 0)
        # Since adj is int8/uint8, tri_matrix is int.
        masked = np.where(adj == 1, tri_matrix, 0)
        return int(np.max(masked))

    # Strategy:
    # 1. Search over circulant graphs defined by subset S of offsets {1..19} of size 10.
    #    This gives a 20-regular graph with 400 edges.
    # 2. For each S, compute max book of the circulant part.
    #    If it's > 7, skip (since we want to minimize, and lower bound is 7, but maybe 7 is achievable?
    #    Actually, the theorem says >= 40/6 = 6.66, so integer bound is 7.
    #    If we find a graph with max book 7, that's optimal.
    #    If circulant part has max book > 7, adding an edge might push it higher or keep it same, but unlikely to reduce it.
    #    So we prioritize S with small max book.
    # 3. For promising S, try adding each non-edge to see if we can keep max book low.
    # 4. Use local search (swap) to refine.
    
    all_offsets = list(range(1, 20))
    best_edges = None
    best_score = float('inf')
    
    # Precompute all combinations of 10 offsets from 19
    # C(19, 10) = 92378. This is manageable.
    
    # Optimization: Symmetry of offsets?
    # Offsets d and N-d are equivalent in terms of graph structure (just reflection).
    # But since we are iterating all, we cover both.
    
    # Let's iterate.
    # To speed up, we can sort S by "potential" but it's hard to know.
    # We'll just iterate and break if time is up.
    
    # We can also try a few random local searches on the best candidates found.
    
    # Precompute a list of all non-edges for a given S?
    # Non-edges are offsets NOT in S.
    # If S has 10 offsets, there are 9 offsets not in S.
    # The non-edges in the graph are pairs (i, j) such that (j-i) mod N is not in S and (i-j) mod N is not in S.
    # Since S is symmetric? No, S is a set of offsets 1..19.
    # The graph is undirected, so edge (i, j) exists if (j-i)%N in S OR (i-j)%N in S.
    # Since offsets are 1..19, (i-j)%N = N - (j-i)%N.
    # If S contains d, it doesn't necessarily contain N-d.
    # Wait, the construction in inspiration 1 uses:
    # for s in S:
    #   j1 = (i + s) % N
    #   j2 = (i - s) % N
    # This effectively uses S and -S.
    # If S = {1, 3, ...}, then edges are added for offsets +1, -1, +3, -3, etc.
    # So the "effective" offset set is S ∪ (-S mod N).
    # Since N=40, -s mod 40 is 40-s.
    # If s in 1..19, 40-s is in 21..39.
    # The set of all offsets used is S ∪ {40-s for s in S}.
    # This set has size 20 (since s != 40-s for s in 1..19).
    # So the graph is 20-regular.
    
    # Let's verify the book size calculation for circulant graphs.
    # For a circulant graph C(N, S_eff), the book size of an edge with offset d is:
    # | (S_eff ∩ (d + S_eff)) | ?
    # Neighbors of 0: S_eff
    # Neighbors of d: d + S_eff
    # Common neighbors: S_eff ∩ (d + S_eff)
    # Size of intersection is the book size for edge (0, d).
    # Due to symmetry, all edges with offset d have the same book size.
    
    # So for a given S, we can compute the max book size of the circulant graph quickly.
    
    for S in combinations(all_offsets, 10):
        if time.time() > deadline - 5:
            break
            
        S_set = set(S)
        # Effective offsets
        # S_eff = {s, N-s for s in S}
        # We can represent S_eff as a boolean array or set.
        # Let's use a set for fast intersection.
        S_eff = set()
        for s in S:
            S_eff.add(s)
            S_eff.add(N - s)
            
        # Calculate max book size for this circulant graph
        max_circ_book = 0
        for d in S: # We only need to check d in S, as S_eff is symmetric
            # Intersection of S_eff and (d + S_eff) mod N
            # (d + S_eff) mod N = { (d + x) % N for x in S_eff }
            # Count how many elements of S_eff are in (d + S_eff)
            # This is equivalent to counting x in S_eff such that x - d is in S_eff (mod N)
            # Or x in S_eff and (x - d) % N in S_eff.
            count = 0
            for x in S_eff:
                if (x - d) % N in S_eff:
                    count += 1
            if count > max_circ_book:
                max_circ_book = count
                
        # If max_circ_book is already > 7, it's likely not optimal.
        # But let's keep track of the best one.
        # If max_circ_book < best_score, it's a candidate.
        # Actually, adding an edge can only increase book sizes of incident edges or keep them same.
        # It cannot decrease the max book size of the base graph.
        # So if max_circ_book > 7, the final score will be at least max_circ_book.
        # We want score 7. So we only care about S with max_circ_book <= 7.
        
        if max_circ_book > 7:
            continue
            
        # Build adjacency matrix for this S
        adj = np.zeros((N, N), dtype=np.uint8)
        for i in range(N):
            for s in S:
                j1 = (i + s) % N
                j2 = (i - s) % N
                adj[i, j1] = 1
                adj[i, j2] = 1
        np.fill_diagonal(adj, 0)
        
        # Get list of edges
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if adj[i, j]:
                    edges.append((i, j))
                    
        # There should be 400 edges.
        # Find non-edges
        edge_set = set(edges)
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        non_edges = [p for p in all_pairs if p not in edge_set]
        
        # Compute base book sizes
        tri_matrix = adj @ adj
        base_books = {}
        for i, j in edges:
            base_books[(i, j)] = int(tri_matrix[i, j])
            
        # Try adding each non-edge
        for u, v in non_edges:
            # New edge (u, v)
            # Common neighbors
            common = np.where(adj[u] & adj[v])[0]
            b_new = len(common)
            
            # Check if adding this edge violates the max book constraint (target 7)
            # New max book = max( max_base_book, b_new, max_{w in common} (book(u,w)+1, book(v,w)+1) )
            # We know max_base_book <= 7.
            # If b_new > 7, skip.
            if b_new > 7:
                continue
                
            violation = False
            for w in common:
                # Edge (u, w)
                bw_u = base_books.get((min(u, w), max(u, w)), 0)
                bw_v = base_books.get((min(v, w), max(v, w)), 0)
                if bw_u + 1 > 7 or bw_v + 1 > 7:
                    violation = True
                    break
                    
            if violation:
                continue
                
            # If we get here, the new max book is <= 7.
            # Since lower bound is 7, this is optimal.
            # Construct final edge list
            final_edges = [list(e) for e in edges]
            final_edges.append([u, v])
            
            # Verify score
            adj[u, v] = 1
            adj[v, u] = 1
            score = get_max_book(adj)
            
            if score < best_score:
                best_score = score
                best_edges = final_edges
                
            if best_score == 7:
                # Found optimal solution
                return best_edges
                
    # If no optimal solution found in circulant search, try local search on the best candidate found
    # Or just return best found so far.
    # If best_edges is None, we might need a fallback.
    # But the loop above should find something if any S has max_circ_book <= 7.
    # If none found, best_edges is None.
    
    if best_edges is None:
        # Fallback: Random graph
        random.seed(42)
        all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_edges)
        best_edges = [list(e) for e in all_edges[:M]]
        
    # Optional: Local search to try to improve if best_score > 7
    # But if we found 7, we are done.
    # If best_score > 7, we might try to reduce it.
    # However, the circulant search is exhaustive for this family.
    # If no 7 found, maybe 8 is the best?
    
    # Let's do a quick local search on the best_edges found to see if we can reduce the score further.
    # But if best_score is already 7, no need.
    if best_score > 7:
        # Local search
        adj = np.zeros((N, N), dtype=np.uint8)
        for u, v in best_edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        current_score = best_score
        best_adj = adj.copy()
        best_tri = (adj @ adj).astype(np.int32)
        
        # Simple swap local search
        while time.time() < deadline:
            # Pick random edge to remove
            while True:
                u1 = random.randint(0, N-1)
                v1 = random.randint(0, N-1)
                if u1 == v1: continue
                if u1 > v1: u1, v1 = v1, u1
                if adj[u1, v1]:
                    break
                    
            # Pick random non-edge to add
            while True:
                u2 = random.randint(0, N-1)
                v2 = random.randint(0, N-1)
                if u2 == v2: continue
                if u2 > v2: u2, v2 = v2, u2
                if not adj[u2, v2]:
                    break
                    
            # Calculate new score
            # This is expensive if done naively.
            # We can use the update logic from the original program.
            # But for simplicity and given the deadline, let's just do a few iterations.
            
            # Save state
            old_u1_v1 = (u1, v1)
            old_u2_v2 = (u2, v2)
            
            # Apply swap
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            
            new_score = get_max_book(adj)
            
            if new_score < current_score:
                current_score = new_score
                best_adj = adj.copy()
                best_tri = (adj @ adj).astype(np.int32)
            else:
                # Revert
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                
            if current_score == 7:
                break

        # Update best_edges from best_adj
        best_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if best_adj[i, j]:
                    best_edges.append([i, j])
                    
    return best_edges
# EVOLVE-BLOCK-END
