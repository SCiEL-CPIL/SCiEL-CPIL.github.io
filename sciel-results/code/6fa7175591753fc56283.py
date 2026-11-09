import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety

    # Strategy:
    # 1. Generate candidate graphs using algebraic constructions (Paley-type).
    # 2. Use a fast bitmask-based verifier.
    # 3. If a candidate fails, attempt local repair on the first few violations.
    
    def is_quadratic_residue(x, p):
        """Check if x is a quadratic residue mod p (x != 0)."""
        if x == 0:
            return False
        # Euler's criterion: x^((p-1)/2) == 1 mod p
        return pow(x, (p - 1) // 2, p) == 1

    def generate_paley_graph(n, p, offset=0):
        """
        Generate a graph on n vertices using quadratic residues mod p.
        Vertices are mapped to elements of GF(p). 
        Edge (u, v) exists if (u - v) is a quadratic residue mod p.
        We map vertices 0..n-1 to 0..n-1 in GF(p) (assuming n <= p).
        """
        adj_masks = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if u == v:
                    continue
                diff = (u - v) % p
                if is_quadratic_residue(diff, p):
                    mask |= (1 << v)
            adj_masks[u] = mask
        return adj_masks

    def verify_graph(n, adj_masks):
        """
        Verify if the graph has no regular induced subgraph of size k >= 6.
        Returns True if valid, False otherwise.
        Also returns the first violation found (subset, degrees) for debugging/repair.
        """
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                # Check if induced subgraph on 'subset' is regular
                # For each vertex v in subset, degree in subset is popcount(adj_masks[v] & subset_mask)
                subset_mask = 0
                for v in subset:
                    subset_mask |= (1 << v)
                
                degrees = []
                for v in subset:
                    # Count bits in intersection
                    # Python int.bit_count() is fast
                    d = (adj_masks[v] & subset_mask).bit_count()
                    degrees.append(d)
                
                # Check if all degrees are equal
                if len(set(degrees)) == 1:
                    return False, (subset, degrees[0])
        return True, None

    def local_search_repair(n, adj_masks, max_iters=20):
        """
        Attempt to fix violations by flipping edges.
        Returns (adj_masks, is_valid).
        """
        current_adj = adj_masks[:]
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            valid, violation = verify_graph(n, current_adj)
            if valid:
                return current_adj, True
            
            if violation is None:
                break
                
            subset, deg = violation
            # Try to flip an edge within the subset to break regularity
            # We need to change the degree of at least one vertex in the subset
            # such that the degrees are no longer all equal.
            
            # Heuristic: Flip an edge (u, v) inside the subset.
            # This changes degree of u and v by +/- 1.
            # We want to make the set of degrees non-uniform.
            
            edges_in_subset = []
            sub_list = list(subset)
            for i in range(len(sub_list)):
                for j in range(i + 1, len(sub_list)):
                    edges_in_subset.append((sub_list[i], sub_list[j]))
            
            random.shuffle(edges_in_subset)
            
            improved = False
            for u, v in edges_in_subset:
                # Flip edge u-v
                # Check current state
                has_edge = (current_adj[u] & (1 << v)) != 0
                
                # Flip
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
                
                # Quick check: Is this specific subset still regular?
                subset_mask = 0
                for x in subset:
                    subset_mask |= (1 << x)
                
                new_degrees = []
                for x in subset:
                    d = (current_adj[x] & subset_mask).bit_count()
                    new_degrees.append(d)
                
                if len(set(new_degrees)) > 1:
                    # This specific violation is broken.
                    # We accept this move and re-verify the whole graph in the next loop iteration
                    # to ensure we didn't create a new violation elsewhere (or that the new state is valid).
                    improved = True
                    break
                else:
                    # Revert
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
            
            if not improved:
                # If we can't fix this specific violation by flipping internal edges,
                # try flipping an edge connecting a vertex in subset to one outside?
                # Or just give up on this candidate.
                break
                
        # Final check
        valid, _ = verify_graph(n, current_adj)
        return current_adj, valid

    def adj_masks_to_edges(n, adj_masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] & (1 << j)) != 0:
                    edges.append([i, j])
        return edges

    best_n = 5
    best_edges = [[0, 1], [1, 2], [2, 3], [3, 4]]
    
    # Try to find a valid graph on 20 vertices
    # Use algebraic constructions with different primes
    # p must be >= n. Let's try p=23, 29, 31, etc.
    # Also try random shifts or different mappings if needed.
    
    candidates = []
    
    # 1. Paley graph on p=23, take first 20 vertices
    # Note: The standard Paley graph is on p vertices. We take an induced subgraph on 20.
    # This might not be optimal, but it's a strong start.
    
    primes = [23, 29, 31, 37, 41, 43, 47]
    
    for p in primes:
        if time.time() > deadline - 20:
            break
            
        # Generate full Paley graph on p vertices, then take first 20
        full_adj = generate_paley_graph(p, p)
        # Take induced subgraph on vertices 0..19
        adj_20 = full_adj[:20]
        
        # Verify
        valid, _ = verify_graph(20, adj_20)
        if valid:
            best_n = 20
            best_edges = adj_masks_to_edges(20, adj_20)
            return {'n': 20, 'edges': best_edges}
        
        # If not valid, try local repair
        # Only do this if we have time
        if time.time() < deadline - 10:
            repaired_adj, is_valid = local_search_repair(20, adj_20, max_iters=5)
            if is_valid:
                best_n = 20
                best_edges = adj_masks_to_edges(20, repaired_adj)
                return {'n': 20, 'edges': best_edges}
                
    # If 20 fails, try 19
    # Use p=19 (Paley graph on 19 vertices is a single graph)
    if time.time() < deadline - 10:
        adj_19 = generate_paley_graph(19, 19)
        valid, _ = verify_graph(19, adj_19)
        if valid:
            best_n = 19
            best_edges = adj_masks_to_edges(19, adj_19)
            return {'n': 19, 'edges': best_edges}
            
        # Try repair on 19
        repaired_adj, is_valid = local_search_repair(19, adj_19, max_iters=5)
        if is_valid:
            best_n = 19
            best_edges = adj_masks_to_edges(19, repaired_adj)
            return {'n': 19, 'edges': best_edges}
            
    # Try random graphs with fast verification as a last resort for 19/20
    # This is slower than algebraic, but might find a solution if algebraic fails.
    # However, algebraic should be strong enough.
    
    # Fallback: Return best found or safe small graph
    if best_n == 5:
         # Try one random 19 graph quickly?
         if time.time() < deadline - 5:
             random.seed(42)
             adj_masks = [0]*19
             for i in range(19):
                 for j in range(i+1, 19):
                     if random.random() < 0.5:
                         adj_masks[i] |= (1 << j)
                         adj_masks[j] |= (1 << i)
             valid, _ = verify_graph(19, adj_masks)
             if valid:
                 best_n = 19
                 best_edges = adj_masks_to_edges(19, adj_masks)
                 return {'n': 19, 'edges': best_edges}
                 
    return {'n': best_n, 'edges': best_edges}

# No test block, just the function
