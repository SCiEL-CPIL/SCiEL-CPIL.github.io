import time
from math import gcd
import sys

def construct():
    # Safety margin: Evaluator kills at 90s. We aim to finish well before.
    deadline = time.time() + 85
    n = 255
    
    # Candidates: numbers in [1, 255] with gcd(v, 255) > 1
    # 255 = 3 * 5 * 17
    # So v must be divisible by 3, 5, or 17
    candidates = [v for v in range(1, n + 1) if v % 3 == 0 or v % 5 == 0 or v % 17 == 0]
    m = len(candidates)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = val_to_idx[255]
    
    # Build adjacency matrix using bitsets (Python integers)
    # adj_bits[i] is an integer where bit j is 1 if candidates[i] and candidates[j] are adjacent (gcd > 1)
    adj_bits = [0] * m
    for i in range(m):
        vi = candidates[i]
        for j in range(i + 1, m):
            vj = candidates[j]
            if gcd(vi, vj) > 1:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    # Initialize best solution with the largest single-prime multiple set
    # Multiples of 3: 85 elements
    # Multiples of 5: 51 elements
    # Multiples of 17: 15 elements
    # We start with multiples of 3 as it's the largest.
    best_bits = 0
    best_sz = 0
    
    # Set of indices for multiples of 3
    mult3_indices = [i for i, v in enumerate(candidates) if v % 3 == 0]
    for i in mult3_indices:
        best_bits |= (1 << i)
    best_sz = len(mult3_indices)
    
    # Ensure 255 is in the best set (it is, since 255 % 3 == 0)
    
    sys.setrecursionlimit(10000)
    
    # Greedy Coloring Upper Bound
    # Returns the number of colors used in a greedy coloring of the subgraph induced by P_bits
    # This is an upper bound on the clique size in P_bits
    def greedy_coloring(P_bits):
        # We want to color the graph induced by P_bits.
        # Vertices are the bits set in P_bits.
        # We process vertices in some order (e.g., degree order or just arbitrary) 
        # and assign the smallest available color.
        # To make the bound tight, we should order vertices by degree in P_bits descending.
        # However, sorting at every node is expensive. A common heuristic is to just iterate 
        # and use the current adjacency structure. 
        # A simpler but effective bound: 
        # The size of a maximal independent set is a lower bound on coloring, 
        # but we need an upper bound on clique, which is the chromatic number.
        # Greedy coloring gives an upper bound on chromatic number, thus an upper bound on clique size?
        # No, Chromatic number >= Clique size. So Greedy Colors >= Chromatic Number >= Clique Size.
        # So the number of colors is a valid upper bound.
        
        # To make it fast, we just iterate through vertices in P_bits and assign colors.
        # We maintain a list of colors for each vertex.
        # Actually, for the bound, we just need to count how many colors are needed.
        
        # Optimization: If P_bits is small, just do it.
        # We'll use a simple greedy: 
        # 1. Get list of vertices in P_bits
        # 2. Sort by degree in P_bits (optional, might be slow)
        # 3. Assign colors
        
        # For performance in tight loops, we can use a fixed order or just the bit order.
        # Let's use the bit order for simplicity and speed, which is still a valid upper bound.
        
        colors = {}
        color_sets = [] # color_sets[c] is a bitset of vertices with color c
        max_color = -1
        
        P_tmp = P_bits
        while P_tmp:
            # Get lowest set bit
            v_bit = P_tmp & (-P_tmp)
            v = v_bit.bit_length() - 1
            P_tmp ^= v_bit
            
            # Find the smallest color not used by neighbors of v
            # Neighbors of v in P_bits: adj_bits[v] & P_bits
            neigh = adj_bits[v] & P_bits
            
            # Check colors 0, 1, 2...
            c = 0
            while c <= max_color:
                # If any neighbor has color c, we can't use c
                if color_sets[c] & neigh:
                    c += 1
                else:
                    break
            
            if c > max_color:
                color_sets.append(0)
                max_color = c
            
            color_sets[c] |= v_bit
            
        return max_color + 1
    
    def bk(R_bits, P_bits, X_bits):
        nonlocal best_bits, best_sz
        
        if time.time() > deadline - 0.5:
            return
        
        # Pruning: If |R| + upper_bound(P) <= best_sz, no need to explore
        # We compute the upper bound on the size of any clique extendable from R using vertices in P.
        # The max clique size in P is at most the chromatic number of G[P].
        ub = greedy_coloring(P_bits)
        if bin(R_bits).count('1') + ub <= best_sz:
            return
        
        if P_bits == 0 and X_bits == 0:
            r_size = bin(R_bits).count('1')
            if r_size > best_sz:
                best_sz = r_size
                best_bits = R_bits
            return
        
        # Pivot selection: choose vertex u in P with max degree in P
        if P_bits:
            P_tmp = P_bits
            u = -1
            max_deg = -1
            while P_tmp:
                v_bit = P_tmp & (-P_tmp)
                v = v_bit.bit_length() - 1
                P_tmp ^= v_bit
                deg = bin(adj_bits[v] & P_bits).count('1')
                if deg > max_deg:
                    max_deg = deg
                    u = v
            # We only need to consider vertices in P that are NOT adjacent to u
            # Because any clique containing u must be a subset of N(u), 
            # and any clique not containing u must be in P \ N(u) ?? 
            # Actually, Bron-Kerbosch with pivot iterates over P \ N(u).
            # Why? Because if a clique C is in P, either it contains u (then C \ {u} is in P \ N(u) ?? No)
            # Standard BK: 
            # If P is empty, return.
            # Choose u in P U X with max degree.
            # For each v in P \ N(u):
            #    BK(R U {v}, P \ N(v), X \ N(v))
            #    P = P \ {v}
            #    X = X U {v}
            # This works because any maximal clique must contain a vertex from P \ N(u).
            # Proof: If a maximal clique C contains no vertex from P \ N(u), 
            # then all vertices in C are in N(u). Since u is in P U X, 
            # if u in P, C U {u} is a larger clique, contradiction to maximality in P.
            # If u in X, C is not maximal in the original graph? 
            # The standard argument holds for finding all maximal cliques.
            # For maximum clique, we just need to search these branches.
            
            Q_bits = P_bits & ~(adj_bits[u])
        else:
            Q_bits = 0
        
        Q_tmp = Q_bits
        while Q_tmp:
            v_bit = Q_tmp & (-Q_tmp)
            v = v_bit.bit_length() - 1
            Q_tmp ^= v_bit
            
            bk(R_bits | v_bit, P_bits & adj_bits[v], X_bits & adj_bits[v])
            P_bits &= ~v_bit
            X_bits |= v_bit
            
            # Check time again in the loop
            if time.time() > deadline - 0.5:
                return
    
    # Start search
    # R must contain 255
    # P is neighbors of 255
    # X is empty
    P0 = adj_bits[idx_255]
    bk(1 << idx_255, P0, 0)
    
    # Convert best_bits back to list of values
    result = []
    for i in range(m):
        if (best_bits >> i) & 1:
            result.append(candidates[i])
    
    result.sort()
    
    # Final verification
    if not result:
        return [255] # Fallback, though best_sz should be > 0
        
    assert 255 in result
    assert all(1 <= x <= 255 for x in result)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) == 1:
                # This should not happen if BK is correct
                # But if it does, fallback to multiples of 3
                fallback = [v for v in range(1, 256) if v % 3 == 0]
                return fallback
    return result
