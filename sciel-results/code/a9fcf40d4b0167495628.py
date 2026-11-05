"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    """
    if q == 3 and n == 6 and radius == 1:
        return _cover_ternary_6_1_mis_based()
    else:
        return _generic_greedy_cover(q, n, radius)

def _cover_ternary_6_1_mis_based():
    """
    Solve K_3(6,1) by finding a Maximum Independent Set in the graph of words
    with Hamming distance <= 2, then covering the remaining space.
    """
    alphabet = range(3)
    all_words = list(itertools.product(alphabet, repeat=6))
    n_words = len(all_words)
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for the conflict graph (distance <= 2)
    # We need neighbors for MIS.
    # Conflict graph G: edge if dist <= 2.
    # Independent set in G: set of words with pairwise dist >= 3.
    
    # Precompute distance <= 2 sets
    # For efficiency, we generate neighbors on the fly or precompute bitsets.
    # n_words = 729. Bitset is 729 bits = 92 bytes. 729*92 ~ 67KB. Very small.
    
    # Let's use Python integers as bitsets for speed.
    # bit i is set if word i is in the set.
    
    neighbors = [0] * n_words
    # Precompute neighbors for dist 1 and 2
    for i, w in enumerate(all_words):
        mask = 1 << i
        neighbors[i] |= mask
        # Dist 1
        for pos in range(6):
            for val in alphabet:
                if val == w[pos]: continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                neighbors[i] |= (1 << j)
        # Dist 2
        for p1 in range(6):
            for p2 in range(p1+1, 6):
                for v1 in alphabet:
                    if v1 == w[p1]: continue
                    for v2 in alphabet:
                        if v2 == w[p2]: continue
                        new_w = list(w)
                        new_w[p1] = v1
                        new_w[p2] = v2
                        j = word_to_idx[tuple(new_w)]
                        neighbors[i] |= (1 << j)
                        
    # Greedy Coloring for Upper Bound in MIS
    # We color the subgraph induced by candidates.
    # The number of colors is an upper bound on the size of the MIS in that subgraph.
    # Actually, we want an upper bound on the independent set size.
    # In a graph, the independence number alpha(G) is the size of the largest independent set.
    # Coloring the *complement* graph gives a clique cover, which is an upper bound on alpha?
    # No.
    # Let's use the standard Tomita bound for Maximum Clique.
    # We are finding MIS in G. This is equivalent to Maximum Clique in G_bar.
    # Or we can run MIS directly.
    # Bound: If we color the vertices of the *candidate subgraph* such that adjacent vertices have different colors,
    # the number of colors is an upper bound on the *clique* size in that subgraph.
    # But we want Independent Set in G.
    # Independent Set in G is Clique in G_bar.
    # So we should color G_bar?
    # G_bar is dense.
    
    # Alternative: 
    # For MIS in G, we can use a greedy algorithm to select vertices.
    # A simple bound: The maximum degree in the candidate subgraph is D.
    # This is weak.
    
    # Let's use a standard Branch and Bound for MIS.
    # We maintain a set of candidates C.
    # We pick a vertex v.
    # Branch 1: Include v. New C = C \ N[v].
    # Branch 2: Exclude v. New C = C \ {v}.
    
    # To prune, we need an upper bound on the size of MIS in C.
    # One simple bound: The number of vertices in C. (Weak)
    # Better: A greedy coloring of the *complement* of the subgraph induced by C?
    # If we color the subgraph induced by C (in G), adjacent nodes get different colors.
    # This partitions C into independent sets.
    # The size of the largest independent set in C is at most the number of colors?
    # No.
    
    # Let's stick to a simpler heuristic:
    # Greedy MIS construction to get a lower bound (initial solution).
    # Then B&B to improve.
    
    # Initial Greedy MIS
    def greedy_mis(candidates_mask):
        # candidates_mask is a bitset of available vertices
        mis = []
        c = candidates_mask
        while c:
            # Pick a vertex with minimum degree in the remaining subgraph?
            # Or just pick the first one?
            # To be fast, pick a random one or first one.
            # Let's pick the one with the fewest neighbors in c to maximize future options.
            best_v = -1
            min_deg = n_words + 1
            # Iterate through set bits
            temp_c = c
            while temp_c:
                lsb = temp_c & -temp_c
                v = lsb.bit_length() - 1
                # Count neighbors of v in c
                deg = (neighbors[v] & c).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp_c &= temp_c - 1
                
            if best_v == -1: break
            
            mis.append(best_v)
            # Remove v and its neighbors from c
            c &= ~(neighbors[best_v])
            
        return mis

    # B&B for MIS
    best_mis = []
    best_size = 0
    
    def solve_mis(candidates_mask, current_mis):
        nonlocal best_mis, best_size
        
        if candidates_mask == 0:
            if len(current_mis) > best_size:
                best_size = len(current_mis)
                best_mis = list(current_mis)
            return
        
        # Upper bound: 
        # A simple bound is the number of bits in candidates_mask.
        # If len(current_mis) + candidates_mask.bit_count() <= best_size:
        #     return
        
        # Better bound: 
        # We can compute a greedy coloring of the subgraph induced by candidates_mask 
        # in the *complement* graph? 
        # Actually, let's just use the count for now, as n=729 is small.
        # We can add a time limit check.
        
        if len(current_mis) + candidates_mask.bit_count() <= best_size:
            return
            
        # Heuristic: Pick a vertex to branch on.
        # Pick a vertex with high degree in the subgraph?
        # Or use a random vertex to break symmetry?
        
        # Let's pick a vertex with the highest degree in the candidate subgraph 
        # to prune the "include" branch faster (removes more nodes).
        # Wait, if we include v, we remove N[v].
        # If we exclude v, we remove {v}.
        # Branching on high degree v makes the "include" branch shrink fast, 
        # but the "exclude" branch stays large.
        # Usually, we branch on a vertex that is "critical".
        
        # Let's just pick the first available vertex for simplicity, 
        # but order them by index to be deterministic?
        # To break symmetry, we can fix the first vertex if it's in the same orbit?
        # Given time, let's just run it.
        
        # Optimization: If we have found a solution of size 56 (theoretical max), we can stop?
        # Max packing size is floor(729/13) = 56.
        
        # Pick a vertex
        # To speed up, we can use a random restart or specific ordering.
        # Let's pick a vertex with the maximum number of neighbors in the candidate set.
        max_deg = -1
        branch_v = -1
        temp_c = candidates_mask
        # Sample a few vertices to find a good one? 
        # Or just iterate all (729 is small).
        while temp_c:
            lsb = temp_c & -temp_c
            v = lsb.bit_length() - 1
            deg = (neighbors[v] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                branch_v = v
            temp_c &= temp_c - 1
            
        if branch_v == -1:
            return
            
        v_mask = 1 << branch_v
        
        # Branch 1: Include branch_v
        # New candidates: candidates_mask without branch_v and its neighbors
        new_c1 = candidates_mask & ~neighbors[branch_v]
        current_mis.append(branch_v)
        solve_mis(new_c1, current_mis)
        current_mis.pop()
        
        # Branch 2: Exclude branch_v
        new_c2 = candidates_mask & ~v_mask
        solve_mis(new_c2, current_mis)

    # Initialize with a greedy solution
    initial_mis = greedy_mis((1 << n_words) - 1)
    best_size = len(initial_mis)
    best_mis = initial_mis
    
    # Run B&B
    start_time = time.time()
    # We might not finish the full B&B in 120s if it's not pruned well.
    # Let's run it with a time limit.
    
    # To make B&B effective, we need good upper bounds.
    # The simple count bound is weak.
    # Let's try to run it. If it's too slow, we fall back to the greedy result.
    
    # Note: The graph is highly symmetric. B&B might explore many symmetric branches.
    # We can enforce symmetry breaking:
    # If we decide to exclude a vertex, we can also exclude its orbit mates?
    # This is complex.
    
    # Let's just run the B&B and hope the greedy bound + small n works.
    # Or, we can use the greedy MIS as the final packing if B&B is too slow.
    
    # Given the 120s limit, and 729 nodes, a naive B&B might be too slow.
    # I will limit the B&B depth or time.
    
    # Let's try a different approach for the "MIS":
    # Use a greedy algorithm with random restarts to find a large independent set.
    # This is much faster and likely finds a near-optimal packing.
    
    def greedy_mis_random(candidates_mask, seed):
        random.seed(seed)
        mis = []
        c = candidates_mask
        while c:
            # Pick a random vertex from c
            # Convert mask to list of indices for random choice?
            # Or use random bit?
            # Let's pick a random index from the set bits.
            # Efficient way:
            # Generate a random number, map to a set bit?
            # Simpler: list of indices.
            indices = []
            temp_c = c
            while temp_c:
                lsb = temp_c & -temp_c
                v = lsb.bit_length() - 1
                indices.append(v)
                temp_c &= temp_c - 1
            
            if not indices: break
            
            # Choose a vertex. 
            # Strategy: Pick one that minimizes the degree in the remaining graph?
            # Or just pick randomly?
            # Let's pick the one with minimum degree to keep options open.
            best_v = indices[0]
            min_deg = n_words + 1
            for v in indices:
                deg = (neighbors[v] & c).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    # Random tie break?
                    if deg == min_deg and random.random() < 0.5:
                        best_v = v # This logic is flawed for random tie break, let's just pick min deg deterministically for now to ensure validity, or random among min degs.
            
            # Let's just pick the min degree vertex.
            # To allow diversity, we can randomize the order of checking?
            # For now, deterministic min-degree greedy.
            
            mis.append(best_v)
            c &= ~neighbors[best_v]
            
        return mis

    # Run multiple greedy restarts
    best_pack_size = 0
    best_pack = []
    
    # We can also try to improve the greedy solution by local search (swap vertices).
    
    for i in range(100): # 100 restarts
        if time.time() - start_time > 100: break
        pack = greedy_mis_random((1 << n_words) - 1, i)
        if len(pack) > best_pack_size:
            best_pack_size = len(pack)
            best_pack = pack
            # If we hit 56, stop
            if best_pack_size >= 56:
                break
                
    # Now we have a packing set best_pack.
    # Compute covered words
    covered_mask = 0
    for v in best_pack:
        # Covered by v are words with dist <= 1.
        # We precomputed neighbors for dist <= 2.
        # We need dist <= 1 set.
        # Let's precompute dist <= 1 sets.
        pass
        
    # Re-precompute dist <= 1 masks
    cover1_masks = [0] * n_words
    for i, w in enumerate(all_words):
        mask = 1 << i
        cover1_masks[i] |= mask
        for pos in range(6):
            for val in alphabet:
                if val == w[pos]: continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                cover1_masks[i] |= (1 << j)
                
    covered_mask = 0
    for v in best_pack:
        covered_mask |= cover1_masks[v]
        
    # Uncovered words
    full_mask = (1 << n_words) - 1
    uncovered_mask = full_mask & ~covered_mask
    
    # Greedy cover the uncovered words
    # We need to select words from the FULL space (not just uncovered) 
    # such that they cover the uncovered words.
    # Actually, we can add any word.
    
    # Greedy Set Cover for the remaining part
    # We want to minimize the number of added words.
    # Candidates: All words.
    # Target: uncovered_mask.
    
    added_words = []
    current_uncovered = uncovered_mask
    
    while current_uncovered:
        best_v = -1
        best_count = 0
        # Iterate through all words? Or just those that cover some uncovered words?
        # To be fast, iterate all.
        for i in range(n_words):
            # Count how many uncovered words i covers
            # Intersection of cover1_masks[i] and current_uncovered
            inter = cover1_masks[i] & current_uncovered
            count = inter.bit_count()
            if count > best_count:
                best_count = count
                best_v = i
                if best_count == current_uncovered.bit_count():
                    break
                    
        if best_v == -1:
            break
            
        added_words.append(best_v)
        current_uncovered &= ~cover1_masks[best_v]
        
    # Combine
    final_codewords = best_pack + added_words
    
    # Local Search to minimize
    # Remove redundancies
    # Check if any word in final_codewords is redundant.
    # A word w is redundant if all words it covers are also covered by other codewords.
    
    def is_redundant(idx, codewords_set):
        # codewords_set is a set of indices
        # Check if cover1_masks[idx] is a subset of union of cover1_masks[c] for c in codewords_set \ {idx}
        # This is expensive.
        # Alternative: 
        # For each word u in cover1_masks[idx], check if u is covered by some c in codewords_set \ {idx}.
        
        # Let's compute the union of covers of other codewords.
        # This is O(N^2) per check.
        
        others = [c for c in codewords_set if c != idx]
        union_cover = 0
        for c in others:
            union_cover |= cover1_masks[c]
            
        # Check if cover1_masks[idx] is subset of union_cover
        return (cover1_masks[idx] & ~union_cover) == 0

    # This is too slow for 73 words * 73 words * 729 bits.
    # Let's use a counting array approach.
    
    counts = np.zeros(n_words, dtype=int)
    for cw in final_codewords:
        # Add coverage
        # We need to add 1 to all bits in cover1_masks[cw]
        # This is slow in Python.
        # Let's use a list of lists for coverage.
        pass
        
    # Let's stick to a simpler local search:
    # Try to remove each word one by one.
    
    improved = True
    while improved:
        improved = False
        for i in range(len(final_codewords)):
            cw = final_codewords[i]
            # Check if removing cw leaves any word uncovered
            # Compute union of all other covers
            union_others = 0
            for j in range(len(final_codewords)):
                if j == i: continue
                union_others |= cover1_masks[final_codewords[j]]
                
            # If cw covers any word not in union_others, it's not redundant
            if (cover1_masks[cw] & ~union_others) != 0:
                continue
            else:
                # Redundant
                final_codewords.pop(i)
                improved = True
                break
                
    # Convert to list of lists
    result = [list(all_words[i]) for i in final_codewords]
    return result

def _generic_greedy_cover(q, n, radius):
    """Generic greedy set cover for arbitrary q, n, radius."""
    alphabet = range(q)
    all_words = list(itertools.product(alphabet, repeat=n))
    n_words = len(all_words)
    
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage sets
    covered_sets = []
    for w in all_words:
        covered = []
        for candidate in all_words:
            dist = sum(1 for a, b in zip(w, candidate) if a != b)
            if dist <= radius:
                covered.append(candidate)
        covered_sets.append(set(word_to_idx[c] for c in covered))
    
    # Greedy set cover
    uncovered = set(range(n_words))
    codewords = []
    
    while uncovered:
        best_word = -1
        best_count = -1
        for i in range(n_words):
            count = len(covered_sets[i] & uncovered)
            if count > best_count:
                best_count = count
                best_word = i
            if best_count == len(uncovered):
                break
        
        codewords.append(best_word)
        uncovered -= covered_sets[best_word]
    
    # Try to remove redundant codewords
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            temp_codewords = codewords[:i] + codewords[i+1:]
            covered = set()
            for cw in temp_codewords:
                covered |= covered_sets[cw]
            if covered == set(range(n_words)):
                codewords = temp_codewords
                improved = True
                break
    
    result = [list(all_words[i]) for i in codewords]
    return result
# EVOLVE-BLOCK-END
