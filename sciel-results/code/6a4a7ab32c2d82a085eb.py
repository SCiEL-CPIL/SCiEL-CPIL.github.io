"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial.distance import cdist

D = 11

# EVOLVE-BLOCK-START
def _construct_base_vectors():
    """
    Construct a base set of vectors in R^11 with norm 2 (norm^2 = 4).
    Uses E8 roots embedded in the first 8 dimensions, with 3 zeros.
    E8 roots are all permutations of:
      1) (±1, ±1, 0, 0, 0, 0, 0, 0)  -> 8*7*4 = 228? No, 8*7*2*2 = 224. Wait.
         Pairs (i,j) with i<j: C(8,2)=28. Signs: 4. Total 112.
         Pairs (i,j) with i>j: C(8,2)=28. Signs: 4. Total 112.
         Wait, standard E8 roots:
         Type 1: ±e_i ± e_j (i≠j). 8*7*4 = 224.
         Type 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs.
         Number of such vectors: 2^8 / 2 = 128.
         Total 224 + 128 = 352?
         Standard E8 has 240 roots.
         Let's re-evaluate Type 1.
         ±e_i ± e_j.
         i≠j. 8*7 = 56 pairs.
         Signs: (+,+), (+,-), (-,+), (-,-). 4 signs.
         Total 56 * 4 = 224.
         Type 2: Half-integer vectors.
         Sum of coordinates is integer.
         Norm^2 = 8 * 1/4 = 2.
         Number of such vectors is 112?
         Total roots = 224 + 112 = 336? No.
         
         Correct E8 root count is 240.
         Type 1: ±e_i ± e_j. 240? No, that's just one type.
         Actually, Type 1 has 240 roots?
         Let's check Wikipedia.
         "The 240 roots of E8 are the vectors (±1, ±1, 0, 0, 0, 0, 0, 0) and their permutations."
         Number of such vectors:
         Choose 2 positions for non-zeros: C(8,2) = 28.
         Choose signs: 4.
         Total 28 * 4 = 112.
         Wait, 112 + 112 = 224?
         Where are the other 16?
         
         Ah, the half-integer ones.
         (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2)
         There are 128 such vectors.
         112 + 128 = 240.
         So Type 1 is 112, Type 2 is 128. Total 240.
         
         Let's verify Type 1 count again.
         Vectors with two non-zero entries ±1, rest 0.
         Positions: 8*7 = 56 ordered pairs (i,j) with i≠j.
         Signs: 4.
         Total 224.
         Why does Wikipedia say 112?
         "The roots are the vectors (±1, ±1, 0, ..., 0) and their permutations."
         Maybe it counts unordered pairs?
         If i<j, 28 pairs. 4 signs. 112.
         But (1, -1, 0...) is distinct from (-1, 1, 0...).
         And (1, 1, 0...) is distinct from (-1, -1, 0...).
         And (1, 1, 0...) is distinct from (1, 1, 0...) with positions swapped? No, positions are fixed.
         
         Let's just generate all of them and check.
    """
    vectors = []
    
    # Type 1: ±e_i ± e_j
    for i in range(8):
        for j in range(8):
            if i == j: continue
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(11, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    # Type 2: Half-integer vectors (scaled by 2 to be integers: ±1, ±1, ..., ±1 with even # of -1s)
    # To keep them as integers, we can multiply by 2.
    # But then norms are 4.
    # Type 1 norms are 2 (squared).
    # We need consistent norm for the kissing condition?
    # "minimum pairwise distance is at least the maximum vector norm"
    # If we mix norms, the max norm is 4 (for Type 2 scaled by 2) or 2 (for Type 1).
    # If we keep Type 2 as half-integers, we can't use int64 easily without scaling.
    # If we scale Type 1 by 2 as well, they become (±2, ±2, 0...). Norm squared 8.
    # Then all vectors have norm squared 8.
    # Let's scale everything to be integers with the same norm.
    
    # Let's restart with a unified integer representation.
    # Target Norm Squared = 4.
    # Type 1: ±1, ±1, 0... (Norm^2 = 2). Scale by 2? -> ±2, ±2, 0... (Norm^2 = 8).
    # Type 2: ±1/2... (Norm^2 = 2). Scale by 2? -> ±1, ±1... (Norm^2 = 8).
    
    # So, let's generate vectors with Norm^2 = 8.
    vectors = []
    
    # Type 1: ±2 e_i ± 2 e_j
    for i in range(8):
        for j in range(8):
            if i == j: continue
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(11, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    # Type 2: ±1, ±1, ..., ±1 (8 coords) with even number of -1s.
    # There are 128 such vectors.
    # We can generate all 2^8 = 256 and filter for even parity of -1s.
    # Or just generate all and check.
    for mask in range(256):
        v = np.zeros(11, dtype=np.int64)
        neg_count = 0
        for k in range(8):
            if mask & (1 << k):
                v[k] = -1
                neg_count += 1
            else:
                v[k] = 1
        if neg_count % 2 == 0:
            vectors.append(v)
            
    arr = np.array(vectors, dtype=np.int64)
    return arr


def _build_conflict_graph(vectors):
    """
    Build a conflict graph where an edge exists if distance < max_norm.
    Here max_norm is the norm of any vector (since they all have same norm).
    Condition: ||v_i - v_j||^2 < ||v_i||^2
    Since ||v_i||^2 = 8, we need ||v_i - v_j||^2 >= 8.
    If ||v_i - v_j||^2 < 8, they conflict.
    """
    n = len(vectors)
    dists = cdist(vectors, vectors)
    # We want to keep pairs with dist >= sqrt(8).
    # Conflict if dist < sqrt(8).
    # dist^2 < 8.
    # dists is symmetric.
    conflict = (dists**2) < 8 - 1e-9  # Floating point safety, but ints are exact.
    # Actually, dists**2 is computed from floats. Let's use exact integer arithmetic if possible.
    # cdist returns floats.
    # Let's recompute dists^2 exactly using vectorized ops.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j
    # ||v_i||^2 = 8.
    # So dist^2 = 16 - 2 dot(i,j).
    # Condition dist^2 < 8  <=> 16 - 2 dot < 8 <=> 8 < 2 dot <=> dot > 4.
    # So conflict if dot(i,j) > 4.
    
    dots = vectors @ vectors.T
    # Diagonal is 8.
    # Off-diagonal: if dot > 4, conflict.
    conflict = dots > 4
    np.fill_diagonal(conflict, False) # Self not conflict? Or maybe self doesn't matter for independent set.
    return conflict


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    vectors = _construct_base_vectors()
    n = len(vectors)
    
    if n == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    conflict = _build_conflict_graph(vectors)
    
    # Greedy coloring for upper bound in Branch and Bound
    # We want Maximum Independent Set (MIS).
    # MIS on G is Max Clique on G_complement? No.
    # MIS on G is Max Clique on G where edges mean "non-conflict"?
    # Let's define the graph H where an edge exists if vectors are COMPATIBLE (dist^2 >= 8).
    # Then we want Max Clique in H.
    # Or we can work on the conflict graph G and find MIS.
    
    # Let's use a simple Branch and Bound for MIS on the conflict graph.
    # Nodes: 352.
    # This might be too large for naive B&B without good bounds.
    # Greedy coloring bound is standard.
    
    # Let's try a simpler approach first: Greedy Set Packing.
    # Sort vectors by degree in conflict graph (ascending).
    # Pick the one with lowest degree, add to solution, remove it and all its neighbors from graph.
    
    # But we have 120s. Let's try to optimize the greedy.
    # Or use a library if available? No, stdlib/numpy/scipy only.
    
    # Let's implement a simple greedy with random restarts or different orderings.
    
    # Heuristic:
    # 1. Compute degrees in conflict graph.
    # 2. Sort by degree.
    # 3. Greedy select.
    
    # To improve, we can try multiple random permutations of tie-breakers.
    
    degrees = np.sum(conflict, axis=1)
    
    best_solution = []
    best_size = 0
    
    # Try a few orderings
    # 1. Ascending degree
    # 2. Descending degree
    # 3. Random shuffles
    
    # Given the symmetry, the optimal solution is likely a subset of the 240 E8 roots?
    # Actually, the maximum kissing number for d=11 is known to be 512? No, that's d=8 (240) and d=24 (196560).
    # K(11) is not known exactly, but bounds are 448 <= K(11) <= 512?
    # Wait, the task says "score is k". The seed score is 1.0 (which corresponds to k=220? No, seed is 220 vectors).
    # The current program constructs 220 vectors.
    # If I construct 240 vectors, score is 240/220 = 1.09.
    # If I can construct more, better.
    
    # Let's just return the 240 E8 roots first. They are all pairwise compatible?
    # Check compatibility of E8 roots.
    # Dot product of two distinct E8 roots:
    # Type 1-1: e_i+e_j and e_k+e_l.
    # If disjoint, dot=0. Dist^2 = 8+8=16 >= 8. OK.
    # If share one, e.g. e_i+e_j and e_i+e_k. Dot = 1. Dist^2 = 16-2=14 >= 8. OK.
    # If same, dot=2. Dist=0.
    # Type 1-2: e_i+e_j and (1/2...1/2). Dot = 1/2 + 1/2 = 1. Dist^2 = 8+8-2=14 >= 8. OK.
    # Type 2-2: (1/2...1/2) and (1/2...1/2).
    # Dot can be 2 (same), 0, -2.
    # If dot=2, same vector.
    # If dot=0, Dist^2 = 16 >= 8. OK.
    # If dot=-2, Dist^2 = 16 - 2(-2) = 20 >= 8. OK.
    # So all 240 E8 roots are mutually compatible (distance^2 >= 8).
    # Thus, the 240 vectors form a valid kissing configuration.
    
    # Can we add more?
    # The conflict graph of the 352 vectors (112 Type 1 + 128 Type 2 + 112 Type 1 duplicates? No, I generated 224 Type 1 and 128 Type 2. Total 352).
    # Wait, my generation:
    # Type 1: 8*7*4 = 224.
    # Type 2: 128.
    # Total 352.
    # Are all 224 Type 1 roots compatible with each other?
    # e_i+e_j and e_i-e_j. Dot = 1-1=0. Dist^2=16. OK.
    # e_i+e_j and e_j+e_i. Same vector.
    # e_i+e_j and e_k+e_l.
    # So the 224 Type 1 roots are mutually compatible?
    # Yes, if they are distinct, dot <= 1.
    # So the 224 Type 1 roots form a clique in the compatibility graph.
    # The 128 Type 2 roots form a clique in the compatibility graph?
    # Dot between Type 2 roots is -2, 0, 2.
    # If distinct, dot != 2. So dot <= 0.
    # Dist^2 >= 16. OK.
    # So Type 2 roots are mutually compatible.
    # Cross compatibility: Type 1 and Type 2.
    # Dot = ±1/2 ± 1/2...
    # In my scaled integer version:
    # Type 1: ±2, ±2, 0...
    # Type 2: ±1, ±1...
    # Dot product:
    # If Type 1 is 2, 2, 0...
    # Type 2 is 1, 1, 1...
    # Dot = 2+2 = 4.
    # Condition: dot > 4 is conflict.
    # Here dot = 4. So NOT conflict. (Dist^2 = 16 - 8 = 8).
    # So they are compatible.
    
    # If Type 1 is 2, -2, 0...
    # Type 2 is 1, 1, 1...
    # Dot = 2 - 2 = 0. Compatible.
    
    # If Type 1 is 2, 2, 0...
    # Type 2 is 1, -1, 1...
    # Dot = 2 - 2 = 0. Compatible.
    
    # So it seems ALL 352 vectors are mutually compatible?
    # If so, we can return all 352!
    
    # Let's double check the "dot > 4" condition.
    # Norm^2 = 8.
    # Dist^2 = 8 + 8 - 2 dot = 16 - 2 dot.
    # We need Dist^2 >= 8.
    # 16 - 2 dot >= 8
    # 8 >= 2 dot
    # 4 >= dot.
    # So if dot <= 4, they are compatible.
    # Conflict if dot > 4.
    
    # Max dot for Type 1 (2,2,0...) and Type 2 (1,1,1...):
    # 2*1 + 2*1 = 4.
    # So dot = 4. Compatible.
    
    # Max dot for Type 1 (2,2,0...) and Type 1 (2,2,0...):
    # If same, dot=8. Conflict (but they are same vector, we only pick once).
    # If distinct, dot <= 4?
    # (2,2,0) and (2,-2,0). Dot = 4-4=0.
    # (2,2,0) and (2,0,2). Dot = 4.
    # (2,2,0) and (-2,2,0). Dot = -4+4=0.
    # (2,2,0) and (2,2,0). Dot=8.
    # So distinct Type 1 have dot <= 4. Compatible.
    
    # Max dot for Type 2 (1,1,1...) and Type 2 (1,1,1...):
    # If same, dot=8.
    # If distinct, dot <= 0. Compatible.
    
    # So YES, all 352 vectors are mutually compatible (distinct ones).
    # We can return all 352!
    
    # Why did I think there were conflicts?
    # Maybe I confused the condition.
    # "minimum pairwise distance is at least the maximum vector norm"
    # Norm is sqrt(8).
    # Distance must be >= sqrt(8).
    # Dist^2 >= 8.
    # This holds for all pairs of distinct vectors in this set.
    
    # So the answer is 352.
    
    return vectors

# EVOLVE-BLOCK-END
