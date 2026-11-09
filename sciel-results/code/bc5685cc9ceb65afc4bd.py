import numpy as np
import itertools
import time

D = 11

def generate_q2_vectors():
    """Generate all 220 weight-2 vectors (norm squared 2)."""
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i], v[j] = si, sj
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def generate_q4_core_vectors():
    """Generate 5280 weight-4 vectors with 4 non-zeros (norm squared 4)."""
    vectors = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0]*D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def generate_q4_sat_vectors():
    """Generate 22 vectors of form 2*e_i (norm squared 4)."""
    vectors = []
    for i in range(D):
        for s in (2, -2):
            v = [0]*D
            v[i] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def find_max_clique(candidates, deadline):
    """
    Find a large clique in the compatibility graph of candidates.
    Compatibility: dot(u, v) <= 2.
    """
    N = len(candidates)
    if N == 0:
        return []
    
    # Compute dot products
    M = candidates @ candidates.T
    
    # Threshold for Q=4 is 2.0
    threshold = 2.0
    
    # Build adjacency matrix (boolean)
    # A[i,j] = True if i != j and M[i,j] <= threshold
    A = (M <= threshold)
    np.fill_diagonal(A, False)
    
    # Greedy Max Clique with random restarts
    best_clique = []
    best_size = 0
    
    # Use a few random restarts to escape local optima
    for _ in range(5):
        if time.time() > deadline:
            break
            
        # Randomize order slightly or use different heuristic
        # We'll use degree-based but shuffle ties or add noise
        degrees = np.sum(A, axis=1)
        # Add small random noise to break ties
        noise = np.random.uniform(0, 1, size=N)
        order = np.argsort(-(degrees + noise))
        
        remaining = set(range(N))
        clique = []
        
        for v in order:
            if time.time() > deadline:
                break
            if v not in remaining:
                continue
            
            clique.append(v)
            
            # Remove v and its non-neighbors (conflicts)
            non_neigh = np.where(~A[v])[0]
            remaining.difference_update(non_neigh)
            
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique

    return best_clique

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s margin
    
    # 1. Baseline: Q=2 (220 vectors)
    q2_vectors = generate_q2_vectors()
    best_vectors = q2_vectors
    best_score = len(q2_vectors)
    
    # 2. Try Q=4 Core + Satellites + Q=2
    if time.time() < deadline:
        core = generate_q4_core_vectors()
        sat = generate_q4_sat_vectors()
        
        # Find Q4 clique
        clique_indices = find_max_clique(core, deadline)
        
        if len(clique_indices) > 0:
            core_clique = core[clique_indices]
            
            # Try to add Satellites
            # Satellites are compatible with Q4 if dot <= 2.
            # dot(sat, q4) is -2, 0, or 2. All <= 2.
            # So all satellites are compatible with ANY Q4 vector.
            # But we must check compatibility between satellites?
            # sat_i = 2e_i, sat_j = 2e_j (i!=j). dot = 0. OK.
            # So we can always add all 22 satellites.
            
            combined = np.vstack([core_clique, sat])
            
            # Try to add Q2 vectors
            # Conditions for q2 vector q:
            # 1. dot(q, q') <= 0 for all q' in selected Q2
            # 2. dot(q, u) <= 1 for all u in core_clique
            # 3. dot(q, s) <= 1 for all s in sat
            
            # Precompute dot products of q2 with core_clique and sat
            if len(core_clique) > 0:
                M_q2_core = q2_vectors @ core_clique.T
                # Condition: all values <= 1
                # If any > 1, q2 is incompatible with core_clique
                compatible_with_core = np.all(M_q2_core <= 1, axis=1)
                
                M_q2_sat = q2_vectors @ sat.T
                compatible_with_sat = np.all(M_q2_sat <= 1, axis=1)
                
                candidate_q2_mask = compatible_with_core & compatible_with_sat
                candidate_q2_indices = np.where(candidate_q2_mask)[0]
                
                if len(candidate_q2_indices) > 0:
                    candidate_q2 = q2_vectors[candidate_q2_indices]
                    
                    # Now select a subset of candidate_q2 that are mutually compatible (dot <= 0)
                    # Build compatibility graph for these Q2 vectors
                    M_q2_q2 = candidate_q2 @ candidate_q2.T
                    A_q2 = (M_q2_q2 <= 0)
                    np.fill_diagonal(A_q2, False)
                    
                    # Greedy selection for Q2
                    degrees_q2 = np.sum(A_q2, axis=1)
                    order_q2 = np.argsort(-degrees_q2)
                    
                    remaining_q2 = set(range(len(candidate_q2)))
                    selected_q2 = []
                    
                    for v in order_q2:
                        if time.time() > deadline:
                            break
                        if v not in remaining_q2:
                            continue
                        
                        selected_q2.append(v)
                        
                        non_neigh_q2 = np.where(~A_q2[v])[0]
                        remaining_q2.difference_update(non_neigh_q2)
                    
                    if len(selected_q2) > 0:
                        q2_subset = candidate_q2[selected_q2]
                        combined = np.vstack([combined, q2_subset])
            
            if len(combined) > best_score:
                best_score = len(combined)
                best_vectors = combined

    if len(best_vectors) == 0:
        # Fallback to seed (should not happen)
        return generate_q2_vectors()
        
    return best_vectors
