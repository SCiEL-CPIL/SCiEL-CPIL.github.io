"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin before 150s kill

    # --- Generate W4 vectors ---
    # Weight 4, entries in {-1, 0, 1}, norm^2 = 4
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # --- Generate W1 vectors ---
    # Weight 1, entries in {-2, 0, 2}, norm^2 = 4
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # --- Identify Conflicts ---
    # Two W4 vectors conflict if their distance is < 2 (i.e., dist^2 < 4).
    # Since they are integer vectors, dist^2 is an integer.
    # dist^2 < 4 implies dist^2 = 2 (since dist^2 >= 1, and distinct vectors have dist^2 >= 2 for W4).
    # Actually, min dist^2 for distinct W4 is 2.
    # So we conflict if dist^2 == 2.
    
    # Efficiently find conflicts:
    # v, w conflict if they differ in exactly 2 positions:
    # one position i where v[i]=1, w[i]=0
    # one position j where v[j]=0, w[j]=1
    # (and same signs elsewhere)
    
    # We can build a dictionary mapping vector tuples to indices
    vec_to_idx = {}
    for i in range(N4):
        vec_to_idx[tuple(vectors_v4[i])] = i

    # Build conflict list
    # For each vector, generate its 28 neighbors and check if they exist
    conflict_edges = []
    for i in range(N4):
        v = vectors_v4[i]
        non_zero_idx = np.nonzero(v)[0]
        zero_idx = np.where(v == 0)[0]
        
        for i_pos in non_zero_idx:
            for j_pos in zero_idx:
                # Create neighbor: set i_pos to 0, j_pos to 1 (keeping sign of v[i_pos])
                # Wait, the sign at j_pos can be +1 or -1?
                # If v[i]=1, w[i]=0.
                # If v[j]=0, w[j]=1.
                # The vector w must have weight 4.
                # If we change v[i] to 0 and v[j] to s (s in {1,-1}), does it remain weight 4?
                # Yes, if we don't touch other positions.
                # But w must be in the candidate list.
                # The candidate list contains ALL sign variants.
                # So for each pair (i_pos, j_pos), there are 2 possible neighbors (sign of j_pos can be 1 or -1? No, the sign is determined by the target vector).
                # Actually, we just need to check if the resulting vector is in the set.
                
                for sign_j in [1, -1]:
                    w = v.copy()
                    # Zero out i_pos
                    w[i_pos] = 0
                    # Set j_pos to sign_j
                    w[j_pos] = sign_j
                    # Check if w is in vec_to_idx
                    key = tuple(w)
                    if key in vec_to_idx:
                        j = vec_to_idx[key]
                        if i < j: # Avoid duplicates
                            conflict_edges.append((i, j))

    # --- Solve ILP ---
    # Maximize sum(x_i)
    # Subject to x_i + x_j <= 1 for all (i, j) in conflict_edges
    # x_i in {0, 1}
    
    c = np.ones(N4) # Objective: maximize sum x_i
    
    # Constraints: x_i + x_j <= 1
    # A_ub @ x <= b_ub
    # A_ub is sparse matrix
    if len(conflict_edges) > 0:
        rows = []
        cols = []
        for u, v in conflict_edges:
            rows.append(u)
            cols.append(v)
            rows.append(v)
            cols.append(u)
        
        A_ub = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(conflict_edges), N4))
        b_ub = np.ones(len(conflict_edges))
        
        constraints = LinearConstraint(A_ub, ub=b_ub)
    else:
        constraints = LinearConstraint(np.zeros((0, N4)), ub=np.zeros(0))
    
    integrality = np.ones(N4) # All variables are binary (0 or 1)
    # milp requires integrality to be 1 for integer, 0 for continuous.
    # For binary, we use bounds 0,1 and integrality 1.
    
    bounds = Bounds(lb=0, ub=1)
    
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': 60, 'presolve': True}
        )
        
        if result.status == 0 or result.status == 2: # Optimal or Feasible
            x = result.x
            # Rounding might be needed if not exactly binary, but integrality=1 should ensure it
            x = (x > 0.5).astype(np.int64)
            selected_indices = np.where(x == 1)[0]
        else:
            # Fallback to heuristic if ILP fails
            selected_indices = heuristic_fallback(vectors_v4, conflict_edges, deadline)
            
    except Exception as e:
        print(f"ILP failed: {e}")
        selected_indices = heuristic_fallback(vectors_v4, conflict_edges, deadline)

    # --- Construct Final Vectors ---
    if len(selected_indices) > 0:
        result_v4 = vectors_v4[selected_indices]
        final_vectors = np.vstack([result_v4, vectors_v1])
    else:
        final_vectors = vectors_v1

    return final_vectors

def heuristic_fallback(vectors_v4, conflict_edges, deadline):
    # Simple greedy fallback
    N = len(vectors_v4)
    in_set = np.zeros(N, dtype=bool)
    
    # Build adjacency list for fast neighbor lookup
    neighbors = [[] for _ in range(N)]
    for u, v in conflict_edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    # Greedy: pick vertex with min degree, add to set, remove neighbors
    degrees = np.array([len(n) for n in neighbors])
    alive = np.ones(N, dtype=bool)
    
    while np.any(alive) and time.time() < deadline - 5:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0: break
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[0]
        
        in_set[chosen] = True
        nbrs = neighbors[chosen]
        to_remove = np.unique(np.concatenate(([chosen], nbrs)))
        alive[to_remove] = False
        
        rem = np.where(alive)[0]
        if len(rem) > 0:
            # Update degrees
            for u in to_remove:
                for w in neighbors[u]:
                    if alive[w]:
                        degrees[w] -= 1

    return np.where(in_set)[0]

# EVOLVE-BLOCK-END
