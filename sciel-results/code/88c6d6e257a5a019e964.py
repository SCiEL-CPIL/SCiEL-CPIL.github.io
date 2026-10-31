"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill
    
    # Generate W4 vectors: weight-4 in {-1,0,1}^11, norm^2 = 4
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    vectors_v4 = np.array(v4_list)
    N4 = len(vectors_v4)
    
    # Generate W1 vectors: weight-1 with entry ±2, norm^2 = 4
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[pos] = 2 * sign
            v1_list.append(v)
    vectors_v1 = np.array(v1_list)
    N1 = len(vectors_v1)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict matrix: conflict if dot > 2 (since dist^2 = 4+4-2*dot >= 4 requires dot <= 2)
    dots = vectors_v4 @ vectors_v4.T
    conflicts = (dots > 2).astype(np.uint8)
    np.fill_diagonal(conflicts, 0)
    
    # Precompute conflict lists
    conflict_lists = [np.where(conflicts[i])[0] for i in range(N4)]
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    rng = np.random.default_rng(42)
    
    def greedy_mis(deadline, rng):
        """Greedy Maximum Independent Set with min-degree ordering."""
        alive = np.ones(N4, dtype=bool)
        deg = degrees.copy()
        selected = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            min_deg = np.min(deg[alive_idx])
            candidates = alive_idx[deg[alive_idx] == min_deg]
            chosen = candidates[rng.integers(len(candidates))]
            selected.append(chosen)
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                deg[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        
        return selected
    
    def kempe_chain_local_search(initial_set, deadline, rng):
        """
        Kempe-chain style local search: try to replace a subset of selected vectors
        with a larger compatible subset, escaping local optima.
        """
        current_set = list(initial_set)
        current_mask = np.zeros(N4, dtype=bool)
        for v in current_set:
            current_mask[v] = True
        best_set = list(current_set)
        best_size = len(current_set)
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve < max_no_improve:
            no_improve += 1
            n = len(current_set)
            if n < 2:
                break
            
            # Try 1-out, 2-in swap (Kempe chain of length 2)
            # Pick a random vertex to remove
            v_remove = current_set[rng.integers(n)]
            
            # Compute forbidden: neighbors of all other selected vertices
            other_mask = current_mask.copy()
            other_mask[v_remove] = False
            
            if np.any(other_mask):
                forbidden = conflicts[:, other_mask].any(axis=1).astype(bool)
            else:
                forbidden = np.zeros(N4, dtype=bool)
            
            # Candidates: not forbidden, not in current set
            candidates = ~forbidden & ~current_mask
            cand_idx = np.where(candidates)[0]
            
            if len(cand_idx) >= 2:
                # Try to find two compatible candidates
                # Sample a few pairs
                found_swap = False
                for attempt in range(20):
                    if time.time() > deadline:
                        break
                    i1 = rng.integers(len(cand_idx))
                    i2 = rng.integers(len(cand_idx))
                    if i1 == i2:
                        continue
                    u1 = cand_idx[i1]
                    u2 = cand_idx[i2]
                    if not conflicts[u1, u2]:
                        # Perform swap
                        current_set.remove(v_remove)
                        current_set.append(u1)
                        current_set.append(u2)
                        current_mask[v_remove] = False
                        current_mask[u1] = True
                        current_mask[u2] = True
                        no_improve = 0
                        found_swap = True
                        if len(current_set) > best_size:
                            best_set = list(current_set)
                            best_size = len(current_set)
                        break
                
                if not found_swap:
                    # Try 1-out, 1-in swap
                    if len(cand_idx) >= 1:
                        u1 = cand_idx[rng.integers(len(cand_idx))]
                        if rng.random() < 0.3:  # Occasionally do neutral swaps for exploration
                            current_set.remove(v_remove)
                            current_set.append(u1)
                            current_mask[v_remove] = False
                            current_mask[u1] = True
                            if len(current_set) > best_size:
                                best_set = list(current_set)
                                best_size = len(current_set)
            else:
                # Try adding a vertex if possible (improvement)
                add_forbidden = conflicts[:, current_mask].any(axis=1).astype(bool)
                add_cand = ~add_forbidden & ~current_mask
                add_idx = np.where(add_cand)[0]
                if len(add_idx) > 0:
                    v_add = add_idx[rng.integers(len(add_idx))]
                    current_set.append(v_add)
                    current_mask[v_add] = True
                    no_improve = 0
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_size = len(current_set)
        
        return best_set
    
    def spectral_greedy_init(rng):
        """Use spectral decomposition to get a good initial independent set."""
        # Build non-conflict adjacency
        non_conf = (1 - conflicts).astype(float)
        np.fill_diagonal(non_conf, 0)
        
        try:
            # Compute Fiedler vector of non-conflict graph
            deg_nc = non_conf.sum(axis=1)
            L = np.diag(deg_nc) - non_conf
            
            # Use sparse-friendly approach: just compute a few eigenvectors
            # For N4=4224, full eigh is feasible
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
            
            # Partition by sign
            pos = fiedler > 0
            neg = ~pos
            
            # Take larger partition, then greedy within it
            if np.sum(pos) >= np.sum(neg):
                partition = np.where(pos)[0]
            else:
                partition = np.where(neg)[0]
            
            # Greedy within partition
            alive_in_part = np.zeros(N4, dtype=bool)
            alive_in_part[partition] = True
            deg_in = conflicts[partition][:, partition].sum(axis=1).astype(np.int32)
            
            order = partition[np.argsort(deg_in)]
            selected = []
            
            for v in order:
                if time.time() > deadline - 30:
                    break
                if alive_in_part[v]:
                    selected.append(v)
                    neighbors = np.where(conflicts[v])[0]
                    alive_in_part[neighbors] = False
                    alive_in_part[v] = False
            
            return selected
        except Exception:
            return greedy_mis(deadline - 30, rng)
    
    # Phase 1: Spectral initialization
    spectral_deadline = start_time + 25
    best_set = spectral_greedy_init(rng)
    
    # Phase 2: Kempe-chain local search
    if len(best_set) > 0:
        ls_deadline = start_time + 70
        improved = kempe_chain_local_search(best_set, ls_deadline, rng)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Phase 3: Multiple greedy restarts with Kempe improvement
    greedy_deadline = deadline - 8
    n_restarts = 0
    
    while time.time() < greedy_deadline:
        # Quick greedy
        g_deadline = min(greedy_deadline, time.time() + 3)
        greedy_result = greedy_mis(g_deadline, rng)
        n_restarts += 1
        
        if len(greedy_result) > len(best_set):
            best_set = greedy_result
        
        # Improve with Kempe search
        if time.time() < deadline - 5:
            k_deadline = min(deadline - 5, time.time() + 12)
            k_result = kempe_chain_local_search(best_set, k_deadline, rng)
            if len(k_result) > len(best_set):
                best_set = k_result
        
        # Check if we should continue
        if time.time() > greedy_deadline:
            break
    
    if not best_set:
        return vectors_v1
    
    result_v4 = vectors_v4[best_set]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
