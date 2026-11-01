"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
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

    # Group W4 vectors by their support (tuple of nonzero positions)
    # For a given support, all 16 sign variants are mutually compatible
    # (dot product between any two distinct sign variants on same support is 0)
    support_to_indices = {}
    for i in range(N4):
        support = tuple(np.nonzero(vectors_v4[i])[0])
        if support not in support_to_indices:
            support_to_indices[support] = []
        support_to_indices[support].append(i)
    
    supports = list(support_to_indices.keys())
    n_supports = len(supports)
    
    # Build support-level conflict matrix:
    # Two supports conflict if they share >= 3 positions
    # (if they share 3 positions, there exist sign variants with dot >= 3 > 2)
    # If they share <= 2 positions, all sign variants are compatible
    support_conflicts = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            if len(si & sj) >= 3:
                support_conflicts[i, j] = True
                support_conflicts[j, i] = True
    
    # Each support contributes 16 vectors if selected
    vectors_per_support = 16
    
    # Also need to check cross-compatibility with W1 vectors
    # W1 vector at position p with value ±2:
    # Compatible with W4 vector if W4 has 0 at position p (dot=0) or |dot|<=2
    # W4 has entries ±1 at 4 positions, so dot with W1 at p is ±1 if p in support, else 0
    # |±1| = 1 <= 2, so all W4 vectors are compatible with all W1 vectors!
    # Great, no need to filter.
    
    # Now solve independent set on support graph
    # This is a much smaller problem: ~330 nodes vs 3328
    
    neighbors_sup = [np.where(support_conflicts[i])[0] for i in range(n_supports)]
    neighbor_counts_sup = np.array([len(n) for n in neighbors_sup], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_support_set = np.zeros(n_supports, dtype=bool)
    
    def greedy_min_degree_supports():
        """Greedy independent set with min-degree heuristic on support graph."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def sa_local_search_supports(in_set_init, max_time=10.0):
        """Improved SA with add/remove/swap moves on support graph."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 2.0
        cooling = 0.999
        min_T = 0.03
        
        # count_conflicts[s] = number of neighbors of s that are in in_set
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors_sup[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            move_type = rng.random()
            
            if move_type < 0.7:
                # Add move: pick a support not in set
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                    
                # Weighted selection: prefer supports with fewer conflicts
                counts = count_conflicts[not_in_set].astype(float)
                # Soft-max weighting: lower conflicts = higher probability
                weights = np.exp(-counts / max(T, 0.1))
                weights /= weights.sum()
                s = not_in_set[rng.choice(len(not_in_set), p=weights)]
                
                k = count_conflicts[s]
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
                if accept:
                    to_remove = np.where(support_conflicts[s] & in_set)[0]
                    
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors_sup[u]] -= 1
                    
                    in_set[s] = True
                    current_size += delta
                    count_conflicts[neighbors_sup[s]] += 1
            elif move_type < 0.9 and current_size > 0:
                # Remove move: pick a support in set to remove
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    continue
                    
                s = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Removing s frees up its neighbors
                delta = -1  # We lose 1, but may gain later
                # Actually, removing always decreases size by 1
                # Only accept if it helps overall (rarely)
                accept = rng.random() < np.exp(-1.0 / T)
                
                if accept:
                    in_set[s] = False
                    current_size -= 1
                    count_conflicts[neighbors_sup[s]] -= 1
            else:
                # Swap move: remove one in-set support, add one not-in-set
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                if len(in_set_indices) == 0 or len(not_in_set) == 0:
                    continue
                
                s_remove = in_set_indices[rng.integers(len(in_set_indices))]
                s_add = not_in_set[rng.integers(len(not_in_set))]
                
                # Check if s_add conflicts with (in_set - s_remove)
                in_set_minus = in_set.copy()
                in_set_minus[s_remove] = False
                conflicts_after = np.sum(support_conflicts[s_add] & in_set_minus)
                
                delta = 1 - 1 - conflicts_after  # +1 for add, -1 for remove, -conflicts for removals
                # Wait: removing s_remove doesn't directly cause removals unless s_add conflicts with s_remove's neighbors
                # Let me simplify: just compute net change
                # Remove s_remove: size -1
                # Add s_add: if no conflicts with remaining, size +1, else need to remove conflicting ones
                # This is complex; let's just use add move for now
                continue
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1
            
            if stuck_counter > 2000:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts with varied strategies
    n_restarts = 0
    max_restarts = 300
    
    def greedy_max_degree_supports():
        """Greedy independent set with max-degree heuristic (opposite of min)."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            max_deg = np.max(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == max_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def random_init_supports(density=0.15):
        """Random initialization with given density, then repair conflicts."""
        in_set = np.zeros(n_supports, dtype=bool)
        candidates = rng.integers(0, n_supports, size=int(n_supports * density * 3))
        
        for s in candidates:
            if not in_set[s] and not np.any(support_conflicts[s] & in_set):
                in_set[s] = True
        
        return in_set
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Choose initialization strategy
        strategy = n_restarts % 3
        if strategy == 0:
            in_set = greedy_min_degree_supports()
        elif strategy == 1:
            in_set = greedy_max_degree_supports()
        else:
            in_set = random_init_supports()
            
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()
        
        # SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()
        
        # Final greedy polish: try to add any remaining non-conflicting supports
        if time.time() < deadline - 3:
            polish_set = best_support_set.copy()
            polished = True
            while polished and time.time() < deadline - 3:
                polished = False
                not_in = np.where(~polish_set)[0]
                if len(not_in) == 0:
                    break
                # Find supports with zero conflicts with current set
                for s in not_in:
                    if not np.any(support_conflicts[s] & polish_set):
                        polish_set[s] = True
                        polished = True
                        if polish_set.sum() > best_size:
                            best_size = polish_set.sum()
                            best_support_set = polish_set.copy()
    
    # Build final vector list from selected supports
    selected_support_indices = np.where(best_support_set)[0]
    final_vectors = []
    
    for si in selected_support_indices:
        support = supports[si]
        for idx in support_to_indices[support]:
            final_vectors.append(vectors_v4[idx])
    
    # Add all W1 vectors (22 of them)
    final_vectors.extend(v1_candidates)
    
    result = np.array(final_vectors, dtype=np.int64)
    
    # Fallback: if something went wrong, use deterministic construction
    if len(result) == 0:
        # Deterministic fallback from inspiration 1
        all_supports = list(itertools.combinations(range(D), 4))
        selected_supports = []
        for s in all_supports:
            valid = True
            for t in selected_supports:
                intersection = set(s).intersection(t)
                if len(intersection) > 2:
                    valid = False
                    break
            if valid:
                selected_supports.append(s)
                if len(selected_supports) >= 32:
                    break
        
        fallback_vectors = []
        for s in selected_supports:
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for pos, sign in zip(s, signs):
                    v[pos] = sign
                fallback_vectors.append(v)
        
        # Add W1 vectors
        for pos in range(D):
            for sign in [1, -1]:
                v = [0] * D
                v[pos] = 2 * sign
                fallback_vectors.append(v)
        
        result = np.array(fallback_vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
